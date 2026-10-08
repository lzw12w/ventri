"""Context: the API surface a plugin sees. One Context per fiber; ``ctx.parent`` walks up."""
from __future__ import annotations

import bisect
import inspect
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar, overload

import anyio

from .errors import ServiceConflict, ServiceNotFound
from .events import Deny, Event, Rewrite, event_name
from .fiber import LIVE, maybe_await
from .plugin import MISSING, keyname

if TYPE_CHECKING:  # pragma: no cover
    from .fiber import Fiber, TaskHandle
    from .kernel import Kernel
    from .transaction import Transaction

T = TypeVar("T")
_MISSING = MISSING


class Context:
    def __init__(self, kernel: Kernel, fiber: Fiber) -> None:
        self.kernel = kernel
        self.fiber = fiber

    def __repr__(self) -> str:
        return f"<Context of {self.fiber.label}>"

    @property
    def parent(self) -> Context | None:
        return self.fiber.parent.ctx if self.fiber.parent else None

    # ---------------------------------------------------------------- plugins
    async def plugin(self, plugin: Any, config: Any = None, *, meta: dict | None = None,
                     timeout: Any = _MISSING, retry: Any = _MISSING) -> Fiber:
        """Load ``plugin`` as a child of this context's fiber.

        Returns the fiber; it is ACTIVE, PENDING (deps missing; see
        ``fiber.pending_reason``) or FAILED (see ``fiber.error``; a load that exceeds
        the load timeout fails with LoadTimeout). ``timeout`` / ``retry`` override
        the plugin's metadata for this fiber. If the *caller* is cancelled mid-load
        the new fiber is disposed and nothing it registered survives.
        """
        from .fiber import Fiber

        if self.fiber.state not in LIVE:
            raise RuntimeError(f"cannot load plugins under {self.fiber!r}")
        return await self._load(Fiber(self.kernel, self.fiber, plugin, config, meta=meta,
                                      timeout=timeout, retry=retry))

    async def scope(self, name: str, isolate: Iterable[Any] = (), *,
                    meta: dict | None = None) -> Fiber:
        """Create a child *scope*: a container fiber that is the realm boundary for
        the keys in ``isolate`` and an event boundary.

        Inside the scope, ``provide``/``get`` of an isolated key use the scope's own
        realm; other keys resolve to the nearest ancestor realm isolating them, else
        the root realm. Sibling scopes never see each other's isolated services or
        events. ``await scope.dispose()`` reclaims everything inside (fibers, tasks,
        listeners, services) deterministically. Use ``scope.ctx`` to load plugins
        into it.
        """
        from .fiber import Fiber

        if self.fiber.state not in LIVE:
            raise RuntimeError(f"cannot create a scope under {self.fiber!r}")
        f = Fiber(self.kernel, self.fiber, ScopePlugin(name), None, scope=True,
                  isolate=frozenset(isolate), meta=meta)
        return await self._load(f)

    async def _load(self, fiber: Fiber) -> Fiber:
        async with self.kernel._op(fiber.tx):
            try:
                await fiber._activate()
            except BaseException:
                await fiber._dispose()  # shielded internally
                raise
        return fiber

    # --------------------------------------------------------------- services
    def provide(self, key: Any, value: Any, *, name: str | None = None) -> Any:
        """Bind ``value`` under ``key`` (a class or a string) for this fiber's lifetime.

        ``name`` additionally enables ``ctx.<name>`` attribute access (string keys
        get it automatically). Inside a transaction the binding is staged and only
        visible to staged fibers until commit.
        """
        from .kernel import Binding

        k, f = self.kernel, self.fiber
        tx = f.tx
        realm = k._realm_of(f, key)
        if k._lookup(realm, key, tx) is not None:
            raise ServiceConflict(f"{keyname(key)} is already provided")
        b = Binding(key, value, f, name or (key if isinstance(key, str) else None), tx, realm)
        f._push_effect(lambda: k._unbind(b), f"service:{keyname(key)}")
        if tx is not None:
            tx._overlay[(realm, key)] = b
        else:
            realm.services[key] = b
        f._bindings.append(b)
        f._can_provide(key)
        k._trace("service.bind", f, key=keyname(key), staged=tx is not None,
                 **({"realm": realm.name} if realm is not k._root_realm else {}))
        k._mark_dirty(realm, tx)
        return value

    @overload
    def get(self, key: type[T]) -> T: ...
    @overload
    def get(self, key: type[T], default: Any) -> T | Any: ...
    @overload
    def get(self, key: str, default: Any = ...) -> Any: ...

    def get(self, key: Any, default: Any = _MISSING) -> Any:
        """Injected keys resolve to the binding the fiber was activated with (a stable
        per-fiber view, even while a replacement is being swapped in); other keys
        resolve against the registry visible to this fiber."""
        f = self.fiber
        b = f._snapshot.get(key)
        if b is None:
            k = self.kernel
            b = k._lookup(k._realm_of(f, key), key, f.tx)
        if b is None:
            if default is _MISSING:
                raise ServiceNotFound(keyname(key))
            return default
        return b.value

    def has(self, key: Any) -> bool:
        k, f = self.kernel, self.fiber
        return k._lookup(k._realm_of(f, key), key, f.tx) is not None

    def __getattr__(self, name: str) -> Any:
        d = self.__dict__
        if name.startswith("_") or "kernel" not in d:
            raise AttributeError(name)
        b = d["kernel"]._find_by_name(d["fiber"], name)
        if b is None:
            raise AttributeError(f"{type(self).__name__} has no attribute or service {name!r}")
        return b.value

    # ---------------------------------------------------------------- effects
    def effect(self, setup: Callable[[], Any]) -> Callable[[], Any]:
        """``setup()`` runs now; if it returns a callable it is run (LIFO) at teardown.
        Returns an async disposer that runs it early."""
        f = self.fiber
        cleanup = setup()
        if not callable(cleanup):
            async def noop() -> None: ...
            return noop
        eff = f._push_effect(cleanup, getattr(setup, "__name__", "effect"))

        async def dispose() -> None:
            if eff in f._effects:
                f._effects.remove(eff)
                await f._run_effect(eff)
        return dispose

    def on_dispose(self, callback: Callable[[], Any]) -> None:
        self.fiber._push_effect(callback, getattr(callback, "__name__", "on_dispose"))

    async def enter(self, cm: Any) -> Any:
        """Enter an (async) context manager now; exit it at teardown."""
        if hasattr(cm, "__aenter__"):
            value = await cm.__aenter__()
            self.fiber._push_effect(lambda: cm.__aexit__(None, None, None), type(cm).__name__)
        else:
            value = cm.__enter__()
            self.fiber._push_effect(lambda: cm.__exit__(None, None, None), type(cm).__name__)
        return value

    def spawn(self, fn: Callable[..., Any], *args: Any, name: str | None = None) -> TaskHandle:
        """Run ``fn(*args)`` in this fiber's task group; cancelled when the fiber unloads.
        An exception in the task marks the fiber FAILED (and tears it down)."""
        return self.fiber.spawn(fn, *args, name=name)

    # ----------------------------------------------------------------- events
    @overload
    def on(self, event: Event[T], handler: Callable[[T], Any], *,
           priority: int = 0) -> Callable[[], None]: ...
    @overload
    def on(self, event: str, handler: Callable[..., Any], *,
           priority: int = 0) -> Callable[[], None]: ...

    def on(self, event: Any, handler: Callable[..., Any], *, priority: int = 0) -> Callable[[], None]:
        """Listen to ``event`` (an ``Event[T]`` key or a string) for this fiber's
        lifetime. Listeners run by ``priority`` (higher first), then registration
        order. Returns a function that removes the listener early."""
        return self._listen(event_name(event), handler, priority)

    def intercept(self, event: Event[T] | str, fn: Callable[[T], Any], *,
                  priority: int = 0) -> Callable[[], None]:
        """Register an interceptor for ``event``: ``fn(value)`` returns ``Deny(reason)``,
        ``Rewrite(new_value)`` or ``None`` (pass). Dispatched by :meth:`check`, never by
        ``emit``. Removed with the fiber."""
        return self._listen(("intercept", event_name(event)), fn, priority)

    def _listen(self, bucket_key: Any, handler: Callable[..., Any], priority: int) -> Callable[[], None]:
        from .kernel import Listener

        k, f = self.kernel, self.fiber
        lst = Listener(handler, f, priority, next(k._listener_seq))
        bucket = k._listeners.setdefault(bucket_key, [])
        bisect.insort(bucket, lst, key=_order)
        label = bucket_key if isinstance(bucket_key, str) else ":".join(bucket_key)

        def remove() -> None:
            b = k._listeners.get(bucket_key, [])
            if lst in b:
                b.remove(lst)

        eff = f._push_effect(remove, f"listener:{label}")

        def off() -> None:
            remove()
            if eff in f._effects:
                f._effects.remove(eff)
        return off

    def _targets(self, event: Any) -> list:
        return self.kernel._listeners_for(event, self.fiber)

    @overload
    async def emit(self, event: Event[T], payload: T, /) -> None: ...
    @overload
    async def emit(self, event: str, /, *args: Any) -> None: ...

    async def emit(self, event: Any, /, *args: Any) -> None:
        """Call listeners sequentially; listener errors are traced, not raised."""
        name = event_name(event)
        for lst in self._targets(name):
            await self._safe(name, lst, args)

    @overload
    async def parallel(self, event: Event[T], payload: T, /) -> None: ...
    @overload
    async def parallel(self, event: str, /, *args: Any) -> None: ...

    async def parallel(self, event: Any, /, *args: Any) -> None:
        """Call listeners concurrently; listener errors are traced, not raised."""
        name = event_name(event)
        async with anyio.create_task_group() as tg:
            for lst in self._targets(name):
                tg.start_soon(self._safe, name, lst, args)

    @overload
    async def serial(self, event: Event[T], payload: T, /) -> Any: ...
    @overload
    async def serial(self, event: str, /, *args: Any) -> Any: ...

    async def serial(self, event: Any, /, *args: Any) -> Any:
        """Await listeners in order; return the first non-None result (errors propagate)."""
        for lst in self._targets(event_name(event)):
            r = await maybe_await(lst.fn(*args))
            if r is not None:
                return r
        return None

    @overload
    def bail(self, event: Event[T], payload: T, /) -> Any: ...
    @overload
    def bail(self, event: str, /, *args: Any) -> Any: ...

    def bail(self, event: Any, /, *args: Any) -> Any:
        """Synchronous ``serial``: first non-None result. Async listeners are rejected."""
        name = event_name(event)
        for lst in self._targets(name):
            r = lst.fn(*args)
            if inspect.isawaitable(r):
                close = getattr(r, "close", None)
                if close:
                    close()
                raise TypeError(f"bail({name!r}) got an async listener; use serial()")
            if r is not None:
                return r
        return None

    async def check(self, event: Event[T] | str, value: T) -> T | Deny:
        """Run the interceptors of ``event`` (priority order) on ``value``.

        ``Deny`` stops the chain and is returned; ``Rewrite(v)`` replaces the value
        and the chain continues; ``None`` passes. Returns the (possibly rewritten)
        value if nobody denied. Interceptor errors propagate (callers fail closed).
        Unlike ``serial``, a ``Rewrite`` does not end the chain, so auditing
        interceptors registered after a rewriting one see the final value."""
        name = event_name(event)
        for lst in self._targets(("intercept", name)):
            verdict = await maybe_await(lst.fn(value))
            if verdict is None:
                continue
            if isinstance(verdict, Deny):
                return verdict
            if isinstance(verdict, Rewrite):
                value = verdict.value
                continue
            raise TypeError(f"interceptor for {name!r} returned {verdict!r}; "
                            "expected Deny, Rewrite or None")
        return value

    async def _safe(self, event: str, lst: Any, args: tuple) -> None:
        try:
            await maybe_await(lst.fn(*args))
        except Exception as e:  # noqa: BLE001 - listener errors are isolated by contract
            self.kernel._trace("event.error", lst.fiber, event=event, error=repr(e))

    # ----------------------------------------------------------- transactions
    def transaction(self, *, wait: bool = True, strict: bool = False,
                    origin: str | None = None, reason: str | None = None,
                    timeout: float | None = None, dry_run: bool = False,
                    probe: Any = None) -> Transaction:
        """``async with ctx.transaction() as tx: ...`` -- see transaction.py.

        A transaction opened inside a scope only locks that scope (plus shared
        locks on its ancestors), so sessions do not block each other.
        ``dry_run=True`` stages and settles everything for real, runs ``probe``
        (a callable ``probe(tx)`` or a ``{name: callable}`` dict; use ``tx.get``
        to see the staged world), then *always* rolls back; ``tx.report`` is the
        resulting TxReport. Every transaction fills ``tx.report``."""
        from .transaction import Transaction

        return Transaction(self, wait=wait, strict=strict, origin=origin, reason=reason,
                           timeout=timeout, dry_run=dry_run, probe=probe)

    async def replace(self, fiber: Fiber, plugin_or_config: Any = _MISSING, *,
                      plugin: Any = _MISSING, config: Any = _MISSING,
                      strategy: str | None = None) -> Fiber:
        """Hot-swap ``fiber`` (new plugin and/or config) in a one-op transaction.
        A positional callable is taken as the plugin, anything else as config.
        On failure the old fiber keeps running untouched and the error is raised
        (for a stop-first replacement it is restarted instead; see
        ``Transaction.replace``)."""
        if plugin_or_config is not _MISSING:
            if callable(plugin_or_config):
                plugin = plugin_or_config
            else:
                config = plugin_or_config
        async with self.transaction() as tx:
            new = await tx.replace(fiber, plugin=plugin, config=config, strategy=strategy)
        return new


class ScopePlugin:
    """The (empty) plugin behind a scope fiber; it only carries the scope's name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, ctx: Context) -> None:
        return None

    def __repr__(self) -> str:
        return f"<scope {self.name}>"


def _order(lst: Any) -> tuple[int, int]:
    return (-lst.priority, lst.seq)
