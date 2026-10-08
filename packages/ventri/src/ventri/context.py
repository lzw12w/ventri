"""Context: the API surface a plugin sees. One Context per fiber; ``ctx.parent`` walks up."""
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar, overload

import anyio

from .errors import ServiceConflict, ServiceNotFound
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
    def on(self, event: str, handler: Callable[..., Any]) -> Callable[[], None]:
        from .kernel import Listener

        k, f = self.kernel, self.fiber
        lst = Listener(handler, f)
        k._listeners.setdefault(event, []).append(lst)

        def remove() -> None:
            bucket = k._listeners.get(event, [])
            if lst in bucket:
                bucket.remove(lst)

        eff = f._push_effect(remove, f"listener:{event}")

        def off() -> None:
            remove()
            if eff in f._effects:
                f._effects.remove(eff)
        return off

    def _targets(self, event: str) -> list:
        return self.kernel._listeners_for(event, self.fiber)

    async def emit(self, event: str, *args: Any) -> None:
        """Call listeners sequentially; listener errors are traced, not raised."""
        for lst in self._targets(event):
            await self._safe(event, lst, args)

    async def parallel(self, event: str, *args: Any) -> None:
        """Call listeners concurrently; listener errors are traced, not raised."""
        async with anyio.create_task_group() as tg:
            for lst in self._targets(event):
                tg.start_soon(self._safe, event, lst, args)

    async def serial(self, event: str, *args: Any) -> Any:
        """Await listeners in order; return the first non-None result (errors propagate)."""
        for lst in self._targets(event):
            r = await maybe_await(lst.fn(*args))
            if r is not None:
                return r
        return None

    def bail(self, event: str, *args: Any) -> Any:
        """Synchronous ``serial``: first non-None result. Async listeners are rejected."""
        for lst in self._targets(event):
            r = lst.fn(*args)
            if inspect.isawaitable(r):
                close = getattr(r, "close", None)
                if close:
                    close()
                raise TypeError(f"bail({event!r}) got an async listener; use serial()")
            if r is not None:
                return r
        return None

    async def _safe(self, event: str, lst: Any, args: tuple) -> None:
        try:
            await maybe_await(lst.fn(*args))
        except Exception as e:  # noqa: BLE001 - listener errors are isolated by contract
            self.kernel._trace("event.error", lst.fiber, event=event, error=repr(e))

    # ----------------------------------------------------------- transactions
    def transaction(self, *, wait: bool = True, strict: bool = False,
                    origin: str | None = None, reason: str | None = None,
                    timeout: float | None = None) -> Transaction:
        """``async with ctx.transaction() as tx: ...`` -- see transaction.py.

        A transaction opened inside a scope only locks that scope (plus shared
        locks on its ancestors), so sessions do not block each other."""
        from .transaction import Transaction

        return Transaction(self, wait=wait, strict=strict, origin=origin, reason=reason,
                           timeout=timeout)

    async def replace(self, fiber: Fiber, plugin_or_config: Any = _MISSING, *,
                      plugin: Any = _MISSING, config: Any = _MISSING) -> Fiber:
        """Hot-swap ``fiber`` (new plugin and/or config) in a one-op transaction.
        A positional callable is taken as the plugin, anything else as config.
        On failure the old fiber keeps running untouched and the error is raised."""
        if plugin_or_config is not _MISSING:
            if callable(plugin_or_config):
                plugin = plugin_or_config
            else:
                config = plugin_or_config
        async with self.transaction() as tx:
            new = await tx.replace(fiber, plugin=plugin, config=config)
        return new


class ScopePlugin:
    """The (empty) plugin behind a scope fiber; it only carries the scope's name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, ctx: Context) -> None:
        return None

    def __repr__(self) -> str:
        return f"<scope {self.name}>"
