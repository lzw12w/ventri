"""Fiber: the runtime record of one loaded plugin.

Each fiber owns
  * a lock that serialises load / unload / dispose of *this* fiber,
  * a host task (started in the parent's task group) running an anyio task group
    that every ``ctx.spawn`` task lives in -> structured concurrency,
  * a LIFO stack of effects (listeners, services, tasks, user disposers),
  * an ordered list of child fibers.

Teardown order is fixed: children (newest first) -> effects (LIFO) -> task group.
"""
from __future__ import annotations

import enum
import inspect
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import anyio

from .plugin import PluginSpec, describe

if TYPE_CHECKING:  # pragma: no cover
    from .context import Context
    from .kernel import Binding, Kernel, Realm
    from .transaction import Transaction


class State(str, enum.Enum):
    PENDING = "pending"      # waiting for inject deps (or parent) to be available
    LOADING = "loading"      # apply/start running
    ACTIVE = "active"
    FAILED = "failed"        # apply raised or a spawned task crashed; see fiber.error
    UNLOADING = "unloading"  # teardown running
    DISPOSED = "disposed"    # terminal


LIVE = (State.LOADING, State.ACTIVE)


# ---------------------------------------------------------------- task locals
class _TaskLocal:
    """Per-task bookkeeping. Keyed by task id so that tasks which inherit a copied
    contextvars.Context from their spawner do not inherit its state."""

    __slots__ = ("depth", "dirty", "owner", "task_id", "tx")

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id
        self.depth = 0                  # nesting of kernel operations (reconcile once at depth 0)
        self.owner: Fiber | None = None  # fiber whose task group runs this task
        self.tx: Transaction | None = None
        self.dirty: set[Realm] = set()  # realms this task changed; its outermost op settles them


_local: ContextVar[_TaskLocal | None] = ContextVar("ventri_local", default=None)


def task_id() -> int:
    return anyio.get_current_task().id


def task_local() -> _TaskLocal:
    tid = task_id()
    loc = _local.get()
    if loc is None or loc.task_id != tid:
        loc = _TaskLocal(tid)
        _local.set(loc)
    return loc


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class Effect:
    __slots__ = ("fn", "label")

    def __init__(self, fn: Callable[[], Any], label: str) -> None:
        self.fn, self.label = fn, label


class TaskHandle:
    """Returned by ``ctx.spawn``; ``cancel()`` is idempotent, ``await wait()`` joins."""

    def __init__(self, name: str, scope: anyio.CancelScope, done: anyio.Event) -> None:
        self.name, self._scope, self._done = name, scope, done

    def cancel(self) -> None:
        self._scope.cancel()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    async def wait(self) -> None:
        await self._done.wait()


class Fiber:
    def __init__(self, kernel: Kernel, parent: Fiber | None, plugin: Any,
                 config: Any = None, *, tx: Transaction | None = None,
                 ctx: Context | None = None, scope: bool = False,
                 isolate: frozenset = frozenset(), meta: dict | None = None) -> None:
        from .context import Context

        self.kernel = kernel
        self.parent = parent
        self.id = 0 if parent is None else next(kernel._ids)
        self.plugin = plugin
        self.spec: PluginSpec | None = describe(plugin) if plugin is not None else None
        self.raw_config = config
        self.config: Any = None
        self.instance: Any = None
        self.state = State.PENDING
        self.error: BaseException | None = None
        self.pending_reason: str | None = None
        self.meta: dict = meta or {}
        self.children: list[Fiber] = []
        self.ctx: Context = ctx if ctx is not None else Context(kernel, self)
        # scopes: a scope fiber owns a realm for the keys it isolates and bounds events
        self.is_scope = scope
        self.isolate = isolate
        self.realm: Realm | None = None
        if scope and parent is not None:
            from .kernel import Realm
            self.realm = Realm(self.name, self)
        self.scope_fiber: Fiber = self if scope or parent is None else parent.scope_fiber
        if parent is None:
            self.scope_chain: frozenset[Fiber] = frozenset((self,))
        else:
            self.scope_chain = parent.scope_chain | {self} if scope else parent.scope_chain
        self._realm_cache: dict[Any, Realm] = {}
        self._bindings: list[Binding] = []  # live/staged bindings this fiber provides
        self._tx = tx
        self._effects: list[Effect] = []
        self._snapshot: dict[Any, Binding] = {}
        self._lock = anyio.Lock()
        self._lock_owner: int | None = None
        self._load_scope: anyio.CancelScope | None = None
        self._dispose_requested = False
        self._tg: Any = None
        self._stop: anyio.Event | None = None
        self._closed: anyio.Event | None = None
        if parent is not None:
            parent.children.append(self)

    # ------------------------------------------------------------ properties
    @property
    def name(self) -> str:
        return self.spec.name if self.spec else "root"

    @property
    def label(self) -> str:
        return f"{self.name}#{self.id}"

    @property
    def inject(self) -> tuple:
        """Required dependency keys."""
        return self.spec.inject if self.spec else ()

    @property
    def optional(self) -> tuple:
        """Optional dependency keys (``X | None``): bound if present, never block."""
        return self.spec.optional if self.spec else ()

    @property
    def deps(self) -> tuple:
        return self.spec.deps if self.spec else ()

    @property
    def scope(self) -> Fiber:
        """The nearest enclosing scope fiber (the root fiber if none)."""
        return self.scope_fiber

    @property
    def tx(self) -> Transaction | None:
        """The open transaction this fiber is staged in (inherited from ancestors)."""
        f: Fiber | None = self
        while f is not None:
            if f._tx is not None:
                return f._tx
            f = f.parent
        return None

    def __repr__(self) -> str:
        return f"<Fiber {self.label} {self.state.value}>"

    def _set_state(self, state: State, **data: Any) -> None:
        old, self.state = self.state, state
        if state is not State.PENDING:
            self.pending_reason = None
        self.kernel._trace("fiber.state", self, old=old.value, new=state.value, **data)

    @asynccontextmanager
    async def _locked(self) -> AsyncIterator[None]:
        async with self._lock:
            self._lock_owner = task_id()
            try:
                yield
            finally:
                self._lock_owner = None

    # --------------------------------------------------------------- effects
    def _push_effect(self, fn: Callable[[], Any], label: str) -> Effect:
        if self.state not in LIVE:
            raise RuntimeError(f"cannot register {label!r} on {self!r}")
        eff = Effect(fn, label)
        self._effects.append(eff)
        return eff

    async def _run_effect(self, eff: Effect) -> None:
        try:
            await maybe_await(eff.fn())
        except Exception as e:  # noqa: BLE001 - cleanup errors are reported, never propagated
            self.kernel._trace("effect.error", self, effect=eff.label, error=repr(e))

    # ------------------------------------------------------- task group host
    async def _open_scope(self) -> None:
        stop, closed = anyio.Event(), anyio.Event()

        async def host(*, task_status: Any = anyio.TASK_STATUS_IGNORED) -> None:
            tg_ref = None
            try:
                async with anyio.create_task_group() as tg:
                    tg_ref = self._tg = tg
                    task_status.started()
                    await stop.wait()
                    tg.cancel_scope.cancel()
            finally:
                if self._tg is tg_ref:
                    self._tg = None
                closed.set()

        self._stop, self._closed = stop, closed
        assert self.parent is not None and self.parent._tg is not None
        await self.parent._tg.start(host, name=f"fiber:{self.label}")

    async def _close_scope(self) -> None:
        stop, closed = self._stop, self._closed
        if stop is None or closed is None:
            return
        self._stop = self._closed = None
        stop.set()
        # A task inside this fiber's own task group cannot wait for that group to exit.
        if self._tg is not None and not self.kernel._inside(self):
            await closed.wait()

    def spawn(self, fn: Callable[..., Any], *args: Any, name: str | None = None) -> TaskHandle:
        if self.state not in LIVE or self._tg is None:
            raise RuntimeError(f"cannot spawn on {self!r}")
        label = name or getattr(fn, "__name__", "task")
        scope, done = anyio.CancelScope(), anyio.Event()
        runner_id: list[int] = []
        k = self.kernel

        async def stop() -> None:
            scope.cancel()
            if not runner_id or runner_id[0] != task_id():
                await done.wait()

        eff = self._push_effect(stop, f"task:{label}")

        async def runner() -> None:
            loc = task_local()
            loc.owner = self
            runner_id.append(loc.task_id)
            try:
                with scope:
                    await fn(*args)
            except Exception as e:  # noqa: BLE001 - supervisor: a task crash fails its fiber
                k._trace("task.error", self, task=label, error=repr(e))
                k._crash(self, e)  # failure propagates to the owning fiber
            finally:
                done.set()
                if eff in self._effects:
                    self._effects.remove(eff)

        self._tg.start_soon(runner, name=f"{self.label}:{label}")
        k._trace("task.spawn", self, task=label)
        return TaskHandle(label, scope, done)

    # ------------------------------------------------------------- lifecycle
    async def _apply(self) -> None:
        spec = self.spec
        assert spec is not None
        cfg = self.raw_config
        if spec.config_type is not None and (cfg is None or isinstance(cfg, dict)):
            cfg = spec.config_type(**(cfg or {}))
        self.config = cfg
        result = spec.call(self.ctx, cfg)
        if spec.is_class:
            self.instance = result
            start = getattr(result, "start", None)
            if callable(start):
                await maybe_await(start())
            stop = getattr(result, "stop", None)
            if callable(stop):
                self._push_effect(stop, "stop")
        else:
            await maybe_await(result)

    async def _activate(self) -> bool:
        """PENDING -> LOADING -> ACTIVE|FAILED. Returns True if a load was attempted."""
        k = self.kernel
        async with self._locked():
            parent = self.parent
            if self.state is not State.PENDING or self._dispose_requested:
                return False
            if parent is None or parent.state not in LIVE or parent._tg is None:
                return False
            snapshot = k._resolve(self)
            if snapshot is None:
                return False
            self._snapshot, self.error = snapshot, None
            self._set_state(State.LOADING)
            ok, err = False, None
            try:
                with anyio.CancelScope() as scope:
                    self._load_scope = scope
                    await self._open_scope()
                    await self._apply()
                    ok = True
            except Exception as e:  # noqa: BLE001 - any apply error -> FAILED (fiber.error)
                err = e
            finally:
                # Runs on success, error, own cancellation *and* outer cancellation.
                self._load_scope = None
                if not ok or self._dispose_requested:
                    await self._teardown()
                    if self._dispose_requested:
                        self._finalize_dispose()
                    elif err is not None:
                        self.error = err
                        self._set_state(State.FAILED, error=repr(err))
                    else:  # cancelled from outside: nothing leaked, may retry later
                        self._set_state(State.PENDING, reason="cancelled")
            if self.state is State.LOADING:
                self._set_state(State.ACTIVE)
            return True

    async def _teardown(self) -> None:
        with anyio.CancelScope(shield=True):
            for child in reversed(list(self.children)):
                await child._dispose()
            while self._effects:
                await self._run_effect(self._effects.pop())
            await self._close_scope()
            self._snapshot, self.instance = {}, None

    async def _deactivate(self) -> bool:
        """ACTIVE -> UNLOADING -> PENDING (used when a dependency goes away)."""
        if self._lock_owner == task_id():
            return False
        with anyio.CancelScope(shield=True):
            async with self._locked():
                if self.state is not State.ACTIVE:
                    return False
                self._set_state(State.UNLOADING)
                await self._teardown()
                if self._dispose_requested:
                    self._finalize_dispose()
                else:
                    self._set_state(State.PENDING)
                return True

    def _finalize_dispose(self) -> None:
        self._set_state(State.DISPOSED)
        if self.parent is not None and self in self.parent.children:
            self.parent.children.remove(self)

    async def _dispose(self) -> None:
        if self.state is State.DISPOSED:
            return
        if self.parent is None:
            raise RuntimeError("the root fiber is disposed by closing the kernel")
        self._dispose_requested = True
        if self._load_scope is not None:
            self._load_scope.cancel()  # abort an in-flight apply
        if self._lock_owner == task_id():
            return  # re-entrant (e.g. self-dispose inside apply): the lock holder finalizes
        with anyio.CancelScope(shield=True):
            async with self._locked():
                if self.state is State.DISPOSED:
                    return
                if self.state is State.ACTIVE:
                    self._set_state(State.UNLOADING)
                    await self._teardown()
                self._finalize_dispose()

    async def _fail(self, error: BaseException) -> None:
        with anyio.CancelScope(shield=True):
            async with self._locked():
                if self.state is not State.ACTIVE:
                    return
                self._set_state(State.UNLOADING)
                await self._teardown()
                if self._dispose_requested:
                    self._finalize_dispose()
                else:
                    self.error = error
                    self._set_state(State.FAILED, error=repr(error))

    # ------------------------------------------------------------ public API
    async def dispose(self) -> None:
        """Dispose this fiber (children first, then effects in reverse)."""
        async with self.kernel._op(self.tx):
            await self._dispose()

    async def restart(self) -> None:
        """Re-run apply (also recovers a FAILED fiber)."""
        async with self.kernel._op(self.tx):
            await self._deactivate()
            if self.state is State.FAILED:
                self.error = None
                self._set_state(State.PENDING)
            await self._activate()
