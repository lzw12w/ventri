"""Kernel: root context, service registry, reconciler, event bus and trace hook."""
from __future__ import annotations

import itertools
import time
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator

import anyio

from .context import Context
from .errors import KernelError
from .fiber import LIVE, Fiber, State, task_local
from .plugin import keyname

if TYPE_CHECKING:  # pragma: no cover
    from .transaction import Transaction


@dataclass(eq=False)
class Binding:
    key: Any
    value: Any
    owner: Fiber
    name: str | None
    tx: Transaction | None  # transaction it was staged in (None = provided live)


@dataclass(eq=False)
class Listener:
    fn: Callable[..., Any]
    fiber: Fiber


@dataclass
class TraceEvent:
    seq: int
    time: float
    kind: str
    fiber: str | None
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        extra = " ".join(f"{k}={v}" for k, v in self.data.items())
        return f"[{self.seq:04d}] {self.kind:<14} {self.fiber or '-':<16} {extra}"


class Kernel(Context):
    """``async with Kernel() as app:`` -- the root context. Closing it disposes everything."""

    def __init__(self, *, trace_limit: int = 10_000) -> None:
        self._ids = itertools.count(1)
        self._seq = itertools.count(1)
        self._services: dict[Any, Binding] = {}
        self._listeners: dict[str, list[Listener]] = {}
        self._tracers: list[Callable[[TraceEvent], Any]] = []
        self.trace_log: deque[TraceEvent] = deque(maxlen=trace_limit)
        self._tx_lock = anyio.Lock()
        self._tg: Any = None
        self._stack: AsyncExitStack | None = None
        self._dirty = False
        root = Fiber(self, None, None, ctx=self)
        Context.__init__(self, self, root)
        root.state = State.ACTIVE

    # ------------------------------------------------------------- lifecycle
    async def __aenter__(self) -> Kernel:
        self._stack = AsyncExitStack()
        self._tg = await self._stack.enter_async_context(anyio.create_task_group())
        self.fiber._tg = self._tg
        self._trace("kernel.start")
        return self

    async def __aexit__(self, *exc: Any) -> bool | None:
        root = self.fiber
        with anyio.CancelScope(shield=True):
            for child in reversed(list(root.children)):
                await child._dispose()
            while root._effects:
                await root._run_effect(root._effects.pop())
        root._set_state(State.DISPOSED)
        self._tg.cancel_scope.cancel()
        assert self._stack is not None
        return await self._stack.__aexit__(*exc)

    async def settle(self) -> None:
        """Run the reconciler until no PENDING fiber can activate and no ACTIVE fiber
        is stale. Only needed after a ``provide`` made outside any kernel operation."""
        await self._settle(None)

    # ---------------------------------------------------------- observability
    def on_trace(self, callback: Callable[[TraceEvent], Any]) -> Callable[[], None]:
        self._tracers.append(callback)
        return lambda: self._tracers.remove(callback) if callback in self._tracers else None

    def _trace(self, kind: str, fiber: Fiber | None = None, **data: Any) -> None:
        ev = TraceEvent(next(self._seq), time.monotonic(), kind,
                        fiber.label if fiber else None, data)
        self.trace_log.append(ev)
        for cb in list(self._tracers):
            try:
                cb(ev)
            except Exception:  # a broken tracer must not break the kernel
                pass

    def snapshot(self) -> dict:
        from .observe import snapshot
        return snapshot(self)

    def tree(self) -> str:
        from .observe import render_tree
        return render_tree(self)

    # ------------------------------------------------------------- registry
    def _lookup(self, key: Any, tx: Transaction | None = None) -> Binding | None:
        if tx is not None and key in tx._overlay:
            return tx._overlay[key]  # may be None: tombstone
        return self._services.get(key)

    def _resolve(self, keys: tuple, tx: Transaction | None) -> dict | None:
        out = {}
        for key in keys:
            b = self._lookup(key, tx)
            if b is None:
                return None
            out[key] = b
        return out

    def _effective(self, tx: Transaction | None) -> dict[Any, Binding]:
        view = dict(self._services)
        if tx is not None:
            for key, b in tx._overlay.items():
                if b is None:
                    view.pop(key, None)
                else:
                    view[key] = b
        return view

    def _find_by_name(self, name: str, tx: Transaction | None) -> Binding | None:
        for b in self._effective(tx).values():
            if b.name == name:
                return b
        return None

    def _deps_current(self, f: Fiber) -> bool:
        tx = f.tx
        return all(self._lookup(k, tx) is f._snapshot.get(k) for k in f.inject)

    async def _unbind(self, b: Binding) -> None:
        """Remove a binding. Dependents are torn down *first*, so they can still use
        the service in their own cleanup."""
        if self._services.get(b.key) is b:
            store, scope = self._services, None
        elif b.tx is not None and b.tx._overlay.get(b.key) is b:
            store, scope = b.tx._overlay, b.tx
        else:
            return  # already replaced by a committed transaction
        for f in reversed(list(self._walk())):
            if f.state is State.ACTIVE and f is not b.owner and b in f._snapshot.values():
                await f._deactivate()
        if store.get(b.key) is b:
            del store[b.key]
            self._trace("service.unbind", b.owner, key=keyname(b.key), staged=scope is not None)
        self._mark_dirty(scope)

    # ----------------------------------------------------------- reconciler
    def _walk(self) -> Iterator[Fiber]:
        stack = list(reversed(self.fiber.children))
        while stack:
            f = stack.pop()
            yield f
            stack.extend(reversed(f.children))

    def _mark_dirty(self, tx: Transaction | None) -> None:
        self._dirty = True
        if tx is None and self._tg is not None and task_local().depth == 0:
            self._tg.start_soon(self._settle, None)  # provide() from plain user code / tasks

    @asynccontextmanager
    async def _op(self, tx: Transaction | None = None) -> AsyncIterator[None]:
        """Wrap a public mutation: nested ops are inline, the outermost one settles."""
        loc = task_local()
        loc.depth += 1
        try:
            yield
        finally:
            loc.depth -= 1
        if loc.depth == 0:
            await self._settle(tx)

    async def _settle(self, tx: Transaction | None) -> None:
        await self._reconcile(None)
        if tx is not None and tx.state == "open":
            await self._reconcile(tx)

    async def _reconcile(self, tx: Transaction | None) -> None:
        """Fixpoint loop over fibers belonging to ``tx`` (None = live world):
        deactivate stale ACTIVE fibers (deepest/newest first), then activate
        PENDING fibers whose deps resolve (tree order)."""
        loc = task_local()
        loc.depth += 1
        try:
            for _ in range(100):
                self._dirty = False
                changed = False
                fibers = [f for f in self._walk() if f.tx is tx]
                for f in reversed(fibers):
                    if f.state is State.ACTIVE and not self._deps_current(f):
                        changed |= await f._deactivate()
                for f in fibers:
                    if f.state is State.PENDING and not f._dispose_requested:
                        changed |= await f._activate()
                if not changed and not self._dirty:
                    return
            raise KernelError("reconcile did not converge (dependency flapping?)")
        finally:
            loc.depth -= 1

    # ----------------------------------------------------- structured errors
    def _inside(self, fiber: Fiber) -> bool:
        """Is the current task running inside ``fiber``'s (or a descendant's) task group?"""
        f = task_local().owner
        while f is not None:
            if f is fiber:
                return True
            f = f.parent
        return False

    def _crash(self, fiber: Fiber, error: BaseException) -> None:
        async def fail() -> None:
            async with self._op(fiber.tx):
                await fiber._fail(error)
        if self._tg is not None:
            self._tg.start_soon(fail)

    # ---------------------------------------------------------------- events
    def _listeners_for(self, event: str, emitter: Fiber) -> list[Listener]:
        """Staged (uncommitted) fibers only hear events from the same transaction."""
        etx = emitter.tx
        out = []
        for lst in self._listeners.get(event, ()):
            ltx = lst.fiber.tx
            if lst.fiber.state in LIVE and (ltx is None or ltx is etx):
                out.append(lst)
        return out
