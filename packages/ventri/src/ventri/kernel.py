"""Kernel: root context, realm-aware service registry, reconciler, event bus, trace hook."""
from __future__ import annotations

import itertools
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import anyio

from .context import Context
from .errors import KernelError
from .fiber import LIVE, Fiber, State, task_local
from .plugin import Retry, keyname

if TYPE_CHECKING:  # pragma: no cover
    from .transaction import Transaction


class Realm:
    """A service namespace. The root realm holds every key that no enclosing scope
    isolates; each ``ctx.scope(name, isolate=...)`` owns a realm for its keys."""

    __slots__ = ("name", "owner", "services")

    def __init__(self, name: str, owner: Fiber) -> None:
        self.name = name
        self.owner = owner
        self.services: dict[Any, Binding] = {}

    def __repr__(self) -> str:
        return f"<Realm {self.name}>"


@dataclass(eq=False, slots=True)
class Binding:
    key: Any
    value: Any
    owner: Fiber
    name: str | None
    tx: Transaction | None  # transaction it was staged in (None = provided live)
    realm: Realm | None = None


@dataclass(eq=False, slots=True)
class Listener:
    fn: Callable[..., Any]
    fiber: Fiber
    priority: int = 0
    seq: int = 0


@dataclass
class TraceEvent:
    """One trace record. ``fiber`` is the fiber label (``name#id``); :meth:`to_dict`
    produces the versioned, exportable form (see ``ventri.trace``)."""

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

    def __init__(self, *, trace_limit: int = 10_000, load_timeout: float | None = 30.0,
                 retry: Retry | dict | None = None) -> None:
        self.load_timeout = load_timeout     # default for plugins without ``timeout``
        self.retry = Retry.coerce(retry)     # default restart policy (None: never retry)
        self._ids = itertools.count(1)
        self._seq = itertools.count(1)
        self._listener_seq = itertools.count(1)
        self._listeners: dict[Any, list[Listener]] = {}
        self._tracers: list[Callable[[TraceEvent], Any]] = []
        self.trace_log: deque[TraceEvent] = deque(maxlen=trace_limit)
        self._tx_ids = itertools.count(1)
        self._tg: Any = None
        self._stack: AsyncExitStack | None = None
        root = Fiber(self, None, None, ctx=self, scope=True)
        self._root_realm = root.realm = Realm("root", root)
        self._services = self._root_realm.services  # root realm (kept for introspection)
        Context.__init__(self, self, root)
        root.state = State.ACTIVE

    # ------------------------------------------------------------- lifecycle
    async def __aenter__(self) -> Self:
        self._stack = AsyncExitStack()
        self._tg = await self._stack.enter_async_context(anyio.create_task_group())
        self.fiber._tg = self._tg
        self._trace("kernel.start")
        return self

    async def __aexit__(self, *exc: object) -> bool | None:
        root = self.fiber
        with anyio.CancelScope(shield=True):
            for child in reversed(list(root.children)):
                await child._dispose()
            while root._effects:
                await root._run_effect(root._effects.pop())
        root._set_state(State.DISPOSED)
        self._tg.cancel_scope.cancel()
        assert self._stack is not None
        return await self._stack.__aexit__(*exc)  # type: ignore[arg-type]

    async def settle(self) -> None:
        """Run the reconciler over the whole tree until no PENDING fiber can activate
        and no ACTIVE fiber is stale. Only needed after a ``provide`` made outside
        any kernel operation."""
        await self._settle(None, full=True)

    # ---------------------------------------------------------- observability
    def on_trace(self, callback: Callable[[TraceEvent], Any]) -> Callable[[], None]:
        self._tracers.append(callback)
        return lambda: self._tracers.remove(callback) if callback in self._tracers else None

    def _trace(self, kind: str, fiber: Fiber | None = None, **data: Any) -> None:
        ev = TraceEvent(next(self._seq), time.time(), kind, fiber.label if fiber else None, data)
        self.trace_log.append(ev)
        for cb in list(self._tracers):
            try:
                cb(ev)
            except Exception:  # noqa: BLE001, S110 - a broken tracer must not break the kernel
                pass

    def snapshot(self) -> dict:
        from .observe import snapshot
        return snapshot(self)

    def tree(self) -> str:
        from .observe import render_tree
        return render_tree(self)

    # ------------------------------------------------------------- registry
    def _realm_of(self, fiber: Fiber, key: Any) -> Realm:
        """The realm ``key`` resolves to for ``fiber``: the nearest enclosing scope
        (inclusive) that isolates ``key``, else the root realm. Ancestry never
        changes, so the answer is cached per fiber."""
        cache = fiber._realm_cache
        realm = cache.get(key)
        if realm is None:
            f: Fiber | None = fiber
            while f is not None:
                if f.realm is not None and key in f.isolate:
                    realm = f.realm
                    break
                f = f.parent
            else:
                realm = self._root_realm
            cache[key] = realm
        return realm

    def _lookup(self, realm: Realm, key: Any, tx: Transaction | None = None) -> Binding | None:
        if tx is not None and tx._overlay:
            rk = (realm, key)
            if rk in tx._overlay:
                return tx._overlay[rk]  # may be None: tombstone
        return realm.services.get(key)

    def _resolve(self, f: Fiber) -> dict | None:
        """Bindings for ``f``'s dependencies, or None if a required one is missing."""
        tx, out = f.tx, {}
        for key in f.inject:
            b = self._lookup(self._realm_of(f, key), key, tx)
            if b is None:
                return None
            out[key] = b
        for key in f.optional:
            b = self._lookup(self._realm_of(f, key), key, tx)
            if b is not None:
                out[key] = b
        return out

    def _visible(self, fiber: Fiber, tx: Transaction | None) -> Iterator[Binding]:
        """Every binding ``fiber`` can see (nearest realm first)."""
        seen: set = set()
        f: Fiber | None = fiber
        while f is not None:
            realm = f.realm
            if realm is not None:
                items: dict[Any, Binding | None] = dict(realm.services)
                if tx is not None:
                    for (r, key), b in tx._overlay.items():
                        if r is realm:
                            items[key] = b
                for key, b in items.items():
                    if b is not None and key not in seen and self._realm_of(fiber, key) is realm:
                        seen.add(key)
                        yield b
            f = f.parent

    def _find_by_name(self, fiber: Fiber, name: str) -> Binding | None:
        for b in self._visible(fiber, fiber.tx):
            if b.name == name:
                return b
        return None

    def _deps_current(self, f: Fiber) -> bool:
        tx = f.tx
        snap = f._snapshot
        return all(self._lookup(self._realm_of(f, k), k, tx) is snap.get(k) for k in f.deps)

    async def _unbind(self, b: Binding) -> None:
        """Remove a binding. Dependents are torn down *first*, so they can still use
        the service in their own cleanup."""
        realm = b.realm
        assert realm is not None
        if realm.services.get(b.key) is b:
            store: dict = realm.services
            skey: Any = b.key
            tx = None
        elif b.tx is not None and b.tx._overlay.get((realm, b.key)) is b:
            store, skey, tx = b.tx._overlay, (realm, b.key), b.tx
        else:
            return  # already replaced by a committed transaction
        for f in reversed(list(self._walk(realm.owner))):
            if f.state is State.ACTIVE and f is not b.owner and b in f._snapshot.values():
                await f._deactivate()
        if store.get(skey) is b:
            del store[skey]
            if b in b.owner._bindings:
                b.owner._bindings.remove(b)
            self._trace("service.unbind", b.owner, key=keyname(b.key), staged=tx is not None)
        self._mark_dirty(realm, tx)

    # ----------------------------------------------------------- reconciler
    def _walk(self, start: Fiber | None = None) -> Iterator[Fiber]:
        """Pre-order walk of ``start``'s descendants (default: the whole tree)."""
        stack = list(reversed((start or self.fiber).children))
        while stack:
            f = stack.pop()
            yield f
            stack.extend(reversed(f.children))

    def _region(self, realms: Iterable[Realm]) -> list[Fiber]:
        """Fibers whose dependencies can be affected by changes in ``realms``: a
        change in a scope realm is only visible inside that scope's subtree."""
        owners = {r.owner for r in realms}
        if self.fiber in owners:
            return list(self._walk())
        roots = []
        for o in owners:
            if o.state is State.DISPOSED:
                continue
            p = o.parent
            while p is not None and p not in owners:
                p = p.parent
            if p is None:
                roots.append(o)
        out: list[Fiber] = []
        for r in roots:
            out.extend(self._walk(r))
        return out

    def _mark_dirty(self, realm: Realm, tx: Transaction | None) -> None:
        loc = task_local()
        if tx is None and loc.depth == 0:
            if self._tg is not None:  # provide() from plain user code / tasks
                self._tg.start_soon(self._settle_all)
            return
        loc.dirty.add(realm)

    async def _settle_all(self) -> None:
        await self._settle(None, full=True)

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

    async def _settle(self, tx: Transaction | None, *, full: bool = False) -> None:
        """Reconcile the regions this task dirtied (or everything with ``full``)."""
        loc = task_local()
        if full:
            loc.dirty.add(self._root_realm)
        if not loc.dirty:
            return
        loc.depth += 1
        try:
            region: set[Realm] = set()
            while loc.dirty:
                region |= loc.dirty
                loc.dirty = set()
                await self._reconcile(None, region)
                if tx is not None and tx.state == "open":
                    await self._reconcile(tx, region)
        finally:
            loc.depth -= 1

    async def _reconcile(self, tx: Transaction | None, region: set[Realm]) -> None:
        """Fixpoint loop over the fibers of ``tx`` (None = live world) in ``region``:
        deactivate stale ACTIVE fibers (deepest/newest first), then activate PENDING
        fibers whose deps resolve (tree order)."""
        loc = task_local()
        for _ in range(100):
            changed = False
            fibers = [f for f in self._region(region) if f.tx is tx]
            for f in reversed(fibers):
                if f.state is State.ACTIVE and not self._deps_current(f):
                    changed |= await f._deactivate()
            for f in fibers:
                if f.state is State.PENDING and not f._dispose_requested:
                    changed |= await f._activate()
            if loc.dirty:
                region |= loc.dirty
                loc.dirty = set()
                changed = True
            if not changed:
                return
        raise KernelError("reconcile did not converge (dependency flapping?)")

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

    def _schedule_retry(self, fiber: Fiber, delay: float, gen: int) -> None:
        async def retry() -> None:
            await anyio.sleep(delay)
            async with self._op(None):
                await fiber._retry_now(gen)
        if self._tg is not None:
            self._tg.start_soon(retry, name=f"retry:{fiber.label}")

    # ---------------------------------------------------------------- events
    def _listeners_for(self, event: Any, emitter: Fiber) -> list[Listener]:
        """Listeners that hear ``event`` emitted from ``emitter``:
        * staged (uncommitted) fibers only hear events from the same transaction;
        * scope filter: the listener's scope and the emitter's scope must lie on one
          root-to-leaf chain (ancestor, same, or descendant scope) -- sibling scopes
          never hear each other."""
        etx = emitter.tx
        echain, escope = emitter.scope_chain, emitter.scope_fiber
        out = []
        for lst in self._listeners.get(event, ()):
            lf = lst.fiber
            ltx = lf.tx
            if lf.state not in LIVE or (ltx is not None and ltx is not etx):
                continue
            if lf.scope_fiber in echain or escope in lf.scope_chain:
                out.append(lst)
        return out
