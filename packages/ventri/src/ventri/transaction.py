"""Transactions: stage plugin changes off to the side, then swap them in atomically.

Model (blue/green staging):
  * ``tx.plugin`` / ``tx.replace`` create *staged* fibers. They really run (apply,
    spawn, listeners) but their services live in a private overlay, visible only
    to other staged fibers of the same transaction; their listeners only hear
    events emitted from inside the transaction.
  * ``tx.dispose`` / ``tx.replace`` do NOT stop the old fiber. They tombstone its
    services in the overlay, so staged fibers no longer see them. The live world
    keeps using the old fiber, untouched.
  * Commit = (1) settle staged fibers, fail if any is FAILED (or not ACTIVE when
    ``strict``); (2) validate the overlay against the live registry; (3) apply the
    overlay to the live registry in ONE synchronous step (no await -> no other task
    can observe a partial registry); (4) restart live dependents whose binding
    changed, dispose removed/replaced fibers, settle.
  * Rollback = dispose staged fibers (newest first), drop the overlay. Removed and
    replaced fibers were never stopped, so nothing has to be "restored".
Locking (scope-aware G4): a transaction opened on a context whose nearest scope
is S takes S's lock exclusively and every ancestor scope's lock (up to the root)
shared. A root transaction therefore excludes every other transaction, while
transactions in sibling sessions run concurrently. Waiters queue FIFO
(``wait=True``) or fail fast with TransactionBusy (``wait=False``). A scope
transaction may only stage or remove fibers inside its scope; bindings it
stages into an outer realm are checked for conflicts at commit (G5).
"""
from __future__ import annotations

from collections import deque
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import anyio

from .errors import (
    PluginError,
    TransactionBusy,
    TransactionConflict,
    TransactionError,
    TransactionTimeout,
)
from .fiber import Fiber, State, task_local
from .plugin import MISSING, keyname

if TYPE_CHECKING:  # pragma: no cover
    from .context import Context
    from .kernel import Binding, Realm

_MISSING = MISSING


class RWLock:
    """FIFO reader/writer lock (readers = transactions in descendant scopes)."""

    __slots__ = ("_queue", "_readers", "_writer")

    def __init__(self) -> None:
        self._readers = 0
        self._writer = False
        self._queue: deque[list] = deque()  # [write, event, granted]

    def _free(self, write: bool) -> bool:
        return not self._writer and (self._readers == 0 if write else True)

    def _grant(self, write: bool) -> None:
        if write:
            self._writer = True
        else:
            self._readers += 1

    def try_acquire(self, write: bool) -> bool:
        if not self._queue and self._free(write):
            self._grant(write)
            return True
        return False

    async def acquire(self, write: bool) -> None:
        if self.try_acquire(write):
            return
        waiter = [write, anyio.Event(), False]
        self._queue.append(waiter)
        try:
            await waiter[1].wait()
        except BaseException:
            if waiter[2]:
                self.release(write)
            else:
                self._queue.remove(waiter)
                self._wake()
            raise

    def release(self, write: bool) -> None:
        if write:
            self._writer = False
        else:
            self._readers -= 1
        self._wake()

    def _wake(self) -> None:
        while self._queue and self._free(self._queue[0][0]):
            waiter = self._queue.popleft()
            self._grant(waiter[0])
            waiter[2] = True
            waiter[1].set()


def _scope_lock(fiber: Fiber) -> RWLock:
    if fiber._txlock is None:
        fiber._txlock = RWLock()
    return fiber._txlock


class Transaction:
    """``async with ctx.transaction(...) as tx`` -- see the module docstring.

    ``origin`` / ``reason`` are free-form metadata recorded in the trace (e.g.
    ``origin="config"``); ``timeout`` (seconds) bounds the time until commit
    preparation is done: when it elapses the transaction is rolled back and
    TransactionTimeout is raised (the commit itself is never interrupted).
    """

    def __init__(self, ctx: Context, *, wait: bool = True, strict: bool = False,
                 origin: str | None = None, reason: str | None = None,
                 timeout: float | None = None) -> None:
        self.ctx, self.kernel = ctx, ctx.kernel
        self.wait, self.strict = wait, strict
        self.origin, self.reason, self.timeout = origin, reason, timeout
        self.id = next(self.kernel._tx_ids)
        self.scope: Fiber = ctx.fiber.scope_fiber
        self.state = "new"   # new -> open -> committed | rolled_back
        self.error: BaseException | None = None
        self._locks: list[tuple[RWLock, bool]] = []
        self._deadline: float | None = None
        self._cs: anyio.CancelScope | None = None
        self._overlay: dict[tuple[Realm, Any], Binding | None] = {}
        self._staged: list[Fiber] = []     # staged roots, in creation order
        self._removals: list[Fiber] = []   # live fibers to dispose on commit
        self._replaces: dict[Fiber, Fiber] = {}  # new -> old

    def __repr__(self) -> str:
        return (f"<Transaction #{self.id} {self.state} staged={len(self._staged)} "
                f"removals={len(self._removals)}>")

    def _meta(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tx": self.id}
        if self.origin is not None:
            out["origin"] = self.origin
        if self.reason is not None:
            out["reason"] = self.reason
        if self.scope is not self.kernel.fiber:
            out["scope"] = self.scope.label
        return out

    # ------------------------------------------------------------ enter/exit
    async def __aenter__(self) -> Self:
        loc = task_local()
        if loc.tx is not None or self.ctx.fiber.tx is not None:
            raise TransactionError("nested transactions are not supported")
        if self.state != "new":
            raise TransactionError(f"transaction is {self.state}")
        chain, f = [], self.scope
        while f is not None:
            if f.is_scope:
                chain.append(f)
            f = f.parent
        try:
            for sf in reversed(chain):  # root first: a fixed order cannot deadlock
                lock, write = _scope_lock(sf), sf is self.scope
                if self.wait:
                    await lock.acquire(write)
                elif not lock.try_acquire(write):
                    raise TransactionBusy("another transaction is in progress")
                self._locks.append((lock, write))
        except BaseException:
            self._release()
            raise
        loc.tx, self.state = self, "open"
        if self.timeout is not None:
            self._deadline = anyio.current_time() + self.timeout
            self._cs = anyio.CancelScope(deadline=self._deadline)
            self._cs.__enter__()
        self.kernel._trace("tx.begin", self.ctx.fiber, **self._meta())
        return self

    def _release(self) -> None:
        while self._locks:
            lock, write = self._locks.pop()
            lock.release(write)

    async def __aexit__(self, et: type[BaseException] | None, exc: BaseException | None,
                        tb: TracebackType | None) -> bool:
        task_local().tx = None
        timed_out = False
        if self._cs is not None:
            cs, self._cs = self._cs, None
            if cs.__exit__(et, exc, tb):  # our own deadline cancelled the block
                timed_out, et = True, TransactionTimeout
                exc = TransactionTimeout(f"transaction #{self.id} timed out after {self.timeout}s")
        try:
            with anyio.CancelScope(shield=True):
                if timed_out:
                    await self._rollback(exc)
                    raise exc  # type: ignore[misc]
                if et is not None:
                    await self._rollback(exc)
                    return False
                try:
                    with anyio.CancelScope(deadline=self._deadline or float("inf")) as prep:
                        await self._prepare()
                    if prep.cancelled_caught:
                        raise TransactionTimeout(
                            f"transaction #{self.id} timed out after {self.timeout}s (during commit)")
                    self._validate()
                except BaseException as e:
                    await self._rollback(e)
                    raise
                self._swap()           # point of no return, synchronous
                await self._finish()
                return False
        finally:
            self._release()

    # ----------------------------------------------------------- operations
    def _check(self) -> None:
        if self.state != "open":
            raise TransactionError(f"transaction is {self.state}")

    def _in_scope(self, fiber: Fiber) -> bool:
        """Is ``fiber`` strictly inside this transaction's scope?"""
        if self.scope is self.kernel.fiber:
            return fiber.parent is not None
        p = fiber.parent
        while p is not None:
            if p is self.scope:
                return True
            p = p.parent
        return False

    async def _stage(self, parent: Fiber, plugin: Any, config: Any) -> Fiber:
        if parent is not self.scope and not self._in_scope(parent):
            raise TransactionError(f"{parent!r} is outside the transaction's scope {self.scope!r}")
        f = Fiber(self.kernel, parent, plugin, config, tx=self)
        self._staged.append(f)
        async with self.kernel._op(self):
            await f._activate()
        if f.state is State.FAILED:
            raise PluginError(f"{f.label} failed: {f.error!r}") from f.error
        return f

    async def plugin(self, plugin: Any, config: Any = None) -> Fiber:
        """Stage a new plugin under the transaction's context."""
        self._check()
        return await self._stage(self.ctx.fiber, plugin, config)

    async def dispose(self, fiber: Fiber) -> None:
        """Stage removal of a live fiber (it keeps running until commit)."""
        self._check()
        if fiber.tx is self:  # staged in this very transaction: just drop it
            if fiber in self._staged:
                self._staged.remove(fiber)
            await fiber.dispose()
            return
        if fiber.tx is not None or fiber.parent is None or not self._in_scope(fiber):
            raise TransactionError(f"cannot dispose {fiber!r} in this transaction")
        if fiber in self._removals or fiber.state is State.DISPOSED:
            return
        self._removals.append(fiber)
        k = self.kernel
        async with k._op(self):  # staged dependents of tombstoned services get restarted
            for f in self._subtree(fiber):
                for b in f._bindings:
                    realm = b.realm
                    assert realm is not None
                    rk = (realm, b.key)
                    if realm.services.get(b.key) is b and rk not in self._overlay:
                        self._overlay[rk] = None
                        k._mark_dirty(realm, self)

    async def replace(self, fiber: Fiber, plugin: Any = _MISSING, config: Any = _MISSING) -> Fiber:
        """Stage ``fiber`` -> new fiber with new plugin and/or config, same parent."""
        self._check()
        plugin = fiber.plugin if plugin is _MISSING else plugin
        config = fiber.raw_config if config is _MISSING else config
        parent = fiber.parent
        if parent is None:
            raise TransactionError("cannot replace the root fiber")
        await self.dispose(fiber)
        new = await self._stage(parent, plugin, config)
        if fiber.tx is None:
            self._replaces[new] = fiber
        return new

    async def reconfigure(self, fiber: Fiber, config: Any) -> Fiber:
        return await self.replace(fiber, config=config)

    # --------------------------------------------------------------- commit
    def _subtree(self, root: Fiber) -> set[Fiber]:
        out, stack = set(), [root]
        while stack:
            f = stack.pop()
            out.add(f)
            stack.extend(f.children)
        return out

    def _staged_fibers(self) -> list[Fiber]:
        return [f for f in self.kernel._walk() if f.tx is self]

    async def _prepare(self) -> None:
        await self.kernel._settle(self, full=True)
        for f in self._staged_fibers():
            if f.state is State.FAILED:
                raise TransactionError(f"{f.label} failed: {f.error!r}") from f.error
            if self.strict and f.state is not State.ACTIVE:
                raise TransactionError(f"{f.label} is {f.state.value} (strict transaction)")

    def _removed(self) -> set[Fiber]:
        out: set[Fiber] = set()
        for f in self._removals:
            out |= self._subtree(f)
        return out

    def _validate(self) -> None:
        removed = self._removed()
        for (realm, key), b in self._overlay.items():
            cur = realm.services.get(key)
            if b is not None and cur is not None and cur.owner not in removed:
                raise TransactionConflict(
                    f"{keyname(key)} was provided by {cur.owner.label} during the transaction")

    def _swap(self) -> None:
        """Apply the overlay to the live registry. MUST stay free of awaits."""
        k, removed = self.kernel, self._removed()
        changed = []
        for (realm, key), b in self._overlay.items():
            cur = realm.services.get(key)
            if b is None:
                if cur is not None and cur.owner in removed:
                    del realm.services[key]
                    changed.append(keyname(key))
            else:
                realm.services[key] = b
                b.tx = None  # the binding is live now
                changed.append(keyname(key))
        for f in self._staged:
            f._tx = None
        for new, old in self._replaces.items():  # new fiber takes the old one's slot
            siblings = new.parent.children if new.parent else []
            if old in siblings and new in siblings:
                siblings.remove(new)
                siblings.insert(siblings.index(old), new)
        self._overlay.clear()
        self.state = "committed"
        k._trace("tx.commit", self.ctx.fiber, **self._meta(), services=changed,
                 added=[f.label for f in self._staged], removed=[f.label for f in self._removals])

    async def _finish(self) -> None:
        k = self.kernel
        # 1. live dependents bound to a swapped-out binding restart (old instance still alive)
        for f in reversed(list(k._walk())):
            if f.state is State.ACTIVE and f.tx is None and not k._deps_current(f):
                await f._deactivate()
        # 2. stop what the transaction removed/replaced
        for f in reversed(self._removals):
            await f._dispose()
        # 3. reactivate everything that can run now
        await k._settle(None, full=True)

    async def _rollback(self, error: BaseException | None) -> None:
        self.error = error
        for f in reversed(self._staged):
            await f._dispose()
        for f in self._staged_fibers():  # defensive: anything left staged
            await f._dispose()
        self._overlay.clear()
        self.state = "rolled_back"
        self.kernel._trace("tx.rollback", self.ctx.fiber, **self._meta(), error=repr(error))
        await self.kernel._settle(None, full=True)
