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
Only one transaction is open per kernel (``wait=True`` queues FIFO, ``wait=False``
raises TransactionBusy).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio

from .errors import PluginError, TransactionBusy, TransactionConflict, TransactionError
from .fiber import Fiber, State, task_local
from .plugin import MISSING, keyname

if TYPE_CHECKING:  # pragma: no cover
    from .context import Context
    from .kernel import Binding

_MISSING = MISSING


class Transaction:
    def __init__(self, ctx: Context, *, wait: bool = True, strict: bool = False) -> None:
        self.ctx, self.kernel = ctx, ctx.kernel
        self.wait, self.strict = wait, strict
        self.state = "new"   # new -> open -> committed | rolled_back
        self.error: BaseException | None = None
        self._overlay: dict[Any, Binding | None] = {}
        self._staged: list[Fiber] = []     # staged roots, in creation order
        self._removals: list[Fiber] = []   # live fibers to dispose on commit
        self._replaces: dict[Fiber, Fiber] = {}  # new -> old

    def __repr__(self) -> str:
        return f"<Transaction {self.state} staged={len(self._staged)} removals={len(self._removals)}>"

    # ------------------------------------------------------------ enter/exit
    async def __aenter__(self) -> Transaction:
        loc = task_local()
        if loc.tx is not None or self.ctx.fiber.tx is not None:
            raise TransactionError("nested transactions are not supported")
        if self.wait:
            await self.kernel._tx_lock.acquire()
        else:
            try:
                self.kernel._tx_lock.acquire_nowait()
            except anyio.WouldBlock:
                raise TransactionBusy("another transaction is in progress") from None
        loc.tx, self.state = self, "open"
        self.kernel._trace("tx.begin", self.ctx.fiber)
        return self

    async def __aexit__(self, et: Any, exc: Any, tb: Any) -> bool:
        task_local().tx = None
        try:
            with anyio.CancelScope(shield=True):
                if et is not None:
                    await self._rollback(exc)
                    return False
                try:
                    await self._prepare()
                    self._validate()
                except BaseException as e:
                    await self._rollback(e)
                    raise
                self._swap()           # point of no return, synchronous
                await self._finish()
                return False
        finally:
            self.kernel._tx_lock.release()

    # ----------------------------------------------------------- operations
    def _check(self) -> None:
        if self.state != "open":
            raise TransactionError(f"transaction is {self.state}")

    async def _stage(self, parent: Fiber, plugin: Any, config: Any) -> Fiber:
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
        if fiber.tx is not None or fiber.parent is None:
            raise TransactionError(f"cannot dispose {fiber!r} in this transaction")
        if fiber in self._removals or fiber.state is State.DISPOSED:
            return
        self._removals.append(fiber)
        subtree = self._subtree(fiber)
        for key, b in self.kernel._services.items():
            if b.owner in subtree and key not in self._overlay:
                self._overlay[key] = None
        async with self.kernel._op(self):
            pass  # staged dependents of tombstoned services get restarted

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
        await self.kernel._settle(self)
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
        for key, b in self._overlay.items():
            cur = self.kernel._services.get(key)
            if b is not None and cur is not None and cur.owner not in removed:
                raise TransactionConflict(
                    f"{keyname(key)} was provided by {cur.owner.label} during the transaction")

    def _swap(self) -> None:
        """Apply the overlay to the live registry. MUST stay free of awaits."""
        k, removed = self.kernel, self._removed()
        changed = []
        for key, b in self._overlay.items():
            cur = k._services.get(key)
            if b is None:
                if cur is not None and cur.owner in removed:
                    del k._services[key]
                    changed.append(keyname(key))
            else:
                k._services[key] = b
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
        k._dirty = True
        k._trace("tx.commit", self.ctx.fiber, services=changed,
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
        await k._settle(None)

    async def _rollback(self, error: BaseException | None) -> None:
        self.error = error
        for f in reversed(self._staged):
            await f._dispose()
        for f in self._staged_fibers():  # defensive: anything left staged
            await f._dispose()
        self._overlay.clear()
        self.state = "rolled_back"
        self.kernel._trace("tx.rollback", self.ctx.fiber, error=repr(error))
        await self.kernel._settle(None)
