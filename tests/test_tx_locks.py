"""Scope-aware transaction locks, transaction metadata and timeouts (DESIGN.md 4.5.4, 4.6)."""
import anyio
import pytest

from ventri import (
    Kernel,
    State,
    TransactionBusy,
    TransactionConflict,
    TransactionError,
    plugin,
)
from ventri.errors import TransactionTimeout

pytestmark = pytest.mark.anyio


def provider(key, value):
    @plugin(name=f"p-{key}")
    def p(ctx):
        ctx.provide(key, value)
    return p


async def test_session_transactions_run_concurrently_root_excludes_them():
    order = []
    async with Kernel() as app:
        a = await app.scope("a", isolate={"log"})
        b = await app.scope("b", isolate={"log"})
        both_open, release = anyio.Event(), anyio.Event()
        opened = []

        async def session_tx(s, tag):
            async with s.ctx.transaction() as tx:
                opened.append(tag)
                if len(opened) == 2:
                    both_open.set()
                await tx.plugin(provider("log", tag))
                await release.wait()
            order.append(f"{tag}-commit")

        async def root_tx():
            async with app.transaction():
                order.append("root-begin")

        async with anyio.create_task_group() as tg:
            tg.start_soon(session_tx, a, "a")
            tg.start_soon(session_tx, b, "b")
            with anyio.fail_after(1):
                await both_open.wait()  # two session transactions are open at once
            tg.start_soon(root_tx)
            await anyio.sleep(0.01)
            assert "root-begin" not in order  # root waits for the sessions
            with pytest.raises(TransactionBusy):
                async with a.ctx.transaction(wait=False):
                    pass
            release.set()
        assert order.index("root-begin") > max(order.index("a-commit"), order.index("b-commit"))
        assert a.ctx.get("log") == "a" and b.ctx.get("log") == "b"


async def test_root_transaction_blocks_new_session_transactions_fifo():
    async with Kernel() as app:
        s = await app.scope("s", isolate={"x"})
        order = []
        gate = anyio.Event()

        async def root():
            async with app.transaction():
                order.append("root")
                await gate.wait()

        async def session():
            async with s.ctx.transaction():
                order.append("session")

        async with anyio.create_task_group() as tg:
            tg.start_soon(root)
            await anyio.sleep(0.01)
            tg.start_soon(session)
            await anyio.sleep(0.01)
            assert order == ["root"]
            with pytest.raises(TransactionBusy):
                async with s.ctx.transaction(wait=False):
                    pass
            gate.set()
        assert order == ["root", "session"]


async def test_scope_transaction_cannot_touch_fibers_outside_its_scope():
    async with Kernel() as app:
        outside = await app.plugin(provider("g", 1))
        s = await app.scope("s")
        before = app.snapshot()
        with pytest.raises(TransactionError, match="cannot dispose"):
            async with s.ctx.transaction() as tx:
                await tx.dispose(outside)
        with pytest.raises(TransactionError, match="cannot dispose"):
            async with s.ctx.transaction() as tx:
                await tx.dispose(s)  # the scope itself is not inside the scope
        assert app.snapshot() == before and outside.state is State.ACTIVE


async def test_concurrent_session_transactions_conflict_on_shared_root_key():
    """Non-guarantee made explicit: session transactions are not serialised against
    each other, so two that stage the same *root-realm* key conflict at commit."""
    async with Kernel() as app:
        a = await app.scope("a")
        b = await app.scope("b")
        staged, go = anyio.Event(), anyio.Event()
        result = {}

        async def first():
            async with a.ctx.transaction() as tx:
                await tx.plugin(provider("shared", "a"))
                staged.set()
                await go.wait()

        async def second():
            await staged.wait()
            try:
                async with b.ctx.transaction() as tx:
                    await tx.plugin(provider("shared", "b"))
                    go.set()
                    await anyio.sleep(0.01)  # let the first one commit
            except TransactionConflict as e:
                result["err"] = e

        async with anyio.create_task_group() as tg:
            tg.start_soon(first)
            tg.start_soon(second)
        assert app.get("shared") == "a" and isinstance(result["err"], TransactionConflict)


async def test_metadata_is_traced():
    async with Kernel() as app:
        async with app.transaction(origin="config", reason="apply ventri.yml") as tx:
            await tx.plugin(provider("k", 1))
        begin = next(e for e in app.trace_log if e.kind == "tx.begin")
        commit = next(e for e in app.trace_log if e.kind == "tx.commit")
        assert begin.data["origin"] == "config" and begin.data["reason"] == "apply ventri.yml"
        assert begin.data["tx"] == commit.data["tx"] == tx.id
        s = await app.scope("s")
        with pytest.raises(KeyError):
            async with s.ctx.transaction(origin="user") as tx2:
                raise KeyError("x")
        rb = [e for e in app.trace_log if e.kind == "tx.rollback"][-1]
        assert rb.data["tx"] == tx2.id and rb.data["scope"] == s.label and rb.data["origin"] == "user"


async def test_timeout_rolls_back_and_releases_lock():
    async with Kernel() as app:
        before = app.snapshot()
        with pytest.raises(TransactionTimeout):
            async with app.transaction(timeout=0.05) as tx:
                staged = await tx.plugin(provider("k", 1))
                await anyio.sleep(10)
        assert tx.state == "rolled_back" and staged.state is State.DISPOSED
        assert app.snapshot() == before
        async with app.transaction(wait=False, timeout=5) as tx:  # lock released; no timeout
            await tx.plugin(provider("k", 2))
        assert app.get("k") == 2


async def test_timeout_during_staged_apply_and_during_commit():
    async def slow(ctx):
        ctx.provide("slow", 1)
        await anyio.sleep(10)

    @plugin(name="late", inject=["dep"])
    async def late(ctx):
        await anyio.sleep(10)  # only becomes loadable at commit preparation

    async with Kernel() as app:
        before = app.snapshot()
        with pytest.raises(TransactionTimeout):
            async with app.transaction(timeout=0.05) as tx:
                await tx.plugin(slow)
        assert app.snapshot() == before and not app.has("slow")

        with pytest.raises(TransactionTimeout, match="during commit"):
            async with app.transaction(timeout=0.05) as tx:
                f = await tx.plugin(late)
                dep = await app.plugin(provider("dep", 1))  # live: staged fibers settle at commit
                assert f.state is State.PENDING
        assert f.state is State.DISPOSED
        await dep.dispose()
        assert app.snapshot() == before


async def test_outer_cancellation_is_not_reported_as_timeout():
    async with Kernel() as app:
        with anyio.move_on_after(0.02) as scope:
            async with app.transaction(timeout=10) as tx:
                await anyio.sleep(1)
        assert scope.cancelled_caught and tx.state == "rolled_back"


async def test_concurrent_session_transactions_conflict_on_root_keys_at_commit():
    """Documented non-guarantee (DESIGN 8, M1 note 5): session transactions only share the
    root lock, so two of them staging the same *root-realm* key are not serialized --
    the second to commit fails with TransactionConflict and rolls back."""
    from ventri import TransactionConflict, TransactionError

    async with Kernel() as app:
        a, b = await app.scope("a"), await app.scope("b")
        res = {}

        async def run(s, name, delay):
            try:
                async with s.ctx.transaction() as tx:
                    await tx.plugin(lambda ctx, config: ctx.provide("cache", name))
                    await anyio.sleep(delay)
                res[name] = "ok"
            except TransactionError as e:
                res[name] = e

        async with anyio.create_task_group() as tg:
            tg.start_soon(run, a, "a", 0.01)
            tg.start_soon(run, b, "b", 0.05)
        assert res["a"] == "ok" and isinstance(res["b"], TransactionConflict)
        assert app.get("cache") == "a" and len(b.children) == 0
