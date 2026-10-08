import anyio
import pytest

from pykernel import (Kernel, PluginError, State, TransactionBusy, TransactionConflict,
                      TransactionError, plugin)

from .conftest import LLM, llm_plugin, make_tool

pytestmark = pytest.mark.anyio


async def test_commit_is_atomic_and_isolated():
    log, heard = [], []
    async with Kernel() as app:
        ext = await app.plugin(make_tool(log, "ext"))  # live dependent, PENDING

        def listener(ctx):
            ctx.on("ping", lambda: heard.append("staged-listener"))

        async with app.transaction() as tx:
            llm = await tx.plugin(llm_plugin, {"model": "v1"})
            inner = await tx.plugin(make_tool(log, "inner"))
            await tx.plugin(listener)
            assert llm.state is State.ACTIVE and inner.state is State.ACTIVE
            # the live world sees nothing yet
            assert not app.has(LLM) and ext.state is State.PENDING
            assert llm.ctx.get(LLM).model == "v1"
            await app.emit("ping")
            assert heard == []
            assert app.snapshot()["fibers"]["children"][1]["staged"] is True
        assert tx.state == "committed"
        assert app.get(LLM).model == "v1"
        assert ext.state is State.ACTIVE
        await app.emit("ping")
        assert heard == ["staged-listener"]
        assert not any(c["staged"] for c in app.snapshot()["fibers"]["children"])
        assert ("apply", "ext", "v1") in log


async def test_rollback_when_plugin_apply_raises_midway():
    log, cleaned = [], []

    async def boom(ctx):
        ctx.on_dispose(lambda: cleaned.append("boom"))
        ctx.spawn(anyio.sleep_forever)
        raise RuntimeError("apply failed")

    async def extra(ctx):
        ctx.provide("extra", 1)
        ctx.on_dispose(lambda: cleaned.append("extra"))

    async with Kernel() as app:
        llm = await app.plugin(llm_plugin, {"model": "v1"})
        tool = await app.plugin(make_tool(log))
        old_svc = app.get(LLM)
        before = app.snapshot()
        with pytest.raises(PluginError) as ei:
            async with app.transaction() as tx:
                new_llm = await tx.replace(llm, config={"model": "v2"})
                x = await tx.plugin(extra)
                assert llm.state is State.ACTIVE  # old one keeps serving during the tx
                await tx.plugin(boom)
        assert isinstance(ei.value.__cause__, RuntimeError)
        assert tx.state == "rolled_back"
        assert app.snapshot() == before
        assert llm.state is State.ACTIVE and llm.raw_config == {"model": "v1"}
        assert app.get(LLM) is old_svc and not old_svc.closed
        assert new_llm.state is State.DISPOSED and x.state is State.DISPOSED
        assert not app.has("extra")
        assert sorted(cleaned) == ["boom", "extra"]
        assert log == [("apply", "tool", "v1")]  # the live dependent never restarted
        assert tool.state is State.ACTIVE


async def test_rollback_when_user_code_raises():
    log = []
    async with Kernel() as app:
        await app.plugin(llm_plugin)
        tool = await app.plugin(make_tool(log))
        before = app.snapshot()
        with pytest.raises(KeyError):
            async with app.transaction() as tx:
                await tx.dispose(tool)
                assert tool.state is State.ACTIVE  # removal is deferred to commit
                staged = await tx.plugin(make_tool(log, "staged"))
                raise KeyError("user code")
        assert staged.state is State.DISPOSED
        assert tool.state is State.ACTIVE
        assert app.snapshot() == before
        assert log == [("apply", "tool", "v1"), ("apply", "staged", "v1"), ("dispose", "staged", "v1")]


async def test_rollback_at_commit_time_strict_and_deferred_failure():
    async with Kernel() as app:
        before = app.snapshot()
        # a staged fiber stays PENDING -> strict commit refuses
        with pytest.raises(TransactionError, match="strict"):
            async with app.transaction(strict=True) as tx:
                await tx.plugin(make_tool([]))
        assert app.snapshot() == before

        # a staged fiber only becomes loadable (and fails) during commit preparation
        @plugin(name="late", inject=["dep"])
        def late(ctx):
            raise ValueError("late failure")

        with pytest.raises(TransactionError) as ei:
            async with app.transaction() as tx:
                f = await tx.plugin(late)
                assert f.state is State.PENDING
                await tx.plugin(lambda ctx: ctx.provide("dep", 1))
        assert isinstance(ei.value.__cause__, ValueError)
        assert app.snapshot() == before and not app.has("dep")


async def test_replace_hot_swap_and_rollback():
    log, violations = [], []
    async with Kernel() as app:
        llm = await app.plugin(llm_plugin, {"model": "v1"})
        tool = await app.plugin(make_tool(log))
        await app.plugin(lambda ctx: None)
        idx = app.fiber.children.index(llm)

        # invariant: the live registry always has an LLM during the swap
        app.on_trace(lambda e: violations.append(e) if not app.has(LLM) else None)
        new = await app.replace(llm, {"model": "v2"})
        assert violations == []
        assert llm.state is State.DISPOSED and new.state is State.ACTIVE
        assert app.fiber.children.index(new) == idx
        assert app.get(LLM).model == "v2"
        assert tool.state is State.ACTIVE
        # tool's teardown still saw v1, then it restarted with v2
        assert log == [("apply", "tool", "v1"), ("dispose", "tool", "v1"), ("apply", "tool", "v2")]

        before = app.snapshot()
        with pytest.raises(PluginError):
            await app.replace(new, {"model": "v3", "fail": True})
        assert app.snapshot() == before
        assert new.state is State.ACTIVE and app.get(LLM).model == "v2"
        assert len(log) == 3 and violations == []

        # replace with a different plugin implementation
        @plugin(name="llm2")
        def llm2(ctx):
            ctx.provide(LLM, LLM("other"))
        newer = await app.replace(new, llm2)
        assert newer.name == "llm2" and app.get(LLM).model == "other"
        assert violations == []


async def test_concurrent_transactions_serialized_or_rejected():
    order = []
    gate, began = anyio.Event(), anyio.Event()

    async with Kernel() as app:
        async def tx1():
            async with app.transaction() as tx:
                order.append("tx1-begin")
                began.set()
                await tx.plugin(lambda ctx: ctx.provide("a", 1))
                await gate.wait()
                order.append("tx1-end")

        async def tx2():
            async with app.transaction() as tx:
                order.append("tx2-begin")
                assert app.has("a")  # sees tx1's committed result
                await tx.plugin(lambda ctx: ctx.provide("b", 2))
                order.append("tx2-end")

        async with anyio.create_task_group() as tg:
            tg.start_soon(tx1)
            await began.wait()
            tg.start_soon(tx2)
            await anyio.sleep(0.01)
            assert order == ["tx1-begin"]  # tx2 is queued
            with pytest.raises(TransactionBusy):
                async with app.transaction(wait=False):
                    pass
            gate.set()
        assert order == ["tx1-begin", "tx1-end", "tx2-begin", "tx2-end"]
        assert app.get("a") == 1 and app.get("b") == 2

        async with app.transaction():
            with pytest.raises(TransactionError, match="nested"):
                async with app.transaction():
                    pass


async def test_commit_conflict_with_live_change_rolls_back():
    async with Kernel() as app:
        with pytest.raises(TransactionConflict):
            async with app.transaction() as tx:
                staged = await tx.plugin(lambda ctx: ctx.provide("x", "staged"))
                live = await app.plugin(lambda ctx: ctx.provide("x", "live"))
        assert staged.state is State.DISPOSED and live.state is State.ACTIVE
        assert app.get("x") == "live"


async def test_transaction_unload_commits_and_unloads_dependents():
    log = []
    async with Kernel() as app:
        llm = await app.plugin(llm_plugin)
        tool = await app.plugin(make_tool(log))
        async with app.transaction() as tx:
            await tx.dispose(llm)
            staged_tool = await tx.plugin(make_tool(log, "staged"))
            assert staged_tool.state is State.PENDING  # tombstoned for staged fibers
        assert llm.state is State.DISPOSED and tool.state is State.PENDING
        assert staged_tool.state is State.PENDING and not app.has(LLM)
        await app.plugin(llm_plugin, {"model": "v9"})
        assert tool.state is State.ACTIVE and staged_tool.state is State.ACTIVE


async def test_cancelled_transaction_rolls_back():
    async with Kernel() as app:
        before = app.snapshot()
        with anyio.move_on_after(0.05):
            async with app.transaction() as tx:
                await tx.plugin(lambda ctx: ctx.provide("x", 1))
                await anyio.sleep_forever()
        assert tx.state == "rolled_back" and app.snapshot() == before
        async with app.transaction(wait=False):  # lock was released
            pass


async def test_replace_recovers_failed_fiber():
    async with Kernel() as app:
        bad = await app.plugin(llm_plugin, {"fail": True})
        assert bad.state is State.FAILED
        good = await app.replace(bad, {"model": "fixed"})
        assert bad.state is State.DISPOSED and good.state is State.ACTIVE
        assert app.get(LLM).model == "fixed"
