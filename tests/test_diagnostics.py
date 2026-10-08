"""Dependency diagnostics: no silent PENDING (DESIGN.md 4.9)."""
import pytest

from ventri import Kernel, State, TransactionError, plugin
from ventri.errors import DependencyCycle

from .conftest import LLM

pytestmark = pytest.mark.anyio


class A:
    pass


class B:
    pass


@plugin(name="a", provides=[A])
def plugin_a(ctx, config, b: B):
    ctx.provide(A, A())


@plugin(name="b", provides={"b": B})
def plugin_b(ctx, config, a: A):
    ctx.provide(B, B())


async def test_missing_provider_is_explained():
    async with Kernel() as app:
        f = await app.plugin(plugin(lambda ctx: None, name="tool", inject=[LLM, "cfg"]))
        assert f.pending_reason == ("missing: LLM (no plugin provides it); "
                                    "missing: cfg (no plugin provides it)")
        assert "(missing: LLM" in app.tree()
        assert app.snapshot()["fibers"]["children"][0]["pending_reason"] == f.pending_reason
        app.provide(LLM, LLM("x"))
        app.provide("cfg", 1)
        await app.settle()
        assert f.state is State.ACTIVE and f.pending_reason is None


async def test_waiting_on_failed_or_pending_provider():
    @plugin(name="llm", provides=[LLM])
    def broken(ctx):
        raise RuntimeError("bad key")

    async with Kernel() as app:
        await app.plugin(broken)
        f = await app.plugin(plugin(lambda ctx: None, name="tool", inject=[LLM]))
        assert f.pending_reason.startswith("waiting: LLM (provider llm#1 failed: RuntimeError('bad key')")


async def test_cycle_is_detected_traced_once_and_resolves():
    async with Kernel() as app:
        a = await app.plugin(plugin_a)
        b = await app.plugin(plugin_b)
        c = await app.plugin(plugin(lambda ctx: None, name="c", inject=[A]))
        assert a.pending_reason == f"cycle: {a.label} → {b.label} → {a.label}"
        assert b.pending_reason == f"cycle: {b.label} → {a.label} → {b.label}"
        assert c.pending_reason == f"waiting: A (provider {a.label} is pending)"
        await app.settle()
        await app.plugin(lambda ctx: None)
        cycles = [e for e in app.trace_log if e.kind == "dep.cycle"]
        assert len(cycles) == 1 and cycles[0].data["cycle"] == [a.label, b.label, a.label]
        assert "cycle: a#1" in app.tree()
        # break the cycle from outside: both activate, reasons clear
        app.provide(A, A())
        await app.settle()
        assert b.state is State.ACTIVE and a.state is State.FAILED  # A is taken -> conflict
        assert a.pending_reason is None and b.pending_reason is None


async def test_self_cycle():
    @plugin(name="selfish", provides=["k"], inject=["k"])
    def selfish(ctx):
        ctx.provide("k", 1)

    async with Kernel() as app:
        f = await app.plugin(selfish)
        assert f.pending_reason == f"cycle: {f.label} → {f.label}"


async def test_strict_transaction_fails_on_cycle():
    async with Kernel() as app:
        before = app.snapshot()
        with pytest.raises(DependencyCycle, match="cycle: a#"):
            async with app.transaction(strict=True) as tx:
                await tx.plugin(plugin_a)
                await tx.plugin(plugin_b)
        assert app.snapshot() == before
        # non-strict: a cycle is allowed to commit as PENDING (diagnosed, not silent)
        async with app.transaction() as tx:
            fa = await tx.plugin(plugin_a)
            await tx.plugin(plugin_b)
        assert fa.state is State.PENDING and fa.pending_reason.startswith("cycle:")
        with pytest.raises(TransactionError, match="missing: LLM"):
            async with app.transaction(strict=True) as tx:
                await tx.plugin(plugin(lambda ctx: None, inject=[LLM]))


async def test_undeclared_cycle_reads_as_missing():
    """Documented non-guarantee: without `provides` (and before any activation) the
    kernel cannot know who would provide a key."""
    async with Kernel() as app:
        x = await app.plugin(plugin(lambda ctx: ctx.provide("x", 1), name="x", inject=["y"]))
        y = await app.plugin(plugin(lambda ctx: ctx.provide("y", 1), name="y", inject=["x"]))
        assert x.pending_reason == "missing: y (no plugin provides it)"
        assert y.pending_reason == "missing: x (no plugin provides it)"


async def test_previous_bindings_count_as_providers():
    async with Kernel() as app:
        llm = await app.plugin(plugin(lambda ctx: ctx.provide(LLM, LLM("v")), name="llm",
                                      inject=["token"]))
        app.provide("token", 1)
        await app.settle()
        tool = await app.plugin(plugin(lambda ctx: None, name="tool", inject=[LLM]))
        assert tool.state is State.ACTIVE
        # the token goes away: llm unloads; tool now waits on llm (it provided LLM before)
        app._services.pop("token")
        await app.settle()
        assert llm.state is State.PENDING and tool.state is State.PENDING
        assert tool.pending_reason == f"waiting: LLM (provider {llm.label} is pending)"


async def test_diagnostics_respect_realms_and_transactions():
    @plugin(name="log", provides=["log"])
    def log(ctx):
        ctx.provide("log", 1)

    async with Kernel() as app:
        a = await app.scope("a", isolate={"log"})
        b = await app.scope("b", isolate={"log"})
        await a.ctx.plugin(plugin(log, inject=["never"]))
        f = await b.ctx.plugin(plugin(lambda ctx: None, name="agent", inject=["log"]))
        # a's declared provider lives in another realm: not a candidate for b
        assert f.pending_reason == "missing: log (no plugin provides it)"

        live = await app.plugin(plugin(lambda ctx: None, name="live", inject=["svc"]))
        async with app.transaction() as tx:
            p = await tx.plugin(plugin(lambda ctx: None, name="p", provides=["svc"], inject=["z"]))
            await app.settle()
            # a staged provider is invisible to live fibers
            assert live.pending_reason == "missing: svc (no plugin provides it)"
        assert live.pending_reason == f"waiting: svc (provider {p.label} is pending)"
