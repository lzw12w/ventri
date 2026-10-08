"""Dry-run transactions and TxReport (DESIGN.md 4.5.4)."""
import json

import pytest

from ventri import Kernel, State, TransactionError, plugin

from .conftest import LLM, llm_plugin, make_tool

pytestmark = pytest.mark.anyio


async def test_dry_run_stages_for_real_reports_and_always_rolls_back():
    log, effects = [], []

    def extra(ctx):
        effects.append("side effect")  # N1: not undone by the rollback
        ctx.provide("extra", 42)
        ctx.on_dispose(lambda: effects.append("cleanup"))

    async with Kernel() as app:
        llm = await app.plugin(llm_plugin, {"model": "v1"})
        await app.plugin(make_tool(log))
        waiting = await app.plugin(plugin(lambda ctx: None, name="waiting", inject=["extra"]))
        gone = await app.plugin(lambda ctx: ctx.provide("old", 1))
        before = app.snapshot()

        def probe(tx):
            return {"model": tx.get(LLM).model, "extra": tx.get("extra"), "old": tx.get("old", None)}

        async with app.transaction(dry_run=True, probe=probe, origin="evolution:17",
                                   reason="try v2") as tx:
            new = await tx.replace(llm, config={"model": "v2"})
            x = await tx.plugin(extra)
            await tx.dispose(gone)
            assert new.state is State.ACTIVE and x.state is State.ACTIVE  # really started
        r = tx.report
        assert app.snapshot() == before and llm.state is State.ACTIVE
        assert new.state is State.DISPOSED and x.state is State.DISPOSED
        assert log == [("apply", "tool", "v1")]  # the live dependent never restarted
        assert effects == ["side effect", "cleanup"]
        assert r.outcome == "dry_run" and r.dry_run and r.ok and r.error is None
        assert r.origin == "evolution:17" and r.reason == "try v2"
        assert r.added == {new.label: "active", x.label: "active"}
        assert r.replaced == {new.label: llm.label}
        assert set(r.removed) == {llm.label, gone.label}
        assert r.restarted == ["tool#2"] and r.activated == [waiting.label]
        assert r.services == {"added": ["extra"], "removed": ["old"], "replaced": ["LLM"]}
        assert r.probes["probe"] == {"ok": True, "error": None,
                                     "value": {"model": "v2", "extra": 42, "old": None}}
        assert any(e.kind == "tx.rollback" and e.data.get("dry_run") for e in app.trace_log)
        assert json.loads(r.to_json())["ok"] is True
        assert "replaced:" in str(r) and "probe probe: ok" in str(r)


async def test_dry_run_reports_failures_instead_of_raising():
    async def broken(ctx):
        raise ValueError("bad config")

    async with Kernel() as app:
        before = app.snapshot()
        async with app.transaction(dry_run=True,
                                   probe={"fails": lambda tx: 1 / 0, "fine": lambda tx: "ok"}) as tx:
            f = await tx.plugin(broken)  # does not raise in a dry run
            assert f.state is State.FAILED
            await tx.plugin(plugin(lambda ctx: None, name="w", inject=["nope"]))
        r = tx.report
        assert not r.ok and r.failures == {f.label: "ValueError('bad config')"}
        assert r.pending == {"w#2": "missing: nope (no plugin provides it)"}
        assert r.error.startswith("TransactionError")
        assert r.probes["fails"]["ok"] is False and "ZeroDivisionError" in r.probes["fails"]["error"]
        assert r.probes["fine"]["ok"] is True
        assert app.snapshot() == before


async def test_dry_run_strict_reports_cycle_and_body_errors_propagate():
    @plugin(name="a", provides=["a"], inject=["b"])
    def a(ctx):
        ctx.provide("a", 1)

    @plugin(name="b", provides=["b"], inject=["a"])
    def b(ctx):
        ctx.provide("b", 1)

    async with Kernel() as app:
        before = app.snapshot()
        async with app.transaction(dry_run=True, strict=True) as tx:
            await tx.plugin(a)
            await tx.plugin(b)
        assert tx.report.error.startswith("DependencyCycle") and not tx.report.ok
        with pytest.raises(KeyError):
            async with app.transaction(dry_run=True) as tx:
                await tx.plugin(lambda ctx: ctx.provide("x", 1))
                raise KeyError("user")
        assert tx.report.outcome == "rolled_back" and app.snapshot() == before


async def test_probe_failure_aborts_a_real_transaction():
    async with Kernel() as app:
        before = app.snapshot()

        def health(tx):
            assert tx.get("svc") == "healthy", "unhealthy"

        with pytest.raises(TransactionError, match="probe 'health' failed"):
            async with app.transaction(probe=health) as tx:
                await tx.plugin(lambda ctx: ctx.provide("svc", "sick"))
        assert app.snapshot() == before and tx.report.outcome == "rolled_back"
        async with app.transaction(probe=health) as tx:
            await tx.plugin(lambda ctx: ctx.provide("svc", "healthy"))
        assert tx.report.outcome == "committed" and tx.report.probes["health"]["ok"]
        assert app.get("svc") == "healthy"


async def test_every_transaction_has_a_report():
    async with Kernel() as app:
        async with app.transaction(origin="user") as tx:
            f = await tx.plugin(lambda ctx: ctx.provide("k", 1))
        assert tx.report.outcome == "committed" and tx.report.added == {f.label: "active"}
        assert tx.report.services["added"] == ["k"]
        with pytest.raises(RuntimeError):
            async with app.transaction() as tx2:
                raise RuntimeError("boom")
        assert tx2.report.outcome == "rolled_back" and tx2.report.error == "RuntimeError('boom')"
