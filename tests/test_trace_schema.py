"""Trace schema v1 (DESIGN.md 4.10): every kernel kind, validation, export."""
import json

import anyio
import pytest

from ventri import KERNEL_KINDS, Kernel, PluginError, Retry, Secret, plugin
from ventri.trace import TraceSchemaError, dumps, read_jsonl, validate

from .test_diagnostics import plugin_a, plugin_b

pytestmark = pytest.mark.anyio


async def _workload(app: Kernel) -> None:
    def tools(ctx, config):
        ctx.provide("tools", {"api_token": "tok-1"})

        async def boom():
            raise RuntimeError("task boom")
        ctx.spawn(boom, name="boom")

        def eff():
            def cleanup():
                raise RuntimeError("effect boom")
            return cleanup
        ctx.effect(eff)

        def bad_listener(_):
            raise RuntimeError("listener boom")
        ctx.on("ping", bad_listener)

    await app.plugin(tools, meta={"id": "tools"})
    await app.emit("ping", 1)
    await anyio.sleep(0)

    calls = {"n": 0}

    def flaky(ctx, config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first load fails")
    await app.plugin(flaky, retry=Retry(max=1, base=0.001))
    for _ in range(50):
        await anyio.sleep(0.002)
        if calls["n"] > 1:
            break

    async with app.transaction(reason="ok") as tx:
        await tx.plugin(lambda ctx, config: ctx.provide("svc", Secret("s3cret")))
    with pytest.raises(PluginError):
        async with app.transaction(reason="fail") as tx:
            await tx.plugin(lambda ctx, config: (_ for _ in ()).throw(RuntimeError("load fails")))
    await app.plugin(plugin_a)
    await app.plugin(plugin_b)
    s = await app.scope("session:a1", isolate=["llm"])
    await s.ctx.plugin(lambda ctx, config: ctx.provide("llm", 1), meta={"id": "llm"})
    s.ctx.trace("agent.turn", turn=1, password="hunter2")
    for f in list(app.fiber.children):
        if f.label.startswith("tools"):
            await f._dispose()


async def test_every_kernel_kind_is_emitted_and_valid():
    events = []
    async with Kernel(load_timeout=1) as app:
        app.on_trace(events.append)
        await _workload(app)
    kinds = {e.kind for e in events}
    assert kinds - {"agent.turn"} <= KERNEL_KINDS, kinds - KERNEL_KINDS
    assert KERNEL_KINDS - kinds == {"kernel.start"}  # emitted before on_trace was registered
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    for e in events:
        rec = validate(json.loads(dumps(e.to_dict())))
        assert rec["v"] == 1 and rec["kind"] == e.kind
    blob = "".join(dumps(e.to_dict()) for e in events)
    assert "tok-1" not in blob and "s3cret" not in blob and "hunter2" not in blob


async def test_fields_path_scope_tx():
    events = []
    async with Kernel() as app:
        app.on_trace(events.append)
        group = await app.plugin(lambda ctx, config: None, meta={"id": "tools"})
        await group.ctx.plugin(plugin(lambda ctx, config: None, name="github"))
        s = await app.scope("session:a1")
        await s.ctx.plugin(lambda ctx, config: None, meta={"id": "llm"})
        async with app.transaction() as tx:
            staged = await tx.plugin(lambda ctx, config: None, meta={"id": "staged"})
    recs = [e.to_dict() for e in events]
    paths = {r["fiber"] for r in recs}
    assert {"root/tools", "root/tools/github", "root/session:a1/llm"} <= paths
    llm = next(r for r in recs if r["fiber"] == "root/session:a1/llm")
    assert llm["scope"] == "session:a1" and llm["attrs"]["label"].startswith("<lambda>#")
    assert next(r for r in recs if r["fiber"] == "root/tools")["scope"] is None
    st = [r for r in recs if r["fiber"] == "root/staged" and r["kind"] == "fiber.state"]
    assert st[0]["tx"] == tx.id
    begin = next(r for r in recs if r["kind"] == "tx.begin")
    assert begin["tx"] == tx.id and begin["fiber"] == "root"
    assert staged.path == "root/staged"


async def test_custom_trace_kind_rules():
    async with Kernel() as app:
        app.trace("config.apply", ok=True)
        assert app.trace_log[-1].kind == "config.apply"
        for bad in ("fiber.state", "nodot", "tx.commit"):
            with pytest.raises(ValueError, match="custom trace kind"):
                app.trace(bad)


def test_validate_rejects_bad_records(tmp_path):
    good = {"v": 1, "seq": 1, "ts": 1.0, "kind": "x.y", "fiber": None, "scope": None,
            "tx": None, "attrs": {}}
    assert validate(dict(good)) == good
    for patch in ({"v": 2}, {"seq": 0}, {"ts": "now"}, {"kind": "nodot"}, {"tx": "3"},
                  {"attrs": []}, {"seq": True}):
        with pytest.raises(TraceSchemaError):
            validate({**good, **patch})
    with pytest.raises(TraceSchemaError):
        validate({**good, "kind": "fiber.state"})  # lacks attrs.old/new
    bad = dict(good)
    del bad["scope"]
    with pytest.raises(TraceSchemaError):
        validate(bad)
    p = tmp_path / "t.jsonl"
    p.write_text(dumps(good) + "\n\n" + dumps({**good, "seq": 2}) + "\n")
    assert [r["seq"] for r in read_jsonl(p)] == [1, 2]
