"""Long-term memory v1 (SQLite FTS5) and session scopes (DESIGN.md 5.3, 5.7)."""
from __future__ import annotations

import anyio
import pytest

from ventri import State
from ventri_agent.memory import LongTermMemory, WorkingMemory
from ventri_agent.session import Budget, SessionInfo
from ventri_agent.sessions import SESSION_KEYS, SessionError

from .harness import Env, call

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------ memory
def test_memory_add_dedupe_search_and_pending(tmp_path):
    m = LongTermMemory(tmp_path / "m.db")
    a, new = m.add("Jeff 喜欢简洁的中文回答", "preference")
    assert new
    b, new2 = m.add("Jeff 喜欢简洁的中文回答。", "preference")  # near duplicate -> merged
    assert not new2 and b.id == a.id
    m.add("Weekly review happens on Sunday evening", "procedure", confidence=0.9)
    m.add("My DeepSeek api key is sk-abcdefghijklmnopqrstuvwx", "fact")
    assert [x.text for x in m.search("简洁")] == ["Jeff 喜欢简洁的中文回答。"]  # 2-char CJK: LIKE fallback
    assert [x.kind for x in m.search("中文回答")] == ["preference"]          # FTS trigram
    assert m.search("Sunday review")[0].kind == "procedure"
    assert m.search("api key") == []  # pending items are not searchable
    pend = m.list(status="pending")
    assert len(pend) == 1 and pend[0].sensitive
    assert m.confirm(pend[0].id) and m.search("DeepSeek")
    assert [x.kind for x in m.top(2)] == ["preference", "fact"]
    md = m.export_markdown()
    assert "## preference" in md and "sensitive" in md
    m2 = LongTermMemory(tmp_path / "m2.db")
    assert m2.import_markdown(md) == 3
    assert m2.import_markdown(md) == 0  # idempotent
    assert m.forget(a.id) and m.get(a.id) is None
    m.close()
    # persisted
    m3 = LongTermMemory(tmp_path / "m.db")
    assert len(m3.list()) == 2


def test_working_memory():
    w = WorkingMemory()
    w.write("plan: 1")
    assert w.write("step 2", "append") == "plan: 1\nstep 2"


async def test_session_end_extracts_memories(tmp_path):
    extraction = {"json": {"memories": [
        {"kind": "preference", "text": "Jeff wants answers in Chinese", "confidence": 0.9},
        {"kind": "fact", "text": "Jeff's bank PIN is 1234", "sensitive": True},
        {"kind": "weird", "text": "ignored"}]}}
    async with Env(tmp_path, [{"content": "好的"}, extraction]) as env:
        s = await env.open()
        await s.turn("以后请用中文回答")
        created = await s.end()
        assert [c.text for c in created] == ["Jeff wants answers in Chinese", "Jeff's bank PIN is 1234"]
        mem = env.kernel.get(LongTermMemory)
        assert [x.text for x in mem.list()] == ["Jeff wants answers in Chinese"]
        assert [x.status for x in mem.list(status="pending")] == ["pending"]
        req = env.provider.requests[-1]
        assert req.json_output and req.thinking is False and "json" in req.messages[0].content.lower()
        assert not s.alive and s.scope.state is State.DISPOSED
        assert [r["state"] for r in env.log_records(s.id) if r["t"] == "state"][-1] == "ended"
        # a later session sees the memory in its frozen snapshot
        env.provider.add({"content": "ok"})
        s2 = await env.open()
        await s2.turn("hi")
        assert "answers in Chinese" in env.provider.requests[-1].messages[1].content


# ---------------------------------------------------------------- sessions
async def test_sessions_are_isolated_scopes(tmp_path):
    async with Env(tmp_path, [{"content": "a"}, {"content": "b"}]) as env:
        a = await env.open()
        b = await env.open()
        for key in SESSION_KEYS:
            if key in (SessionInfo, Budget, WorkingMemory):
                assert a.ctx.get(key) is not b.ctx.get(key)
        assert not env.kernel.has(SessionInfo)  # isolated: invisible from the root
        a.ctx.get(WorkingMemory).write("only in a")
        assert b.ctx.get(WorkingMemory).text == ""
        await a.turn("one")
        assert b.loop.builder.history == []


async def test_dispose_returns_to_baseline_snapshot(tmp_path):
    async with Env(tmp_path, [{"tool_calls": [call("t.write", {"path": "x"})]}, {"content": "a"}]) as env:
        before = env.kernel.snapshot()
        s = await env.open()
        env.choices = ["session"]
        await s.turn("go")
        assert env.kernel.snapshot() != before
        await env.mgr.suspend(s.id)
        assert env.kernel.snapshot() == before
        s2 = await env.open(s.id)
        await s2.end(extract=False)
        assert env.kernel.snapshot() == before


async def test_idle_sweep_suspends_and_resume_restores(tmp_path):
    async with Env(tmp_path, [{"content": "a"}, {"content": "b"}], idle_timeout=0.0) as env:
        s = await env.open()
        await s.turn("one")
        s.ctx.get(WorkingMemory).write("keep me")
        from ventri_agent.session import SessionLog
        s.ctx.get(SessionLog).append("work", text="keep me")
        assert await env.mgr.sweep_idle() == [s.id]
        assert not s.alive and env.mgr.get(s.id) is None
        s2 = await env.open(s.id)
        assert s2.ctx.get(WorkingMemory).text == "keep me"
        r = await s2.turn("two")
        assert r.n == 2


async def test_retention_sweep_ends_old_sessions(tmp_path):
    async with Env(tmp_path, [{"content": "a"}, {"json": {"memories": []}}]) as env:
        env.mgr.cfg.retention_days = 0
        s = await env.open()
        await s.turn("one")
        await env.mgr.suspend(s.id)
        await anyio.sleep(0.01)
        assert await env.mgr.sweep_retention() == [s.id]
        assert env.mgr.list()[0]["state"] == "ended"


async def test_unknown_preset_and_broken_loop_plugin(tmp_path):
    async with Env(tmp_path, [], agents={"bad": {"loop": "no_such_module_xyz:plugin"}}) as env:
        with pytest.raises(SessionError, match="unknown agent preset"):
            await env.open(agent="nope")
        with pytest.raises(Exception):  # noqa: B017 - resolve error surfaces, scope cleaned up
            await env.open(agent="bad")
        assert [c for c in env.mgr_fiber.children] == []


async def test_custom_loop_plugin_via_preset(tmp_path):
    async with Env(tmp_path, [{"content": "x"}],
                   agents={"alt": {"loop": "tests.agent.alt_loop:plugin"}}) as env:
        s = await env.open(agent="alt")
        r = await s.turn("hi")
        assert r.text == "ALT: hi"


async def test_session_survives_provider_replacement(tmp_path):
    """Hot config edit replacing the model provider restarts dependents; the
    session comes back from its log (DESIGN 5.7: crash/restart = resume)."""
    from ventri_agent.providers.fake import FakeProvider

    from .harness import provider_plugin
    async with Env(tmp_path, [{"content": "first"}]) as env:
        s = await env.open()
        await s.turn("one")
        prov_fiber = next(c for c in env.kernel.fiber.children if c.name == "provider:test")
        await env.kernel.replace(prov_fiber, provider_plugin(FakeProvider([{"content": "second"}])))
        s2 = await env.open(s.id)
        r = await s2.turn("two")
        assert r.text == "second"
        assert [m.content for m in s2.loop.builder.history if m.role == "user"] == ["one", "two"]


async def test_sessions_list(tmp_path):
    async with Env(tmp_path, [{"content": "a"}]) as env:
        s = await env.open()
        await s.turn("first question")
        items = env.mgr.list()
        assert items[0]["id"] == s.id and items[0]["title"] == "first question" and items[0]["turns"] == 1
        assert env.mgr.last_id() == s.id
