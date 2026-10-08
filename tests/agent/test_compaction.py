"""Compaction: cross-turn + intra-turn (one long agentic turn), cut selection,
replay byte-equality, the loop guard, the hard context-window check, and old
logs with (removed) ``prune`` records."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from ventri_agent.messages import ChatRequest, Message, ToolCall
from ventri_agent.providers.fake import FakeProvider
from ventri_agent.sessions import SessionError

from .harness import Env, call

pytestmark = pytest.mark.anyio

SUMMARY = "## Goal\nbuild it\n## Done so far\n- ran steps\n## Key findings\n- /app/x.py\n## Current state\nok\n" \
          "## Remaining plan\n- finish"


class SumProvider(FakeProvider):
    """Scripted agent steps; summary calls (no tools, thinking off) get SUMMARY."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.summaries: list[ChatRequest] = []

    def _next(self, req: ChatRequest) -> dict[str, Any]:
        if not req.tools and req.thinking is False:
            self.summaries.append(req)
            return {"content": SUMMARY}
        return super()._next(req)


def provider(soft: int, **caps: Any) -> SumProvider:
    p = SumProvider()
    p._caps = replace(p._caps, soft_context=soft, **caps)
    return p


def echo_steps(n: int, size: int = 2000) -> list[dict[str, Any]]:
    return [{"reasoning": f"step {i} thinking", "tool_calls": [call("t.echo", {"text": f"s{i} " + "e" * size},
                                                                    id=f"c{i}")]} for i in range(n)]


def agent_msgs(builder: Any, n: int, *, calls: int = 2, size: int = 1500, note_every: int = 4) -> None:
    for i in range(n):
        tcs = [ToolCall(f"k{i}_{j}", "t.echo", json.dumps({"text": "a" * 50})) for j in range(calls)]
        builder.append(Message.assistant(None, reasoning=f"r{i}", tool_calls=tcs))
        for tc in tcs:
            builder.append(Message.tool(tc.id, "x" * size, tool="t.echo"))
        if i % note_every == 3:
            builder.append(Message.system(f"[note {i}]", tail="time"))


def well_formed(msgs: list[Message]) -> None:
    """Every tool message follows the assistant message that called it."""
    open_ids: set[str] = set()
    for m in msgs:
        if m.role == "assistant":
            open_ids = {tc.id for tc in m.tool_calls}
        elif m.role == "tool":
            assert m.tool_call_id in open_ids, f"orphan tool result {m.tool_call_id}"
        else:
            open_ids = set() if m.role == "user" else open_ids


# ------------------------------------------------------------------ cut selection
async def test_plan_keeps_user_message_and_last_steps_and_never_orphans(tmp_path):
    async with Env(tmp_path, provider=provider(256_000)) as env:
        s = await env.open()
        b = s.loop.builder
        b.append(Message.user("THE TASK"))
        agent_msgs(b, 12)
        u = next(i for i, m in enumerate(b.history) if m.content == "THE TASK")
        assert b.compaction_plan(10**9, keep_turns=4, keep_steps=3) is None   # under the goal: nothing
        plan = b.compaction_plan(8_000, keep_turns=4, keep_steps=3)
        assert plan is not None and plan.drop == 0 and plan.span is not None
        a, z = plan.span
        assert a == u + 1 and b.history[z].role == "assistant" and plan.kept_steps == 3
        kept = b.history[z:]
        assert sum(m.role == "assistant" for m in kept) == 3
        well_formed(kept)
        b.apply_compaction(0, "", span=plan.span, progress="[progress] P", steps=9, trims=plan.trims)
        h = b.history
        assert h[u].content == "THE TASK" and h[u + 1].role == "system" and h[u + 1].content == "[progress] P"
        assert h[u + 2].role == "assistant" and all(m.reasoning_content for m in h if m.role == "assistant")
        well_formed(h)


async def test_plan_force_without_older_turns_and_fallbacks(tmp_path):
    async with Env(tmp_path, provider=provider(256_000)) as env:
        s = await env.open()
        b = s.loop.builder
        for t in range(3):                                   # three earlier small turns
            b.append(Message.user(f"old {t}"))
            b.append(Message.assistant(f"answer {t}", reasoning=""))
        b.append(Message.user("NOW"))
        agent_msgs(b, 8, size=200)
        # under the goal, force: compacts the current turn's earlier steps
        plan = b.compaction_plan(10**9, keep_turns=4, keep_steps=2, force=True)
        assert plan is not None and plan.drop == 0 and plan.kept_steps == 2
        # cross-turn alone is not enough -> both parts in one plan
        plan = b.compaction_plan(3_500, keep_turns=1, keep_steps=6)
        assert plan is not None and plan.drop > 0 and plan.span is not None and plan.kept_steps < 6
        # nothing fits: fewer kept steps, then all earlier turns, then trimmed kept results
        b.append(Message.assistant(None, reasoning="r", tool_calls=[ToolCall("huge", "t.echo", "{}")]))
        b.append(Message.tool("huge", "H" * 60_000, tool="t.echo"))
        plan = b.compaction_plan(3_000, keep_turns=4, keep_steps=4)
        assert plan is not None and plan.kept_steps == 1
        assert b.history[plan.drop].content == "NOW"          # every earlier turn summarised
        assert plan.trims and "artifact" in plan.trims[0][1] and len(plan.trims[0][1]) < 6_000
        assert (s.info.dir / "artifacts" / "huge-full.txt").read_text() == "H" * 60_000


# ------------------------------------------------------------ long single turn
async def test_long_single_turn_compacts_inside_the_turn_and_keeps_working(tmp_path):
    prov = provider(20_000)        # trigger 12K, goal 6K
    prov.add(*echo_steps(30), {"reasoning": "done", "content": "ALL DONE"})
    events: list[Any] = []
    async with Env(tmp_path, provider=prov) as env:
        s = await env.open()
        r = await s.turn("TASK-XYZ: run thirty steps", events.append)
        assert r.status == "ok" and r.text == "ALL DONE" and r.steps == 31
        recs = env.log_records(s.id)
        compacts = [x for x in recs if x["t"] == "compact"]
        assert compacts and all(c["drop"] == 0 and c["span"] and c["steps"] > 0 for c in compacts)
        assert "## Goal" in compacts[0]["progress"]
        agent_reqs = [q for q in prov.requests if q.tools]
        for q in agent_reqs:
            users = [m.content for m in q.messages if m.role == "user"]
            assert users == ["TASK-XYZ: run thirty steps"]
            well_formed(q.messages)
        assert max(sum(len(json.dumps(m.to_api())) for m in q.messages) for q in agent_reqs) < 60_000
        notices = [e.text for e in events if e.kind == "notice" and "compacted context" in e.text]
        assert len(notices) == len(compacts) and "earlier steps of this turn" in notices[0]
        # the progress summary request saw the task and the clipped steps
        assert "TASK-XYZ" in (prov.summaries[0].messages[1].content or "")
        epochs = [x for x in recs if x["t"] == "prefix"]
        assert len(epochs) == 1 + len(compacts)


async def test_resume_after_intra_turn_compaction_rebuilds_the_same_bytes(tmp_path):
    prov = provider(20_000)
    prov.add(*echo_steps(14), {"reasoning": "", "content": "done"})
    async with Env(tmp_path, provider=prov) as env:
        s = await env.open()
        await s.turn("long task")
        b = s.loop.builder
        assert any(x["t"] == "compact" and x.get("span") for x in env.log_records(s.id))
        before = [m.to_api() for m in b.build(s.loop.route).messages]
        epoch = b.epoch.hash
        await env.mgr.suspend(s.id)
        s2 = await env.open(s.id)
        b2 = s2.loop.builder
        assert b2.epoch.hash == epoch
        assert [m.to_api() for m in b2.build(s2.loop.route).messages] == before
        prov.add({"content": "next"})
        assert (await s2.turn("more")).status == "ok"


# ----------------------------------------------------------------- loop guard
async def test_oversized_kept_steps_are_trimmed_and_compaction_runs_at_most_once_per_step(tmp_path):
    prov = provider(20_000)        # each t.big result (~17K tokens, kept inline) alone exceeds the trigger
    prov.add(*[{"reasoning": "", "tool_calls": [call("t.big", {}, id=f"b{i}")]} for i in range(4)],
             {"reasoning": "", "content": "ok"})
    async with Env(tmp_path, provider=prov, agents={"default": {"inline_tokens": 100_000}}) as env:
        s = await env.open()
        r = await s.turn("big outputs")
        assert r.status == "ok"
        recs = env.log_records(s.id)
        compacts = [x for x in recs if x["t"] == "compact"]
        assert compacts and any(c.get("trim") for c in compacts)
        assert len(compacts) <= r.steps
        # between two agent requests there is at most one compaction (<= 1 progress + 1 earlier summary)
        seq = ["S" if not q.tools else "A" for q in prov.requests]
        assert "SSS" not in "".join(seq)
        assert s.loop.builder.estimate_tokens() < 20_000


async def test_request_over_the_context_window_is_not_sent(tmp_path):
    prov = provider(20_000, context=20_000, max_output=1_000)
    async with Env(tmp_path, provider=prov) as env:
        s = await env.open()
        r = await s.turn("q " + "w" * 100_000)               # ~30K tokens in one user message
        assert r.status == "error" and "context too large" in (r.reason or "")
        assert prov.requests == []


# ------------------------------------------------------------- config / old logs
async def test_compaction_preset_keys_and_validation(tmp_path):
    async with Env(tmp_path, agents={"a": {"compact_at": 0.5, "compact_keep_turns": 2,
                                           "compact_keep_steps": 3}}) as env:
        s = await env.open(agent="a")
        assert (s.info.agent.compact_at, s.info.agent.compact_keep_turns, s.info.agent.compact_keep_steps) == (0.5, 2, 3)
    async with Env(tmp_path, agents={"bad": {"compact_at": 1.5}}) as env:
        with pytest.raises(SessionError, match="compact_at"):
            await env.open(agent="bad")


async def test_old_log_with_prune_records_loads_with_full_messages(tmp_path):
    prov = FakeProvider([{"tool_calls": [call("t.echo", {"text": "long " * 200}, id="p1")]}, {"content": "a"}])
    async with Env(tmp_path, provider=prov) as env:
        s = await env.open()
        await s.turn("first")
        sid = s.id
        seq = next(x["m"]["meta"]["seq"] for x in env.log_records(sid) if x["t"] == "msg" and x["m"]["role"] == "tool")
        await env.mgr.suspend(sid)
        log = tmp_path / "sessions" / f"{sid}.jsonl"
        with log.open("a") as f:     # what a 0.2.0a1 dev build wrote
            f.write(json.dumps({"t": "prune", "ts": 0, "edits": [{"seq": seq, "content": "[stub]"}], "saved": 300}) + "\n")
        s2 = await env.open(sid)
        tool = [m for m in s2.loop.builder.history if m.role == "tool"]
        assert tool and (tool[0].content or "").startswith("echo: long")
        prov.add({"content": "b"})
        assert (await s2.turn("again")).status == "ok"


async def test_failed_summary_is_not_retried_every_step(tmp_path):
    class Failing(SumProvider):
        def _next(self, req: ChatRequest) -> dict[str, Any]:
            if not req.tools and req.thinking is False:
                self.summaries.append(req)
                return {"error": 400}
            return FakeProvider._next(self, req)
    prov = Failing()
    prov._caps = replace(prov._caps, soft_context=20_000)
    prov.add(*echo_steps(16), {"reasoning": "", "content": "done"})
    events: list[Any] = []
    async with Env(tmp_path, provider=prov) as env:
        s = await env.open()
        r = await s.turn("task", events.append)
        assert r.status == "ok"
        assert any(e.kind == "error" and "compaction failed" in e.text for e in events)
        assert 1 <= len(prov.summaries) <= 4        # once per ~soft/10 of growth, not once per step
