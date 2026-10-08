"""Cache-friendly ContextBuilder (DESIGN.md 5.3) and the M2 criterion "input
cache hit rate >= 70% over sessions of >= 5 turns", measured with the fake
provider's simulation of DeepSeek's prefix-unit disk cache."""
from __future__ import annotations

import itertools
import json

import pytest

from ventri_agent.tools.registry import Risk, Tool

from .harness import Env, call

pytestmark = pytest.mark.anyio


def is_prefix(a, b) -> bool:
    """Request ``a`` is a strict prefix of request ``b`` (tools and messages)."""
    if a.tools != b.tools:
        return False
    am = [json.dumps(m.to_api(), sort_keys=True) for m in a.messages]
    bm = [json.dumps(m.to_api(), sort_keys=True) for m in b.messages]
    return bm[:len(am)] == am


def realistic_turn(i: int) -> list:
    return [{"reasoning": f"plan {i}", "tool_calls": [call("t.echo", {"text": f"note {i}"})]},
            {"reasoning": f"answer {i}", "content": f"Here is answer {i}. " + "detail " * 40}]


async def test_every_request_extends_the_previous_one(tmp_path):
    script = [s for i in range(6) for s in realistic_turn(i)]
    async with Env(tmp_path, script) as env:
        s = await env.open()
        for i in range(6):
            await s.turn(f"question {i}: " + "context " * 30)
        reqs = env.provider.requests
        assert len(reqs) == 12
        for a, b in itertools.pairwise(reqs):
            assert is_prefix(a, b)


async def test_cache_hit_rate_at_least_70_percent_over_5_turns(tmp_path):
    script = [s for i in range(8) for s in realistic_turn(i)]
    async with Env(tmp_path, script) as env:
        s = await env.open()
        results = [await s.turn(f"q{i} " + "words " * 20) for i in range(8)]
        hit = sum(r.usage.cache_hit for r in results)
        total = sum(r.usage.prompt_tokens for r in results)
        print(f"simulated cache hit rate over 8 turns: {hit / total:.1%}")
        assert total > 0 and hit / total >= 0.70, hit / total
        assert s.loop.totals.hit_rate >= 0.70
        # the per-call usage records (what `va cost` reports) agree
        recs = [r for r in env.log_records(s.id) if r["t"] == "usage"]
        rate = sum(r["usage"]["cache_hit"] for r in recs) / sum(r["usage"]["prompt_tokens"] for r in recs)
        assert rate >= 0.70


async def test_time_notes_and_notices_are_appended_not_rewritten(tmp_path):
    async with Env(tmp_path, [{"content": "a"}, {"content": "b"}, {"content": "c"}]) as env:
        s = await env.open()
        await s.turn("one")
        s.loop._last_time_note = 0  # force a new time note on the next turn
        await s.turn("two")
        a, b = env.provider.requests[:2]
        assert is_prefix(a, b)
        tails = [m for m in b.messages if m.meta.get("tail") == "time"]
        assert len(tails) == 2 and "current time" in tails[1].content


async def test_tool_set_change_waits_for_the_next_epoch(tmp_path):
    async with Env(tmp_path, [{"content": "a"}, {"content": "b"}, {"content": "c"}]) as env:
        s = await env.open()
        await s.turn("one")
        epoch_tools = list(s.loop.builder.epoch.tool_names)
        # load a new tool plugin and unload the test tools (hot config edit)
        newt = Tool("t.new", "new tool", lambda a, tc: "new", None, Risk.READ)
        env.registry.register(None, newt)
        await env.tools_fiber.dispose()
        events = []
        await s.turn("two", events.append)
        a, b = env.provider.requests[:2]
        assert is_prefix(a, b)  # the prefix (incl. tool list) did not change mid-epoch
        notice = [e.text for e in events if e.kind == "notice"]
        assert notice and "t.new" in notice[0] and "t.echo" in notice[0]
        assert s.loop.builder.epoch.tool_names == epoch_tools
        # a removed tool is no longer callable even though it is still listed
        assert s.loop.builder.tool("t.echo") is None
        assert s.loop.builder.tool("t.new") is None  # not in this epoch yet
        s.loop.builder.new_epoch()
        assert "t.new" in s.loop.builder.epoch.tool_names and "t.echo" not in s.loop.builder.epoch.tool_names
        await s.turn("three")
        c = env.provider.requests[2]
        assert [t["function"]["name"] for t in c.tools] == sorted(t["function"]["name"] for t in c.tools)
        assert "t__new" in [t["function"]["name"] for t in c.tools]


async def test_tools_are_sorted_and_canonical(tmp_path):
    async with Env(tmp_path, [{"content": "a"}]) as env:
        s = await env.open()
        await s.turn("one")
        tools = env.provider.requests[0].tools
        names = [t["function"]["name"] for t in tools]
        assert names == sorted(names)
        for t in tools:
            assert json.dumps(t, sort_keys=True) == json.dumps(t)  # keys already sorted


async def test_memory_snapshot_is_frozen_for_the_epoch(tmp_path):
    async with Env(tmp_path, [{"content": "a"}, {"content": "b"}]) as env:
        from ventri_agent.memory import LongTermMemory
        mem = env.kernel.get(LongTermMemory)
        mem.add("Jeff prefers concise answers", "preference")
        s = await env.open()
        await s.turn("one")
        mem.add("Jeff lives in Shanghai", "fact")
        await s.turn("two")
        a, b = env.provider.requests
        assert is_prefix(a, b)
        assert "concise" in b.messages[1].content and "Shanghai" not in b.messages[1].content


async def test_agent_preset_selects_tools_and_persona(tmp_path):
    agents = {"reader": {"persona": "You only read.", "tools": ["t.echo", "t.web"], "route": "cheap"}}
    async with Env(tmp_path, [{"content": "a"}], agents=agents) as env:
        s = await env.open(agent="reader")
        await s.turn("one")
        r = env.provider.requests[0]
        assert sorted(t["function"]["name"] for t in r.tools) == ["t__echo", "t__web"]
        assert r.messages[0].content.startswith("You only read.")
        assert r.thinking is False  # cheap route
