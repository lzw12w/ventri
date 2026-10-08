"""Opt-in contract tests against the real DeepSeek API (DESIGN.md 8, M2: "DeepSeek
contract tests, nightly, real API, small budget").

Skipped unless ``DEEPSEEK_API_KEY`` is set. Prompts are tiny and ``max_tokens``
is low: a full run costs well under 0.01 USD. Run them alone with
``pytest -m live -s`` (``-s`` prints the measured usage / cache numbers).
The key is read from the environment only and never printed.
"""
from __future__ import annotations

import json
import os

import httpx
import pytest
from pydantic import BaseModel, Field

from ventri_agent.messages import ChatRequest, ContentDelta, Done, Message, ReasoningDelta, ToolCallStart
from ventri_agent.providers.base import ReasoningContentMissing, collect, complete_json
from ventri_agent.providers.deepseek import DeepSeekProvider
from ventri_agent.tools.registry import Tool

from .harness import Env

KEY = os.environ.get("DEEPSEEK_API_KEY")
pytestmark = [pytest.mark.anyio, pytest.mark.live,
              pytest.mark.skipif(not KEY, reason="DEEPSEEK_API_KEY not set (live contract tests are opt-in)")]

ROUTES = {"default": {"model": "deepseek-flash", "thinking": True, "effort": "low", "max_tokens": 1024},
          "plan": {"model": "deepseek-v4-pro", "thinking": True, "effort": "high", "max_tokens": 1024},
          "cheap": {"model": "deepseek-flash", "thinking": False, "max_tokens": 400}}


@pytest.fixture
async def ds():
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15)) as http:
        yield DeepSeekProvider(http, api_key=KEY, routes=ROUTES, max_retries=2)


def report(name: str, **kv) -> None:
    print(f"\n[live] {name}: " + json.dumps(kv, ensure_ascii=False, default=str))


class WeatherArgs(BaseModel):
    city: str = Field(description="City name, e.g. Shanghai")
    unit: str = Field("celsius", description="celsius or fahrenheit")


WEATHER = Tool("weather.get", "Get the current weather for a city.", lambda a, tc: None, WeatherArgs)


async def test_models_endpoint(ds):
    warnings = await ds.probe()
    caps = ds.caps_for("deepseek-flash")
    report("models", warnings=warnings, flash_context=caps.context, flash_max_output=caps.max_output,
           flash_effort=caps.effort_levels, pro_vision=ds.caps_for("deepseek-v4-pro").vision)
    assert not [w for w in warnings if "not listed" in w]
    assert caps.context >= 1_000_000


async def test_thinking_mode_streams_reasoning_and_usage(ds):
    req = ds.route("default").request([Message.user("What is 17 * 23? Reply with the number only.")],
                                      effort="high", max_tokens=800)
    events = [e async for e in ds.stream(req)]
    done = events[-1]
    assert isinstance(done, Done)
    n_r = sum(isinstance(e, ReasoningDelta) for e in events)
    n_c = sum(isinstance(e, ContentDelta) for e in events)
    u = done.usage
    report("thinking", model=done.model, finish=done.finish_reason, reasoning_deltas=n_r, content_deltas=n_c,
           reasoning_chars=len(done.message.reasoning_content or ""), content=done.message.content,
           usage=u.to_json())
    assert "391" in (done.message.content or "")
    assert done.message.reasoning_content and n_r >= 1 and n_c >= 1 and len(events) > 3  # streamed
    assert u.prompt_tokens == u.cache_hit + u.cache_miss and u.reasoning_tokens > 0


async def test_non_thinking_mode(ds):
    done = await collect(ds.stream(ds.route("cheap").request([Message.user("Say OK.")], max_tokens=10)))
    report("non-thinking", content=done.message.content, reasoning=done.message.reasoning_content,
           usage=done.usage.to_json())
    assert done.message.reasoning_content is None and done.usage.reasoning_tokens == 0


@pytest.mark.parametrize("strict", [False, True], ids=["plain-tools", "strict-tools"])
async def test_reasoning_content_round_trip_with_tools(ds, strict):
    tools = [WEATHER.spec(strict=True)]
    msgs = [Message.system("Use the weather tool to answer weather questions."),
            Message.user("What's the weather in Shanghai right now? Use the tool.")]
    r1 = ds.route("default").request(msgs, tools=tools, strict=strict, effort="high")
    events = [e async for e in ds.stream(r1)]
    d1 = events[-1]
    assert isinstance(d1, Done)
    assert d1.finish_reason == "tool_calls" and d1.message.tool_calls, d1.message.content
    tc = d1.message.tool_calls[0]
    args = WEATHER.parse(tc.args())  # validates against the schema
    assert tc.name == "weather__get" and "shanghai" in args.city.lower()
    assert any(isinstance(e, ToolCallStart) for e in events)
    assert d1.message.reasoning_content is not None
    # follow-up: the assistant message goes back *with* reasoning_content
    hist = [*msgs, d1.message, Message.tool(tc.id, json.dumps({"city": "Shanghai", "temp_c": 21, "sky": "cloudy"}))]
    r2 = ds.route("default").request(hist, tools=tools, strict=strict, effort="high")
    d2 = await collect(ds.stream(r2))
    report("tool round-trip" + (" (strict, /beta)" if strict else ""), call=tc.name, args=tc.args(),
           reasoning_chars_1=len(d1.message.reasoning_content or ""), answer=d2.message.content,
           usage_1=d1.usage.to_json(), usage_2=d2.usage.to_json())
    assert d2.finish_reason == "stop" and "21" in (d2.message.content or "")
    # the local validator refuses the request DeepSeek documents as a 400
    bad = [*msgs, Message.assistant(None, tool_calls=d1.message.tool_calls), hist[-1]]
    with pytest.raises(ReasoningContentMissing):
        await collect(ds.stream(ds.route("default").request(bad, tools=tools)))


async def test_forced_tool_choice_in_thinking_mode_is_a_400(ds):
    """Contract check of the documented restriction (sent raw, bypassing local validation)."""
    body = ds.body(ChatRequest(messages=[Message.user("weather in Paris?")], model="deepseek-flash",
                               tools=[WEATHER.spec()], tool_choice="required", thinking=True, effort="low",
                               max_tokens=50))
    r = await ds.http.post(f"{ds.base_url}/chat/completions", json=body, headers=ds.headers())
    report("tool_choice=required + thinking", status=r.status_code, message=r.json().get("error", {}).get("message"))
    assert r.status_code == 400


class Colours(BaseModel):
    colours: list[str]


async def test_json_output(ds):
    req = ds.route("cheap").request([Message.system("Reply in json."),
                                     Message.user('List three primary colours as {"colours": [...]}.')],
                                    max_tokens=60)
    value, dones = await complete_json(ds, req, Colours)
    report("json", value=value.model_dump(), calls=len(dones))
    assert len(value.colours) == 3


async def test_agent_session_cache_hit_rate_over_5_turns(tmp_path):
    """M2 criterion on the real API: >= 5 turns, input cache hit rate from `usage`."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15)) as http:
        prov = DeepSeekProvider(http, api_key=KEY, routes=ROUTES, max_retries=2)
        async with Env(tmp_path, provider=prov, extract_memory=False) as env:
            s = await env.open()
            questions = ["Hi! In one short sentence: what is a prefix cache?",
                         "Use t.echo to echo the word 'ventri', then tell me what it returned.",
                         "In one short sentence: why does appending instead of rewriting help caching?",
                         "What time is it in Shanghai? If you have no tool for that, just say so briefly.",
                         "Summarise our conversation in one sentence.",
                         "Thanks, that's all. Reply with 'bye'."]
            results = [await s.turn(q) for q in questions]
            per = [(r.usage.prompt_tokens, r.usage.cache_hit) for r in results]
            hit = sum(h for _, h in per)
            total = sum(p for p, _ in per)
            recs = [r for r in env.log_records(s.id) if r["t"] == "usage"]
            report("agent session", turns=len(results), calls=len(recs), statuses=[r.status for r in results],
                   per_turn_prompt_hit=per, hit_rate=round(hit / total, 3),
                   cost_usd=round(sum(r.cost_usd for r in results), 6),
                   reasoning_tokens=sum(r.usage.reasoning_tokens for r in results))
            assert all(r.status == "ok" for r in results)
            assert any(c[0] == "t.echo" for c in env.probe.calls)
            assert hit / total >= 0.70
