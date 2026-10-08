"""DeepSeek / OpenAI-compatible adapter against a recorded-style fake HTTP
transport (httpx.MockTransport). Wire facts follow api-docs.deepseek.com
(verified 2026-10-08); see providers/deepseek.py."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import anyio
import httpx
import pytest
from pydantic import BaseModel

from ventri import Kernel, State
from ventri_agent.messages import (
    ChatRequest,
    ContentDelta,
    Done,
    Message,
    ReasoningDelta,
    ToolCall,
    ToolCallStart,
    Usage,
)
from ventri_agent.providers.base import (
    ModelProvider,
    ProviderError,
    ReasoningContentMissing,
    RequestInvalid,
    collect,
    complete_json,
)
from ventri_agent.providers.deepseek import DeepSeekProvider
from ventri_agent.providers.deepseek import plugin as deepseek_plugin
from ventri_agent.providers.pricing import PeakSchedule, PriceTable

pytestmark = pytest.mark.anyio

USAGE = {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
         "prompt_cache_hit_tokens": 768, "prompt_cache_miss_tokens": 232,
         "prompt_tokens_details": {"cached_tokens": 768},
         "completion_tokens_details": {"reasoning_tokens": 20}}


def chunk(delta=None, finish=None, usage=None, model="deepseek-flash"):
    d = {"id": "x", "object": "chat.completion.chunk", "model": model,
         "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        d["usage"] = usage
    return "data: " + json.dumps(d) + "\n\n"


def sse(*parts: str, done=True) -> bytes:
    body = "".join(parts) + ("data: [DONE]\n\n" if done else "")
    return body.encode()


THINK_STREAM = sse(
    ": keep-alive\n\n",
    chunk({"role": "assistant", "content": None, "reasoning_content": "Let me "}),
    ": keep-alive\n\n",
    chunk({"reasoning_content": "think."}),
    chunk({"content": "Hello"}),
    chunk({"content": " world"}),
    chunk({}, finish="stop", usage=USAGE),
)

TOOL_STREAM = sse(
    chunk({"reasoning_content": "need a tool"}),
    chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                           "function": {"name": "fs__read", "arguments": ""}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\": "}}]}),
    chunk({"tool_calls": [{"index": 1, "id": "call_2", "type": "function",
                           "function": {"name": "time__now", "arguments": "{}"}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": "\"a.md\"}"}}]}),
    chunk({}, finish="tool_calls", usage=USAGE),
)


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if callable(r):
            return r(request)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, httpx.Response):
            return r
        return httpx.Response(200, content=r, headers={"content-type": "text/event-stream"})

    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


def provider(rec, **kw) -> DeepSeekProvider:
    http = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    p = DeepSeekProvider(http, api_key="test-key", **kw)
    p.sleeps = []

    async def sleep(d):
        p.sleeps.append(d)
    p.sleep = sleep
    return p


def req(**kw) -> ChatRequest:
    kw.setdefault("messages", [Message.system("sys"), Message.user("hi")])
    kw.setdefault("model", "deepseek-flash")
    return ChatRequest(**kw)


# ------------------------------------------------------------------ stream
async def test_stream_parses_reasoning_content_usage_and_keepalive():
    rec = Recorder(THINK_STREAM)
    p = provider(rec)
    events = [e async for e in p.stream(req(thinking=True, effort="high"))]
    assert [type(e) for e in events] == [ReasoningDelta, ReasoningDelta, ContentDelta, ContentDelta, Done]
    done = events[-1]
    assert done.message.content == "Hello world"
    assert done.message.reasoning_content == "Let me think."
    assert done.finish_reason == "stop"
    u = done.usage
    assert (u.prompt_tokens, u.cache_hit, u.cache_miss, u.reasoning_tokens) == (1000, 768, 232, 20)
    assert u.hit_rate == pytest.approx(0.768)
    h = rec.requests[0].headers
    assert h["authorization"] == "Bearer test-key" and rec.requests[0].url.path == "/chat/completions"


async def test_tool_call_fragments_are_merged_by_index():
    p = provider(Recorder(TOOL_STREAM))
    events = [e async for e in p.stream(req(thinking=True, tools=[{"type": "function", "function": {"name": "fs__read"}}]))]
    starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert [(s.index, s.name) for s in starts] == [(0, "fs__read"), (1, "time__now")]
    done = events[-1]
    assert [(t.id, t.name, t.args()) for t in done.message.tool_calls] == [
        ("call_1", "fs__read", {"path": "a.md"}), ("call_2", "time__now", {})]
    assert done.message.reasoning_content == "need a tool"
    assert done.message.content is None and done.finish_reason == "tool_calls"
    api = done.message.to_api()
    assert api["content"] == "" and api["reasoning_content"] == "need a tool"


async def test_request_body_thinking_effort_and_options():
    rec = Recorder(THINK_STREAM)
    p = provider(rec, user_id="u-jeff")
    await collect(p.stream(req(thinking=True, effort="max", temperature=0.3, max_tokens=64, stop=["END"])))
    await collect(p.stream(req(thinking=False, temperature=0.3, json_output=True,
                               messages=[Message.user("answer in json")])))
    b1, b2 = rec.bodies()
    assert b1["stream"] is True and b1["stream_options"] == {"include_usage": True}
    assert b1["thinking"] == {"type": "enabled"} and b1["reasoning_effort"] == "max"
    assert "temperature" not in b1  # no effect in thinking mode
    assert b1["max_tokens"] == 64 and b1["stop"] == ["END"] and b1["user_id"] == "u-jeff"
    assert b2["thinking"] == {"type": "disabled"} and "reasoning_effort" not in b2
    assert b2["temperature"] == 0.3 and b2["response_format"] == {"type": "json_object"}


async def test_effort_none_disables_thinking():
    rec = Recorder(THINK_STREAM)
    await collect(provider(rec).stream(req(effort="none")))
    assert rec.bodies()[0]["thinking"] == {"type": "disabled"}


async def test_strict_tools_use_the_beta_endpoint():
    rec = Recorder(TOOL_STREAM)
    tools = [{"type": "function", "function": {"name": "a", "parameters": {"type": "object", "properties": {},
                                                                           "required": [], "additionalProperties": False}}}]
    await collect(provider(rec).stream(req(tools=tools, strict=True, thinking=True)))
    r = rec.requests[0]
    assert r.url.path == "/beta/chat/completions"
    assert json.loads(r.content)["tools"][0]["function"]["strict"] is True
    rec2 = Recorder(TOOL_STREAM)
    await collect(provider(rec2).stream(req(tools=tools, thinking=True)))
    assert rec2.requests[0].url.path == "/chat/completions"
    assert "strict" not in json.loads(rec2.requests[0].content)["tools"][0]["function"]


# -------------------------------------------------------------- validation
async def test_reasoning_content_must_be_resent_with_tools_in_thinking_mode():
    rec = Recorder(THINK_STREAM)
    p = provider(rec)
    tools = [{"type": "function", "function": {"name": "a"}}]
    tc = ToolCall("c1", "a", "{}")
    hist = [Message.user("q"), Message.assistant(None, tool_calls=[tc]), Message.tool("c1", "r")]
    with pytest.raises(ReasoningContentMissing):
        await collect(p.stream(req(messages=hist, tools=tools, thinking=True)))
    assert rec.requests == []  # rejected locally, nothing sent
    hist[1].reasoning_content = "because"
    await collect(p.stream(req(messages=hist, tools=tools, thinking=True)))
    sent = rec.bodies()[0]["messages"][1]
    assert sent["reasoning_content"] == "because" and sent["tool_calls"][0]["id"] == "c1"
    # without tools (or without thinking) the rule does not apply
    await collect(p.stream(req(messages=[Message.user("q"), Message.assistant("a"), Message.user("b")], thinking=True)))
    await collect(p.stream(req(messages=hist[:1] + [Message.assistant(None, tool_calls=[tc]), hist[2]],
                               tools=tools, thinking=False)))


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "a"}}])
async def test_forced_tool_choice_is_invalid_in_thinking_mode(choice):
    p = provider(Recorder(THINK_STREAM))
    with pytest.raises(RequestInvalid):
        await collect(p.stream(req(tools=[{"type": "function", "function": {"name": "a"}}],
                                   tool_choice=choice, thinking=True)))
    await collect(p.stream(req(tools=[{"type": "function", "function": {"name": "a"}}],
                               tool_choice=choice, thinking=False)))


async def test_json_output_needs_the_word_json():
    p = provider(Recorder(THINK_STREAM))
    with pytest.raises(RequestInvalid):
        await collect(p.stream(req(json_output=True, messages=[Message.user("list three colours")])))


async def test_dangling_tool_message_is_rejected_locally():
    p = provider(Recorder(THINK_STREAM))
    with pytest.raises(RequestInvalid):
        await collect(p.stream(req(messages=[Message.user("q"), Message.tool("nope", "r")])))


def test_retired_models_rejected_and_legacy_aliases_warned():
    http = httpx.AsyncClient(transport=httpx.MockTransport(Recorder(THINK_STREAM)))
    with pytest.raises(ValueError, match="retired"):
        DeepSeekProvider(http, api_key="k", routes={"default": {"model": "deepseek-chat"}})
    with pytest.raises(ValueError, match="effort"):
        DeepSeekProvider(http, api_key="k", routes={"default": {"model": "deepseek-flash", "effort": "ultra-max"}})
    p = DeepSeekProvider(http, api_key="k", routes={"default": {"model": "deepseek-v4-flash"}})
    assert p.warnings and "legacy" in p.warnings[0]
    assert p.caps_for("deepseek-v4-flash").context == 1_048_576


# ----------------------------------------------------------------- retries
async def test_429_is_retried_honouring_retry_after():
    rec = Recorder(httpx.Response(429, json={"error": {"message": "rate limited"}}, headers={"retry-after": "2"}),
                   THINK_STREAM)
    p = provider(rec)
    done = await collect(p.stream(req()))
    assert done.message.content == "Hello world"
    assert p.sleeps == [2.0] and p.stats.retries == 1 and len(rec.requests) == 2


async def test_5xx_retried_with_backoff_then_gives_up():
    rec = Recorder(httpx.Response(503, json={"error": {"message": "busy"}}))
    p = provider(rec, max_retries=3)
    with pytest.raises(ProviderError) as ei:
        await collect(p.stream(req()))
    assert ei.value.status == 503 and ei.value.retryable
    assert len(rec.requests) == 4 and len(p.sleeps) == 3
    assert p.sleeps[0] <= p.sleeps[1] <= p.sleeps[2] <= 20  # exponential, jittered, capped


@pytest.mark.parametrize("status", [400, 401, 402, 422])
async def test_client_errors_are_not_retried(status):
    rec = Recorder(httpx.Response(status, json={"error": {"message": "bad"}}))
    p = provider(rec)
    with pytest.raises(ProviderError) as ei:
        await collect(p.stream(req()))
    assert ei.value.status == status and not ei.value.retryable and len(rec.requests) == 1
    assert "bad" in str(ei.value)


async def test_transport_errors_are_retried():
    rec = Recorder(httpx.ConnectError("refused"), THINK_STREAM)
    p = provider(rec)
    assert (await collect(p.stream(req()))).message.content == "Hello world"
    assert len(rec.requests) == 2


async def test_mid_stream_break_is_not_replayed():
    broken = sse(chunk({"content": "partial"}), done=False)  # connection drops: no finish_reason
    rec = Recorder(broken, THINK_STREAM)
    p = provider(rec)
    got = []
    with pytest.raises(ProviderError) as ei:
        async for e in p.stream(req()):
            got.append(e)
    assert ei.value.retryable and [type(e) for e in got] == [ContentDelta]
    assert len(rec.requests) == 1  # never re-sent after output was yielded


@pytest.mark.parametrize("finish", ["insufficient_system_resource", "aborted"])
async def test_server_interrupted_generation_is_an_error(finish):
    rec = Recorder(sse(chunk({"content": "half an ans"}), chunk({}, finish=finish, usage=USAGE)))
    with pytest.raises(ProviderError, match=finish) as ei:
        await collect(provider(rec).stream(req()))
    assert ei.value.retryable and len(rec.requests) == 1


async def test_stream_error_chunk_raises():
    rec = Recorder(sse('data: {"error": {"message": "overloaded"}}\n\n', done=False))
    p = provider(rec, max_retries=0)
    with pytest.raises(ProviderError, match="overloaded"):
        await collect(p.stream(req()))


async def test_per_model_concurrency_limit():
    active = 0
    peak = 0

    async def slow(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await anyio.sleep(0.02)
        active -= 1
        return httpx.Response(200, content=THINK_STREAM)

    class AsyncTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return await slow(request)

    p = DeepSeekProvider(httpx.AsyncClient(transport=AsyncTransport()), api_key="k",
                         concurrency={"deepseek-flash": 2})
    async with anyio.create_task_group() as tg:
        for _ in range(6):
            tg.start_soon(collect, p.stream(req()))
    assert peak == 2


# ------------------------------------------------------------------- probe
async def test_probe_updates_caps_and_warns_about_unknown_models():
    models = {"object": "list", "data": [
        {"id": "deepseek-flash", "object": "model", "context_window": 1048576, "max_output_tokens": 393216,
         "input_modalities": ["text", "image"], "effort": {"supported_levels": ["low", "high", "max"]}},
        {"id": "deepseek-v4-pro", "object": "model", "context_window": 1048576, "max_output_tokens": 393216,
         "input_modalities": ["text"]}]}
    rec = Recorder(httpx.Response(200, json=models))
    http = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    p = DeepSeekProvider(http, api_key="k", routes={"default": {"model": "deepseek-flash"},
                                                    "x": {"model": "deepseek-v9"}})
    w = await p.probe()
    assert any("deepseek-v9" in x for x in w)
    assert p.caps_for("deepseek-flash").vision and not p.caps_for("deepseek-v4-pro").vision
    assert rec.requests[0].url.path == "/models"
    p2 = DeepSeekProvider(httpx.AsyncClient(transport=httpx.MockTransport(Recorder(httpx.ConnectError("x")))), api_key="k")
    assert any("probe failed" in x for x in await p2.probe())


# ----------------------------------------------------------------- pricing
def test_peak_and_off_peak_pricing():
    t = PriceTable()
    u = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, cache_hit=500_000, cache_miss=500_000)
    peak = datetime(2026, 10, 8, 2, 30, tzinfo=UTC)       # Thu 10:30 Beijing
    off = datetime(2026, 10, 8, 5, 0, tzinfo=UTC)         # Thu 13:00 Beijing (lunch gap)
    weekend = datetime(2026, 10, 10, 2, 30, tzinfo=UTC)   # Saturday
    full = 0.5 * 0.006 + 0.5 * 0.30 + 1.20
    assert t.price(u, peak, "deepseek-flash").usd == pytest.approx(full)
    assert t.price(u, off, "deepseek-flash").usd == pytest.approx(full / 2)
    assert t.price(u, weekend, "deepseek-flash").usd == pytest.approx(full / 2)
    assert t.price(u, peak, "deepseek-v4-flash").usd == pytest.approx(full)  # legacy alias
    assert t.price(u, peak, "deepseek-v4-pro").usd == pytest.approx(0.5 * 0.044 + 0.5 * 1.32 + 3.96)
    assert t.price(u, peak, "unknown").usd == 0.0
    hol = PriceTable.from_config(holidays=["2026-10-08"])
    assert hol.price(u, peak, "deepseek-flash").usd == pytest.approx(full / 2)
    s = PeakSchedule()
    assert s.next_off_peak(peak) == datetime(2026, 10, 8, 4, 0, tzinfo=UTC)
    assert s.next_off_peak(off) == off
    assert PriceTable.from_config({"deepseek-flash": {"cache_hit": 0, "cache_miss": 1, "output": 1}}) \
        .price(u, off, "deepseek-flash").usd == pytest.approx(0.5 * (0.5 + 1))


# --------------------------------------------------------------- helpers
class Answer(BaseModel):
    answer: int


async def test_complete_json_repairs_once():
    bad = sse(chunk({"content": "{answer: 4"}), chunk({}, finish="stop", usage=USAGE))
    good = sse(chunk({"content": '{"answer": 4}'}), chunk({}, finish="stop", usage=USAGE))
    rec = Recorder(bad, good)
    p = provider(rec)
    value, dones = await complete_json(p, req(messages=[Message.user("reply in json")], thinking=False), Answer)
    assert value.answer == 4 and len(dones) == 2
    assert "invalid" in rec.bodies()[1]["messages"][-1]["content"]


# ------------------------------------------------------------------ plugin
async def test_plugin_reads_key_from_env_and_fails_without(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-test-value")
    async with Kernel() as app:
        f = await app.plugin(deepseek_plugin, {"probe": False})
        assert f.state is State.ACTIVE
        p = app.get(ModelProvider)
        assert isinstance(p, DeepSeekProvider) and p.route("plan").model == "deepseek-v4-pro"
        assert "dummy-test-value" not in app.tree()
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    async with Kernel() as app:
        f = await app.plugin(deepseek_plugin, {"probe": False})
        assert f.state is State.FAILED and "no DeepSeek API key" in str(f.error)
