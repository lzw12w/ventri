"""OpenAI-compatible Chat Completions adapter (httpx, no ``openai`` SDK).

Shared by the DeepSeek adapter and usable on its own for any OpenAI-compatible
endpoint (vLLM, Ollama, ...): ``use: ventri_agent.providers.openai_compat``.

Behaviour:

* always streams (``stream: true`` + ``stream_options.include_usage``) and
  parses SSE by hand: ``: keep-alive`` comments and blank lines are skipped,
  ``data: [DONE]`` ends the stream, tool-call fragments are merged by index;
* ``reasoning_content`` deltas (DeepSeek, vLLM reasoning parsers) are surfaced
  as ``ReasoningDelta`` and kept on the final message;
* 429 / 5xx / transport errors are retried with exponential backoff and
  jitter (``Retry-After`` honoured) **only before the first event was
  yielded** -- a stream that breaks mid-way raises ``ProviderError(retryable=True)``
  so the caller (the agent loop) decides; nothing half-received is replayed;
* one ``anyio.Semaphore`` per model bounds concurrency (DeepSeek account
  limits: Flash 2500, Pro 500).
"""
from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import anyio
import httpx
from pydantic import BaseModel, Field

import ventri
from ventri import Secret

from ..messages import (
    ChatEvent,
    ChatRequest,
    ContentDelta,
    Done,
    Message,
    Money,
    ReasoningDelta,
    ToolCall,
    ToolCallStart,
    Usage,
)
from .base import CallStats, ModelCaps, ModelProvider, ProviderError, RequestInvalid, Route, route_from
from .pricing import PriceTable

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass
class _Partial:
    id: str = ""
    name: str = ""
    args: list[str] = field(default_factory=list)


class OpenAICompatProvider:
    """``ModelProvider`` over an OpenAI-compatible ``/chat/completions`` endpoint."""

    name = "openai-compat"

    def __init__(self, http: httpx.AsyncClient, *, base_url: str, api_key: str | None,
                 routes: dict[str, Any], caps: dict[str, ModelCaps] | None = None,
                 default_caps: ModelCaps | None = None, prices: PriceTable | None = None,
                 concurrency: dict[str, int] | None = None, default_concurrency: int = 64,
                 max_retries: int = 4, backoff_base: float = 0.5, backoff_cap: float = 20.0,
                 user_id: str | None = None, strict_tools: bool = False) -> None:
        self.http = http
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.routes: dict[str, Route] = {n: route_from(n, s) for n, s in routes.items()}
        if "default" not in self.routes:
            raise ValueError("routes must define 'default'")
        self._caps = dict(caps or {})
        self._default_caps = default_caps or ModelCaps()
        self.prices = prices or PriceTable(prices={}, aliases={})
        self._concurrency = dict(concurrency or {})
        self._default_concurrency = default_concurrency
        self._sems: dict[str, anyio.Semaphore] = {}
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.user_id = user_id
        self.strict_tools = strict_tools  # ContextBuilder sends strict tool schemas when True
        self.stats = CallStats()
        self.sleep = anyio.sleep  # patched by tests

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.base_url} routes={sorted(self.routes)}>"

    # ------------------------------------------------------------- protocol
    @property
    def caps(self) -> ModelCaps:
        return self.caps_for(self.routes["default"].model)

    def caps_for(self, model: str) -> ModelCaps:
        return self._caps.get(model, self._default_caps)

    def route(self, name: str) -> Route:
        try:
            return self.routes[name]
        except KeyError:
            raise KeyError(f"no model route {name!r} (have: {', '.join(sorted(self.routes))})") from None

    def price(self, usage: Usage, at: datetime, model: str | None = None) -> Money:
        return self.prices.price(usage, at, model or self.routes["default"].model)

    # ---------------------------------------------------------- request body
    def endpoint(self, req: ChatRequest) -> str:
        return f"{self.base_url}/chat/completions"

    def validate(self, req: ChatRequest) -> None:
        if not req.messages:
            raise RequestInvalid("request has no messages")
        ids = {tc.id for m in req.messages if m.role == "assistant" for tc in m.tool_calls}
        for m in req.messages:
            if m.role == "tool" and m.tool_call_id not in ids:
                raise RequestInvalid(f"tool message answers unknown tool_call_id {m.tool_call_id!r}")

    def body(self, req: ChatRequest) -> dict[str, Any]:
        b: dict[str, Any] = {
            "model": req.model,
            "messages": [m.to_api() for m in req.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if req.tools:
            b["tools"] = [dict(t, function={**t["function"], "strict": True}) if req.strict else t
                          for t in req.tools]
            if req.tool_choice is not None:
                b["tool_choice"] = req.tool_choice
        if req.max_tokens is not None:
            b["max_tokens"] = req.max_tokens
        if req.json_output:
            b["response_format"] = {"type": "json_object"}
        if req.temperature is not None:
            b["temperature"] = req.temperature
        if req.stop:
            b["stop"] = req.stop
        b.update(req.extra)
        return b

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _sem(self, model: str) -> anyio.Semaphore:
        s = self._sems.get(model)
        if s is None:
            s = self._sems[model] = anyio.Semaphore(self._concurrency.get(model, self._default_concurrency))
        return s

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(self.backoff_cap, max(0.0, float(retry_after)))
            except ValueError:
                pass
        d = min(self.backoff_cap, self.backoff_base * 2 ** attempt)
        return d * (0.5 + random.random() / 2)

    # -------------------------------------------------------------- stream
    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.validate(req)
        body = self.body(req)
        url = self.endpoint(req)
        attempt = 0
        async with self._sem(req.model):
            while True:
                yielded = False
                try:
                    self.stats.calls += 1
                    async with self.http.stream("POST", url, json=body, headers=self.headers()) as resp:
                        if resp.status_code >= 400:
                            raw = (await resp.aread()).decode("utf-8", "replace")
                            raise _http_error(resp.status_code, raw, resp.headers.get("retry-after"))
                        async for ev in self._parse(resp, req):
                            yielded = True
                            yield ev
                    return
                except _Retryable as e:
                    err: ProviderError = e.error
                except httpx.TransportError as e:
                    err = ProviderError(f"transport error: {type(e).__name__}: {e}", retryable=True)
                except ProviderError as e:
                    err = e
                if not err.retryable or yielded or attempt >= self.max_retries:
                    self.stats.errors += 1
                    raise err
                self.stats.retries += 1
                await self.sleep(self._delay(attempt, getattr(err, "retry_after", None)))
                attempt += 1

    async def _parse(self, resp: httpx.Response, req: ChatRequest) -> AsyncIterator[ChatEvent]:
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, _Partial] = {}
        usage = Usage()
        finish: str | None = None
        model = req.model
        saw_reasoning = False
        async for line in resp.aiter_lines():
            if not line or line.startswith(":"):
                continue  # blank separators and SSE keep-alive comments
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as e:
                raise ProviderError(f"malformed stream chunk: {data[:200]!r}", retryable=True) from e
            if "error" in chunk:
                err = chunk["error"] or {}
                raise ProviderError(f"stream error: {err.get('message', err)}", retryable=True, body=chunk)
            model = chunk.get("model") or model
            if chunk.get("usage"):
                usage = Usage.from_api(chunk["usage"])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                r = delta.get("reasoning_content")
                if r:
                    saw_reasoning = True
                    reasoning.append(r)
                    yield ReasoningDelta(r)
                elif r == "":
                    saw_reasoning = True
                c = delta.get("content")
                if c:
                    content.append(c)
                    yield ContentDelta(c)
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index", len(calls)))
                    p = calls.get(idx)
                    fn = tc.get("function") or {}
                    if p is None:
                        p = calls[idx] = _Partial(tc.get("id") or "", fn.get("name") or "")
                        yield ToolCallStart(idx, p.id, p.name)
                    else:
                        p.id = p.id or tc.get("id") or ""
                        p.name = p.name or fn.get("name") or ""
                    if fn.get("arguments"):
                        p.args.append(fn["arguments"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        if finish is None:
            raise ProviderError("stream ended before finish_reason (connection dropped?)", retryable=True)
        tool_calls = [ToolCall(p.id or f"call_{i}", p.name, "".join(p.args) or "{}")
                      for i, p in sorted(calls.items())]
        thinking = saw_reasoning or bool(req.thinking)
        msg = Message.assistant("".join(content) or None,
                                reasoning="".join(reasoning) if thinking else None,
                                tool_calls=tool_calls, model=model, thinking=thinking)
        self.stats.usage = self.stats.usage + usage
        yield Done(msg, usage, finish, model)

    async def aclose(self) -> None:
        await self.http.aclose()


class _Retryable(Exception):
    def __init__(self, error: ProviderError) -> None:
        self.error = error


def _http_error(status: int, raw: str, retry_after: str | None) -> Exception:
    try:
        detail = json.loads(raw).get("error", {})
        msg = detail.get("message") if isinstance(detail, dict) else str(detail)
    except (ValueError, AttributeError):
        msg = raw[:500]
    err = ProviderError(f"HTTP {status}: {msg or raw[:200]}", status=status,
                        retryable=status in RETRY_STATUS, body=raw[:2000])
    err.retry_after = retry_after  # type: ignore[attr-defined]
    return _Retryable(err) if err.retryable else err


def now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------- plugin
class OpenAICompatConfig(BaseModel):
    base_url: str
    api_key: Secret[str] | None = None
    routes: dict[str, Any] = Field(default_factory=lambda: {"default": {"model": "default"}})
    context: int = 128_000
    max_output: int = 8_192
    concurrency: int = 16
    timeout: float = 600.0
    max_retries: int = 4


@ventri.plugin(name="provider:openai-compat", config=OpenAICompatConfig, provides={"llm": ModelProvider})
async def openai_compat(ctx: Any, cfg: OpenAICompatConfig) -> None:
    """``use: ventri_agent.providers.openai_compat`` -- any OpenAI-compatible endpoint."""
    http = await ctx.enter(httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout, connect=15.0)))
    caps = ModelCaps(context=cfg.context, max_output=cfg.max_output, soft_context=cfg.context)
    prov = OpenAICompatProvider(http, base_url=cfg.base_url,
                                api_key=cfg.api_key.reveal() if cfg.api_key else None,
                                routes=cfg.routes, default_caps=caps,
                                default_concurrency=cfg.concurrency, max_retries=cfg.max_retries)
    ctx.provide(ModelProvider, prov)


plugin = openai_compat
