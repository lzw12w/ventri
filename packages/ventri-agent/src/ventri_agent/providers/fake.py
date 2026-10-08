"""A scripted, offline ``ModelProvider`` for tests, demos and ``va chat --fake``.

Each model call consumes the next scripted step (a dict, or the return value of
a callable ``step(request)``)::

    {"reasoning": "...", "content": "...",
     "tool_calls": [{"name": "fs.read", "arguments": {"path": "a.md"}}],
     "error": 503}            # raise a ProviderError instead (retryable for 429/5xx)

When the script is exhausted it echoes the last user message. It also
*simulates DeepSeek's disk cache*: every request persists two prefix units
(end of input, end of output) and a later request hits the longest persisted
unit it fully extends -- so cache-friendliness of the prompt layout is testable
offline. Token counts are estimated (4 characters per token).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
from pydantic import BaseModel, Field

import ventri

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
    new_id,
)
from .base import CallStats, ModelCaps, ModelProvider, ProviderError, Route, route_from
from .pricing import PriceTable

Step = dict[str, Any] | Callable[[ChatRequest], dict[str, Any]]


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def _seg(obj: Any) -> tuple[str, int]:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()[:16], estimate_tokens(s)


class FakeProvider:
    name = "fake"

    def __init__(self, script: list[Step] | None = None, *, model: str = "deepseek-flash",
                 routes: dict[str, Any] | None = None, caps: ModelCaps | None = None,
                 chunk_delay: float = 0.0, echo: bool = True) -> None:
        self.script: list[Step] = list(script or [])
        self.routes: dict[str, Route] = {n: route_from(n, s) for n, s in (routes or {
            "default": {"model": model, "thinking": True, "effort": "high"},
            "plan": {"model": "deepseek-v4-pro", "thinking": True, "effort": "max"},
            "cheap": {"model": model, "thinking": False}}).items()}
        self._caps = caps or ModelCaps(context=1_048_576, max_output=393_216, thinking=True,
                                       effort_levels=("low", "high", "max"), strict=True,
                                       cache="prefix-disk", soft_context=256_000)
        self.chunk_delay = chunk_delay
        self.echo = echo
        self.requests: list[ChatRequest] = []
        self.units: set[tuple[str, ...]] = set()
        self.prices = PriceTable()
        self.stats = CallStats()
        self.strict_tools = False

    @property
    def caps(self) -> ModelCaps:
        return self._caps

    def caps_for(self, model: str) -> ModelCaps:
        return self._caps

    def route(self, name: str) -> Route:
        return self.routes[name]

    def price(self, usage: Usage, at: datetime, model: str | None = None) -> Money:
        return self.prices.price(usage, at, model or self.routes["default"].model)

    def add(self, *steps: Step) -> FakeProvider:
        self.script.extend(steps)
        return self

    # ------------------------------------------------------------- caching
    def _segments(self, req: ChatRequest) -> list[tuple[str, int]]:
        segs = [_seg(sorted(json.dumps(t, sort_keys=True) for t in req.tools))] if req.tools else []
        return segs + [_seg(m.to_api()) for m in req.messages]

    def _usage(self, req: ChatRequest, out: Message) -> Usage:
        segs = self._segments(req)
        keys = tuple(h for h, _ in segs)
        hit_n = max((n for n in range(len(keys), 0, -1) if keys[:n] in self.units), default=0)
        total = sum(t for _, t in segs)
        hit = sum(t for _, t in segs[:hit_n])
        out_seg = _seg(out.to_api())
        self.units.add(keys)
        self.units.add((*keys, out_seg[0]))
        reasoning = estimate_tokens(out.reasoning_content or "")
        completion = reasoning + estimate_tokens((out.content or "") + "".join(
            t.name + t.arguments for t in out.tool_calls))
        return Usage(total, completion, hit, total - hit, reasoning)

    # -------------------------------------------------------------- stream
    def _next(self, req: ChatRequest) -> dict[str, Any]:
        if self.script:
            step = self.script.pop(0)
            return step(req) if callable(step) else dict(step)
        if not self.echo:
            raise ProviderError("fake provider script exhausted")
        last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        return {"content": f"(fake) {last}"}

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(req)
        self.stats.calls += 1
        step = self._next(req)
        if step.get("error"):
            status = int(step["error"])
            self.stats.errors += 1
            raise ProviderError(f"HTTP {status}: scripted failure", status=status,
                                retryable=status == 429 or status >= 500)
        thinking = req.thinking is not False
        reasoning = step.get("reasoning", "") if thinking else None
        if req.json_output and "json" in step:
            step["content"] = json.dumps(step["json"], ensure_ascii=False)
        for piece in _chunks(reasoning or ""):
            await self._tick()
            yield ReasoningDelta(piece)
        content = step.get("content")
        for piece in _chunks(content or ""):
            await self._tick()
            yield ContentDelta(piece)
        calls = []
        for i, tc in enumerate(step.get("tool_calls") or []):
            args = tc.get("arguments", {})
            call = ToolCall(tc.get("id") or new_id("call_"), tc["name"],
                            args if isinstance(args, str) else json.dumps(args, ensure_ascii=False))
            calls.append(call)
            yield ToolCallStart(i, call.id, call.name)
        msg = Message.assistant(content, reasoning=reasoning, tool_calls=calls, model=req.model,
                                thinking=thinking)
        usage = self._usage(req, msg)
        self.stats.usage = self.stats.usage + usage
        yield Done(msg, usage, step.get("finish_reason") or ("tool_calls" if calls else "stop"), req.model)

    async def _tick(self) -> None:
        if self.chunk_delay:
            await anyio.sleep(self.chunk_delay)


def _chunks(text: str, size: int = 12) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def load_script(path: str | Path) -> list[Step]:
    """A JSON or YAML file holding a list of steps (see module docstring)."""
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yml", ".yaml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"{p}: a fake script is a list of steps")  # noqa: TRY004 - bad file content
    return data


class FakeConfig(BaseModel):
    script: list[dict[str, Any]] = Field(default_factory=list)
    script_file: str | None = None
    chunk_delay: float = 0.0


@ventri.plugin(name="provider:fake", config=FakeConfig, provides={"llm": ModelProvider})
def fake(ctx: Any, cfg: FakeConfig) -> None:
    """``use: ventri_agent.providers.fake`` -- offline scripted model."""
    steps: list[Step] = list(cfg.script)
    if cfg.script_file:
        steps += load_script(cfg.script_file)
    ctx.provide(ModelProvider, FakeProvider(steps, chunk_delay=cfg.chunk_delay))


plugin = fake
