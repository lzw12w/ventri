"""Wire-neutral chat types shared by providers, the context builder, the loop and channels.

``Message`` mirrors the OpenAI Chat Completions message shape (which DeepSeek
uses) plus ``reasoning_content`` (thinking mode) and a ``meta`` dict that is
persisted in the session log but never sent to a model.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:12]


def now_ts() -> float:
    return datetime.now(UTC).timestamp()


@dataclass
class ToolCall:
    """A tool call produced by the model. ``arguments`` is the raw JSON text."""

    id: str
    name: str
    arguments: str = "{}"

    def args(self) -> dict[str, Any]:
        """Parsed arguments; raises ``ValueError`` on invalid JSON / non-object."""
        data = json.loads(self.arguments or "{}")
        if not isinstance(data, dict):
            raise ValueError("tool arguments must be a JSON object")  # noqa: TRY004 - malformed model output
        return data

    def to_api(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class Message:
    role: Role
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------- builders
    @classmethod
    def system(cls, text: str, **meta: Any) -> Message:
        return cls("system", text, meta=meta)

    @classmethod
    def user(cls, text: str, **meta: Any) -> Message:
        return cls("user", text, meta=meta)

    @classmethod
    def assistant(cls, text: str | None, *, reasoning: str | None = None,
                  tool_calls: list[ToolCall] | None = None, **meta: Any) -> Message:
        return cls("assistant", text, reasoning, list(tool_calls or []), meta=meta)

    @classmethod
    def tool(cls, call_id: str, text: str, **meta: Any) -> Message:
        return cls("tool", text, tool_call_id=call_id, meta=meta)

    # ----------------------------------------------------------- conversion
    def to_api(self, *, with_reasoning: bool = True) -> dict[str, Any]:
        """The Chat Completions wire form (``meta`` is never sent)."""
        d: dict[str, Any] = {"role": self.role}
        if self.role == "assistant":
            d["content"] = self.content if self.content is not None else ""
            if with_reasoning and self.reasoning_content is not None:
                d["reasoning_content"] = self.reasoning_content
            if self.tool_calls:
                d["tool_calls"] = [tc.to_api() for tc in self.tool_calls]
        else:
            d["content"] = self.content or ""
        if self.role == "tool":
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            d["tool_calls"] = [{"id": t.id, "name": t.name, "arguments": t.arguments}
                               for t in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Message:
        return cls(d["role"], d.get("content"), d.get("reasoning_content"),
                   [ToolCall(t["id"], t["name"], t.get("arguments", "{}")) for t in d.get("tool_calls") or []],
                   d.get("tool_call_id"), d.get("name"), dict(d.get("meta") or {}))


@dataclass
class Usage:
    """Token usage of one model call. ``cache_hit + cache_miss == prompt_tokens``
    on DeepSeek (``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit: int = 0
    cache_miss: int = 0
    reasoning_tokens: int = 0

    def __add__(self, o: Usage) -> Usage:
        return Usage(self.prompt_tokens + o.prompt_tokens, self.completion_tokens + o.completion_tokens,
                     self.cache_hit + o.cache_hit, self.cache_miss + o.cache_miss,
                     self.reasoning_tokens + o.reasoning_tokens)

    @property
    def hit_rate(self) -> float:
        total = self.cache_hit + self.cache_miss
        return self.cache_hit / total if total else 0.0

    def to_json(self) -> dict[str, int]:
        return dict(vars(self))

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Usage:
        return cls(**{k: int(d.get(k, 0) or 0) for k in
                      ("prompt_tokens", "completion_tokens", "cache_hit", "cache_miss", "reasoning_tokens")})

    @classmethod
    def from_api(cls, u: dict[str, Any] | None) -> Usage:
        """Parse an OpenAI / DeepSeek ``usage`` object."""
        if not u:
            return cls()
        prompt = int(u.get("prompt_tokens") or 0)
        hit = u.get("prompt_cache_hit_tokens")
        if hit is None:
            hit = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        miss = u.get("prompt_cache_miss_tokens")
        if miss is None:
            miss = max(0, prompt - int(hit))
        details = u.get("completion_tokens_details") or {}
        return cls(prompt, int(u.get("completion_tokens") or 0), int(hit), int(miss),
                   int(details.get("reasoning_tokens") or 0))


@dataclass(frozen=True)
class Money:
    """An amount in USD (DeepSeek bills in USD; ``cny`` converts for display/budgets)."""

    usd: float = 0.0

    def __add__(self, o: Money) -> Money:
        return Money(self.usd + o.usd)

    def cny(self, rate: float) -> float:
        return self.usd * rate

    def __str__(self) -> str:
        return f"${self.usd:.4f}"


# ---------------------------------------------------------------- requests
@dataclass
class ChatRequest:
    """One model call. ``tools`` are wire-form function specs (already ordered
    and serialised by the context builder). ``thinking``/``effort`` ``None``
    means "provider default"."""

    messages: list[Message]
    model: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    thinking: bool | None = None
    effort: str | None = None
    max_tokens: int | None = None
    json_output: bool = False
    temperature: float | None = None
    tool_choice: str | dict[str, Any] | None = None
    strict: bool = False
    stop: list[str] | None = None
    user_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------- stream events
@dataclass
class ReasoningDelta:
    text: str
    type: Literal["reasoning"] = "reasoning"


@dataclass
class ContentDelta:
    text: str
    type: Literal["content"] = "content"


@dataclass
class ToolCallStart:
    """The model started emitting a tool call (arguments follow in later chunks)."""

    index: int
    id: str
    name: str
    type: Literal["tool_call"] = "tool_call"


@dataclass
class Done:
    """Final event: the complete assistant message, usage and finish reason."""

    message: Message
    usage: Usage
    finish_reason: str | None
    model: str
    type: Literal["done"] = "done"


ChatEvent = ReasoningDelta | ContentDelta | ToolCallStart | Done
