"""The ModelProvider interface (DESIGN.md 5.2) and helpers shared by adapters."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from ..messages import ChatEvent, ChatRequest, ContentDelta, Done, Message, Money, Usage


@dataclass(frozen=True)
class ModelCaps:
    """Capability descriptor of one model (negotiated by upper layers instead of
    hard-coding model names)."""

    context: int = 128_000
    max_output: int = 8_192
    thinking: bool = False
    effort_levels: tuple[str, ...] = ()
    tools: bool = True
    strict: bool = False
    json: bool = True
    fim: bool = False
    vision: bool = False
    cache: Literal["prefix-disk", "none"] = "none"
    soft_context: int = 128_000  # what the ContextBuilder plans for (<= context)


@dataclass(frozen=True)
class Route:
    """A named model route (``default`` / ``plan`` / ``cheap``): the only place
    model names appear in configuration."""

    name: str
    model: str
    thinking: bool | None = None
    effort: str | None = None
    max_tokens: int | None = None

    def request(self, messages: list[Message], **kw: Any) -> ChatRequest:
        kw.setdefault("thinking", self.thinking)
        kw.setdefault("effort", self.effort)
        kw.setdefault("max_tokens", self.max_tokens)
        return ChatRequest(messages=messages, model=self.model, **kw)


@runtime_checkable
class ModelProvider(Protocol):
    """``caps`` describes the default route's model; ``stream`` yields
    ``ReasoningDelta`` / ``ContentDelta`` / ``ToolCallStart`` events and ends with
    exactly one ``Done``; ``price`` is time-aware (DeepSeek off-peak pricing)."""

    name: str
    caps: ModelCaps
    routes: dict[str, Route]

    def caps_for(self, model: str) -> ModelCaps: ...
    def route(self, name: str) -> Route: ...
    def stream(self, req: ChatRequest) -> AsyncIterator[ChatEvent]: ...
    def price(self, usage: Usage, at: datetime, model: str | None = None) -> Money: ...


# ------------------------------------------------------------------ errors
class ProviderError(Exception):
    """A model call failed. ``retryable`` errors (429, 5xx, transport) were
    already retried by the adapter when this is raised."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False,
                 body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.body = body


class RequestInvalid(ProviderError):
    """Rejected locally before sending (e.g. missing ``reasoning_content``)."""


class ReasoningContentMissing(RequestInvalid):
    pass


# ----------------------------------------------------------------- helpers
def route_from(name: str, spec: Any) -> Route:
    if isinstance(spec, Route):
        return replace(spec, name=name)
    if isinstance(spec, str):
        return Route(name, spec)
    d = dict(spec)
    return Route(name, d.pop("model"), d.pop("thinking", None), d.pop("effort", None),
                 d.pop("max_tokens", None))


async def collect(stream: AsyncIterator[ChatEvent],
                  on_event: Callable[[ChatEvent], Any] | None = None) -> Done:
    """Drain a stream and return its ``Done`` event."""
    done: Done | None = None
    async for ev in stream:
        if on_event is not None:
            on_event(ev)
        if isinstance(ev, Done):
            done = ev
    if done is None:
        raise ProviderError("stream ended without a final event", retryable=True)
    return done


async def complete_json[T](provider: ModelProvider, req: ChatRequest, model: Callable[..., T], *,
                           repair: bool = True) -> tuple[T, list[Done]]:
    """JSON output + validation + one repair retry (DESIGN.md 5.2 "JSON 输出").

    ``model`` is a pydantic model class (or any callable taking the parsed JSON
    as keyword arguments / a dict). Returns ``(value, [done events])``."""
    req = replace(req, json_output=True, tools=[], tool_choice=None)
    dones: list[Done] = []
    last_err: Exception | None = None
    for attempt in range(2 if repair else 1):
        done = await collect(provider.stream(req))
        dones.append(done)
        text = done.message.content or ""
        try:
            data = json.loads(text)
            validate = getattr(model, "model_validate", None)
            value = validate(data) if validate else model(**data) if isinstance(data, dict) else model(data)
            return value, dones
        except Exception as e:  # noqa: BLE001 - one repair round, then give up
            last_err = e
            if attempt == 0 and repair:
                req = replace(req, messages=[*req.messages, Message.assistant(text, reasoning=done.message.reasoning_content),
                                             Message.user(f"The JSON above is invalid ({e}). Reply with corrected JSON only.")])
    raise ProviderError(f"model returned invalid JSON: {last_err}", body=dones[-1].message.content)


def text_of(events: list[ChatEvent]) -> str:
    return "".join(e.text for e in events if isinstance(e, ContentDelta))


@dataclass
class CallStats:
    """Cumulative per-provider counters (exposed for /cost and inspect)."""

    calls: int = 0
    retries: int = 0
    errors: int = 0
    usage: Usage = field(default_factory=Usage)
