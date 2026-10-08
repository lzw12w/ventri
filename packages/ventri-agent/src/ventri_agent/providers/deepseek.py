"""DeepSeek adapter (DESIGN.md 5.2), on top of the OpenAI-compatible adapter.

API facts re-verified 2026-10-08 (https://api-docs.deepseek.com):

* models ``deepseek-flash`` (V4.1-Flash, vision) and ``deepseek-v4-pro`` (no
  vision); 1M context, 384K (393216) max output. ``deepseek-chat`` /
  ``deepseek-reasoner`` are retired (rejected here); ``deepseek-v4-flash`` and
  ``deepseek-v4-flash-vision-exp`` are legacy aliases served by V4.1-Flash
  (accepted with a warning).
* thinking: ``{"thinking": {"type": "enabled"|"disabled"}}`` (default enabled)
  plus ``reasoning_effort`` low/high/max (default high; ``none`` also disables
  thinking; minimal/medium/xhigh/ultra are mapped server-side). Temperature has
  no effect in thinking mode. ``tool_choice`` ``required`` / named is a 400 in
  thinking mode.
* when a request carries ``tools``, the ``reasoning_content`` of *every* earlier
  assistant message must be sent back, else HTTP 400 -- checked locally here
  (``ReasoningContentMissing``) instead of waiting for the 400.
* ``strict`` tool calls are Beta: base URL ``/beta`` and ``strict: true`` on
  every function; schemas must be all-required + ``additionalProperties:false``.
* JSON output: ``response_format: {"type": "json_object"}`` and the prompt must
  contain the word "json".
* usage: ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``; in a
  stream the usage rides on the last chunk (no separate usage chunk).
* 429 when over the account concurrency (Flash 2500, Pro 500); 500/503 retry.
"""
from __future__ import annotations

import os
from dataclasses import replace
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel, Field

import ventri
from ventri import Secret

from ..messages import ChatRequest
from .base import ModelCaps, ModelProvider, ReasoningContentMissing, RequestInvalid
from .openai_compat import OpenAICompatProvider
from .pricing import DEEPSEEK_ALIASES, PriceTable

BASE_URL = "https://api.deepseek.com"
RETIRED = {"deepseek-chat": "retired 2026-07-24", "deepseek-reasoner": "retired 2026-07-24"}
EFFORTS = ("low", "high", "max")
_FLASH = ModelCaps(context=1_048_576, max_output=393_216, thinking=True, effort_levels=EFFORTS,
                   tools=True, strict=True, json=True, fim=True, vision=True, cache="prefix-disk",
                   soft_context=256_000)
DEEPSEEK_CAPS = {"deepseek-flash": _FLASH, "deepseek-v4-pro": replace(_FLASH, vision=False)}
DEFAULT_ROUTES: dict[str, Any] = {
    "default": {"model": "deepseek-flash", "thinking": True, "effort": "high"},
    "plan": {"model": "deepseek-v4-pro", "thinking": True, "effort": "max"},
    "cheap": {"model": "deepseek-flash", "thinking": False},
}
CONCURRENCY = {"deepseek-flash": 2500, "deepseek-v4-pro": 500}


class DeepSeekProvider(OpenAICompatProvider):
    name = "deepseek"

    def __init__(self, http: httpx.AsyncClient, *, api_key: str | None, base_url: str = BASE_URL,
                 beta_url: str | None = None, routes: dict[str, Any] | None = None,
                 prices: PriceTable | None = None, soft_context: int = 256_000, **kw: Any) -> None:
        caps = {m: replace(c, soft_context=min(soft_context, c.context)) for m, c in DEEPSEEK_CAPS.items()}
        super().__init__(http, base_url=base_url, api_key=api_key, routes=routes or DEFAULT_ROUTES,
                         caps=caps, default_caps=caps["deepseek-flash"], prices=prices or PriceTable(),
                         concurrency=kw.pop("concurrency", None) or CONCURRENCY, **kw)
        self.beta_url = (beta_url or f"{self.base_url}/beta").rstrip("/")
        self.warnings: list[str] = []
        for r in self.routes.values():
            if r.model in RETIRED:
                raise ValueError(f"route {r.name!r}: model {r.model!r} is {RETIRED[r.model]}; "
                                 "use deepseek-flash or deepseek-v4-pro")
            if r.model in DEEPSEEK_ALIASES:
                self.warnings.append(f"route {r.name!r}: {r.model!r} is a legacy name served by "
                                     f"{DEEPSEEK_ALIASES[r.model]}; update the route")
            if r.effort is not None and r.effort not in (*EFFORTS, "none"):
                raise ValueError(f"route {r.name!r}: effort must be one of {EFFORTS}")

    def caps_for(self, model: str) -> ModelCaps:
        return super().caps_for(DEEPSEEK_ALIASES.get(model, model))

    # ------------------------------------------------------------ request
    @staticmethod
    def thinking_on(req: ChatRequest) -> bool:
        return req.thinking is not False and req.effort != "none"  # the API default is enabled

    def endpoint(self, req: ChatRequest) -> str:
        return f"{self.beta_url if req.strict else self.base_url}/chat/completions"

    def validate(self, req: ChatRequest) -> None:
        super().validate(req)
        if req.model in RETIRED:
            raise RequestInvalid(f"model {req.model!r} is {RETIRED[req.model]}")
        thinking = self.thinking_on(req)
        if req.tools and thinking:
            for i, m in enumerate(req.messages):
                if m.role == "assistant" and m.reasoning_content is None:
                    raise ReasoningContentMissing(
                        f"messages[{i}]: assistant message without reasoning_content; DeepSeek "
                        "requires every earlier reasoning_content when the request carries tools")
        if thinking and (req.tool_choice == "required" or isinstance(req.tool_choice, dict)):
            raise RequestInvalid("tool_choice 'required' / named is not supported in thinking mode")
        if req.json_output and not any("json" in (m.content or "").lower()
                                       for m in req.messages if m.role in ("system", "user")):
            raise RequestInvalid("JSON output needs the word 'json' in a system or user message")

    def body(self, req: ChatRequest) -> dict[str, Any]:
        b = super().body(req)
        if req.thinking is not None or req.effort is not None:
            on = self.thinking_on(req)
            b["thinking"] = {"type": "enabled" if on else "disabled"}
            if on and req.effort:
                b["reasoning_effort"] = req.effort
        if self.thinking_on(req):
            b.pop("temperature", None)  # no effect in thinking mode
        uid = req.user_id or self.user_id
        if uid:
            b["user_id"] = uid
        return b

    # -------------------------------------------------------------- probe
    async def probe(self) -> list[str]:
        """``GET /models``: refresh capabilities and return warnings (unknown route
        models, legacy names). Never raises for transport problems."""
        warnings = list(self.warnings)
        try:
            r = await self.http.get(f"{self.base_url}/models", headers=self.headers())
            r.raise_for_status()
            data = r.json().get("data") or []
        except (httpx.HTTPError, ValueError) as e:
            return [*warnings, f"model probe failed: {type(e).__name__}: {e}"]
        ids = {d.get("id") for d in data}
        for d in data:
            mid = d.get("id")
            base = self._caps.get(mid, self._default_caps)
            eff = tuple((d.get("effort") or {}).get("supported_levels") or base.effort_levels)
            ctxw = int(d.get("context_window") or base.context)
            self._caps[mid] = replace(base, context=ctxw,
                                      max_output=int(d.get("max_output_tokens") or base.max_output),
                                      vision="image" in (d.get("input_modalities") or ["text"]),
                                      effort_levels=eff, soft_context=min(base.soft_context, ctxw))
        for r in self.routes.values():
            if r.model not in ids and DEEPSEEK_ALIASES.get(r.model) not in ids:
                warnings.append(f"route {r.name!r}: model {r.model!r} is not listed by /models")
        return warnings


class DeepSeekConfig(BaseModel):
    api_key: Secret[str] | None = None   # None: $DEEPSEEK_API_KEY
    base_url: str = BASE_URL
    beta_url: str | None = None
    routes: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_ROUTES))
    strict_tools: bool = False           # Beta endpoint; schemas are strict-compatible either way
    soft_context: int = 256_000
    timeout: float = 900.0               # read timeout between stream chunks
    max_retries: int = 4
    probe: bool = True
    user_id: str | None = None
    prices: dict[str, dict[str, float]] = Field(default_factory=dict)
    holidays: list[date] = Field(default_factory=list)


@ventri.plugin(name="provider:deepseek", config=DeepSeekConfig, provides={"llm": ModelProvider})
async def deepseek(ctx: Any, cfg: DeepSeekConfig) -> None:
    """``use: ventri_agent.providers.deepseek``; provides ``ModelProvider`` (``ctx.llm``)."""
    key = cfg.api_key.reveal() if cfg.api_key is not None else os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("no DeepSeek API key: set api_key (e.g. ${secret:deepseek}) or $DEEPSEEK_API_KEY")
    http = await ctx.enter(httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout, connect=15.0)))
    prov = DeepSeekProvider(http, api_key=key, base_url=cfg.base_url, beta_url=cfg.beta_url,
                            routes=cfg.routes, prices=PriceTable.from_config(cfg.prices, cfg.holidays),
                            soft_context=cfg.soft_context, max_retries=cfg.max_retries,
                            user_id=cfg.user_id, strict_tools=cfg.strict_tools)
    for w in prov.warnings:
        ctx.trace("model.warning", warning=w)
    ctx.provide(ModelProvider, prov)
    if cfg.probe:
        async def run_probe() -> None:
            try:
                found = await prov.probe()
            except Exception as e:  # noqa: BLE001 - a probe never fails the provider
                found = [f"model probe failed: {e!r}"]
            for w in found:
                if w not in prov.warnings:
                    ctx.trace("model.warning", warning=w)
        ctx.spawn(run_probe, name="probe")


plugin = deepseek
