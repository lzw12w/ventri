"""``inspect.*`` -- read-only, redacted self-inspection (DESIGN.md 4.10).

The agent can read its own plugin tree and recent trace (e.g. to diagnose "why
did a tool disappear"); writes only ever go through evolution proposals (M3)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

import ventri
from ventri import redact

from .registry import Tool, ToolContext, ToolRegistry


class TraceArgs(BaseModel):
    limit: int = Field(40, description="Most recent N records")
    kind: str | None = Field(None, description="Prefix filter, e.g. 'fiber.' or 'tool.'")
    contains: str | None = Field(None, description="Substring filter on the fiber path")


def make_tools(kernel: Any) -> list[Tool]:
    def tree(a: Any, tc: ToolContext) -> str:
        return kernel.tree()

    def trace(a: TraceArgs, tc: ToolContext) -> str:
        rows = []
        for ev in reversed(list(kernel.trace_log)):
            d = ev.to_dict()
            if a.kind and not d["kind"].startswith(a.kind):
                continue
            if a.contains and a.contains not in (d.get("fiber") or ""):
                continue
            rows.append(f"#{d['seq']} {d['kind']} {d.get('fiber')} {redact(d.get('attrs'))}")
            if len(rows) >= min(a.limit, 500):
                break
        return "\n".join(reversed(rows)) or "(no matching trace records)"

    return [
        Tool("inspect.tree", "Show Ventri's live plugin tree (states, services, errors; secrets redacted).",
             tree, None, parallel_safe=True),
        Tool("inspect.trace", "Show recent trace records (redacted).", trace, TraceArgs, parallel_safe=True),
    ]


@ventri.plugin(name="tool:inspect")
def inspect_tools(ctx: Any, config: Any, registry: ToolRegistry) -> None:
    """``use: ventri_agent.tools.inspect``."""
    for t in make_tools(ctx.kernel):
        registry.register(ctx, t)


plugin = inspect_tools
