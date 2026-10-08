"""``memory.search`` (read) and ``memory.remember`` (write-local, asked by default)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

import ventri

from ..memory import LongTermMemory
from .registry import Risk, Tool, ToolContext, ToolRegistry


class SearchArgs(BaseModel):
    query: str = Field(description="Words to look for (Chinese works; >= 3 characters is best)")
    k: int = 8


class RememberArgs(BaseModel):
    text: str = Field(description="One durable fact / preference / procedure about the user")
    kind: Literal["fact", "preference", "procedure", "episode"] = "fact"


def make_tools(mem: LongTermMemory) -> list[Tool]:
    def search(a: SearchArgs, tc: ToolContext) -> str:
        items = mem.search(a.query, a.k)
        return "\n".join(f"- [{m.kind}] {m.text} (#{m.id})" for m in items) or "no memories found"

    def remember(a: RememberArgs, tc: ToolContext) -> str:
        item, new = mem.add(a.text, a.kind, source_session=tc.session_id)
        state = "stored" if new else "merged with an existing memory"
        return f"{state}: #{item.id}" + (" (pending user confirmation)" if item.status == "pending" else "")

    return [
        Tool("memory.search", "Search long-term memory about the user.", search, SearchArgs,
             parallel_safe=True, untrusted=True),
        Tool("memory.remember", "Store a durable memory about the user.", remember, RememberArgs,
             risk=Risk.WRITE_LOCAL, idempotent=False, subject=lambda a: {"kind": a.kind}),
    ]


@ventri.plugin(name="tool:memory")
def memory_tools(ctx: Any, config: Any, registry: ToolRegistry, mem: LongTermMemory) -> None:
    """``use: ventri_agent.tools.memory``."""
    for t in make_tools(mem):
        registry.register(ctx, t)


plugin = memory_tools
