"""Core tools: ``time.now``, ``artifact.read``, ``work.read`` / ``work.write``.

``work.*`` edits the session's working memory (in-session state, logged; no
effect outside the session, hence risk ``read``)."""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

import ventri

from ..memory import WorkingMemory
from ..providers.pricing import PeakSchedule
from ..session import SessionLog
from .registry import Tool, ToolContext, ToolError, ToolRegistry


class NowArgs(BaseModel):
    tz: str = Field("Asia/Shanghai", description="IANA time zone")


class ArtifactArgs(BaseModel):
    handle: str = Field(description="Artifact handle from a truncated tool result")
    offset: int = 0
    limit: int = 8_000


class WorkWriteArgs(BaseModel):
    text: str
    mode: Literal["replace", "append"] = "replace"


def now(a: NowArgs, tc: ToolContext) -> dict[str, Any]:
    try:
        tz = ZoneInfo(a.tz)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ToolError(f"unknown time zone {a.tz!r}") from e
    t = datetime.now(UTC)
    local = t.astimezone(tz)
    return {"iso": local.isoformat(timespec="seconds"), "weekday": local.strftime("%A"), "tz": a.tz,
            "deepseek_pricing": "peak" if PeakSchedule().is_peak(t) else "off-peak"}


def artifact_read(a: ArtifactArgs, tc: ToolContext) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-.]+", a.handle) or ".." in a.handle:
        raise ToolError("invalid artifact handle")
    p = tc.workdir / "artifacts" / f"{a.handle}.txt"
    if not p.exists():
        raise ToolError(f"no artifact {a.handle!r} in this session")
    text = p.read_text(encoding="utf-8")
    chunk = text[a.offset:a.offset + a.limit]
    rest = len(text) - a.offset - len(chunk)
    return chunk + (f"\n[... {rest} more chars; offset={a.offset + len(chunk)}]" if rest > 0 else "")


def work_read(a: Any, tc: ToolContext) -> str:
    return tc.get(WorkingMemory).text or "(working memory is empty)"


def work_write(a: WorkWriteArgs, tc: ToolContext) -> str:
    text = tc.get(WorkingMemory).write(a.text, a.mode)
    log = tc.get(SessionLog)
    if log is not None:
        log.append("work", text=text)
    return f"working memory updated ({len(text)} chars)"


CORE_TOOLS = [
    Tool("time.now", "Current date and time.", now, NowArgs, parallel_safe=True),
    Tool("artifact.read", "Read more of a large tool result stored as an artifact.", artifact_read,
         ArtifactArgs, parallel_safe=True, untrusted=True),
    Tool("work.read", "Read this session's working memory (task notes / plan).", work_read, None,
         parallel_safe=True),
    Tool("work.write", "Replace or append to this session's working memory.", work_write, WorkWriteArgs),
]


@ventri.plugin(name="tools:core")
def core(ctx: Any, config: Any, registry: ToolRegistry) -> None:
    """``use: ventri_agent.tools.core`` -- time.now, artifact.read, work.*."""
    for t in CORE_TOOLS:
        registry.register(ctx, Tool(**{**t.__dict__, "source": ""}))


plugin = core
