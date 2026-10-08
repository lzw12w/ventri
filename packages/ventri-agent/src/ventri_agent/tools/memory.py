"""Long-term memory tools.

* ``memory.search`` -- read, parallel-safe, output fenced as untrusted.
* ``memory.remember`` -- write-local (asked by default). Exact duplicates are
  not stored twice; a near duplicate *supersedes* the older memory (newer
  wins) and the result says so, naming the old item so the model can restore
  it with ``memory.update`` if both are true.
* ``memory.update`` -- write-local, asked by default, grantable: correct an
  item's text (also restores a superseded item).
* ``memory.forget`` -- write-local, asked every time (not grantable): delete an
  item. The approval prompt shows the item's current text.

Text written by the model is scanned for prompt-injection / exfiltration
payloads (``threat_patterns``, strict scope) because memories are injected into
every later session's system prompt. Credential-looking text is stored
``pending`` until the user confirms it. Updates and deletions are written to
the audit log with before/after text, in addition to the permission gate's own
decision record.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

import ventri

from ..memory import LongTermMemory, MemoryItem
from ..threat_patterns import first_threat_message
from .registry import Risk, Tool, ToolContext, ToolError, ToolRegistry


class SearchArgs(BaseModel):
    query: str = Field(description="Words to look for (Chinese works; >= 3 characters is best)")
    k: int = 8


class RememberArgs(BaseModel):
    text: str = Field(description="One durable fact / preference / procedure about the user")
    kind: Literal["fact", "preference", "procedure", "episode"] = "fact"


class UpdateArgs(BaseModel):
    id: int = Field(description="Memory id (the #number shown by memory.search or the memory snapshot)")
    text: str = Field(description="The corrected full text of that memory")


class ForgetArgs(BaseModel):
    id: int = Field(description="Memory id to delete (the #number shown by memory.search)")


def _snip(text: str, n: int = 80) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


PENDING = (" It looks sensitive, so it is pending: it is not used until the user confirms it "
           "(/memory confirm or `va memory confirm`).")


def make_tools(mem: LongTermMemory) -> list[Tool]:
    def audit(tc: ToolContext, **rec: Any) -> None:
        from ..permission import AuditLog  # lazy: permission imports tools
        log = tc.get(AuditLog) if tc.ctx is not None else None
        if log is not None:
            log.write(session=tc.session_id, call=tc.call_id, origin=tc.origin, **rec)

    def scan(text: str) -> None:
        if msg := first_threat_message(text, scope="strict"):
            raise ToolError(msg + " Nothing was stored.")

    def existing(item_id: int) -> MemoryItem:
        item = mem.get(item_id)
        if item is None:
            raise ToolError(f"no memory #{item_id}; use memory.search to find the right id")
        return item

    def search(a: SearchArgs, tc: ToolContext) -> str:
        items = mem.search(a.query, a.k)
        return "\n".join(f"- [{m.kind}] {m.text} (#{m.id})" for m in items) or "no memories found"

    def remember(a: RememberArgs, tc: ToolContext) -> str:
        scan(a.text)
        try:
            r = mem.remember(a.text, a.kind, source_session=tc.session_id)
        except ValueError as e:
            raise ToolError(str(e)) from None
        pending = PENDING if r.item.status == "pending" else ""
        if r.action == "duplicate":
            return f"already stored as #{r.item.id}; nothing new was written."
        if r.action == "created":
            return f"stored as #{r.item.id}.{pending}"
        old = r.previous
        assert old is not None
        audit(tc, tool="memory.remember", event="memory.supersede", id=r.item.id, supersedes=old.id,
              before=old.text, after=r.item.text)
        when = "once the user confirms it" if pending else "now"
        return (f"stored as #{r.item.id}. It is a near-duplicate of #{old.id} (\"{_snip(old.text)}\"), so it "
                f"supersedes #{old.id} {when}: the newer memory wins and #{old.id} is no longer used. If "
                f"#{old.id} is still a separate true fact, call memory.update(id={old.id}, text=...) to "
                f"restore it.{pending}")

    def update(a: UpdateArgs, tc: ToolContext) -> str:
        before = existing(a.id)
        scan(a.text)
        try:
            item = mem.update(a.id, a.text)
        except ValueError as e:
            raise ToolError(str(e)) from None
        assert item is not None
        audit(tc, tool="memory.update", event="memory.update", id=a.id, before=before.text, after=item.text,
              status_before=before.status, status_after=item.status)
        note = " It was superseded and is active again." if before.status == "superseded" else ""
        pending = PENDING if item.status == "pending" else ""
        return f"updated #{a.id}: \"{_snip(before.text)}\" -> \"{_snip(item.text)}\".{note}{pending}"

    def forget(a: ForgetArgs, tc: ToolContext) -> str:
        before = existing(a.id)
        mem.forget(a.id)
        audit(tc, tool="memory.forget", event="memory.forget", id=a.id, before=before.text,
              status_before=before.status)
        return f"forgot #{a.id} (\"{_snip(before.text)}\")."

    def describe(item_id: int) -> str:
        item = mem.get(item_id)
        return _snip(item.text) if item else "(no such memory)"

    return [
        Tool("memory.search", "Search long-term memory about the user.", search, SearchArgs,
             parallel_safe=True, untrusted=True),
        Tool("memory.remember", "Store a durable memory about the user. A near-duplicate of an existing "
             "memory supersedes it (newer wins).", remember, RememberArgs,
             risk=Risk.WRITE_LOCAL, idempotent=False, subject=lambda a: {"kind": a.kind, "text": _snip(a.text)}),
        Tool("memory.update", "Correct the text of an existing memory by id (also restores a superseded "
             "memory).", update, UpdateArgs, risk=Risk.WRITE_LOCAL, idempotent=True,
             default_action="ask", grantable=True,
             subject=lambda a: {"id": str(a.id), "before": describe(a.id), "text": _snip(a.text)}),
        Tool("memory.forget", "Delete a memory by id (when the user asks to forget it or it is wrong).",
             forget, ForgetArgs, risk=Risk.WRITE_LOCAL, idempotent=True, default_action="ask",
             grantable=False, subject=lambda a: {"id": str(a.id), "text": describe(a.id)}),
    ]


@ventri.plugin(name="tool:memory")
def memory_tools(ctx: Any, config: Any, registry: ToolRegistry, mem: LongTermMemory) -> None:
    """``use: ventri_agent.tools.memory``."""
    for t in make_tools(mem):
        registry.register(ctx, t)


plugin = memory_tools
