"""Feishu message rendering: interactive cards (JSON 2.0), progress state,
approval cards and chunking of long replies.

Every message the channel sends is a JSON 2.0 card with ``update_multi: true``
so it can be patched later (PATCH ``/im/v1/messages/:id`` requires it before
and after the update, and a message must keep one card dialect). Card callbacks
are answered with a toast only; card updates always go through PATCH.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ._hermes import fence_segments

CARD_MAX_BYTES = 28_000          # Feishu: card content <= 30 KB (after JSON escaping)
CHUNK_BYTES = 12_000             # Markdown bytes per reply card
PROGRESS_TAIL_CHARS = 1_500      # streamed answer tail shown while a turn runs
TOOL_LINES = 8

_AT_TAG_RE = re.compile(r"<(/?)(at|person)\b", re.IGNORECASE)


def sanitize(md: str) -> str:
    """Neutralise card Markdown that could act on other people: ``<at id=all>``
    / ``<at user_id=...>`` / ``<person>`` from model output become literal text."""
    return _AT_TAG_RE.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", md)


def _md(content: str, *, size: str | None = None) -> dict[str, Any]:
    el: dict[str, Any] = {"tag": "markdown", "content": content}
    if size:
        el["text_size"] = size
    return el


def md_elements(md: str) -> list[dict[str, Any]]:
    return [_md(seg) for seg in fence_segments(sanitize(md)) if seg.strip()] or [_md(" ")]


def card(elements: list[dict[str, Any]], *, title: str | None = None, template: str = "blue") -> dict[str, Any]:
    c: dict[str, Any] = {"schema": "2.0", "config": {"update_multi": True, "width_mode": "fill"},
                         "body": {"elements": elements}}
    if title:
        c["header"] = {"title": {"tag": "plain_text", "content": title}, "template": template}
    return c


def card_json(c: dict[str, Any]) -> str:
    return json.dumps(c, ensure_ascii=False, separators=(",", ":"))


def markdown_card(md: str, *, title: str | None = None, template: str = "blue",
                  note: str = "") -> dict[str, Any]:
    els = md_elements(md)
    if note:
        els.append(_md(f"<font color='grey'>{sanitize(note)}</font>", size="notation"))
    return card(els, title=title, template=template)


# ----------------------------------------------------------------- chunking
def _blen(s: str) -> int:
    return len(s.encode("utf-8"))


def _split_long_line(line: str, limit: int) -> list[str]:
    out, cur = [], ""
    for ch in line:
        if _blen(cur) + _blen(ch) > limit:
            out.append(cur)
            cur = ""
        cur += ch
    return [*out, cur] if cur else out


def chunk_markdown(text: str, limit: int = CHUNK_BYTES) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` UTF-8 bytes, preferring
    paragraph then line boundaries; a code fence cut by a split is closed at
    the end of one piece and reopened (same language) at the start of the next."""
    text = text.strip("\n")
    if _blen(text) <= limit:
        return [text] if text else []
    lines: list[str] = []
    for ln in text.split("\n"):
        lines.extend(_split_long_line(ln, limit - 64) if _blen(ln) > limit - 64 else [ln])
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    fence: str | None = None      # opening line of the fence we are inside
    for ln in lines:
        stripped = ln.strip()
        opens = stripped.startswith("```")
        extra = _blen(ln) + 1
        if cur and size + extra + 8 > limit:
            # prefer to cut at the last blank line outside a code block
            cut = len(cur)
            if fence is None:
                for i in range(len(cur) - 1, max(len(cur) // 2, 0), -1):
                    if not cur[i].strip():
                        cut = i
                        break
            head, rest = cur[:cut], cur[cut:]
            body = "\n".join(head)
            if fence is not None:
                body += "\n```"
            chunks.append(body.strip("\n"))
            while rest and not rest[0].strip():
                rest.pop(0)
            cur = ([fence] if fence is not None else []) + rest
            size = sum(_blen(x) + 1 for x in cur)
        cur.append(ln)
        size += extra
        if opens:
            fence = None if fence is not None else stripped
    if cur and "\n".join(cur).strip():
        chunks.append("\n".join(cur).strip("\n"))
    return chunks


# ----------------------------------------------------------------- progress
@dataclass
class ToolLine:
    text: str
    done: bool = False
    ok: bool = True


@dataclass
class Progress:
    """What the progress card shows while a turn runs."""

    started: float = field(default_factory=time.monotonic)
    header: str = ""
    tools: list[ToolLine] = field(default_factory=list)
    reasoning_chars: int = 0
    reasoning_tail: str = ""
    thinking: bool = False
    content: str = ""
    notices: list[str] = field(default_factory=list)
    waiting: str = ""
    version: int = 0

    def touch(self) -> None:
        self.version += 1

    def tool_start(self, text: str) -> None:
        self.tools.append(ToolLine(" ".join(text.split())[:160]))
        self.thinking = False
        self.touch()

    def tool_end(self, ok: bool) -> None:
        for t in reversed(self.tools):
            if not t.done:
                t.done, t.ok = True, ok
                break
        self.waiting = ""
        self.touch()

    def tools_md(self, limit: int = TOOL_LINES) -> str:
        if not self.tools:
            return ""
        shown = self.tools[-limit:]
        lines = []
        if len(self.tools) > limit:
            lines.append(f"<font color='grey'>… 另有 {len(self.tools) - limit} 次较早的工具调用</font>")
        for t in shown:
            mark = ("✓" if t.ok else "✗") if t.done else "…"
            lines.append(f"⚙ `{_code_safe(t.text)}` {mark}")
        return "\n".join(lines)


def _code_safe(s: str) -> str:
    return s.replace("`", "'")


def progress_card(p: Progress, *, show_thinking: bool = False) -> dict[str, Any]:
    els: list[dict[str, Any]] = []
    if p.header:
        els.append(_md(f"<font color='grey'>{sanitize(p.header)}</font>", size="notation"))
    for n in p.notices[-3:]:
        els.append(_md(f"<font color='grey'>· {sanitize(n)}</font>", size="notation"))
    tools = p.tools_md()
    if tools:
        els.append(_md(sanitize(tools)))
    if p.waiting:
        els.append(_md(f"⏸ 等待审批：`{_code_safe(p.waiting)}`"))
    if p.thinking or (show_thinking and p.reasoning_tail and not p.content):
        tail = f"\n{sanitize(p.reasoning_tail[-600:])}" if show_thinking and p.reasoning_tail else ""
        els.append(_md(f"<font color='grey'>💭 思考中…（{p.reasoning_chars} 字）{tail}</font>"))
    if p.content:
        tail = p.content[-PROGRESS_TAIL_CHARS:]
        if len(p.content) > PROGRESS_TAIL_CHARS:
            tail = "…" + tail
        els.extend(md_elements(tail))
    if not els:
        els.append(_md("<font color='grey'>…</font>"))
    secs = int(time.monotonic() - p.started)
    return card(els, title=f"⏳ 处理中 · {secs}s", template="blue")


def final_card(p: Progress, answer: str, *, status: str, footer: str, error: str = "",
               more: int = 0) -> dict[str, Any]:
    """The progress card's last state: compact tool summary + the answer (or
    its first chunk when ``more`` further chunks follow as separate cards)."""
    els: list[dict[str, Any]] = []
    if p.header:
        els.append(_md(f"<font color='grey'>{sanitize(p.header)}</font>", size="notation"))
    if p.tools:
        names: dict[str, int] = {}
        for t in p.tools:
            n = t.text.split(" ", 1)[0]
            names[n] = names.get(n, 0) + 1
        failed = sum(1 for t in p.tools if t.done and not t.ok)
        summary = "、".join(f"{n}×{c}" if c > 1 else n for n, c in names.items())
        els.append(_md(f"<font color='grey'>⚙ {len(p.tools)} 次工具调用：{sanitize(summary)}"
                       f"{f'（{failed} 次失败）' if failed else ''}</font>", size="notation"))
    if answer.strip():
        els.extend(md_elements(answer))
    if error:
        els.append(_md(f"**{sanitize(error)}**"))
    if more:
        els.append(_md(f"<font color='grey'>（续 {more} 条消息）</font>", size="notation"))
    els.append(_md(f"<font color='grey'>{sanitize(footer)}</font>", size="notation"))
    title, tpl = {"ok": (None, "blue"), "error": ("⚠️ 出错", "red"),
                  "budget": ("⚠️ 预算用尽", "orange"), "cancelled": ("已取消", "grey")}.get(status, (None, "blue"))
    return card(els, title=title, template=tpl)


# ---------------------------------------------------------------- approvals
CHOICE_LABEL = {"once": "已允许一次", "session": "已在本会话允许", "deny": "已拒绝"}


def approval_card(*, summary: str, risk: str, args_preview: str, grantable: bool, timeout: float,
                  values: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``values``: choice -> callback value (carries the request id and HMAC)."""
    def button(label: str, kind: str, choice: str) -> dict[str, Any]:
        return {"tag": "column", "width": "auto", "elements": [{
            "tag": "button", "text": {"tag": "plain_text", "content": label}, "type": kind,
            "behaviors": [{"type": "callback", "value": values[choice]}]}]}

    cols = [button("允许一次", "primary", "once")]
    if grantable:
        cols.append(button("本会话允许", "default", "session"))
    cols.append(button("拒绝", "danger", "deny"))
    body = (f"**{sanitize(summary)}**  （风险：{sanitize(risk)}）\n"
            f"```json\n{args_preview.replace('```', '`‵`')}\n```")
    els = [*md_elements(body),
           {"tag": "column_set", "flex_mode": "none", "horizontal_spacing": "default", "columns": cols},
           _md(f"<font color='grey'>{int(timeout)} 秒内未处理视为拒绝；只有发起请求的人可以审批。</font>",
               size="notation")]
    return card(els, title="🔐 需要审批", template="orange")


def approval_result_card(*, summary: str, risk: str, outcome: str, by: str = "") -> dict[str, Any]:
    ok = outcome in ("once", "session")
    label = CHOICE_LABEL.get(outcome, outcome)
    who = f"（{sanitize(by)}）" if by else ""
    els = [_md(f"**{sanitize(summary)}**  （风险：{sanitize(risk)}）"), _md(f"{'✅' if ok else '❌'} {label}{who}")]
    return card(els, title="🔐 审批" + ("：已允许" if ok else "：已拒绝"), template="green" if ok else "red")


def fits(c: dict[str, Any]) -> bool:
    return _blen(card_json(c)) <= CARD_MAX_BYTES
