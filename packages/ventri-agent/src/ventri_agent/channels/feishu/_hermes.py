"""Pieces of the Feishu adapter ported from Hermes Agent.

Adapted from Hermes Agent ``plugins/platforms/feishu/adapter.py``
(https://github.com/NousResearch/hermes-agent, commit a28a5d03), Copyright (c)
2025 Nous Research, MIT License -- see THIRD_PARTY_NOTICES.md. Ported:

* inbound text normalisation: ``@_user_N`` mention placeholders -> names, the
  bot's own leading/trailing mention stripped (``normalize_text``,
  ``strip_edge_self_mentions``), rich-text ``post`` flattening to Markdown
  (``parse_post``: text styles, links, mentions, code blocks, images/files as
  placeholders, locale wrappers);
* outbound: splitting Markdown at fenced code blocks so every fence gets its
  own element (Feishu's Markdown renderer can swallow text that follows a fence
  inside one large element) -- ``fence_segments``;
* the ``lark_oapi.ws.client`` workarounds: the SDK keeps its asyncio loop in a
  *module-level global* ``loop``, so a per-thread proxy lets every WebSocket
  client run on the loop of the thread that owns it (``install_loop_proxy`` /
  ``bind_thread_loop``); and a receive loop that dies after the SDK's own
  reconnect ladder gave up stops the worker loop, so ``start()`` returns and
  the supervisor rebuilds the client instead of leaving a deaf socket
  (``ws_client_class``);
* reply-target-gone error codes that fall back to a new message.

Changes for Ventri: no Hermes gateway types (``MessageEvent``, profiles,
multiplexing, i18n); media are placeholders only (no download); the
receive-loop guard is a subclass instead of a global patch of the SDK class;
the ping / connect-kwargs overrides are not ported; post text styles are
also read in Feishu's list form (``"style": ["bold"]``), not only as a dict.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# reply target withdrawn / missing -> post a new message to the chat instead
REPLY_FALLBACK_CODES = frozenset({230011, 231003})

_MENTION_PLACEHOLDER_RE = re.compile(r"@_user_\d+")
_MENTION_BOUNDARY_CHARS = frozenset(" \t\n\r.,;:!?、，。；：！？()[]{}<>\"'`")
_TRAILING_TERMINAL_PUNCT = frozenset(" \t\n\r.!?。！？")
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_FENCE_OPEN_RE = re.compile(r"^```([^\n`]*)\s*$")
_FENCE_CLOSE_RE = re.compile(r"^```\s*$")
_PREFERRED_LOCALES = ("zh_cn", "en_us")
_STATIC_POST_TAGS = {"br": "\n", "hr": "\n\n---\n\n", "divider": "\n\n---\n\n"}
_TEXT_STYLE_WRAPPERS = (("bold", "**", "**"), ("italic", "*", "*"), ("underline", "<u>", "</u>"),
                        ("strikethrough", "~~", "~~"))


@dataclass(frozen=True)
class MentionRef:
    key: str = ""
    name: str = ""
    open_id: str = ""
    is_all: bool = False
    is_self: bool = False


def mentions_map(mentions: list[MentionRef]) -> dict[str, MentionRef]:
    return {m.key: m for m in mentions if m.key}


# ------------------------------------------------------------------ inbound
def normalize_text(text: str, mentions: dict[str, MentionRef] | None = None) -> str:
    """Replace ``@_user_N`` placeholders with ``@name`` and tidy whitespace
    (Hermes ``_normalize_feishu_text``; newlines are kept)."""
    def sub(m: re.Match[str]) -> str:
        ref = (mentions or {}).get(m.group(0))
        return " " if ref is None else f"@{ref.name or ref.open_id or 'user'}"

    cleaned = _MENTION_PLACEHOLDER_RE.sub(sub, text or "")
    cleaned = cleaned.replace("@_all", "@all").replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n")]
    return "\n".join(lines).strip()


def strip_edge_self_mentions(text: str, mentions: list[MentionRef]) -> str:
    """Drop the bot's own ``@name`` at the start (unconditionally, on a word
    boundary) and at the end (only before whitespace / terminal punctuation)."""
    if not text:
        return text
    names = [f"@{m.name or m.open_id or 'user'}" for m in mentions if m.is_self]
    if not names:
        return text
    remaining = text.lstrip()
    while True:
        for nm in names:
            if not remaining.startswith(nm):
                continue
            after = remaining[len(nm):]
            if after and after[0] not in _MENTION_BOUNDARY_CHARS:
                continue
            remaining = after.lstrip()
            break
        else:
            break
    while True:
        i = len(remaining)
        while i > 0 and remaining[i - 1] in _TRAILING_TERMINAL_PUNCT:
            i -= 1
        body, tail = remaining[:i], remaining[i:]
        for nm in names:
            if body.endswith(nm):
                remaining = body[: -len(nm)].rstrip() + tail
                break
        else:
            return remaining


def _wrap_inline_code(text: str) -> str:
    max_run = max([0, *[len(run) for run in re.findall(r"`+", text)]])
    fence = "`" * (max_run + 1)
    body = f" {text} " if text.startswith("`") or text.endswith("`") else text
    return f"{fence}{body}{fence}"


def _on(style: Any, key: str) -> bool:
    # Feishu sends ``"style": ["bold", ...]``; Hermes only read the dict form, both are accepted here
    if isinstance(style, list | tuple | set):
        return key in style
    return isinstance(style, dict) and style.get(key) in (True, 1, "true")


def _render_text_element(el: dict[str, Any]) -> str:
    text = str(el.get("text", "") or "")
    style = el.get("style")
    if _on(style, "code"):
        return _wrap_inline_code(text)
    if not text:
        return ""
    for key, prefix, suffix in _TEXT_STYLE_WRAPPERS:
        if _on(style, key):
            text = f"{prefix}{text}{suffix}"
    return text


def _render_code_block(el: dict[str, Any]) -> str:
    lang = (str(el.get("language", "") or "") or str(el.get("lang", "") or "")).strip().replace("\n", " ")
    code = (str(el.get("text", "") or "") or str(el.get("content", "") or "")).replace("\r\n", "\n")
    return f"```{lang}\n{code}{'' if code.endswith(chr(10)) else chr(10)}```"


def _render_post_element(el: Any, mentions: dict[str, MentionRef]) -> str:
    if isinstance(el, str):
        return el
    if not isinstance(el, dict):
        return ""
    tag = str(el.get("tag", "")).strip().lower()
    if tag in _STATIC_POST_TAGS:
        return _STATIC_POST_TAGS[tag]
    if tag == "text":
        return _render_text_element(el)
    if tag in ("code_block", "pre"):
        return _render_code_block(el)
    if tag == "md":
        return str(el.get("text", "") or "")
    if tag == "a":
        href = str(el.get("href", "")).strip()
        label = str(el.get("text", href) or "").strip()
        return f"[{label}]({href})" if label and href else label
    if tag == "at":
        placeholder = str(el.get("user_id", "")).strip()
        if placeholder == "@_all":
            return "@all"
        ref = mentions.get(placeholder)
        name = (ref.name or ref.open_id or "user") if ref is not None else (
            str(el.get("user_name", "")).strip() or "user")
        return f"@{name}"
    if tag in ("img", "image"):
        return "[图片]"
    if tag in ("media", "file", "audio", "video"):
        name = next((str(el.get(k, "")).strip() for k in ("file_name", "title", "text") if el.get(k)), "")
        return f"[附件: {name}]" if name else "[附件]"
    if tag in ("emotion", "emoji"):
        label = str(el.get("text", "")).strip() or str(el.get("emoji_type", "")).strip()
        return f":{label}:" if label else ""
    if tag == "code":
        code = str(el.get("text", "") or "") or str(el.get("content", "") or "")
        return _wrap_inline_code(code) if code else ""
    return ""


def _to_post(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict) or not isinstance(candidate.get("content"), list):
        return {}
    return {"title": str(candidate.get("title", "") or ""), "content": candidate["content"]}


def _resolve_post(payload: Any) -> dict[str, Any]:
    direct = _to_post(payload)
    if direct or not isinstance(payload, dict):
        return direct
    for outer in (payload.get("post"), payload):
        if not isinstance(outer, dict):
            continue
        direct = _to_post(outer)
        if direct:
            return direct
        preferred = (outer.get(k) for k in _PREFERRED_LOCALES)
        for cand in map(_to_post, itertools.chain(preferred, outer.values())):
            if cand:
                return cand
    return {}


def parse_post(payload: Any, mentions: dict[str, MentionRef]) -> str:
    """Flatten a ``post`` (rich text) message to Markdown-ish text."""
    post = _resolve_post(payload)
    if not post:
        return ""
    parts: list[str] = []
    title = normalize_text(post["title"])
    if title:
        parts.append(title)
    for row in post["content"]:
        if not isinstance(row, list):
            continue
        line = "".join(_render_post_element(item, mentions) for item in row)
        if "```" not in line:
            line = normalize_text(line, mentions)
        if line.strip():
            parts.append(line)
    return "\n".join(parts).strip()


def load_content(raw: str) -> dict[str, Any]:
    try:
        v = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


# ----------------------------------------------------------------- outbound
def fence_segments(content: str) -> list[str]:
    """Split Markdown so each fenced code block is its own segment (Hermes
    ``_build_markdown_post_rows``)."""
    if not content or "```" not in content:
        return [content]
    out: list[str] = []
    cur: list[str] = []
    in_code = False

    def flush() -> None:
        nonlocal cur
        seg = "\n".join(cur)
        if seg.strip():
            out.append(seg)
        cur = []

    for line in content.splitlines():
        is_fence = bool((_FENCE_CLOSE_RE if in_code else _FENCE_OPEN_RE).match(line.strip()))
        if is_fence and not in_code:
            flush()
        cur.append(line)
        if is_fence:
            in_code = not in_code
            if not in_code:
                flush()
    flush()
    return out or [content]


# ------------------------------------------------- lark_oapi.ws.client shims
_PROXY_LOCK = threading.Lock()
_thread_state = threading.local()


class _ThreadLoopProxy:
    """Stands in for ``lark_oapi.ws.client.loop``: forwards to the loop the
    current thread registered (``bind_thread_loop``), else the SDK's own."""

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback

    def __getattr__(self, name: str) -> Any:
        return getattr(getattr(_thread_state, "loop", None) or self._fallback, name)


def install_loop_proxy(ws_module: Any) -> None:
    """Idempotent, process-wide."""
    with _PROXY_LOCK:
        if not isinstance(ws_module.loop, _ThreadLoopProxy):
            ws_module.loop = _ThreadLoopProxy(ws_module.loop)


def bind_thread_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    _thread_state.loop = loop


def ws_client_class(ws_module: Any, *, on_link_up: Callable[[], None],
                    on_dead: Callable[[BaseException], None]) -> Any:
    """A ``lark_oapi.ws.Client`` subclass whose receive loop reports a fresh
    link and, when it dies for good (the SDK's reconnect ladder raised), stops
    the worker loop so ``start()`` returns instead of hanging deaf forever."""
    base = ws_module.Client

    class GuardedClient(base):  # type: ignore[misc, valid-type]
        async def _receive_message_loop(self) -> None:
            on_link_up()
            try:
                await super()._receive_message_loop()
            except Exception as e:  # noqa: BLE001 - reported, then the supervisor rebuilds
                on_dead(e)
                asyncio.get_running_loop().stop()

    return GuardedClient
