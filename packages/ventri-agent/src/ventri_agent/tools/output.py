"""Shared handling for large tool output (shell.run, web.fetch).

A tool that can produce a lot of text keeps the full (capped) output in an
artifact file and returns a head + tail preview that stays under the loop's
spill threshold, with an ``artifact.read`` pointer to the rest. Sizes are
measured with :func:`ventri_agent.tokens.estimate_tokens` so Chinese output is
budgeted correctly.

The ANSI / control-character / Unicode-tag stripping and the 40% head / 60%
tail split are adapted from Hermes Agent (tools/ansi_strip.py,
tools/tool_output_truncate.py), Copyright (c) 2025 Nous Research, MIT License
-- see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import re
import secrets

from ..tokens import estimate_tokens, prefix_within, suffix_within
from .registry import ToolContext

# The loop spills results above 8000 tokens; previews stay well below that so a
# tool's own head+tail never gets re-spilled (which would hide the pointer).
PREVIEW_TOKENS = 6_000
ARTIFACT_MAX_CHARS = 4_000_000      # full output kept on disk (≈ a few MB)

# Full ECMA-48 escape coverage (from Hermes tools/ansi_strip.py): CSI, OSC (BEL/ST),
# DCS/SOS/PM/APC, nF, Fp/Fe/Fs and 8-bit C1 controls.
_ANSI = re.compile(
    r"\x1b"
    r"(?:"
    r"\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"     # CSI sequence
    r"|\][\s\S]*?(?:\x07|\x1b\\)"                  # OSC (BEL or ST terminator)
    r"|[PX^_][\s\S]*?(?:\x1b\\)"                   # DCS/SOS/PM/APC strings
    r"|[\x20-\x2f]+[\x30-\x7e]"                    # nF escape sequences
    r"|[\x30-\x7e]"                                 # Fp/Fe/Fs single-byte
    r")"
    r"|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"       # 8-bit CSI
    r"|\x9d[\s\S]*?(?:\x07|\x9c)"                       # 8-bit OSC
    r"|[\x80-\x9f]",                                    # other 8-bit C1 controls
    re.DOTALL)
# C0 controls except tab/newline (CR handled separately) and DEL.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Unicode TAG characters: invisible to people, visible to tokenizers ("ASCII
# smuggling"); emoji tag sequences (U+1F3F4 ... U+E007F) are kept.
_TAGS = re.compile(r"(\U0001F3F4[\U000E0020-\U000E007E]+\U000E007F)|[\U000E0000-\U000E007F]")
_HEAD_RATIO = 0.4


def strip_ansi(text: str) -> str:
    """Remove terminal escapes, bare control characters and Unicode tag
    characters; ``\\r``-overwrites become newlines so nothing hides behind them."""
    if not text:
        return text
    text = _ANSI.sub("", text)
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    return _TAGS.sub(lambda m: m.group(1) or "", text)


def write_artifact(tc: ToolContext, text: str, kind: str) -> str:
    """Store ``text`` under the session's artifact directory; return the handle."""
    d = tc.workdir / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    base = re.sub(r"[^A-Za-z0-9_\-.]", "_", tc.call_id) or secrets.token_hex(6)
    handle = f"{base}-{kind}"
    (d / f"{handle}.txt").write_text(text, encoding="utf-8")
    return handle


def head_tail(text: str, budget: int = PREVIEW_TOKENS) -> tuple[str, str, int]:
    """Split ``text`` into a head (40% of ``budget``: errors surface early) and a
    tail (the rest: the most recent lines matter most).

    Returns ``(head, tail, omitted_chars)``; the caller only uses this when the
    whole text is over budget, so ``omitted_chars`` is > 0 in practice.
    """
    head = prefix_within(text, int(budget * _HEAD_RATIO))
    rest = text[len(head):]
    tail = suffix_within(rest, budget - estimate_tokens(head))
    return head, tail, len(rest) - len(tail)


def preview(tc: ToolContext, text: str, kind: str, *, budget: int = PREVIEW_TOKENS,
            note: str = "") -> str:
    """Return ``text`` unchanged when it fits ``budget``; otherwise write it to an
    artifact and return head + tail with a pointer."""
    if estimate_tokens(text) <= budget:
        return text
    handle = write_artifact(tc, text, kind)
    head, tail, omitted = head_tail(text, budget)
    return (f"{head}\n\n[... {omitted} chars omitted; the full output ({len(text)} chars, "
            f"~{estimate_tokens(text)} tokens{note}) is artifact {handle!r}: "
            f"artifact.read(handle={handle!r}, offset={len(head)}) ...]\n\n{tail}")
