"""CJK-aware token estimate, used for every size decision (artifact spill,
context budget, compaction, the fake provider's usage numbers).

DeepSeek documents the ratio (https://api-docs.deepseek.com/quick_start/token_usage,
checked 2026-10-08): "1 English character ≈ 0.3 token. 1 Chinese character ≈ 0.6
token." The old ``len / 4`` estimate (0.25 per character) under-counted Chinese
by more than 2x. Here CJK ideographs, kana, hangul and CJK/full-width
punctuation count 0.6 and everything else 0.3, rounded up. It is an estimate:
the provider's ``usage`` remains the source of truth for billing.
"""
from __future__ import annotations

import math
import re

CJK_TOKENS_PER_CHAR = 0.6
OTHER_TOKENS_PER_CHAR = 0.3

_CJK = re.compile(
    "[\u2e80-\u2fdf"        # CJK radicals, Kangxi radicals
    "\u3000-\u303f"         # CJK symbols and punctuation
    "\u3040-\u30ff"         # hiragana, katakana
    "\u3100-\u31ff"         # bopomofo, hangul compatibility jamo, kanbun, katakana ext.
    "\u3200-\u33ff"         # enclosed CJK, CJK compatibility
    "\u3400-\u4dbf"         # CJK extension A
    "\u4e00-\u9fff"         # CJK unified ideographs
    "\uac00-\ud7af"         # hangul syllables
    "\uf900-\ufaff"         # CJK compatibility ideographs
    "\ufe30-\ufe4f"         # CJK compatibility forms
    "\uff00-\uffef"         # half-/full-width forms
    "\U00020000-\U0003134f"  # CJK extensions B-G
    "]")


def count_cjk(text: str) -> int:
    return len(_CJK.findall(text))


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = count_cjk(text)
    return max(1, math.ceil(cjk * CJK_TOKENS_PER_CHAR + (len(text) - cjk) * OTHER_TOKENS_PER_CHAR))


def prefix_within(text: str, max_tokens: int) -> str:
    """Longest prefix of ``text`` whose estimate stays within ``max_tokens``."""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:   # binary search on the (monotonic) estimate
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def suffix_within(text: str, max_tokens: int) -> str:
    """Longest suffix of ``text`` whose estimate stays within ``max_tokens``."""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[len(text) - mid:]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[len(text) - lo:] if lo else ""
