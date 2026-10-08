"""Result dataclasses, text helpers, limits and extension tables.

Portions adapted from Hermes Agent (https://github.com/NousResearch/hermes-agent),
Copyright (c) 2025 Nous Research, MIT License. Sources: ``tools/file_operations_common.py``,
``tools/binary_extensions.py``, ``tools/tool_output_limits.py``, ``agent/search_policy.py``.
Leaf module: imports nothing else from this package.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- limits
# Hermes reads these from ``tool_output`` in config.yaml; Ventri keeps the
# defaults as module constants (tests may monkeypatch them).
MAX_LINES = 2000            # read pagination cap (lines per call)
MAX_LINE_LENGTH = 2000      # per-line clamp before '... [truncated]'
MAX_READ_CHARS = 100_000    # char budget per read, cut on a line boundary
DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 2000
DEFAULT_SEARCH_OFFSET = 0
DEFAULT_SEARCH_LIMIT = 50


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_read_pagination(offset: Any = DEFAULT_READ_OFFSET,
                              limit: Any = DEFAULT_READ_LIMIT) -> tuple[int, int]:
    """Clamp read pagination: offset >= 1, 1 <= limit <= MAX_LINES."""
    normalized_offset = max(1, _coerce_int(offset, DEFAULT_READ_OFFSET))
    normalized_limit = max(1, min(_coerce_int(limit, DEFAULT_READ_LIMIT), MAX_LINES))
    return normalized_offset, normalized_limit


def normalize_search_pagination(offset: Any = DEFAULT_SEARCH_OFFSET,
                                limit: Any = DEFAULT_SEARCH_LIMIT) -> tuple[int, int]:
    return (max(0, _coerce_int(offset, DEFAULT_SEARCH_OFFSET)),
            max(1, _coerce_int(limit, DEFAULT_SEARCH_LIMIT)))


# ---------------------------------------------------------------- results
@dataclass
class ReadResult:
    content: str = ""
    total_lines: int = 0
    file_size: int = 0
    truncated: bool = False
    truncated_lines: bool | None = None
    hint: str | None = None
    is_binary: bool = False
    error: str | None = None
    not_found: bool = False
    similar_files: list[str] = field(default_factory=lambda: list[str]())
    encoding: str | None = None          # set when transcoded (UTF-16 rescue)
    # (dev, ino, size, mtime_ns, ctime_ns, sha256 digest) of the bytes this read saw
    snapshot: tuple[Any, ...] | None = None
    undecodable: int = 0                 # bytes in the page that are not valid UTF-8 (shown as U+FFFD)


@dataclass
class WriteResult:
    bytes_written: int = 0
    dirs_created: bool = False
    verified: bool | None = None
    content_sha256: str | None = None
    lint: dict[str, Any] | None = None
    error: str | None = None
    warning: str | None = None


@dataclass
class PatchResult:
    success: bool = False
    diff: str = ""
    lint: dict[str, Any] | None = None
    error: str | None = None
    no_change: bool = False
    note: str | None = None
    strategy: str | None = None
    replacements: int = 0
    # (read sha256, written sha256) of this edit; used to carry full-content baselines
    read_sha256: str | None = None
    written_sha256: str | None = None
    error_detail: str | None = None      # file-derived text (snippets): untrusted data


@dataclass
class SearchMatch:
    path: str
    line_number: int
    content: str
    mtime: float = 0.0


@dataclass
class SearchResult:
    matches: list[SearchMatch] = field(default_factory=lambda: list[SearchMatch]())
    files: list[str] = field(default_factory=lambda: list[str]())
    counts: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    total_count: int = 0
    truncated: bool = False
    limit_reason: str | None = None
    warning: str | None = None
    error: str | None = None
    engine: str = ""


@dataclass
class LintResult:
    success: bool = True
    skipped: bool = False
    output: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        if self.skipped:
            return {"status": "skipped", "message": self.message}
        result: dict[str, Any] = {"status": "ok" if self.success else "error", "output": self.output}
        if self.message:
            result["message"] = self.message
        return result


# ---------------------------------------------------------------- text helpers
_CONFLICT_OPEN = re.compile(r"^\s*\d+\|<<<<<<< ", re.MULTILINE)
_CONFLICT_CLOSE = re.compile(r"^\s*\d+\|>>>>>>> ", re.MULTILINE)


def count_conflict_blocks(formatted_content: str) -> int:
    """Unresolved git merge-conflict blocks in a ``LINE|CONTENT`` read (balanced pairs only)."""
    opens = len(_CONFLICT_OPEN.findall(formatted_content))
    return min(opens, len(_CONFLICT_CLOSE.findall(formatted_content))) if opens else 0


def detect_line_ending(sample: str | None) -> str | None:
    """``\\r\\n`` if any CRLF in the first 4KB, else ``\\n``; None for single-line content."""
    head = sample[:4096] if sample else ""
    if "\r\n" in head:
        return "\r\n"
    if "\n" in head:
        return "\n"
    return None


def normalize_line_endings(text: str, target: str) -> str:
    """Convert every line ending (CRLF, lone CR, LF) to ``target``; idempotent."""
    lf_normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if target == "\n":
        return lf_normalized
    if target == "\r\n":
        return lf_normalized.replace("\n", "\r\n")
    return text


UTF8_BOM = "\ufeff"


def has_bom(text: str | None) -> bool:
    return bool(text) and text is not None and text.startswith(UTF8_BOM)


def strip_bom(text: str) -> tuple[str, bool]:
    """(text without a leading BOM, had_bom); mid-content U+FEFF is data."""
    if has_bom(text):
        return text[len(UTF8_BOM):], True
    return text, False


# ---------------------------------------------------------------- extensions
# Source: tools/binary_extensions.py (itself ported from free-code).
BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg",
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz", ".z", ".tgz", ".iso",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".obj", ".lib", ".app", ".msi", ".deb", ".rpm",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".odp",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear", ".node", ".wasm", ".rlib",
    ".sqlite", ".sqlite3", ".db", ".mdb", ".idx",
    ".psd", ".ai", ".eps", ".sketch", ".fig", ".xd", ".blend", ".3ds", ".max",
    ".swf", ".fla", ".lockb", ".dat", ".data",
})
# Container documents a plain-text write can never produce validly.
OPAQUE_DOCUMENT_EXTENSIONS = frozenset({
    ".doc", ".docx", ".docm", ".xls", ".xlsx", ".xlsm", ".xlsb",
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".odt", ".ods", ".odp", ".rtf", ".epub",
})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"})
_SQLITE_SIDECAR_MARKERS = ("-wal", "-shm", "-journal")
_SQLITE_EXTENSIONS = frozenset({".db", ".sqlite", ".sqlite3"})


def _lower_suffix(path: str) -> str:
    base = path.replace("\\", "/").rsplit("/", 1)[-1]
    dot = base.rfind(".")
    return "" if dot == -1 else base[dot:].lower()


def _strip_sidecar_marker(suffix: str) -> str | None:
    for marker in _SQLITE_SIDECAR_MARKERS:
        if suffix.endswith(marker):
            base = suffix[: -len(marker)]
            if base in _SQLITE_EXTENSIONS:
                return base
    return None


def _has_extension_in(path: str, extensions: frozenset[str]) -> bool:
    suffix = _lower_suffix(path)
    return (_strip_sidecar_marker(suffix) or suffix) in extensions


def is_sqlite_sidecar(path: str) -> bool:
    return _strip_sidecar_marker(_lower_suffix(path)) is not None


def has_binary_extension(path: str) -> bool:
    return _has_extension_in(path, BINARY_EXTENSIONS)


def has_opaque_document_extension(path: str) -> bool:
    return _has_extension_in(path, OPAQUE_DOCUMENT_EXTENSIONS)


def is_pdf_path(path: str) -> bool:
    return path.lower().endswith(".pdf")


def is_image_path(path: str) -> bool:
    return _lower_suffix(path) in IMAGE_EXTENSIONS


# Source: agent/search_policy.py -- heavyweight trees pruned by broad scans.
SEARCH_PRUNE_DIR_NAMES = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "venv", ".venv", "site-packages", "dist-packages",
    "vendor", "third_party",
    "build", "dist", "target", "out", "coverage",
    ".next", ".turbo", ".parcel-cache", ".nuxt", ".svelte-kit",
    ".archive",
    "__pycache__", ".cache", ".Trash", ".tox", ".nox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".npm", ".yarn", ".pnpm-store",
    ".gradle", ".m2", ".nuget",
    "backups", "backup", ".backups",
})


def human_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} bytes"
