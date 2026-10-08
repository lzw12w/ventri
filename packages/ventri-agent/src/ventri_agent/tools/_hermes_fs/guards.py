"""Read- and write-side guards for the file tools.

Portions adapted from Hermes Agent (https://github.com/NousResearch/hermes-agent),
Copyright (c) 2025 Nous Research, MIT License. Sources: ``tools/file_tools.py``
(device-path blocklist, stat-based special-file guard, whole-file rewrite hint)
and ``tools/file_tools_write_guards.py`` (binary-document write guard,
read_file display-text refusal). Hermes's sensitive-path denylist, protected
instruction files, approval gate and sandbox-mirror guards are not ported:
Ventri confines every path to configured roots and decides writes with its own
permission engine.
"""
from __future__ import annotations

import os
import stat
from collections import Counter
from itertools import pairwise
from pathlib import Path

from .common import has_binary_extension, has_opaque_document_extension, is_pdf_path, is_sqlite_sidecar

# Device/fd paths whose reads hang the process (path-only check, no I/O).
BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/tty", "/dev/console",
    "/dev/stdout", "/dev/stderr",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})
# /proc/<pid>/... files that leak secrets, argv, memory layout or raw memory.
BLOCKED_PROC_SUFFIXES = (
    "/fd/0", "/fd/1", "/fd/2",
    "/environ", "/cmdline", "/maps", "/smaps", "/smaps_rollup", "/numa_maps",
    "/mem", "/auxv", "/pagemap")


def is_blocked_device_path(path: str) -> bool:
    normalized = os.path.normpath(path)
    if normalized in BLOCKED_DEVICE_PATHS:
        return True
    return normalized.startswith("/proc/") and normalized.endswith(BLOCKED_PROC_SUFFIXES)


def is_blocked_device(path: str) -> bool:
    """True if the literal path, any symlink hop, or the realpath is a blocked device."""
    normalized = os.path.normpath(os.path.expanduser(path))
    if is_blocked_device_path(normalized):
        return True
    seen: set[str] = set()
    current = normalized
    for _ in range(20):
        try:
            target = os.readlink(current)
        except OSError:
            break
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(current), target)
        target = os.path.normpath(target)
        if is_blocked_device_path(target):
            return True
        if target in seen:
            break
        seen.add(target)
        current = target
    try:
        return is_blocked_device_path(os.path.normpath(os.path.realpath(normalized)))
    except (OSError, ValueError):
        return False


_SPECIAL_FILE_KINDS = (
    (stat.S_ISFIFO, "a FIFO (named pipe)"),
    (stat.S_ISSOCK, "a socket"),
    (stat.S_ISCHR, "a character device"),
    (stat.S_ISBLK, "a block device"))


def special_file_kind(path: str | Path) -> str | None:
    """Human name for a non-regular file type that would hang a read, else None."""
    try:
        mode = os.stat(os.fspath(path)).st_mode
    except OSError:
        return None
    if stat.S_ISREG(mode) or stat.S_ISDIR(mode):
        return None
    return next((label for predicate, label in _SPECIAL_FILE_KINDS if predicate(mode)),
                "a special (non-regular) file")


def check_binary_document_write(display: str, resolved: str) -> str | None:
    """Refuse text writes that would corrupt a binary document: opaque document
    formats and SQLite sidecars always; .pdf and other binary extensions only when
    OVERWRITING an existing file (raw PDF syntax is text-authorable)."""
    ext = os.path.splitext(resolved)[1].lower()
    if has_opaque_document_extension(resolved):
        return (f"Refusing to write plain text to binary document '{display}' ({ext}). A text write "
                "cannot produce a valid document container and would corrupt the file. Use a "
                "library such as python-docx/openpyxl/python-pptx via shell.run to edit it.")
    if is_sqlite_sidecar(resolved):
        return (f"Refusing to write plain text to binary SQLite sidecar '{display}' ({ext}). A "
                "-wal/-shm/-journal file holds raw database pages; text there corrupts the database. "
                "Use the sqlite3 CLI via shell.run instead.")
    pdf = is_pdf_path(resolved)
    if (pdf or has_binary_extension(resolved)) and os.path.isfile(resolved):
        if pdf:
            return (f"Refusing to overwrite existing PDF '{display}' with plain text -- writing text "
                    "back would destroy the document. (Creating a NEW .pdf file is allowed.)")
        return (f"Refusing to overwrite existing binary file '{display}' ({ext}) with plain text -- "
                "writing text back would destroy the file. Use a binary-aware tool via shell.run. "
                "(Creating a NEW file with this extension is allowed.)")
    return None


def looks_like_line_numbered_read(content: str) -> bool:
    """True for content dominated by fs.read's ``LINE_NUM|CONTENT`` display
    (>= 60% of non-empty lines are consecutive numbered lines)."""
    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    numbered: list[int] = []
    for line in lines:
        prefix, sep, _rest = line.lstrip().partition("|")
        if sep and prefix.isdigit():
            numbered.append(int(prefix))
    if len(numbered) < 2 or len(numbered) / len(lines) < 0.6:
        return False
    consecutive = sum(1 for prev, cur in pairwise(numbered) if cur == prev + 1)
    return consecutive >= len(numbered) - 1


# Whole-file rewrite hint thresholds (Hermes: a re-sent file that is >= 80%
# unchanged is a patch written the expensive way).
REWRITE_HINT_MIN_CHARS = 20_000
REWRITE_HINT_MAX_CHARS = 400_000
REWRITE_HINT_MIN_UNCHANGED = 0.80


def whole_file_rewrite_hint(old: str | None, new_content: str) -> str | None:
    if old is None or not (REWRITE_HINT_MIN_CHARS <= len(new_content) <= REWRITE_HINT_MAX_CHARS):
        return None
    if not (REWRITE_HINT_MIN_CHARS <= len(old) <= REWRITE_HINT_MAX_CHARS):
        return None
    old_lines, new_lines = old.splitlines(), new_content.splitlines()
    if not old_lines:
        return None
    unchanged = sum((Counter(old_lines) & Counter(new_lines)).values())
    ratio = unchanged / max(len(old_lines), len(new_lines))
    if ratio < REWRITE_HINT_MIN_UNCHANGED:
        return None
    changed = max(len(old_lines), len(new_lines)) - unchanged
    return (f"{unchanged:,} of {len(new_lines):,} lines were already on disk ({ratio:.0%} unchanged); "
            f"~{changed:,} line(s) actually changed. Re-sending a {len(new_content):,}-char file costs output "
            "tokens for every unchanged line; for edits like this use fs.edit (old_string/new_string), "
            "which sends only the changed region.")
