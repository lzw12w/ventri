"""``fs.read / fs.list / fs.search`` (read) and ``fs.write / fs.edit`` (write-local).

Ventri's architecture is kept: one ``tool:fs`` plugin (config ``roots`` and
``write: ask | allow | deny``), every path resolved (``~`` expanded, symlinks
followed) and confined to the configured roots by the tool itself (so the
default ``allow`` for read tools means "read inside roots"), permission
metadata (risk, subject, default action) and untrusted-output fencing.

The file *behaviour* is ported from Hermes Agent's file tools (MIT; see
``_hermes_fs`` and THIRD_PARTY_NOTICES.md): line-numbered paged reads with a
character budget, binary/device/special-file guards, read-before-write and
stale-file detection, fuzzy ``old_string`` matching with already-applied
detection and match-location errors, unified diffs, CRLF/BOM and non-UTF-8
byte preservation, atomic verified writes with an in-process syntax check,
and ripgrep-backed search with a pure-Python fallback.

All filesystem work runs in a worker thread (``anyio.to_thread``) so a slow
disk or a huge tree never blocks the event loop and tool timeouts / parallel
reads keep working. Read-before-write state lives in the session-scoped
:class:`~._hermes_fs.state.FileState` service (provided by ``sessions.py``).
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import BaseModel, Field

import ventri

from ..tokens import CJK_TOKENS_PER_CHAR, OTHER_TOKENS_PER_CHAR, count_cjk, estimate_tokens, prefix_within
from ._hermes_fs import guards, ops
from ._hermes_fs import search as hsearch
from ._hermes_fs.common import (
    MAX_LINE_LENGTH,
    MAX_LINES,
    UTF8_BOM,
    SearchResult,
    count_conflict_blocks,
    human_size,
    normalize_read_pagination,
)
from ._hermes_fs.state import FileState, file_metadata
from .registry import Risk, Tool, ToolContext, ToolError, ToolRegistry

# The per-read budget. Hermes uses 100K chars; Ventri moves any tool result above
# 8K estimated tokens into an artifact, which would hide most of a read from the
# model while the read-tracker counted it as seen -- so stay below that. Measured
# with the CJK-aware estimate (ventri_agent.tokens): ~23K ASCII chars or ~11K
# Chinese characters.
READ_TOKEN_BUDGET = 7_000
LARGE_FILE_HINT_BYTES = 512_000
LIST_LIMIT = 500

class Roots:
    def __init__(self, roots: list[str]) -> None:
        self.roots = [Path(os.path.realpath(Path(r).expanduser())) for r in roots]

    def contains(self, p: str | Path) -> bool:
        real = Path(os.path.realpath(p))
        return any(real == r or r in real.parents for r in self.roots)

    def resolve(self, p: str, *, must_exist: bool = False) -> Path:
        if not self.roots:
            raise ToolError("no roots configured for this tool")
        if not p or not p.strip():
            raise ToolError("path is empty")
        raw = Path(p).expanduser()
        if not raw.is_absolute():
            raw = self.roots[0] / raw
        real = Path(os.path.realpath(raw))
        if not any(real == r or r in real.parents for r in self.roots):
            raise ToolError(f"{p}: outside the allowed roots ({', '.join(map(str, self.roots))})")
        if must_exist and not real.exists():
            raise ToolError(f"{p}: no such file or directory")
        return real

    def display(self, p: str | Path) -> str:
        """Path as shown to the model: relative to the first root when inside it."""
        path = Path(p)
        if self.roots:
            try:
                rel = path.relative_to(self.roots[0])
                return rel.as_posix() if rel.parts else "."
            except ValueError:
                pass
        return str(path)


def walk(base: Path, pattern: str, recursive: bool, limit: int = 2000) -> list[Path]:
    """Sorted entries below ``base`` matching ``pattern`` (hidden entries and
    symlinked directories skipped; never leaves ``base``)."""
    out: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted([*dirnames, *filenames]):
                if name.startswith("."):
                    continue
                if fnmatch.fnmatch(name, pattern):
                    out.append(Path(dirpath) / name)
                    if len(out) >= limit:
                        return sorted(out)
        return sorted(out)
    for p in sorted(base.iterdir()):
        if not p.name.startswith(".") and fnmatch.fnmatch(p.name, pattern):
            out.append(p)
            if len(out) >= limit:
                break
    return out


def file_state(tc: ToolContext) -> FileState:
    """The session's read-before-write state (a fallback handle keyed by the
    session id when the tool runs outside a session scope, e.g. in tests)."""
    st = None
    if tc.ctx is not None:
        try:
            st = tc.get(FileState)
        except Exception:  # noqa: BLE001 - no session scope
            st = None
    return st if isinstance(st, FileState) else FileState(tc.session_id)


async def in_thread[T](fn: Callable[[], T], *, cancellable: bool) -> T:
    """Run blocking file work off the event loop. Reads may be abandoned on
    cancellation (timeout); writes always run to completion so a timeout can
    never leave a half-applied edit behind."""
    return await anyio.to_thread.run_sync(fn, abandon_on_cancel=cancellable)


# ====================================================================== core operations
# Shared by fs.* and notes.* (``display`` maps resolved paths to model-facing names).

def _truncate_to_budget(content: str, max_tokens: int) -> tuple[str, int, bool]:
    """Trim line-numbered content to the last COMPLETE line within ``max_tokens``
    (Hermes ``_truncate_to_char_budget``, measured in estimated tokens); a first
    line longer than the budget is clamped so the read always advances."""
    if estimate_tokens(content) <= max_tokens:
        return content, (content.count("\n") + 1 if content else 0), False
    lines = content.split("\n")
    kept: list[str] = []
    running = 0.0
    for line in lines:
        cjk = count_cjk(line)
        addition = cjk * CJK_TOKENS_PER_CHAR + (len(line) - cjk + (1 if kept else 0)) * OTHER_TOKENS_PER_CHAR
        if running + addition > max_tokens:
            break
        kept.append(line)
        running += addition
    if not kept:
        kept.append(prefix_within(lines[0], max_tokens))
    return "\n".join(kept), len(kept), True


def _render(head: str, body: str, notes: list[str]) -> str:
    out = head
    if body:
        out += "\n" + body
    for n in notes:
        out += f"\n[{n}]"
    return out


def _check_target(raw: str, resolved: Path, display: str, verb: str) -> None:
    if guards.is_blocked_device(raw) or guards.is_blocked_device(str(resolved)):
        raise ToolError(f"Cannot {verb} '{display}': device/proc paths would block or produce endless output.")
    kind = guards.special_file_kind(resolved)
    if kind:
        raise ToolError(f"Cannot {verb} '{display}': it is {kind}, not a regular file; "
                        "opening it could block forever.")


def read_impl(roots: Roots, fstate: FileState, raw: str, resolved: Path, offset: int, limit: int,
              display: Callable[[Path], str]) -> str:
    """Paged, line-numbered read (Hermes ``read_file_tool`` minus the parts that
    need Hermes's environment: document extraction, secret redaction, read dedup)."""
    shown = display(resolved)
    _check_target(raw, resolved, shown, "read")
    if resolved.is_dir():
        raise ToolError(f"'{shown}' is a directory. Use fs.list to see its entries or "
                        "fs.search(target='files') to find files by name.")
    offset, limit = normalize_read_pagination(offset, limit)
    version_before = file_metadata(str(resolved))
    r = ops.read_file(str(resolved), offset, limit, allow=roots.contains)
    if r.error:
        msg = r.error.replace(str(resolved), shown)
        if r.not_found and r.similar_files:
            msg += "\nSimilar files: " + ", ".join(display(Path(f)) for f in r.similar_files)
        raise ToolError(msg)
    notes: list[str] = []
    content = r.content
    truncated = r.truncated
    end_line = min(offset + limit - 1, r.total_lines) if r.total_lines else offset + limit - 1
    if estimate_tokens(content) > READ_TOKEN_BUDGET:
        content, kept, _ = _truncate_to_budget(content, READ_TOKEN_BUDGET)
        next_offset = offset + kept
        end_line = next_offset - 1
        truncated = True
        notes.append(f"Output truncated at the ~{READ_TOKEN_BUDGET:,}-token read budget after {kept} line(s) "
                     f"(showing lines {offset}-{end_line} of {r.total_lines}). Use offset={next_offset} "
                     "to continue.")
    elif r.hint:
        notes.append(r.hint)
    if r.undecodable:
        notes.append(f"{r.undecodable} byte(s) in this range are not valid UTF-8 and are shown as U+FFFD. "
                     "fs.edit keeps those bytes intact; a whole-file fs.write overwrite would replace them, "
                     "so this view does not count as a full read for overwriting.")
    if r.truncated_lines:
        notes.append(f"Some lines were longer than {MAX_LINE_LENGTH} chars and were clamped; "
                     "fs.search can locate text inside them.")
    conflicts = count_conflict_blocks(content) if content else 0
    if conflicts:
        notes.append(f"{conflicts} unresolved git merge-conflict block(s) (<<<<<<< / ======= / >>>>>>>) in "
                     "this range. Resolve them (keep one side or combine, delete the markers) before editing "
                     "around them.")
    if r.file_size > LARGE_FILE_HINT_BYTES and limit > 200 and truncated:
        notes.append(f"This file is large ({r.file_size:,} bytes). Consider reading only the section you need "
                     "with offset and limit to keep context usage efficient.")
    past_eof = r.total_lines > 0 and offset > r.total_lines
    if not past_eof:
        fstate.record_read(str(resolved), version_before=version_before, snapshot=r.snapshot,
                           partial=(offset > 1) or truncated,
                           redacted=bool(r.truncated_lines) or r.undecodable > 0, offset=offset,
                           end_line=end_line,
                           total_lines=r.total_lines)
    head = f"{shown} ({r.total_lines} lines, {human_size(r.file_size)}"
    head += f", decoded from {r.encoding}" if r.encoding else ""
    head += ")"
    return _render(head, content, notes)


def search_impl(spec: hsearch.SearchSpec, display: Callable[[Path], str]) -> SearchResult:
    return hsearch.search(spec, lambda p: display(Path(p)))


def render_search(res: SearchResult, spec: hsearch.SearchSpec, display: Callable[[Path], str],
                  extra_notes: list[str] | None = None) -> str:
    if res.error:
        raise ToolError(res.error)
    notes = list(extra_notes or [])
    shown = display(Path(spec.base))
    total = f"{res.total_count}+" if res.truncated else str(res.total_count)
    rows: list[str]
    if spec.target == "files":
        rows = [display(Path(f)) + ("/" if os.path.isdir(f) else "") for f in res.files]
        head = (f"{total} path(s) matching {spec.pattern!r} under {shown}" if rows
                else f"No files matching {spec.pattern!r} under {shown}")
    elif spec.output_mode == "files_only":
        rows = [display(Path(f)) for f in res.files]
        head = (f"{total} file(s) containing {spec.pattern!r} under {shown}" if rows
                else f"No matches for {spec.pattern!r} under {shown}")
    elif spec.output_mode == "count":
        rows = [f"{display(Path(p))}: {n}" for p, n in sorted(res.counts.items())]
        head = (f"{res.total_count} matching line(s) in {len(res.counts)} file(s) under {shown}" if rows
                else f"No matches for {spec.pattern!r} under {shown}")
    else:
        rows = [f"{display(Path(m.path))}:{m.line_number}: {m.content}" for m in res.matches]
        head = (f"{total} match(es) for {spec.pattern!r} under {shown}" if rows
                else f"No matches for {spec.pattern!r} under {shown}")
    if res.truncated:
        if res.limit_reason:
            notes.append(f"Search stopped early ({res.limit_reason}); results may be incomplete. Narrow the "
                         "path, pattern or file_glob.")
        else:
            notes.append(f"Results truncated. Use offset={spec.offset + spec.limit} to see more, or narrow "
                         "with a more specific pattern or file_glob.")
    if res.warning:
        notes.append(res.warning)
    return _render(head, "\n".join(rows), notes)


def _lint_notes(lint: dict[str, Any] | None) -> list[str]:
    if not lint or lint.get("status") in (None, "ok", "skipped"):
        return []
    out = str(lint.get("output") or "").strip()
    msg = str(lint.get("message") or "").strip()
    return [f"Syntax check failed: {(msg + ' ') if msg else ''}{out}".strip()]


def write_impl(fstate: FileState, raw: str, resolved: Path, content: str, mode: str,
               display: Callable[[Path], str]) -> str:
    """create | overwrite | append. Overwrite follows Hermes ``write_file``: an
    existing file is replaced only when this session holds its full current
    content (read in full / written by it, unchanged since); refusals happen
    before any disk mutation."""
    shown = display(resolved)
    _check_target(raw, resolved, shown, "write")
    if resolved.is_dir():
        raise ToolError(f"'{shown}' is a directory; give a file path.")
    refused = guards.check_binary_document_write(shown, str(resolved))
    if refused:
        raise ToolError(refused)
    if guards.looks_like_line_numbered_read(content):
        raise ToolError("content looks like fs.read output (every line starts with a 'N|' line-number "
                        "prefix). Strip the line-number prefixes and send only the file text.")
    rp = str(resolved)
    notes: list[str] = []
    with fstate.lock(rp):
        exists = os.path.lexists(rp)
        if mode == "create" and exists:
            raise ToolError(f"{shown} already exists. Use fs.edit for a targeted change, mode='append' to add "
                            "to the end, or read it in full and use mode='overwrite'.")
        pre: str | None = None
        read_sha: str | None = ""
        known: str | None = None
        if exists:
            try:
                data = ops.read_exact_bytes(rp)
            except OSError as e:
                raise ToolError(f"Cannot read existing '{shown}': {e}") from e
            pre = ops.decode_exact(data)
            read_sha = hashlib.sha256(data).hexdigest()
        if mode == "overwrite" and exists:
            blocker = fstate.stale_overwrite_blocker(rp)
            if blocker:
                raise ToolError(f"Refusing to overwrite {shown} (file untouched): "
                                + blocker.replace(rp, shown))
            hint = guards.whole_file_rewrite_hint(pre, content)
            if hint:
                notes.append(hint)
        text = content
        if mode == "append" and pre is not None:
            known = fstate.known_full_sha256(rp)
            stale = fstate.check_stale(rp)
            if stale and "not read by this session" not in stale:
                notes.append("Warning: " + stale.replace(rp, shown))
            sep = ""
            body = pre.removeprefix(UTF8_BOM)
            if body and not body.endswith(("\n", "\r")) and not content.startswith(("\n", "\r")):
                sep = "\n"
                notes.append("The file did not end with a newline; one was added before the appended text.")
            text = pre + sep + content
        res = ops.write_file(rp, text, pre_content=pre)
        if res.error:
            raise ToolError(res.error.replace(rp, shown))
        fstate.note_write(rp)
        fstate.reset_patch_failures(rp)
        if mode == "append" and pre is not None:
            fstate.carry_baseline(rp, known, read_sha, res.content_sha256)
        else:
            fstate.carry_baseline(rp, None, "", res.content_sha256)   # the session wrote all of it
    verb = {"overwrite": "Wrote", "append": "Appended to"}.get(mode, "Created") if exists else "Created"
    head = f"{verb} {shown} ({res.bytes_written} bytes"
    head += ", created parent directories" if res.dirs_created else ""
    head += ", verified on disk)" if res.verified else ")"
    notes += _lint_notes(res.lint)
    if res.warning:
        notes.append(res.warning)
    return _render(head, "", notes)


def edit_impl(fstate: FileState, raw: str, resolved: Path, edits: list[tuple[str, str, bool]],
              display: Callable[[Path], str]) -> str:
    """Fuzzy find-and-replace (Hermes ``patch_tool`` replace mode). Stale reads
    only warn here, as in Hermes: the edit re-reads the file and anchors on
    ``old_string``, so it cannot clobber unseen changes."""
    shown = display(resolved)
    _check_target(raw, resolved, shown, "edit")
    if resolved.is_dir():
        raise ToolError(f"'{shown}' is a directory; give a file path.")
    if not os.path.exists(resolved):
        raise ToolError(f"File not found: {shown}. Use fs.write to create a new file.")
    refused = guards.check_binary_document_write(shown, str(resolved))
    if refused:
        raise ToolError(refused)
    rp = str(resolved)
    with fstate.lock(rp):
        stale = fstate.check_stale(rp)
        known = fstate.known_full_sha256(rp)
        r = ops.patch_replace(rp, edits, display=shown)
        if r.error:
            msg = r.error.replace(rp, shown)
            if "Could not find" in msg:
                n = fstate.record_patch_failure(rp)
                if n >= 3:
                    msg += (f"\nThis is failure #{n} editing {shown!r}. Stop retrying with variations of the same "
                            "old_string. Either: (1) re-read the file fresh to verify current content, (2) use a "
                            "longer / more unique old_string with surrounding context lines, or (3) read the file "
                            "in full, then use fs.write mode='overwrite' if the region is hard to anchor.")
                elif not r.error_detail or "Did you mean" not in r.error_detail:
                    msg += "\nold_string not found. Use fs.read to verify the current content, or fs.search to locate the text."
            raise ToolError(msg, untrusted=r.error_detail)
        fstate.reset_patch_failures(rp)
        if r.no_change:
            return _render(f"No change to {shown}", "", [r.note or "Edit already applied."])
        fstate.note_write(rp)
        fstate.carry_baseline(rp, known, r.read_sha256, r.written_sha256)
    notes: list[str] = []
    strategies = [s for s in (r.strategy or "").split(",") if s and s not in ("exact", "already_applied")]
    if strategies:
        notes.append(f"old_string matched via fuzzy strategy: {', '.join(sorted(set(strategies)))}. "
                     "Check the diff.")
    notes += _lint_notes(r.lint)
    if stale:
        notes.append("Warning: " + stale.replace(rp, shown))
    head = f"Edited {shown}: {r.replacements} replacement(s)"
    return _render(head, r.diff.rstrip("\n"), notes)


# ====================================================================== argument models
class ReadArgs(BaseModel):
    path: str = Field(description="File path (absolute, ~/..., or relative to the first root)")
    offset: int = Field(1, description="Line number to start reading from (1-indexed, default: 1)")
    limit: int = Field(MAX_LINES, description=(
        f"Maximum number of lines to read (default and max: {MAX_LINES}). Reads are additionally capped at "
        f"a ~{READ_TOKEN_BUDGET // 1000}K-token budget with an offset= continuation."))


class ListArgs(BaseModel):
    path: str = Field(".", description="Directory")
    pattern: str = Field("*", description="Glob for names, e.g. *.md")
    recursive: bool = Field(False, description="Recurse into subdirectories")


class SearchArgs(BaseModel):
    pattern: str = Field(description=(
        "Regex pattern for content search, or glob pattern (e.g. '*.py', '*config*') for file search"))
    target: Literal["content", "files"] = Field(
        "content", description="'content' searches inside file contents, 'files' searches for files by name")
    path: str = Field(".", description=(
        "Directory or file to search in (default: the first root). Several paths may be given "
        "comma-separated."))
    file_glob: str | None = Field(None, description=(
        "Content search: only search files matching this glob (e.g. '*.py')"))
    limit: int = Field(50, description="Maximum number of results to return (default: 50)")
    offset: int = Field(0, description="Skip first N results for pagination (default: 0)")
    output_mode: Literal["content", "files_only", "count"] = Field("content", description=(
        "Content search output: 'content' = matching lines with line numbers, 'files_only' = file paths, "
        "'count' = match counts per file"))
    context: int = Field(0, description="Content search: lines of context before and after each match")
    order: Literal["discovery", "modified"] = Field("discovery", description=(
        "File search order: 'discovery' (fast, bounded) or 'modified' (newest first; may scan the whole "
        "tree). Ignored for content search."))
    ignore_case: bool = Field(False, description="Case-insensitive matching")
    literal: bool = Field(False, description="Treat pattern as a literal string instead of a regex")


class WriteArgs(BaseModel):
    path: str = Field(description="File path (created with parent directories if missing)")
    content: str = Field(description="Text to write (the complete file for create/overwrite)")
    mode: Literal["create", "overwrite", "append"] = Field("create", description=(
        "create: new file, fails if it exists; overwrite: replace the whole file (an existing file must "
        "have been read in full first); append: add to the end"))


class EditItem(BaseModel):
    old_string: str = Field(description="Text to find (unique unless replace_all)")
    new_string: str = Field(description="Replacement text ('' deletes)")
    replace_all: bool | None = Field(None, description="Replace every occurrence (default false)")


class EditArgs(BaseModel):
    path: str = Field(description="File to edit")
    old_string: str | None = Field(None, description=(
        "Exact text to find and replace. Must be unique in the file unless replace_all=true. Include "
        "surrounding context lines to ensure uniqueness. Copy it from fs.read output WITHOUT the 'N|' "
        "line-number prefix."))
    new_string: str | None = Field(None, description=(
        "Replacement text; it must differ from old_string. Pass '' to delete the matched text."))
    replace_all: bool = Field(False, description=(
        "Replace all occurrences instead of requiring a unique match (default: false)"))
    edits: list[EditItem] | None = Field(None, description=(
        "Several replacements in this file, applied in order and atomically (all or nothing). Use "
        "instead of old_string/new_string."))


READ_DESC = (
    "Read a text file with line numbers and pagination. Output format: 'LINE_NUM|CONTENT' (the prefix is "
    "not part of the file). Suggests similar filenames if not found. Use offset and limit for large files; "
    f"reads exceeding ~{READ_TOKEN_BUDGET // 1000}K tokens are truncated on a line boundary with an "
    "offset= continuation. Cannot read images/binary files (they are identified instead). Read a file "
    "before overwriting it with fs.write.")
LIST_DESC = "List a directory (hidden entries skipped). For finding files by name across a tree prefer fs.search target='files'."
SEARCH_DESC = (
    "Search file contents or find files by name (ripgrep-backed when available). Hidden directories and "
    "gitignored files are skipped by default.\n\n"
    "Content search (target='content'): regex search inside files. Output modes: matching lines with line "
    "numbers, file paths only, or match counts.\n\n"
    "File search (target='files'): find files by glob pattern (e.g. '*.py', '*config*'); also lists "
    "matching directories.")
WRITE_DESC = (
    "Create, overwrite or append to a text file; creates parent directories automatically. mode='overwrite' "
    "REPLACES the entire file -- use fs.edit for targeted changes. For an EXISTING file, read it first: "
    "overwrite refuses (file untouched) when this session has no current full read/write of the file or the "
    "file changed on disk since; on refusal, fs.read, merge, then retry. append adds a newline first when the "
    "file does not end with one. CRLF line endings, a UTF-8 BOM and undecodable bytes of an existing file are "
    "preserved; the write is atomic and verified on disk (do NOT re-read to check it landed). .py/.json/.yaml/"
    ".toml content is syntax-checked (JSON/YAML/TOML syntax errors are refused).")
EDIT_DESC = (
    "Targeted find-and-replace edits in a file. Uses fuzzy matching (9 strategies) so minor whitespace/"
    "indentation differences won't break it; if old_string matches several places the error lists their "
    "line numbers; an edit that is already applied is reported as such instead of failing. Returns a "
    "unified diff (do not re-read the file to verify). Several replacements in one file: pass edits=[...].")


# ====================================================================== tools
def make_tools(roots: Roots, *, write: str = "ask", prefix: str = "fs") -> list[Tool]:
    def show(p: Path) -> str:
        return roots.display(p)

    async def read(a: ReadArgs, tc: ToolContext) -> str:
        st = file_state(tc)

        def work() -> str:
            if guards.is_blocked_device(a.path):
                raise ToolError(f"Cannot read '{a.path}': device/proc paths would block or produce endless output.")
            return read_impl(roots, st, a.path, roots.resolve(a.path), a.offset, a.limit, show)
        return await in_thread(work, cancellable=True)

    async def ls(a: ListArgs, tc: ToolContext) -> str:
        def work() -> str:
            base = roots.resolve(a.path, must_exist=True)
            if not base.is_dir():
                raise ToolError(f"{a.path}: not a directory (use fs.read for files)")
            rows: list[str] = []
            entries = walk(base, a.pattern, a.recursive, LIST_LIMIT + 1)
            for p in entries[:LIST_LIMIT]:
                rel = p.relative_to(base).as_posix()
                try:
                    if p.is_symlink():
                        rows.append(f"{rel} -> (symlink)")
                    elif p.is_dir():
                        rows.append(f"{rel}/")
                    else:
                        rows.append(f"{rel}  ({human_size(p.stat().st_size)})")
                except OSError:
                    rows.append(rel)
            out = f"{show(base)}:\n" + ("\n".join(rows) or "(empty)")
            if len(entries) > LIST_LIMIT:
                out += (f"\n[Listing truncated at {LIST_LIMIT} entries. Narrow it with pattern= or a "
                        "subdirectory path, or use fs.search target='files'.]")
            return out
        return await in_thread(work, cancellable=True)

    async def search(a: SearchArgs, tc: ToolContext) -> str:
        def work() -> str:
            return search_paths(roots, a, show)
        return await in_thread(work, cancellable=True)

    async def write_(a: WriteArgs, tc: ToolContext) -> str:
        st = file_state(tc)

        def work() -> str:
            if guards.is_blocked_device(a.path):
                raise ToolError(f"Cannot write '{a.path}': device/proc paths are refused.")
            return write_impl(st, a.path, roots.resolve(a.path), a.content, a.mode, show)
        return await in_thread(work, cancellable=False)

    async def edit(a: EditArgs, tc: ToolContext) -> str:
        st = file_state(tc)
        edits = edit_list(a)

        def work() -> str:
            if guards.is_blocked_device(a.path):
                raise ToolError(f"Cannot edit '{a.path}': device/proc paths are refused.")
            return edit_impl(st, a.path, roots.resolve(a.path), edits, show)
        return await in_thread(work, cancellable=False)

    def subj(a: Any) -> dict[str, str]:
        try:
            return {"path": str(roots.resolve(a.path))}
        except ToolError:
            return {"path": a.path}

    w: Any = write if write in ("ask", "allow", "deny") else "ask"
    return [
        Tool(f"{prefix}.read", READ_DESC, read, ReadArgs, parallel_safe=True, subject=subj, untrusted=True),
        Tool(f"{prefix}.list", LIST_DESC, ls, ListArgs, parallel_safe=True, subject=subj),
        Tool(f"{prefix}.search", SEARCH_DESC, search, SearchArgs, parallel_safe=True, subject=subj,
             untrusted=True),
        Tool(f"{prefix}.write", WRITE_DESC, write_, WriteArgs,
             risk=Risk.WRITE_LOCAL, idempotent=False, default_action=w, subject=subj),
        Tool(f"{prefix}.edit", EDIT_DESC, edit, EditArgs,
             risk=Risk.WRITE_LOCAL, idempotent=False, default_action=w, subject=subj, untrusted=True),
    ]


def edit_list(a: EditArgs) -> list[tuple[str, str, bool]]:
    if a.edits:
        if a.old_string is not None or a.new_string is not None:
            raise ToolError("pass either old_string/new_string or edits, not both")
        return [(e.old_string, e.new_string, bool(e.replace_all)) for e in a.edits]
    if a.old_string is None or a.new_string is None:
        raise ToolError("old_string and new_string are required (or pass edits=[...])")
    return [(a.old_string, a.new_string, a.replace_all)]


def search_paths(roots: Roots, a: SearchArgs, show: Callable[[Path], str]) -> str:
    """Resolve (possibly comma-separated) search roots, run the search on each
    and merge (Hermes multi-path recovery: missing paths are skipped with a note)."""
    raw_paths = [a.path]
    if "," in a.path and not os.path.exists(roots.resolve(a.path)):
        raw_paths = [p.strip() for p in a.path.split(",") if p.strip()]
    bases: list[Path] = []
    missing: list[str] = []
    for rp in raw_paths:
        if guards.is_blocked_device(rp):
            raise ToolError(f"Cannot search '{rp}': device/proc paths are refused.")
        base = roots.resolve(rp)
        if not base.exists():
            missing.append(rp)
        else:
            bases.append(base)
    if not bases:
        if len(raw_paths) == 1:
            similar = ops.suggest_similar_files(str(roots.resolve(raw_paths[0]))).similar_files
            msg = f"Path not found: {raw_paths[0]}"
            if similar:
                msg += "\nSimilar paths: " + ", ".join(show(Path(s)) for s in similar)
            raise ToolError(msg)
        raise ToolError(f"None of the search paths exist: {', '.join(raw_paths)}")
    notes = [f"Skipped missing path(s): {', '.join(missing)}"] if missing else []

    def spec_for(base: Path, offset: int, limit: int) -> hsearch.SearchSpec:
        return hsearch.SearchSpec(
            pattern=a.pattern, base=str(base), target=a.target, file_glob=a.file_glob or None,
            limit=limit, offset=offset, output_mode=a.output_mode, context=max(0, min(a.context, 20)),
            order=a.order, ignore_case=a.ignore_case, fixed_strings=a.literal)

    if len(bases) == 1:
        spec = spec_for(bases[0], a.offset, a.limit)
        return render_search(search_impl(spec, show), spec, show, notes)
    offset, limit = max(0, a.offset), max(1, min(a.limit, 1000))
    merged = SearchResult()
    for base in bases:
        r = search_impl(spec_for(base, 0, offset + limit), show)
        if r.error:
            raise ToolError(r.error)
        merged.matches += r.matches
        merged.files += r.files
        merged.counts.update(r.counts)
        merged.total_count += r.total_count
        merged.truncated = merged.truncated or r.truncated
        merged.limit_reason = merged.limit_reason or r.limit_reason
        if r.warning:
            merged.warning = f"{merged.warning} {r.warning}" if merged.warning else r.warning
    if len(merged.matches) > offset + limit or len(merged.files) > offset + limit:
        merged.truncated = True
    merged.matches = merged.matches[offset:offset + limit]
    merged.files = merged.files[offset:offset + limit]
    combined = spec_for(Path(os.path.commonpath([str(b) for b in bases])), offset, limit)
    return render_search(merged, combined, show, notes)


class FsConfig(BaseModel):
    roots: list[str] = Field(default_factory=lambda: ["~/.ventri/workspace"])
    write: Literal["ask", "allow", "deny"] = "ask"


@ventri.plugin(name="tool:fs", config=FsConfig)
def fs(ctx: Any, cfg: FsConfig, registry: ToolRegistry) -> None:
    """``use: ventri_agent.tools.fs`` (config ``roots``, ``write``)."""
    from ..paths import expand

    for r in cfg.roots:
        if r.startswith("~/.ventri"):
            expand(r).mkdir(parents=True, exist_ok=True)
    roots = Roots([str(expand(r)) for r in cfg.roots])
    for t in make_tools(roots, write=cfg.write):
        registry.register(ctx, t)


plugin = fs
