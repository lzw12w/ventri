"""Content / file-name search: ripgrep with a pure-Python fallback.

Portions adapted from Hermes Agent (https://github.com/NousResearch/hermes-agent),
Copyright (c) 2025 Nous Research, MIT License. Source: ``tools/file_operations_search.py``
(rg argument construction, bounded native rg runner, output parsing with the
diagnostics split and context-line parser, zero-match steering probes,
newline/multiline handling, files search with directory supplement, macOS
TCC-protected folder pruning).

Adaptation: rg runs as an argv list (no shell; ``--no-config`` so a user's
ripgreprc cannot change the output format) and never follows symlinks, so a
search cannot leave the (already confined) search root. Hermes's
``grep``/``find`` fallbacks are replaced by an in-process walker with the same
contract (hidden directories pruned, binary and huge files skipped, explicit
truncation reasons), which also keeps the tools working where rg is absent.
"""
from __future__ import annotations

import bisect
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .common import SearchMatch, SearchResult, normalize_search_pagination

USE_RG = True                       # tests flip this to exercise the Python fallback
RG_TIMEOUT = 60
FALLBACK_MAX_FILES = 20_000         # files scanned by the Python walker before it stops (reported)
FALLBACK_MAX_FILE_BYTES = 10_000_000
# The Python walker cannot read .gitignore; it prunes these dependency/cache trees
# (a conservative subset of Hermes's SEARCH_PRUNE_DIR_NAMES) in addition to every
# hidden directory, which rg skips by default.
FALLBACK_PRUNE_DIRS = frozenset({"node_modules", "__pycache__", "site-packages", "dist-packages", "venv"})

_MACOS_TCC_PROTECTED_HOME_DIRS = ("Desktop", "Documents", "Downloads", "Library", "Movies", "Music", "Pictures")


def rg_executable() -> str | None:
    return shutil.which("rg") if USE_RG else None


def macos_protected_exclusions(path: str, *, home: str | None = None, platform: str | None = None) -> list[str]:
    """Protected home dirs (relative to ``path``) below a broad macOS search root:
    only an ANCESTOR search ($HOME, /Users) gets exclusions, so recursive tools never
    trigger unattended privacy prompts; a search rooted inside one stays allowed."""
    if (platform or sys.platform) != "darwin":
        return []
    root = Path(os.path.normpath(path))
    home_path = Path(os.path.normpath(str(Path(home or Path.home()).expanduser())))
    out: list[str] = []
    for dirname in _MACOS_TCC_PROTECTED_HOME_DIRS:
        try:
            relative = (home_path / dirname).relative_to(root)
        except ValueError:
            continue
        if relative.parts:
            out.append(relative.as_posix())
    return out


def macos_warning(skipped: list[str]) -> str:
    return ("Skipped macOS protected folders during broad search to avoid an unattended privacy prompt: "
            f"{', '.join(os.path.basename(s) for s in skipped)}. Search a protected folder directly when "
            "access is intentional.")


# ------------------------------------------------------------------ rg output parsing
_SEARCH_TIMEOUT_MARKER_RE = re.compile(r"\n?\[Command timed out after \d+s\]\s*$")
_SEARCH_OUTPUT_RE = re.compile(r"^([A-Za-z]:)?[^\s:][^\n]*?[:\-]\d|^[^\s:][^\s]*$")
_MATCH_LINE_RE = re.compile(r"^([A-Za-z]:)?(.*?):(\d+):(.*)$")
_REGEX_NEWLINE_ESCAPE_RE = re.compile(r"(?<!\\)(?:\\\\)*\\n")
_OUTPUT_MODE_FLAGS = {"files_only": "-l", "count": "-c"}


@dataclass
class _Exec:
    stdout: str
    exit_code: int


def _stdout_and_limit(result: _Exec) -> tuple[str, str | None]:
    if result.exit_code == 124:
        return _SEARCH_TIMEOUT_MARKER_RE.sub("", result.stdout), "search_timeout"
    return result.stdout, None


def split_tool_diagnostics(output: str) -> tuple[str, str]:
    """Separate rg diagnostics from match output -> (diagnostics, payload)."""
    diagnostics: list[str] = []
    payload: list[str] = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        if line.lstrip().startswith(("rg: ", "grep: ")):
            diagnostics.append(line)
        elif line == "--" or _SEARCH_OUTPUT_RE.match(line):
            payload.append(line)
        else:
            diagnostics.append(line)
    return "\n".join(diagnostics), "\n".join(payload)


def parse_context_line(line: str) -> tuple[str, int, str] | None:
    """``path-line-content`` using the RIGHTMOST numeric separator (filenames may
    contain ``-<digits>-``)."""
    if not line or line == "--":
        return None
    match = None
    for candidate in re.finditer(r"-(\d+)-", line):
        match = candidate
    if match is None or match.start() == 0:
        return None
    return line[:match.start()], int(match.group(1)), line[match.end():]


def pattern_has_regex_newline(pattern: str) -> bool:
    """A literal newline, or ``\\n`` with an odd number of backslashes."""
    return "\n" in pattern or bool(_REGEX_NEWLINE_ESCAPE_RE.search(pattern))


def parse_search_output(result: _Exec, output_mode: str, limit: int, offset: int, context: int,
                        warning: str | None = None) -> SearchResult:
    """rg exit codes: 0 matches, 1 none, 2 error -- but 2 also on PARTIAL errors
    (one unreadable file), so an error surfaces only with no usable payload."""
    stdout, limit_reason = _stdout_and_limit(result)
    diagnostics, payload = split_tool_diagnostics(stdout)
    if result.exit_code == 2 and not payload.strip():
        error_msg = diagnostics.strip() or result.stdout.strip() or "Search error"
        return SearchResult(error=f"Search failed: {error_msg}", total_count=0)
    lines = [ln for ln in payload.strip().split("\n") if ln]
    if output_mode == "files_only":
        return SearchResult(files=lines[offset:offset + limit], total_count=len(lines),
                            truncated=len(lines) > offset + limit or bool(limit_reason),
                            limit_reason=limit_reason, warning=warning)
    if output_mode == "count":
        counts: dict[str, int] = {}
        for line in lines:
            if ":" in line:
                path, n = line.rsplit(":", 1)
                try:
                    counts[path] = int(n)
                except ValueError:
                    pass
        return SearchResult(counts=counts, total_count=sum(counts.values()),
                            truncated=bool(limit_reason), limit_reason=limit_reason, warning=warning)
    matches: list[SearchMatch] = []
    for line in lines:
        if line == "--":
            continue
        m = _MATCH_LINE_RE.match(line)
        if m:
            matches.append(SearchMatch(path=(m.group(1) or "") + m.group(2), line_number=int(m.group(3)),
                                       content=m.group(4)[:500]))
            continue
        if context > 0:
            parsed = parse_context_line(line)
            if parsed:
                matches.append(SearchMatch(path=parsed[0], line_number=parsed[1], content=parsed[2][:500]))
    total = len(matches)
    return SearchResult(matches=matches[offset:offset + limit], total_count=total,
                        truncated=total > offset + limit or bool(limit_reason),
                        limit_reason=limit_reason, warning=warning)


# ------------------------------------------------------------------ rg runner
def run_rg(argv: list[str], fetch_limit: int, timeout: float = RG_TIMEOUT, *, cwd: str | None = None,
           merge_stderr: bool = False) -> _Exec:
    """Run rg and stop reading after ``fetch_limit`` lines (``| head -n`` without a
    shell). Exit codes follow rg (0/1/2), 124 on timeout with partial output; at the
    bound rg is killed like ``head`` closing the pipe would."""
    try:
        proc = subprocess.Popen(  # argv list, no shell
            argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        return _Exec(stdout=f"rg: {exc}", exit_code=2)
    lines: list[bytes] = []
    bounded = threading.Event()
    stream = proc.stdout
    assert stream is not None

    def drain() -> None:
        for raw in stream:
            lines.append(raw)
            if len(lines) >= fetch_limit:
                bounded.set()
                break

    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    deadline = time.monotonic() + timeout
    timed_out = False
    while True:
        drainer.join(0.05)
        if not drainer.is_alive() or bounded.is_set():
            break
        if time.monotonic() > deadline:
            timed_out = True
            break
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    drainer.join()
    stream.close()
    stdout = b"".join(lines).decode("utf-8", errors="replace")
    if timed_out:
        return _Exec(stdout=stdout + f"\n[Command timed out after {int(timeout)}s]", exit_code=124)
    return _Exec(stdout=stdout, exit_code=0 if bounded.is_set() else proc.returncode)


# ------------------------------------------------------------------ options
@dataclass
class SearchSpec:
    pattern: str
    base: str                       # absolute, resolved, confined search root (dir or file)
    target: str = "content"         # content | files
    file_glob: str | None = None
    limit: int = 50
    offset: int = 0
    output_mode: str = "content"    # content | files_only | count
    context: int = 0
    order: str = "discovery"        # discovery | modified (files target)
    ignore_case: bool = False
    fixed_strings: bool = False


def _rg_common_globs(spec: SearchSpec) -> list[str]:
    out: list[str] = []
    for item in macos_protected_exclusions(spec.base):
        out += ["--glob", f"!{item}/**"]
    return out


def _search_content_rg(rg: str, spec: SearchSpec) -> SearchResult:
    argv = [rg, "--no-config", "--line-number", "--no-heading", "--with-filename"]
    if spec.output_mode not in ("files_only", "count"):
        argv += ["--max-columns", "2000", "--max-columns-preview"]   # giant single lines stay bounded
    multiline = pattern_has_regex_newline(spec.pattern) and not spec.fixed_strings
    if multiline:
        argv.append("--multiline")
    if spec.context > 0:
        argv += ["-C", str(spec.context)]
    argv += _rg_common_globs(spec)
    if spec.file_glob:
        argv += ["--glob", spec.file_glob]
    if spec.ignore_case:
        argv.append("-i")
    if spec.fixed_strings:
        argv.append("-F")
    if spec.output_mode in _OUTPUT_MODE_FLAGS:
        argv.append(_OUTPUT_MODE_FLAGS[spec.output_mode])
    argv += ["-e", spec.pattern, "--", spec.base]
    note = ("Pattern contains \\n -- multiline mode (-U) was enabled automatically so the regex can match "
            "across line boundaries.") if multiline else None
    fetch_limit = spec.limit + spec.offset + 1 + (200 if spec.context > 0 else 0)
    result = run_rg(argv, fetch_limit, merge_stderr=True)
    parsed = parse_search_output(result, spec.output_mode, spec.limit, spec.offset, spec.context, warning=note)
    parsed.engine = "rg"
    return parsed


# (rg flags, message) zero-match probes, in order; the fixed-string probe only for regex metachars.
_ZERO_MATCH_PROBES = (
    ("-i", ("0 exact matches, but {total} case-insensitive match(es) in {n} file(s): {paths} -- the "
            "pattern's casing may be wrong.")),
    ("--hidden --no-ignore", ("0 matches in visible files, but {total} match(es) in {n} hidden or "
                              "gitignored file(s): {paths} -- these are excluded by default.")),
    ("-F", ("0 regex matches, but {total} literal match(es) in {n} file(s): {paths} -- the pattern contains "
            "regex metacharacters that likely need escaping (or pass a simpler substring).")),
)
_PROBE_PRUNE = sorted({"node_modules", ".git", ".venv", "venv", "__pycache__", "site-packages", ".cache"})
# the Python walker's hidden probe prunes the same dependency/cache trees
_PROBE_PRUNE_SET = frozenset(_PROBE_PRUNE)


def _format_probe(template: str, per_file: dict[str, int], display: Callable[[str], str]) -> str | None:
    total = sum(per_file.values())
    if total <= 0:
        return None
    names = [display(p) for p in per_file]
    extra = len(names) - 5
    paths = ", ".join(names[:5]) + (f" (+{extra} more)" if extra > 0 else "")
    return template.format(total=total, n=len(per_file), paths=paths)


def _zero_match_probe_rg(rg: str, spec: SearchSpec, display: Callable[[str], str]) -> str | None:
    has_meta = bool(re.search(r"[.\[\](){}?*+^$\\|]", spec.pattern))
    for flags, template in _ZERO_MATCH_PROBES:
        if flags == "-F" and (not has_meta or spec.fixed_strings):
            continue
        if flags == "-i" and spec.ignore_case:
            continue
        argv = [rg, "--no-config", *flags.split(), "--count-matches"]
        if spec.file_glob:
            argv += ["--glob", spec.file_glob]
        if flags.startswith("--hidden"):
            for d in _PROBE_PRUNE:
                argv += ["--glob", f"!{d}/**", "--glob", f"!**/{d}/**"]
        if spec.ignore_case and flags != "-i":
            argv.append("-i")
        argv += ["-e", spec.pattern, "--", spec.base]
        probe = run_rg(argv, 50, timeout=30)
        per_file: dict[str, int] = {}
        for line in probe.stdout.strip().splitlines():
            p, _sep, n = line.rpartition(":")
            if n.isdigit() and p:
                per_file[p] = per_file.get(p, 0) + int(n)
        hint = _format_probe(template, per_file, display)
        if hint:
            return hint
    return None


def _glob_for_files(pattern: str) -> str:
    """Wrap bare names so the glob matches at any depth (Hermes ``_search_files_rg``)."""
    return f"*{pattern}" if ("/" not in pattern and not pattern.startswith("*")) else pattern


def _search_files_rg(rg: str, spec: SearchSpec) -> SearchResult:
    glob_pattern = _glob_for_files(spec.pattern)
    fetch_limit = spec.limit + spec.offset + 1
    argv = [rg, "--no-config", "--files", "-g", glob_pattern, *_rg_common_globs(spec), "--", spec.base]
    if spec.order == "modified":
        fetch_limit = 100_000          # exact global order needs the whole listing
    result = run_rg(argv, fetch_limit)
    stdout, limit_reason = _stdout_and_limit(result)
    files = [f for f in stdout.splitlines() if f]
    if result.exit_code not in {0, 1, 124}:
        return SearchResult(error="File search failed while running ripgrep.")
    if limit_reason != "search_timeout" and os.path.isdir(spec.base):
        # rg --files lists files only: supplement matching directories so empty
        # directories stay discoverable (Hermes #54347).
        dirs = _walk_dirs(spec.base, glob_pattern, fetch_limit)
        seen = set(files)
        files += [d for d in dirs if d not in seen]
    if spec.order == "modified":
        files = _sort_by_mtime(files)
    return SearchResult(files=files[spec.offset:spec.offset + spec.limit], total_count=len(files),
                        truncated=len(files) > spec.offset + spec.limit or bool(limit_reason),
                        limit_reason=limit_reason, engine="rg")


def _sort_by_mtime(paths: list[str]) -> list[str]:
    def mtime(p: str) -> float:
        try:
            return os.stat(p).st_mtime
        except OSError:
            return 0.0
    return sorted(paths, key=mtime, reverse=True)


# ------------------------------------------------------------------ Python fallback
def _is_hidden(name: str) -> bool:
    return name.startswith(".") and name not in (".", "..")


def _walk(base: str, *, hidden: bool = False, prune_dirs: frozenset[str] = FALLBACK_PRUNE_DIRS,
          ) -> tuple[list[tuple[str, list[str], list[str]]], str | None]:
    """Bounded topdown walk below ``base`` (symlinks are never followed). Prunes hidden
    descendants (unless ``hidden``), dependency trees and macOS protected folders."""
    excluded = {os.path.normpath(os.path.join(base, rel)) for rel in macos_protected_exclusions(base)}
    out: list[tuple[str, list[str], list[str]]] = []
    seen_files = 0
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames
                             if (hidden or not _is_hidden(d)) and d not in prune_dirs
                             and os.path.normpath(os.path.join(dirpath, d)) not in excluded
                             and not os.path.islink(os.path.join(dirpath, d)))
        files = sorted(f for f in filenames if (hidden or not _is_hidden(f))
                       and not os.path.islink(os.path.join(dirpath, f)))
        out.append((dirpath, dirnames, files))
        seen_files += len(files)
        if seen_files >= FALLBACK_MAX_FILES:
            return out, f"file_limit ({FALLBACK_MAX_FILES} files scanned)"
    return out, None


def _walk_dirs(base: str, glob_pattern: str, limit: int) -> list[str]:
    out: list[str] = []
    walked, _ = _walk(base, prune_dirs=FALLBACK_PRUNE_DIRS | {".git"})
    for dirpath, dirnames, _files in walked:
        for d in dirnames:
            full = os.path.join(dirpath, d)
            if _glob_match(glob_pattern, os.path.relpath(full, base), d):
                out.append(full)
                if len(out) >= limit:
                    return out
    return out


def _glob_match(glob: str, rel: str, name: str) -> bool:
    """rg ``-g`` semantics, approximately: a glob without '/' matches the basename
    at any depth; with '/' it matches the path relative to the search root."""
    if "/" in glob:
        g = glob.removeprefix("**/")
        return fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(rel, glob)
    return fnmatch.fnmatch(name, glob)


def _iter_files(spec: SearchSpec, *, hidden: bool = False) -> tuple[list[str], str | None]:
    if os.path.isfile(spec.base):
        return [spec.base], None
    walked, reason = _walk(spec.base, hidden=hidden,
                           prune_dirs=FALLBACK_PRUNE_DIRS | _PROBE_PRUNE_SET if hidden else FALLBACK_PRUNE_DIRS)
    files: list[str] = []
    for dirpath, _dirs, names in walked:
        for f in names:
            full = os.path.join(dirpath, f)
            if spec.file_glob and not _glob_match(spec.file_glob, os.path.relpath(full, spec.base), f):
                continue
            files.append(full)
    return files, reason


def _compile(spec: SearchSpec) -> re.Pattern[str]:
    flags = re.MULTILINE | (re.IGNORECASE if spec.ignore_case else 0)
    return re.compile(re.escape(spec.pattern) if spec.fixed_strings else spec.pattern, flags)


def _read_text_for_search(path: str) -> str | None:
    try:
        if os.path.getsize(path) > FALLBACK_MAX_FILE_BYTES:
            return None
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if b"\x00" in data[:8192]:
        return None   # binary, as rg skips it
    return data.decode("utf-8", "replace")


def _search_content_python(spec: SearchSpec, *, hidden: bool = False, count_only: bool = False
                           ) -> SearchResult:
    try:
        rx = _compile(spec)
    except re.error as e:
        return SearchResult(error=f"Search failed: invalid regex: {e}")
    files, reason = _iter_files(spec, hidden=hidden)
    multiline = pattern_has_regex_newline(spec.pattern) and not spec.fixed_strings
    fetch_limit = spec.limit + spec.offset + 1 + (200 if spec.context > 0 else 0)
    matches: list[SearchMatch] = []
    file_hits: list[str] = []
    counts: dict[str, int] = {}
    stop = False
    for path in files:
        text = _read_text_for_search(path)
        if text is None:
            continue
        lines = text.split("\n")
        hit_lines: list[int] = []
        if multiline:
            starts = [0]
            for ln in lines[:-1]:
                starts.append(starts[-1] + len(ln) + 1)
            hit_lines = sorted({bisect.bisect_right(starts, m.start()) for m in rx.finditer(text)})
        else:
            hit_lines = [i + 1 for i, ln in enumerate(lines) if rx.search(ln)]
        if not hit_lines:
            continue
        if count_only or spec.output_mode == "count":
            counts[path] = len(hit_lines)
            continue
        if spec.output_mode == "files_only":
            file_hits.append(path)
            if len(file_hits) >= fetch_limit:
                stop = True
        else:
            if spec.context > 0:
                want: set[int] = set()
                for n in hit_lines:
                    want.update(range(max(1, n - spec.context), min(len(lines), n + spec.context) + 1))
                rows = sorted(want)
            else:
                rows = hit_lines
            for n in rows:
                matches.append(SearchMatch(path=path, line_number=n, content=lines[n - 1].rstrip("\r")[:500]))
                if len(matches) >= fetch_limit:
                    stop = True
                    break
        if stop:
            break
    warning = ("Pattern contains \\n -- matched across line boundaries (multiline)." if multiline else None)
    if count_only or spec.output_mode == "count":
        return SearchResult(counts=counts, total_count=sum(counts.values()), truncated=bool(reason),
                            limit_reason=reason, warning=warning, engine="python")
    if spec.output_mode == "files_only":
        return SearchResult(files=file_hits[spec.offset:spec.offset + spec.limit], total_count=len(file_hits),
                            truncated=len(file_hits) > spec.offset + spec.limit or stop or bool(reason),
                            limit_reason=reason, warning=warning, engine="python")
    total = len(matches)
    return SearchResult(matches=matches[spec.offset:spec.offset + spec.limit], total_count=total,
                        truncated=total > spec.offset + spec.limit or stop or bool(reason),
                        limit_reason=reason, warning=warning, engine="python")


def _zero_match_probe_python(spec: SearchSpec, display: Callable[[str], str]) -> str | None:
    has_meta = bool(re.search(r"[.\[\](){}?*+^$\\|]", spec.pattern))
    probes: list[tuple[str, SearchSpec, bool]] = []
    if not spec.ignore_case:
        probes.append((_ZERO_MATCH_PROBES[0][1], SearchSpec(**{**spec.__dict__, "ignore_case": True}), False))
    probes.append((_ZERO_MATCH_PROBES[1][1].replace("hidden or gitignored", "hidden"),
                   SearchSpec(**spec.__dict__), True))
    if has_meta and not spec.fixed_strings:
        probes.append((_ZERO_MATCH_PROBES[2][1], SearchSpec(**{**spec.__dict__, "fixed_strings": True}), False))
    for template, probe_spec, hidden in probes:
        res = _search_content_python(probe_spec, hidden=hidden, count_only=True)
        hint = _format_probe(template, res.counts, display)
        if hint:
            return hint
    return None


def _search_files_python(spec: SearchSpec) -> SearchResult:
    glob_pattern = _glob_for_files(spec.pattern)
    if os.path.isfile(spec.base):
        name = os.path.basename(spec.base)
        hits = [spec.base] if fnmatch.fnmatch(name, glob_pattern) else []
        return SearchResult(files=hits, total_count=len(hits), engine="python")
    walked, reason = _walk(spec.base)
    hits: list[str] = []
    for dirpath, dirnames, files in walked:
        for name in [*dirnames, *files]:
            full = os.path.join(dirpath, name)
            if _glob_match(glob_pattern, os.path.relpath(full, spec.base), name):
                hits.append(full)
    if spec.order == "modified":
        hits = _sort_by_mtime(hits)
    return SearchResult(files=hits[spec.offset:spec.offset + spec.limit], total_count=len(hits),
                        truncated=len(hits) > spec.offset + spec.limit or bool(reason), limit_reason=reason,
                        engine="python")


# ------------------------------------------------------------------ entry point
def search(spec: SearchSpec, display: Callable[[str], str] = lambda p: p) -> SearchResult:
    """Search content (regex, ``target="content"``) or file names (glob,
    ``target="files"``). Attaches zero-match steering hints and the macOS
    protected-folder warning."""
    spec.offset, spec.limit = normalize_search_pagination(spec.offset, spec.limit)
    if spec.target == "files" and spec.order not in {"discovery", "modified"}:
        return SearchResult(error=f"Invalid file search order {spec.order!r}; expected 'discovery' or 'modified'.")
    rg = rg_executable()
    if spec.target == "files":
        result = _search_files_rg(rg, spec) if rg else _search_files_python(spec)
    else:
        result = _search_content_rg(rg, spec) if rg else _search_content_python(spec)
        if (not result.error and result.total_count == 0 and not result.matches and not result.files
                and not result.counts):
            try:
                hint = _zero_match_probe_rg(rg, spec, display) if rg else _zero_match_probe_python(spec, display)
            except Exception:  # noqa: BLE001 - hints are best-effort
                hint = None
            if hint:
                result.warning = hint if not result.warning else f"{result.warning} {hint}"
    excluded = macos_protected_exclusions(spec.base)
    if excluded and not result.error:
        w = macos_warning(excluded)
        result.warning = w if not result.warning else f"{result.warning} {w}"
    return result
