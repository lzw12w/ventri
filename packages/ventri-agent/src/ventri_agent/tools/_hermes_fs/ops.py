"""Local file operations: read (line-numbered, paged), raw read, atomic write, fuzzy patch.

Portions adapted from Hermes Agent (https://github.com/NousResearch/hermes-agent),
Copyright (c) 2025 Nous Research, MIT License. Sources: ``tools/file_operations.py``
(``ShellFileOperations``: the native local read path ``_read_file_native`` /
``_assemble_read_result``, binary identification, UTF-16 rescue, not-found
suggestions, ``read_file_raw``, ``write_file``, ``_atomic_write``,
``patch_replace``, ``_no_match_result``, ``_verify_patch_persisted``) and
``tools/file_operations_lint.py`` (in-process linters, ``_check_lint_delta``).

Adaptation: Hermes shells out through a terminal backend (``sed``/``cut``/
``base64``/``mktemp``/``mv``); everything here runs in-process on the local
filesystem -- the byte-for-byte contract of Hermes's native read path is kept,
the shell transport, LSP diagnostics and external linters are dropped. Paths
passed in are already resolved and confined by the caller (``fs.Roots``).
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import stat
import tempfile
import tomllib
import unicodedata
from collections.abc import Callable

from .common import (
    MAX_LINE_LENGTH,
    UTF8_BOM,
    LintResult,
    PatchResult,
    ReadResult,
    WriteResult,
    detect_line_ending,
    has_binary_extension,
    has_bom,
    human_size,
    is_image_path,
    normalize_line_endings,
    normalize_read_pagination,
    strip_bom,
)
from .fuzzy_match import format_no_match_hint, fuzzy_find_and_replace, is_already_applied

# ------------------------------------------------------------------ binary identification
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "PNG image data"),
    (b"\xff\xd8\xff", "JPEG image data"),
    (b"GIF87a", "GIF image data"),
    (b"GIF89a", "GIF image data"),
    (b"RIFF", "RIFF container (WAV/AVI/WebP family)"),
    (b"%PDF-", "PDF document"),
    (b"PK\x03\x04", "ZIP archive (also docx/xlsx/jar/apk)"),
    (b"PK\x05\x06", "ZIP archive (empty)"),
    (b"\x1f\x8b", "gzip compressed data"),
    (b"BZh", "bzip2 compressed data"),
    (b"\xfd7zXZ\x00", "xz compressed data"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip archive"),
    (b"\x7fELF", "ELF executable"),
    (b"MZ", "Windows PE executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable (64-bit)"),
    (b"\xca\xfe\xba\xbe", "Mach-O universal binary / Java class"),
    (b"SQLite format 3\x00", "SQLite database"),
    (b"OggS", "Ogg container"),
    (b"fLaC", "FLAC audio"),
    (b"ID3", "MP3 audio (ID3 tag)"),
    (b"\x00\x00\x00", "ISO media container (MP4/MOV family)"),  # ftyp at +4
    (b"BM", "BMP image data"),
    (b"II*\x00", "TIFF image data (little-endian)"),
    (b"MM\x00*", "TIFF image data (big-endian)"),
)


def identify_binary_bytes(sample: bytes) -> str:
    """Best-effort human name for binary content from its magic bytes; never raises."""
    for prefix, name in _MAGIC_SIGNATURES:
        if sample.startswith(prefix):
            if name.startswith("ISO media") and sample[4:8] != b"ftyp":
                continue
            return name
    return "unknown binary"


def describe_binary_file(sample: bytes | None, file_size: int) -> str:
    """One-line refusal naming the type ("PNG image data, 4.1 KB")."""
    return f"Binary file ({identify_binary_bytes(sample or b'')}, {human_size(file_size)}) -- cannot display as text."


def is_likely_binary_bytes(sample: bytes) -> bool:
    """Text iff valid UTF-8 (one incomplete multibyte sequence allowed at the very
    end: the byte-boundary cut). NUL bytes or mid-stream invalid UTF-8 read as
    binary, so a display never shows U+FFFD for bytes an edit would keep."""
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
        return False
    except UnicodeDecodeError as exc:
        if exc.start >= len(sample) - 3:
            try:
                sample[: exc.start].decode("utf-8")
                return False
            except UnicodeDecodeError:
                pass
        return True


def mostly_text(sample: bytes, max_bad_ratio: float = 0.10) -> bool:
    """Ventri: an invalid-UTF-8 sample that is still mostly text (legacy 8-bit
    encodings such as Latin-1) reads as text with U+FFFD for the undecodable
    bytes instead of being refused as binary (Hermes treats any invalid UTF-8
    as binary because its writes could not preserve those bytes; Ventri's can)."""
    text = sample.decode("utf-8", "replace")
    if not text:
        return True
    bad = sum(1 for c in text if c == "\ufffd" or (ord(c) < 32 and c not in "\n\r\t\f\b\x1b"))
    return bad / len(text) <= max_bad_ratio


def add_line_numbers(content: str, start_line: int = 1) -> str:
    """Prefix each line with a compact ``<n>|`` gutter, clamping long lines. A
    trailing newline terminates the last line (no phantom ``<N+1>|`` row)."""
    content = content.removesuffix("\n")
    return "\n".join(
        f"{i}|{line if len(line) <= MAX_LINE_LENGTH else line[:MAX_LINE_LENGTH] + '... [truncated]'}"
        for i, line in enumerate(content.split("\n"), start=start_line))


def unified_diff(old_content: str, new_content: str, filename: str) -> str:
    return "".join(difflib.unified_diff(
        old_content.splitlines(keepends=True), new_content.splitlines(keepends=True),
        fromfile=f"a/{filename}", tofile=f"b/{filename}"))


# ------------------------------------------------------------------ read
def not_regular_error(path: str) -> ReadResult:
    return ReadResult(error=(f"Cannot read '{path}': not a regular file (directory, dangling symlink, "
                             "FIFO, socket, or device). Reading it could block indefinitely."))


def _assemble_read_result(read_output: str, *, offset: int, end_line: int, total_lines: int,
                          file_size: int, file_ends_with_newline: bool | None) -> ReadResult:
    """Turn a raw page into the final result: BOM strip, pagination hint, phantom
    last-line fix, empty-file and past-EOF notes (``_assemble_read_result``)."""
    if file_size > 0 and file_ends_with_newline is False:
        total_lines += 1   # the newline count misses an unterminated last line
    if offset == 1:
        read_output, _ = strip_bom(read_output)
    truncated = total_lines > end_line
    hint = None
    if truncated:
        hint = (f"Use offset={end_line + 1} to continue reading "
                f"(showing {offset}-{end_line} of {total_lines} lines)")
    if not truncated and read_output.endswith("\n") and file_ends_with_newline is False:
        read_output = read_output[:-1]
    if file_size == 0:
        return ReadResult(content="", total_lines=0, file_size=0, hint="File is empty (0 bytes).")
    if offset > total_lines > 0:
        return ReadResult(content="", total_lines=total_lines, file_size=file_size, hint=(
            f"Note: offset {offset} is beyond the end of the file ({total_lines} lines total). "
            f"Retry with offset <= {total_lines}."))
    truncated_lines = any(len(line) > MAX_LINE_LENGTH for line in read_output.split("\n"))
    return ReadResult(content=add_line_numbers(read_output, offset), total_lines=total_lines,
                      file_size=file_size, truncated=truncated, hint=hint,
                      truncated_lines=True if truncated_lines else None)


_UTF16_MAX_BYTES = 10 * 1024 * 1024
_UTF16_SAMPLE_BYTES = 512


def try_read_utf16(path: str, offset: int, limit: int, file_size: int) -> ReadResult | None:
    """UTF-16 rescue (Hermes, after MoonshotAI/kimi-code#2647): trust a BOM, else
    zero-byte parity (zeros at odd indices -> LE, even -> BE). Legacy 8-bit
    encodings are never guessed. In-process here (Hermes runs a python snippet)."""
    if has_binary_extension(path) or file_size > _UTF16_MAX_BYTES:
        return None
    try:
        with open(path, "rb") as f:
            data = f.read(_UTF16_MAX_BYTES + 1)
    except OSError:
        return None
    if len(data) > _UTF16_MAX_BYTES:
        return None
    sample = data[:_UTF16_SAMPLE_BYTES]
    enc = None
    if sample[:2] == b"\xfe\xff":
        enc = "utf-16-be"
    elif sample[:2] == b"\xff\xfe":
        enc = "utf-16-le"
    else:
        odd = sum(1 for i in range(1, len(sample), 2) if sample[i] == 0)
        even = sum(1 for i in range(0, len(sample), 2) if sample[i] == 0)
        if even == 0 and odd >= 2:
            enc = "utf-16-le"
        elif odd == 0 and even >= 2:
            enc = "utf-16-be"
    if enc is None:
        return None
    text = data.decode(enc, "replace")
    if text[:1] == "\ufeff":
        text = text[1:]
    lines = text.replace("\r\n", "\n").split("\n")
    total_lines = len(lines)
    content = "\n".join(lines[offset - 1: offset - 1 + limit])
    end_line = offset + limit - 1
    truncated = total_lines > end_line
    hint_parts = [(f"Transcoded from {enc.upper()} to UTF-8 for display. Text edits via fs.edit/fs.write "
                   "would re-encode the file as UTF-8.")]
    if truncated:
        hint_parts.append(f"Use offset={end_line + 1} to continue reading "
                          f"(showing {offset}-{end_line} of {total_lines} lines)")
    truncated_lines = any(len(line) > MAX_LINE_LENGTH for line in content.split("\n"))
    return ReadResult(content=add_line_numbers(content, offset), total_lines=total_lines, file_size=file_size,
                      truncated=truncated, hint=" ".join(hint_parts), encoding=enc,
                      truncated_lines=True if truncated_lines else None)


def _read_binary_file(path: str, offset: int, limit: int, file_size: int, sample: bytes | None) -> ReadResult:
    utf16 = try_read_utf16(path, offset, limit, file_size)
    if utf16 is not None:
        return utf16
    return ReadResult(is_binary=True, file_size=file_size, error=describe_binary_file(sample, file_size))


_CONFUSABLES = (("\u202f", " "), ("\u00a0", " "), ("\u2019", "'"), ("\u2018", "'"))


def unicode_variant_match(path: str) -> str | None:
    """On-disk spelling of a file whose name is unicode-equivalent to ``path``
    (NFC/NFD, confusable spaces/quotes); only when EXACTLY one entry matches."""
    dir_path = os.path.dirname(path) or "."
    filename = os.path.basename(path)
    if not filename:
        return None

    def canon(name: str) -> str:
        out = unicodedata.normalize("NFC", name)
        for src, dst in _CONFUSABLES:
            out = out.replace(src, dst)
        return out

    target = canon(filename)
    try:
        entries = os.listdir(dir_path)
    except OSError:
        return None
    candidates = [e for e in entries if e != filename and canon(e) == target]
    return os.path.join(dir_path, candidates[0]) if len(candidates) == 1 else None


def suggest_similar_files(path: str) -> ReadResult:
    """"File not found" listing up to 5 similar names from the same directory."""
    dir_path = os.path.dirname(path) or "."
    filename = os.path.basename(path)
    basename_no_ext = os.path.splitext(filename)[0].lower()
    ext = os.path.splitext(filename)[1].lower()
    lower_name = filename.lower()
    try:
        entries = sorted(os.listdir(dir_path))[:1000]
    except OSError:
        entries = []
    scored: list[tuple[int, str]] = []
    for f in entries:
        lf = f.lower()
        score = 0
        if lf == lower_name:
            score = 100
        elif os.path.splitext(f)[0].lower() == basename_no_ext:   # config.yml vs config.yaml
            score = 90
        elif lf.startswith(lower_name) or lower_name.startswith(lf):
            score = 70
        elif lower_name in lf:
            score = 60
        elif lf in lower_name and len(lf) > 2:
            score = 40
        elif ext and os.path.splitext(f)[1].lower() == ext:
            common = set(lower_name) & set(lf)
            if len(common) >= max(len(lower_name), len(lf)) * 0.4:
                score = 30
        if score == 0 and difflib.SequenceMatcher(None, lower_name, lf).ratio() >= 0.8:
            score = 50   # near-miss spelling (AGENT.md -> AGENTS.md)
        if score > 0:
            scored.append((score, os.path.join(dir_path, f)))
    scored.sort(key=lambda x: -x[0])
    return ReadResult(error=f"File not found: {path}", not_found=True, similar_files=[fp for _, fp in scored[:5]])


def read_file(path: str, offset: int = 1, limit: int = 2000, *,
              allow: Callable[[str], bool] | None = None) -> ReadResult:
    """Read with pagination, binary detection and line numbers (``_read_file_native``).

    ``os.stat`` is the regular-file guard (FIFOs/devices are refused before
    anything opens them); the first 1000 bytes drive the byte-layer binary
    check; one chunked pass counts lines and collects the page, each line
    clamped to ``4 * MAX_LINE_LENGTH + 1`` bytes, so neither the file nor a
    pathological line is held whole. ``allow`` vets a unicode-variant
    spelling before it is read (path confinement)."""
    offset, limit = normalize_read_pagination(offset, limit)
    try:
        st = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        if os.path.islink(path):  # dangling: an entry, not an absent path
            return not_regular_error(path)
        variant = unicode_variant_match(path)
        if variant is not None and (allow is None or allow(variant)):
            result = read_file(variant, offset, limit, allow=allow)
            note = (f"Note: '{path}' not found byte-for-byte; resolved to the unicode-equivalent file "
                    f"'{variant}' (invisible encoding difference: NFC/NFD or special space/quote characters).")
            result.hint = f"{note} {result.hint}" if result.hint else note
            return result
        return suggest_similar_files(path)
    except OSError as e:
        return ReadResult(error=f"Cannot read '{path}': {e.strerror or e}")
    if not stat.S_ISREG(st.st_mode):
        return not_regular_error(path)
    file_size = st.st_size
    clamp = 4 * MAX_LINE_LENGTH + 1
    end_line = offset + limit - 1
    page: list[bytes] = []
    total_lines = 0
    lineno = 1
    kept = bytearray()
    have_partial = False
    last_byte = b""
    clamped = False
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            sample = fh.read(1000)
            if (is_image_path(path) or has_binary_extension(path) or b"\x00" in sample
                    or (is_likely_binary_bytes(sample) and not mostly_text(sample))):
                return _read_binary_file(path, offset, limit, file_size, sample)
            fh.seek(0)
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                last_byte = chunk[-1:]
                if lineno > end_line:
                    total_lines += chunk.count(b"\n")
                    have_partial = chunk[-1:] != b"\n"
                    continue
                pos, n = 0, len(chunk)
                while pos < n:
                    nl = chunk.find(b"\n", pos)
                    in_page = offset <= lineno <= end_line
                    if nl < 0:
                        if in_page and len(kept) < clamp:
                            kept += chunk[pos:pos + (clamp - len(kept))]
                        have_partial = True
                        break
                    if in_page:
                        if len(kept) < clamp:
                            kept += chunk[pos:min(nl, pos + (clamp - len(kept)))]
                        clamped = clamped or len(kept) >= clamp
                        page.append(bytes(kept) + b"\n")
                    kept = bytearray()
                    have_partial = False
                    total_lines += 1
                    lineno += 1
                    pos = nl + 1
    except OSError as e:
        return ReadResult(error=f"Failed to read file: {e.strerror or e}")
    if have_partial and offset <= lineno <= end_line:
        page.append(bytes(kept) + b"\n")
    raw_page = b"".join(page)
    # Ventri: undecodable bytes are shown as U+FFFD but counted, so the caller can
    # warn and refuse to treat this view as the file's full content (a whole-file
    # rewrite from it would replace those bytes). Edits keep them (surrogateescape).
    undecodable = 0 if clamped else sum(
        1 for ch in raw_page.decode("utf-8", "surrogateescape") if "\udc80" <= ch <= "\udcff")
    # CRLF files display as LF (writes restore CRLF); the BOM is stripped below.
    read_output = raw_page.decode("utf-8", errors="replace").replace("\r\n", "\n")
    result = _assemble_read_result(read_output, offset=offset, end_line=end_line, total_lines=total_lines,
                                   file_size=file_size,
                                   file_ends_with_newline=(last_byte == b"\n") if file_size else None)
    result.snapshot = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, digest.digest())
    result.undecodable = undecodable
    return result


def read_exact_bytes(path: str) -> bytes:
    """The file's bytes exactly (regular files only; a FIFO is refused, never opened blocking)."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"{path}: not a regular file")
        with open(fd, "rb", closefd=False) as fh:
            return fh.read()
    finally:
        os.close(fd)


def decode_exact(data: bytes) -> str:
    """surrogateescape: ``encode('utf-8', 'surrogateescape')`` restores every byte
    UTF-8 cannot decode, so a read -> edit -> write round-trip is byte-exact."""
    return data.decode("utf-8", "surrogateescape")


# ------------------------------------------------------------------ lint (in-process only)
def _lint_json(content: str) -> tuple[bool, str]:
    try:
        json.loads(content)
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e.msg} (line {e.lineno}, column {e.colno})"
    except Exception as e:  # noqa: BLE001 - any parse failure is a lint failure
        return False, f"{type(e).__name__}: {e}"


def _lint_yaml(content: str) -> tuple[bool, str]:
    """Syntax-only parse (events), not ``safe_load``: tags like ``!Sub`` stay valid.
    ``__SKIP__`` when no YAML library is installed."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return True, "__SKIP__"
    try:
        for _event in yaml.parse(content, Loader=yaml.SafeLoader):  # pyright: ignore
            pass
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _lint_toml(content: str) -> tuple[bool, str]:
    try:
        tomllib.loads(content)
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _lint_python(content: str) -> tuple[bool, str]:
    try:
        ast.parse(content.encode("utf-8", "surrogateescape"))
        return True, ""
    except SyntaxError as e:
        loc = f" (line {e.lineno}, column {e.offset})" if e.lineno else ""
        return False, f"{type(e).__name__}: {e.msg}{loc}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


LINTERS_INPROC: dict[str, Callable[[str], tuple[bool, str]]] = {
    ".py": _lint_python, ".json": _lint_json, ".yaml": _lint_yaml, ".yml": _lint_yaml, ".toml": _lint_toml,
}
# write refuses on a parse failure for these (.py keeps the non-blocking lint report)
FAIL_CLOSED_INPROC_EXTS = frozenset({".json", ".yaml", ".yml", ".toml"})


def check_lint(path: str, content: str) -> LintResult:
    ext = os.path.splitext(path)[1].lower()
    linter = LINTERS_INPROC.get(ext)
    if linter is None:
        return LintResult(skipped=True, message=f"No linter for {ext} files")
    ok, err = linter(content)
    if err == "__SKIP__":
        return LintResult(skipped=True, message=f"No linter available for {ext} (missing dependency)")
    return LintResult(success=ok, output="" if ok else err)


def check_lint_delta(path: str, pre_content: str | None, post_content: str) -> LintResult:
    """Post-write lint; when it fails and the pre-edit content is known, report only
    errors this edit introduced."""
    post = check_lint(path, post_content)
    if post.success or post.skipped or pre_content is None:
        return post
    pre = check_lint(path, pre_content)
    if pre.success or pre.skipped or not pre.output:
        return post
    pre_lines = {ln.strip() for ln in pre.output.splitlines() if ln.strip()}
    post_lines = [ln for ln in post.output.splitlines() if ln.strip() and ln.strip() not in pre_lines]
    if not post_lines:
        return LintResult(success=False, output=post.output, message=(
            "Pre-existing lint errors -- this edit didn't introduce new ones but the file is still broken."))
    return LintResult(success=False, output=(
        "New lint errors introduced by this edit (pre-existing errors filtered out):\n" + "\n".join(post_lines)))


# ------------------------------------------------------------------ write
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udc7f\udd00-\udfff]")


def _reject_unencodable(path: str, content: str) -> WriteResult | None:
    """Refuse a lone surrogate outside the surrogateescape range (U+DC80-U+DCFF
    round-trips; anything else cannot be encoded)."""
    m = _LONE_SURROGATE_RE.search(content)
    if m:
        return WriteResult(error=(f"Refusing to write '{path}': content contains a lone surrogate character "
                                  f"({m.group(0)!r}) that cannot be encoded as UTF-8. The file was NOT "
                                  "created or modified."))
    return None


def _fail_closed_syntax_error(path: str, ext: str, content: str) -> WriteResult | None:
    linter = LINTERS_INPROC.get(ext) if ext in FAIL_CLOSED_INPROC_EXTS else None
    if linter is None:
        return None
    ok, err = linter(strip_bom(content)[0])
    if ok or err == "__SKIP__":
        return None
    return WriteResult(error=(f"Refusing to write '{path}': candidate content fails {ext} syntax validation "
                              f"({err}). The file was NOT created or modified. Fix the content and retry."))


def _json_nonstandard_constant(text: str) -> str | None:
    if "NaN" not in text and "Infinity" not in text:
        return None
    found: list[str] = []

    def note_constant(value: str) -> float:
        found.append(value)
        return float("nan")

    try:
        json.loads(strip_bom(text)[0], parse_constant=note_constant)
    except (ValueError, RecursionError):
        return None
    return found[0] if found else None


def _refuse_introduced_json_constant(path: str, content: str, pre_content: str | None) -> WriteResult | None:
    constant = _json_nonstandard_constant(content)
    if constant is None:
        return None
    if pre_content is not None and _json_nonstandard_constant(pre_content) is not None:
        return None
    return WriteResult(error=(f"Refusing to write '{path}': candidate content uses {constant}, which is not "
                              "valid JSON. The file was NOT created or modified. Use null or a string instead "
                              "and retry."))


def atomic_write(path: str, data: bytes) -> None:
    """Temp file in the SAME directory -> fsync -> ``os.replace`` (same-FS rename is
    atomic). A symlinked target is resolved first (replacing the link would orphan
    the target). Existing target: its mode is copied; new target: umask-default
    perms instead of mkstemp's 0600. The temp file is removed on every failure."""
    target = os.path.realpath(path) if os.path.islink(path) else path
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    try:
        mode: int | None = stat.S_IMODE(os.stat(target).st_mode)
    except OSError:
        mode = None
    fd, tmp = tempfile.mkstemp(prefix=".ventri-tmp.", dir=parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is None:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _probe_write_target(path: str, pre_content: str | None) -> tuple[bool, str | None, str | None]:
    """(has_bom, pre_content, original_line_ending) from disk; the BOM always comes from disk."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return False, pre_content, detect_line_ending(pre_content) if pre_content else None
    bom = head.startswith(UTF8_BOM.encode("utf-8"))
    if pre_content is None:
        try:
            pre_content = decode_exact(read_exact_bytes(path))
        except OSError:
            pre_content = None
    ending = detect_line_ending(pre_content) if pre_content else detect_line_ending(head.decode("utf-8", "replace"))
    return bom, pre_content, ending


def write_file(path: str, content: str, pre_content: str | None = None) -> WriteResult:
    """Write atomically, creating parent directories. Order (Hermes ``write_file``):
    lone-surrogate refusal -> fail-closed syntax gate (JSON/YAML/TOML) -> on-disk
    probe (pre-content, CRLF, BOM) -> JSON NaN/Infinity refusal when introduced ->
    CRLF/BOM preservation -> atomic write -> sha256 verification -> lint delta."""
    refused = _reject_unencodable(path, content)
    if refused is not None:
        return refused
    ext = os.path.splitext(path)[1].lower()
    refused = _fail_closed_syntax_error(path, ext, content)
    if refused is not None:
        return refused
    bom, pre_content, original_ending = _probe_write_target(path, pre_content)
    if ext == ".json":
        refused = _refuse_introduced_json_constant(path, content, pre_content)
        if refused is not None:
            return refused
    # fs.read strips the BOM and models send bare-LF text: keep CRLF files CRLF
    # and restore the BOM (prepend only when absent).
    if original_ending == "\r\n":
        content = normalize_line_endings(content, "\r\n")
    if bom and not has_bom(content):
        content = UTF8_BOM + content
    data = content.encode("utf-8", "surrogateescape")
    parent = os.path.dirname(path)
    dirs_created = bool(parent) and not os.path.isdir(parent)
    try:
        atomic_write(path, data)
    except OSError as e:
        return WriteResult(error=f"Failed to write file: {e.strerror or e}")
    written_sha = hashlib.sha256(data).hexdigest()
    verified: bool | None = None
    try:
        verified = hashlib.sha256(read_exact_bytes(path)).hexdigest() == written_sha
    except OSError:
        verified = None
    if verified is False:
        return WriteResult(error=(f"Post-write verification failed for {path}: on-disk content hash differs "
                                  "from the intended write. Re-read the file and retry."))
    lint = check_lint_delta(path, pre_content=pre_content, post_content=content)
    return WriteResult(bytes_written=len(data), dirs_created=dirs_created, verified=verified,
                       content_sha256=written_sha, lint=lint.to_dict() if ext in LINTERS_INPROC else None)


# ------------------------------------------------------------------ patch (replace mode)
def _no_match_result(path: str, content: str, old_string: str, new_string: str,
                     match_count: int, error: str | None) -> PatchResult:
    """Already-applied detection first (a re-sent edit is the most common failure),
    else the error plus a best-effort "Did you mean?" snippet."""
    if is_already_applied(content, old_string, new_string):
        return PatchResult(success=True, no_change=True, note=(
            f"File already contains the target text -- the edit appears to be already applied to {path}. "
            "No write performed; do not re-send this edit."))
    err = error or f"Could not find match for old_string in {path}"
    hint = ""
    try:
        hint = format_no_match_hint(err, match_count, old_string, content)
    except Exception:  # noqa: BLE001, S110 - hints are best-effort
        pass
    if err.startswith("Found ") and "Matches:\n" in err:
        head, _, rows = err.partition("Matches:\n")
        return PatchResult(error=head.strip(), error_detail="Matches:\n" + rows)
    return PatchResult(error=err, error_detail=hint.strip() or None)


def _verify_patch_persisted(path: str, new_content: str) -> PatchResult | None:
    try:
        data = read_exact_bytes(path)
    except OSError:
        return PatchResult(error=f"Post-write verification failed: could not re-read {path}")
    on_disk = strip_bom(decode_exact(data))[0].replace("\r\n", "\n").replace("\r", "\n")
    intended = new_content.replace("\r\n", "\n").replace("\r", "\n")
    if on_disk != intended:
        return PatchResult(error=(f"Post-write verification failed for {path}: on-disk content differs from "
                                  f"intended write (wrote {len(intended)} chars, read back {len(on_disk)} chars "
                                  "after normalizing line endings). Re-read the file and try again."))
    return None


def apply_replacements(content: str, edits: list[tuple[str, str, bool]]) -> tuple[str, int, list[str], PatchResult | None]:
    """Apply ``(old, new, replace_all)`` edits in order to ``content`` (all or nothing).
    Returns (new_content, total replacements, strategies, failure)."""
    total = 0
    strategies: list[str] = []
    current = content
    for i, (old, new, replace_all) in enumerate(edits):
        new_content, count, strategy, error = fuzzy_find_and_replace(current, old, new, replace_all)
        if error or count == 0:
            fail = _no_match_result("the file", current, old, new, count, error)
            if fail.success:          # already applied: a no-op step
                strategies.append("already_applied")
                continue
            if len(edits) > 1 and fail.error:
                fail.error = f"edit #{i + 1} of {len(edits)}: {fail.error} (no edit was applied)"
            return content, 0, strategies, fail
        current = new_content
        total += count
        strategies.append(strategy or "")
    return current, total, strategies, None


def patch_replace(path: str, edits: list[tuple[str, str, bool]], *, display: str | None = None) -> PatchResult:
    """Fuzzy find-and-replace on ``path`` (``patch_replace``): exact bytes are read
    and decoded with surrogateescape, matching runs on BOM-stripped text, the
    substituted text takes the file's line ending, the write is atomic and
    verified, and the result carries a unified diff and the lint delta."""
    shown = display or path
    try:
        data = read_exact_bytes(path)
    except FileNotFoundError:
        return PatchResult(error=f"File not found: {shown}")
    except OSError as e:
        return PatchResult(error=f"Failed to read file: {shown}: {e.strerror or e}")
    read_sha = hashlib.sha256(data).hexdigest()
    raw_content = decode_exact(data)
    content, _ = strip_bom(raw_content)
    new_content, count, strategies, failure = apply_replacements(content, edits)
    if failure is not None:
        if failure.success:
            failure.note = (failure.note or "").replace("the file", shown)
        else:
            failure.error = (failure.error or "").replace("in the file", f"in {shown}")
        failure.read_sha256 = read_sha
        return failure
    if count == 0:   # every edit was already applied
        return PatchResult(success=True, no_change=True, read_sha256=read_sha, note=(
            f"File already contains the target text -- the edit appears to be already applied to {shown}. "
            "No write performed; do not re-send this edit."))
    if new_content == content:   # Ventri: a (fuzzy) match whose text already equals new_string
        return PatchResult(success=True, no_change=True, read_sha256=read_sha, note=(
            f"The matched text in {shown} already equals new_string -- the edit appears to be already "
            "applied. No write performed; do not re-send this edit."))
    file_ending = detect_line_ending(content)
    if file_ending:
        new_content = normalize_line_endings(new_content, file_ending)
    written = write_file(path, new_content, pre_content=raw_content)
    if written.error:
        return PatchResult(error=f"Failed to write changes: {written.error}", read_sha256=read_sha)
    verify_error = _verify_patch_persisted(path, new_content)
    if verify_error is not None:
        return verify_error
    lint = check_lint_delta(path, pre_content=content, post_content=new_content)
    ext = os.path.splitext(path)[1].lower()
    return PatchResult(success=True, diff=unified_diff(content, new_content, shown),
                       lint=lint.to_dict() if ext in LINTERS_INPROC else None,
                       strategy=",".join(s for s in strategies if s), replacements=count,
                       read_sha256=read_sha, written_sha256=written.content_sha256)
