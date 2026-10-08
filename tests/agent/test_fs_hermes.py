"""File tools ported from Hermes Agent: behaviour of fs.* / notes.* and of the
``_hermes_fs`` core.

Portions adapted from Hermes Agent (https://github.com/NousResearch/hermes-agent),
Copyright (c) 2025 Nous Research, MIT License. Test cases adapted from
``tests/tools/``: test_patch_already_applied.py, test_patch_multimatch_locations.py,
test_read_binary_type_disclosure.py, test_read_file_utf8_binary_regression.py,
test_read_past_eof_note.py, test_read_special_file_guard.py,
test_file_write_surrogate_roundtrip.py, test_file_staleness.py,
test_known_file_write_baseline.py, test_search_auto_multiline.py,
test_search_hidden_dirs.py, test_search_zero_match_and_multipath.py. They call
Ventri's tools (or the ported core) instead of Hermes's ShellFileOperations /
file_tools entry points; the rest of this file is Ventri regression tests.
"""
from __future__ import annotations

import os
import random
import shutil
import socket
import stat
import sys
import time
import uuid
from pathlib import Path

import anyio
import pytest

from ventri_agent.tokens import estimate_tokens
from ventri_agent.tools import fs, notes
from ventri_agent.tools._hermes_fs import ops
from ventri_agent.tools._hermes_fs import search as hsearch
from ventri_agent.tools._hermes_fs.fuzzy_match import (
    _format_match_locations,  # pyright: ignore[reportPrivateUsage]
    fuzzy_find_and_replace,
    is_already_applied,
)
from ventri_agent.tools._hermes_fs.state import FileState, get_registry
from ventri_agent.tools.registry import ToolContext, ToolError, call_handler

pytestmark = pytest.mark.anyio

HAS_RG = shutil.which("rg") is not None


# ----------------------------------------------------------------------- helpers
class FS:
    """fs.* tools on ``root`` for one session id."""

    def __init__(self, root: Path, sid: str | None = None) -> None:
        self.root = root
        self.sid = sid or f"test-{uuid.uuid4().hex[:8]}"
        self.tools = {t.name: t for t in fs.make_tools(fs.Roots([str(root)]), write="allow")}

    async def __call__(self, name: str, **args: object) -> str:
        t = self.tools[f"fs.{name}"]
        return await call_handler(t, t.parse(args), ToolContext(self.sid, None, self.root))  # type: ignore[arg-type]

    def other(self) -> FS:
        return FS(self.root)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "root"
    r.mkdir()
    return r


@pytest.fixture
def t(root: Path) -> FS:
    return FS(root)


@pytest.fixture(params=["rg", "python"])
def engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "rg" and not HAS_RG:
        pytest.skip("ripgrep not installed")
    monkeypatch.setattr(hsearch, "USE_RG", request.param == "rg")
    return request.param


def bump_mtime(p: Path) -> None:
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


# ======================================================== Hermes-adapted: fuzzy patch
class TestIsAlreadyApplied:
    def test_cases(self) -> None:
        assert is_already_applied("x = compute_value(1)\n", "compute_value(1)", "compute_value(1)")
        assert not is_already_applied("y = 2\n", "compute_value(1)", "compute_value(1)")
        assert is_already_applied("def new_name(x):\n    return x\n", "def old_name(x):", "def new_name(x):")
        assert not is_already_applied("def old_name(x):\n    pass\n\ndef new_name(x):\n    pass\n",
                                      "def old_name(x):", "def new_name(x):")
        assert not is_already_applied("def old_name(x):\n", "def old_name(x):", "def new_name(x):")
        assert not is_already_applied("x = 1\n", "y = 2", "x = 1")      # trivial target never matches
        assert not is_already_applied("def new_name( x ):\n", "def old_name(x):", "def new_name(x):")

    async def test_identical_old_new_present_is_success_noop(self, t: FS, root: Path) -> None:
        f = root / "a.py"
        f.write_text("value = compute_total(items)\n")
        out = await t("edit", path="a.py", old_string="value = compute_total(items)",
                      new_string="value = compute_total(items)")
        assert out.startswith("No change") and f.read_text() == "value = compute_total(items)\n"

    async def test_replay_of_landed_edit_is_success_noop(self, t: FS, root: Path) -> None:
        f = root / "b.py"
        f.write_text("import os\n\nRETRY_LIMIT_SECONDS = 30\n")
        out = await t("edit", path="b.py", old_string="TIMEOUT_WINDOW_MS = 9000",
                      new_string="RETRY_LIMIT_SECONDS = 30")
        assert out.startswith("No change") and "already applied" in out

    async def test_fuzzy_match_onto_the_new_text_is_a_noop(self, t: FS, root: Path) -> None:
        f = root / "c.py"
        f.write_text("def new_name(x):\n    return x\n")
        before = f.stat().st_mtime_ns
        out = await t("edit", path="c.py", old_string="def old_name(x):", new_string="def new_name(x):")
        assert out.startswith("No change") and "already" in out and f.stat().st_mtime_ns == before

    async def test_genuine_no_match_still_errors(self, t: FS, root: Path) -> None:
        (root / "a.py").write_text("x = 1\n")
        with pytest.raises(ToolError, match="Could not find"):
            await t("edit", path="a.py", old_string="totally_absent_line()", new_string="y")


class TestMultiMatchLocations:
    def test_caps_at_five_with_overflow_note(self) -> None:
        line = "x = do_thing()\n"
        content = line * 9
        matches = [(i * len(line), (i + 1) * len(line) - 1) for i in range(9)]
        out = _format_match_locations(content, matches)
        assert out.count("L") == 5 and "... and 4 more" in out

    def test_long_lines_truncated(self) -> None:
        out = _format_match_locations("y = " + "z" * 200 + "\n", [(0, 5)])
        assert "..." in out and len(out.splitlines()[0]) < 100

    async def test_ambiguous_edit_lists_locations_as_untrusted_detail(self, t: FS, root: Path) -> None:
        content = ("def block_a(v):\n    value = value + 1\n    return v\n\n"
                   "def block_b(v):\n    value = value + 1\n    return v\n")
        _new, count, _s, error = fuzzy_find_and_replace(content, "    value = value + 1", "    value = value + 2")
        assert count == 0 and error and "Found 2 matches" in error and "L2:" in error and "L6:" in error
        (root / "b.py").write_text(content)
        with pytest.raises(ToolError, match="Found 2 matches") as ei:
            await t("edit", path="b.py", old_string="    value = value + 1", new_string="    value = value + 2")
        assert ei.value.untrusted and "L2:" in ei.value.untrusted and "L6:" in ei.value.untrusted
        assert "L2:" not in str(ei.value)        # file text never enters the trusted error text


# ======================================================== Hermes-adapted: reads
class TestBinaryDisclosure:
    @pytest.mark.parametrize(("prefix", "expected"), [
        (b"\x89PNG\r\n\x1a\n", "PNG image data"), (b"%PDF-", "PDF document"), (b"PK\x03\x04", "ZIP archive"),
        (b"\x7fELF", "ELF executable"), (b"SQLite format 3\x00", "SQLite database")])
    def test_known_signatures(self, prefix: bytes, expected: str) -> None:
        assert expected in ops.identify_binary_bytes(prefix + bytes(range(64)))

    def test_sizes_and_unknown(self) -> None:
        assert ops.identify_binary_bytes(b"") == "unknown binary"
        assert "4.0 KB" in ops.describe_binary_file(b"\x7fELF", 4096)

    async def test_lying_extension_names_real_type(self, t: FS, root: Path) -> None:
        rng = random.Random(20260810)
        (root / "notes.txt").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(rng.getrandbits(8) for _ in range(4096)))
        with pytest.raises(ToolError, match="PNG image data"):
            await t("read", path="notes.txt")


class TestUtf8BinaryClassification:
    async def test_cjk_cut_mid_character_reads_as_text(self, t: FS, root: Path) -> None:
        (root / "cjk.md").write_bytes(("漢字テキスト" * 200).encode())
        assert "漢字テキスト" in await t("read", path="cjk.md")

    async def test_bom_cyrillic_reads_as_text_without_bom(self, t: FS, root: Path) -> None:
        (root / "bom.txt").write_bytes(("Привет мир\n" * 100).encode("utf-8-sig"))
        out = await t("read", path="bom.txt")
        assert "1|Привет" in out and "\ufeff" not in out

    async def test_nul_byte_stays_binary(self, t: FS, root: Path) -> None:
        (root / "nul.txt").write_bytes(b"abc\x00def" * 10)
        with pytest.raises(ToolError, match="Binary file"):
            await t("read", path="nul.txt")

    async def test_invalid_utf8_noise_without_nul_stays_binary(self, t: FS, root: Path) -> None:
        rng = random.Random(7)
        (root / "noise.txt").write_bytes(bytes(rng.randrange(0x80, 0x100) for _ in range(2000)))
        with pytest.raises(ToolError, match="Binary file"):
            await t("read", path="noise.txt")

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
    async def test_utf16_transcodes_to_readable_text(self, t: FS, root: Path, encoding: str) -> None:
        (root / "u16.txt").write_bytes("hello\nwörld\n".encode(encoding))
        out = await t("read", path="u16.txt")
        assert "1|hello" in out and "2|wörld" in out and "Transcoded from UTF-16" in out


class TestReadPaging:
    async def test_offset_beyond_eof_names_recovery(self, t: FS, root: Path) -> None:
        (root / "f.txt").write_text("a\nb\nc\n")
        out = await t("read", path="f.txt", offset=10)
        assert "beyond the end of the file (3 lines total)" in out and "offset <= 3" in out

    async def test_empty_file_says_so(self, t: FS, root: Path) -> None:
        (root / "e.txt").write_text("")
        assert "File is empty" in await t("read", path="e.txt")

    async def test_pagination_hint(self, t: FS, root: Path) -> None:
        (root / "f.txt").write_text("".join(f"line {i}\n" for i in range(1, 11)))
        out = await t("read", path="f.txt", offset=3, limit=2)
        assert "3|line 3" in out and "4|line 4" in out and "5|" not in out
        assert "Use offset=5 to continue" in out

    async def test_char_budget_truncates_on_line_boundary(self, t: FS, root: Path) -> None:
        (root / "big.txt").write_text("".join(f"{i:06d} " + "x" * 90 + "\n" for i in range(1, 1001)))
        out = await t("read", path="big.txt")
        assert estimate_tokens(out) < fs.READ_TOKEN_BUDGET + 300
        assert "read budget" in out and "Use offset=" in out

    async def test_budget_counts_chinese_heavier(self, t: FS, root: Path) -> None:
        (root / "zh.txt").write_text("".join(f"第{i}行：" + "中文内容" * 20 + "\n" for i in range(1, 400)),
                                     encoding="utf-8")
        out = await t("read", path="zh.txt")
        assert "read budget" in out and estimate_tokens(out) < fs.READ_TOKEN_BUDGET + 300
        assert len(out) < 13_000   # ~0.6 token per Chinese char: far fewer chars than an ASCII read

    async def test_long_line_clamped(self, t: FS, root: Path) -> None:
        (root / "long.txt").write_text("a" * 10_000 + "\nshort\n")
        out = await t("read", path="long.txt")
        assert "2|short" in out and "clamped" in out

    async def test_similar_file_suggestion(self, t: FS, root: Path) -> None:
        (root / "AGENTS.md").write_text("x")
        with pytest.raises(ToolError, match="Similar files: AGENTS.md"):
            await t("read", path="AGENT.md")

    async def test_directory_points_to_list(self, t: FS, root: Path) -> None:
        (root / "d").mkdir()
        with pytest.raises(ToolError, match="is a directory"):
            await t("read", path="d")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX special files")
class TestSpecialFileGuard:
    async def test_fifo_read_returns_instantly(self, t: FS, root: Path) -> None:
        os.mkfifo(root / "pipe")
        with anyio.fail_after(5):
            with pytest.raises(ToolError, match="FIFO"):
                await t("read", path="pipe")
            with pytest.raises(ToolError, match="FIFO"):
                await t("write", path="pipe", content="x", mode="overwrite")

    async def test_socket(self, t: FS, root: Path) -> None:
        path = root / "s.sock"
        if len(str(path)) > 100:
            pytest.skip("AF_UNIX path too long")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(path))
            with pytest.raises(ToolError, match="socket"):
                await t("read", path="s.sock")
        finally:
            sock.close()


# ======================================================== Hermes-adapted: writes
class TestSurrogateRoundtrip:
    def test_roundtrips_surrogateescape_bytes(self, tmp_path: Path) -> None:
        raw = b"caf\xe9 \xff\xfe end\n"
        p = tmp_path / "x.txt"
        res = ops.write_file(str(p), raw.decode("utf-8", "surrogateescape"))
        assert res.error is None and res.verified and p.read_bytes() == raw

    @pytest.mark.parametrize("bad", ["\ud800", "x\udfffy"])
    def test_unencodable_surrogate_rejected_and_target_unchanged(self, tmp_path: Path, bad: str) -> None:
        p = tmp_path / "x.txt"
        p.write_text("keep me\n")
        res = ops.write_file(str(p), bad)
        assert res.error and p.read_text() == "keep me\n"

    def test_syntax_gate_refuses_broken_json_and_leaves_file(self, tmp_path: Path) -> None:
        p = tmp_path / "c.json"
        p.write_text('{"a": 1}\n')
        res = ops.write_file(str(p), '{"a": 1,}')
        assert res.error and "NOT created or modified" in res.error and p.read_text() == '{"a": 1}\n'


class TestStaleness:
    async def test_overwrite_refused_when_never_read(self, t: FS, root: Path) -> None:
        (root / "f.txt").write_text("original\n")
        with pytest.raises(ToolError, match="Refusing to overwrite"):
            await t("write", path="f.txt", content="new\n", mode="overwrite")
        assert (root / "f.txt").read_text() == "original\n"

    async def test_overwrite_refused_before_mutation_when_modified_externally(self, t: FS, root: Path) -> None:
        f = root / "f.txt"
        f.write_text("original\n")
        await t("read", path="f.txt")
        f.write_text("changed by someone else\n")
        bump_mtime(f)
        with pytest.raises(ToolError, match="modified since you last read it"):
            await t("write", path="f.txt", content="mine\n", mode="overwrite")
        assert f.read_text() == "changed by someone else\n"
        await t("read", path="f.txt")
        await t("write", path="f.txt", content="mine\n", mode="overwrite")
        assert f.read_text() == "mine\n"

    async def test_overwrite_requires_every_page(self, t: FS, root: Path) -> None:
        f = root / "f.txt"
        f.write_text("".join(f"{i}\n" for i in range(1, 11)))
        await t("read", path="f.txt", offset=1, limit=5)
        with pytest.raises(ToolError, match="partial view"):
            await t("write", path="f.txt", content="x\n", mode="overwrite")
        await t("read", path="f.txt", offset=6, limit=5)
        await t("write", path="f.txt", content="x\n", mode="overwrite")
        assert f.read_text() == "x\n"

    async def test_other_session_write_makes_my_read_stale(self, t: FS, root: Path) -> None:
        f = root / "f.txt"
        f.write_text("v1\n")
        other = t.other()
        await t("read", path="f.txt")
        await other("read", path="f.txt")
        await anyio.sleep(0.01)
        await other("edit", path="f.txt", old_string="v1", new_string="v2")
        with pytest.raises(ToolError, match="another session"):
            await t("write", path="f.txt", content="v3\n", mode="overwrite")
        assert f.read_text() == "v2\n"

    async def test_edit_warns_on_stale_file_but_applies(self, t: FS, root: Path) -> None:
        f = root / "f.txt"
        f.write_text("alpha\nbeta\n")
        await t("read", path="f.txt")
        f.write_text("alpha\nbeta\ngamma\n")
        bump_mtime(f)
        out = await t("edit", path="f.txt", old_string="beta", new_string="BETA")
        assert "Warning:" in out and "modified since you last read it" in out
        assert f.read_text() == "alpha\nBETA\ngamma\n"

    async def test_edit_after_full_read_keeps_baseline(self, t: FS, root: Path) -> None:
        f = root / "f.txt"
        f.write_text("one\ntwo\n")
        await t("read", path="f.txt")
        await t("edit", path="f.txt", old_string="one", new_string="ONE")
        await t("write", path="f.txt", content="rewritten\n", mode="overwrite")   # still allowed
        assert f.read_text() == "rewritten\n"

    async def test_blind_edit_never_gains_a_baseline(self, t: FS, root: Path) -> None:
        f = root / "f.txt"
        f.write_text("one\ntwo\n")
        await t("edit", path="f.txt", old_string="one", new_string="ONE")
        with pytest.raises(ToolError, match="without a full view"):
            await t("write", path="f.txt", content="x\n", mode="overwrite")

    async def test_own_create_then_overwrite_is_allowed(self, t: FS, root: Path) -> None:
        await t("write", path="new.txt", content="a\n")
        await t("write", path="new.txt", content="b\n", mode="overwrite")
        assert (root / "new.txt").read_text() == "b\n"

    async def test_third_patch_failure_escalates(self, t: FS, root: Path) -> None:
        (root / "f.txt").write_text("hello\n")
        for _ in range(2):
            with pytest.raises(ToolError) as ei:
                await t("edit", path="f.txt", old_string="nope nope", new_string="x")
            assert "failure #" not in str(ei.value)
        with pytest.raises(ToolError, match="failure #3"):
            await t("edit", path="f.txt", old_string="nope nope", new_string="x")

    async def test_line_numbered_content_refused(self, t: FS) -> None:
        with pytest.raises(ToolError, match="line-number"):
            await t("write", path="n.txt", content="1|first\n2|second\n3|third\n")


# ======================================================== Hermes-adapted: search
class TestSearch:
    @pytest.fixture
    def proj(self, root: Path) -> Path:
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("def Handler():\n    return compute(1)\n")
        (root / "README.md").write_text("Use compute(1) here.\n")
        (root / ".hub").mkdir()
        (root / ".hub" / "blob.txt").write_text("needle in hidden dir\n")
        return root

    async def test_content_matches(self, t: FS, proj: Path, engine: str) -> None:
        out = await t("search", pattern="compute")
        assert "src/app.py:2:" in out and "README.md:1:" in out

    async def test_newline_regex_matches_across_lines(self, t: FS, proj: Path, engine: str) -> None:
        out = await t("search", pattern=r"Handler\(\):\n\s+return")
        assert "src/app.py:1:" in out and "multiline" in out

    async def test_hidden_dirs_excluded_with_hint(self, t: FS, proj: Path, engine: str) -> None:
        out = await t("search", pattern="needle")
        assert out.startswith("No matches") and ".hub/blob.txt" in out and "hidden" in out

    async def test_case_mismatch_hint(self, t: FS, proj: Path, engine: str) -> None:
        out = await t("search", pattern="HANDLER")
        assert out.startswith("No matches") and "case-insensitive" in out and "src/app.py" in out
        assert "src/app.py:1:" in await t("search", pattern="HANDLER", ignore_case=True)

    async def test_regex_metachar_literal_hint(self, t: FS, proj: Path, engine: str) -> None:
        out = await t("search", pattern="compute(1)")
        assert out.startswith("No matches") and "literal match" in out
        assert "README.md:1:" in await t("search", pattern="compute(1)", literal=True)

    async def test_true_zero_match_has_no_hint(self, t: FS, proj: Path, engine: str) -> None:
        assert (await t("search", pattern="zzz_absent")).strip().endswith("under .")

    async def test_files_only_count_and_context(self, t: FS, proj: Path, engine: str) -> None:
        assert "src/app.py" in await t("search", pattern="compute", output_mode="files_only")
        assert "src/app.py: 1" in await t("search", pattern="compute", output_mode="count")
        out = await t("search", pattern="return", context=1, file_glob="*.py")
        assert "src/app.py:1: def Handler():" in out and "src/app.py:2:" in out

    async def test_files_target_lists_dirs_and_files(self, t: FS, proj: Path, engine: str) -> None:
        (proj / "config_dir").mkdir()
        (proj / "src" / "config.py").write_text("")
        out = await t("search", pattern="*config*", target="files")
        assert "config_dir/" in out and "src/config.py" in out

    async def test_pagination(self, t: FS, root: Path, engine: str) -> None:
        (root / "many.txt").write_text("".join(f"hit {i}\n" for i in range(30)))
        out = await t("search", pattern="hit", limit=10)
        assert out.count("many.txt:") == 10 and "offset=10" in out
        out2 = await t("search", pattern="hit", limit=10, offset=10)
        assert "many.txt:11: hit 10" in out2

    async def test_multi_path_merge_and_missing(self, t: FS, proj: Path, engine: str) -> None:
        out = await t("search", pattern="compute", path="src,README.md,nope")
        assert "src/app.py:2:" in out and "README.md:1:" in out and "Skipped missing path(s): nope" in out
        with pytest.raises(ToolError, match="None of the search paths exist"):
            await t("search", pattern="compute", path="nope1,nope2")

    async def test_search_does_not_follow_symlinks_out_of_root(self, t: FS, root: Path, tmp_path: Path,
                                                               engine: str) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("top secret token\n")
        os.symlink(outside, root / "linkdir")
        os.symlink(outside / "secret.txt", root / "link.txt")
        out = await t("search", pattern="secret")
        assert "top secret" not in out
        with pytest.raises(ToolError, match="outside the allowed roots"):
            await t("search", pattern="secret", path="linkdir")


# ======================================================== Ventri regressions
class TestEncodingPreservation:
    async def test_non_utf8_bytes_survive_edit_and_append(self, t: FS, root: Path) -> None:
        f = root / "latin1.txt"
        f.write_bytes(b"caf\xe9 au lait\nsecond line\n\xff\xfe raw\n")
        out = await t("read", path="latin1.txt")
        assert "1|caf\ufffd au lait" in out and "2|second line" in out and "not valid UTF-8" in out
        with pytest.raises(ToolError, match="Refusing to overwrite"):     # would replace the raw bytes
            await t("write", path="latin1.txt", content="clean\n", mode="overwrite")
        await t("edit", path="latin1.txt", old_string="second line", new_string="SECOND LINE")
        assert f.read_bytes() == b"caf\xe9 au lait\nSECOND LINE\n\xff\xfe raw\n"
        await t("write", path="latin1.txt", content="appended\n", mode="append")
        assert f.read_bytes() == b"caf\xe9 au lait\nSECOND LINE\n\xff\xfe raw\nappended\n"

    async def test_crlf_and_bom_preserved_by_edit_overwrite_append(self, t: FS, root: Path) -> None:
        f = root / "win.txt"
        f.write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\n")
        out = await t("read", path="win.txt")
        assert "1|one" in out and "\ufeff" not in out and "\r" not in out
        await t("edit", path="win.txt", old_string="two", new_string="TWO")
        assert f.read_bytes() == b"\xef\xbb\xbfone\r\nTWO\r\n"
        await t("write", path="win.txt", content="three\n", mode="append")
        assert f.read_bytes() == b"\xef\xbb\xbfone\r\nTWO\r\nthree\r\n"
        await t("write", path="win.txt", content="a\nb\n", mode="overwrite")
        assert f.read_bytes() == b"\xef\xbb\xbfa\r\nb\r\n"


class TestFuzzyEditThroughTool:
    async def test_indentation_drift_matches(self, t: FS, root: Path) -> None:
        f = root / "m.py"
        f.write_text("def f():\n    if x:\n        return 1\n")
        out = await t("edit", path="m.py", old_string="if x:\n    return 1", new_string="if x:\n    return 2")
        assert "fuzzy strategy" in out and "+        return 2" in out
        assert f.read_text() == "def f():\n    if x:\n        return 2\n"

    async def test_replace_all_and_multi_edit_atomicity(self, t: FS, root: Path) -> None:
        f = root / "m.txt"
        f.write_text("a1\na1\nb2\n")
        await t("edit", path="m.txt", old_string="a1", new_string="A1", replace_all=True)
        assert f.read_text() == "A1\nA1\nb2\n"
        with pytest.raises(ToolError, match="edit #2 of 2"):
            await t("edit", path="m.txt", edits=[{"old_string": "b2", "new_string": "B2"},
                                                 {"old_string": "zzz-missing", "new_string": "q"}])
        assert f.read_text() == "A1\nA1\nb2\n"      # all or nothing
        out = await t("edit", path="m.txt", edits=[{"old_string": "b2", "new_string": "B2", "replace_all": None},
                                                   {"old_string": "B2", "new_string": "C3"}])
        assert "2 replacement(s)" in out and f.read_text() == "A1\nA1\nC3\n"

    async def test_bad_argument_combinations(self, t: FS, root: Path) -> None:
        (root / "m.txt").write_text("x\n")
        with pytest.raises(ToolError, match="required"):
            await t("edit", path="m.txt", old_string="x")
        with pytest.raises(ToolError, match="File not found"):
            await t("edit", path="missing.txt", old_string="x", new_string="y")


class TestConfinement:
    async def test_symlink_and_dotdot_escapes_refused(self, t: FS, root: Path, tmp_path: Path) -> None:
        (tmp_path / "secret.txt").write_text("secret\n")
        os.symlink(tmp_path / "secret.txt", root / "link.txt")
        os.symlink(tmp_path, root / "up")
        for name, args in [("read", {"path": "link.txt"}), ("read", {"path": "../secret.txt"}),
                           ("read", {"path": "up/secret.txt"}),
                           ("write", {"path": "link.txt", "content": "x", "mode": "overwrite"}),
                           ("write", {"path": "up/new.txt", "content": "x"}),
                           ("edit", {"path": "link.txt", "old_string": "secret", "new_string": "x"}),
                           ("list", {"path": "up"})]:
            with pytest.raises(ToolError, match="outside the allowed roots"):
                await t(name, **args)
        assert (tmp_path / "secret.txt").read_text() == "secret\n" and not (tmp_path / "new.txt").exists()

    async def test_unicode_variant_filename_resolves(self, t: FS, root: Path) -> None:
        (root / "caf\u00e9.txt").write_text("nfc\n")
        fs_normalizes = (root / "cafe\u0301.txt").exists()   # APFS/HFS+ are normalization-insensitive
        out = await t("read", path="cafe\u0301.txt")       # NFD spelling of the same name
        assert "1|nfc" in out
        assert fs_normalizes or "unicode-equivalent" in out


class TestAtomicWrite:
    async def test_failed_replace_leaves_original_and_no_temp(self, t: FS, root: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
        f = root / "f.txt"
        f.write_text("original\n")
        await t("read", path="f.txt")

        def boom(src: str, dst: str) -> None:
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(ToolError, match="No space left"):
            await t("write", path="f.txt", content="new\n", mode="overwrite")
        monkeypatch.undo()
        assert f.read_text() == "original\n"
        assert [p.name for p in root.iterdir()] == ["f.txt"]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
    async def test_mode_preserved_and_no_temp_left(self, t: FS, root: Path) -> None:
        f = root / "run.sh"
        f.write_text("echo hi\n")
        f.chmod(0o750)
        await t("edit", path="run.sh", old_string="echo hi", new_string="echo bye")
        assert stat.S_IMODE(f.stat().st_mode) == 0o750 and f.read_text() == "echo bye\n"
        assert [p.name for p in root.iterdir()] == ["run.sh"]

    async def test_write_into_symlink_inside_root_updates_target(self, t: FS, root: Path) -> None:
        (root / "real.txt").write_text("v1\n")
        os.symlink(root / "real.txt", root / "alias.txt")
        await t("read", path="alias.txt")
        await t("write", path="alias.txt", content="v2\n", mode="overwrite")
        assert (root / "alias.txt").is_symlink() and (root / "real.txt").read_text() == "v2\n"


class TestNonBlocking:
    async def test_file_work_runs_off_the_event_loop(self, t: FS, root: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
        (root / "f.txt").write_text("x\n")
        real = fs.read_impl

        def slow_read(*a: object, **kw: object) -> str:
            time.sleep(0.3)
            return real(*a, **kw)  # type: ignore[arg-type]
        monkeypatch.setattr(fs, "read_impl", slow_read)
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await anyio.sleep(0.01)
        async with anyio.create_task_group() as tg:
            tg.start_soon(ticker)
            assert "1|x" in await t("read", path="f.txt")
            tg.cancel_scope.cancel()
        assert ticks >= 10          # the loop kept running while the read slept

    async def test_timeout_cancels_a_slow_read(self, t: FS, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (root / "f.txt").write_text("x\n")
        monkeypatch.setattr(fs, "read_impl", lambda *a, **kw: time.sleep(2) or "late")
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            with anyio.fail_after(0.2):
                await t("read", path="f.txt")
        assert time.monotonic() - t0 < 1.5


class TestNotes:
    async def test_notes_paging_search_and_stale_guard(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "long.md").write_text("".join(f"line {i}\n" for i in range(1, 21)))
        tools = {x.name: x for x in notes.make_tools(fs.Roots([str(vault)]), "allow")}
        sid = f"notes-{uuid.uuid4().hex[:6]}"

        async def n(tool_name: str, **args: object) -> str:
            tool = tools[f"notes.{tool_name}"]
            return await call_handler(tool, tool.parse(args), ToolContext(sid, None, tmp_path))  # type: ignore[arg-type]
        out = await n("read", name="long", offset=5, limit=3)
        assert "5|line 5" in out and "7|line 7" in out and "8|" not in out and "offset=8" in out
        assert "long:12: line 12" in await n("search", query="LINE 12")
        with pytest.raises(ToolError, match="Refusing to overwrite"):
            await n("write", name="long", content="short\n", mode="overwrite")   # only a partial read
        await n("read", name="long")
        await n("write", name="long", content="short\n", mode="overwrite")
        assert (vault / "long.md").read_text() == "short\n"
        with pytest.raises(ToolError, match="no note"):
            await n("read", name="missing")


async def test_file_state_is_session_scoped_and_forgotten(tmp_path: Path) -> None:
    from tests.agent.harness import Env, call
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("hello\nhello\n")
    script = [{"tool_calls": [call("fs.read", {"path": "a.txt"})]},
              {"tool_calls": [call("fs.edit", {"path": "a.txt", "old_string": "hello", "new_string": "bye"})]},
              {"content": "done"}]
    async with Env(tmp_path, script) as env:
        await env.kernel.plugin(fs.plugin, {"roots": [str(root)], "write": "allow"})
        s = await env.open()
        await s.turn("go")
        st = s.ctx.get(FileState)
        assert isinstance(st, FileState) and st.session_id == s.id
        assert s.id in get_registry()._sessions     # pyright: ignore[reportPrivateUsage]
        tool_out = [str(m.content) for m in s.loop.builder.history if m.role == "tool"]
        assert tool_out[0].startswith('<tool-output tool="fs.read" trust="untrusted">')
        # the ambiguous edit's match listing is fenced as untrusted data after the error
        assert tool_out[1].startswith("ERROR: Found 2 matches") and 'trust="untrusted"' in tool_out[1]
        await s.end(extract=False)
        assert s.id not in get_registry()._sessions  # pyright: ignore[reportPrivateUsage]
