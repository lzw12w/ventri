"""CJK-aware token estimate, shared output helpers and shell.run's bounded,
streamed output (head + tail + artifact, partial output on timeout)."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import anyio
import pytest

from ventri_agent.providers.fake import estimate_tokens as fake_estimate
from ventri_agent.tokens import count_cjk, estimate_tokens, prefix_within, suffix_within
from ventri_agent.tools import core, shell
from ventri_agent.tools.output import head_tail, preview, strip_ansi
from ventri_agent.tools.registry import ToolContext, ToolError, call_handler

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------ tokens
def test_estimate_follows_deepseek_ratio():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("x" * 1000) == 300            # 0.3 / English char
    assert estimate_tokens("中" * 1000) == 600           # 0.6 / Chinese char
    assert estimate_tokens("中文" * 500 + "ab" * 500) == 900
    assert count_cjk("日本語カタカナ한국어，。！") == 13
    assert fake_estimate is estimate_tokens              # one estimate everywhere
    # the old len/4 estimate said 250 for 1000 Chinese chars: under-counted 2.4x
    assert estimate_tokens("中" * 1000) > 2 * ((1000 + 3) // 4)


def test_prefix_suffix_within():
    s = "中文abc" * 100
    p, q = prefix_within(s, 50), suffix_within(s, 50)
    assert estimate_tokens(p) <= 50 < estimate_tokens(s[:len(p) + 1])
    assert estimate_tokens(q) <= 50 < estimate_tokens(s[len(s) - len(q) - 1:])
    assert s.startswith(p) and s.endswith(q)
    assert prefix_within("short", 50) == "short" and suffix_within("", 5) == ""


def test_strip_ansi():
    raw = "\x1b[1;31mFAIL\x1b[0m done\x1b]0;title\x07 \x1b(B\x07bell\rover \x9b2Kx \U000E0041hidden"
    assert strip_ansi(raw) == "FAIL done bell\nover x hidden"
    flag = "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F"
    assert strip_ansi(flag) == flag                      # emoji tag sequences survive


def test_head_tail_and_preview(tmp_path):
    text = "".join(f"row {i}\n" for i in range(20000))
    head, tail, omitted = head_tail(text, 1000)
    assert text.startswith(head) and text.endswith(tail) and omitted == len(text) - len(head) - len(tail)
    assert estimate_tokens(head) <= 400 and estimate_tokens(head) + estimate_tokens(tail) <= 1000
    tc = ToolContext("s", None, tmp_path, call_id="call_9")  # type: ignore[arg-type]
    assert preview(tc, "small", "x") == "small"
    out = preview(tc, text, "x", budget=1000)
    assert "artifact 'call_9-x'" in out and f"offset={len(head)}" in out
    assert (tmp_path / "artifacts" / "call_9-x.txt").read_text() == text


# ------------------------------------------------------------------ shell
def tool(tmp_path: Path, **kw):
    (tmp_path / "wd").mkdir(exist_ok=True)
    return shell.make_tool(shell.ShellConfig(cwd=str(tmp_path / "wd"), **kw))


async def run(t, args, tmp_path, call_id="call_s"):
    return await call_handler(t, t.parse(args), ToolContext("s1", None, tmp_path, call_id=call_id))  # type: ignore[arg-type]


async def test_shell_large_output_head_tail_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path / "home"))
    t = tool(tmp_path, timeout=20)
    assert t.untrusted                                    # output is outside data now
    out = await run(t, {"command": "seq 1 200000; echo DONE >&2"}, tmp_path)
    assert out.startswith("exit code 0\n1\n2\n3\n") and out.rstrip().endswith("DONE")
    assert estimate_tokens(out) < 8000 and "artifact 'call_s-shell'" in out
    full = (tmp_path / "artifacts" / "call_s-shell.txt").read_text()
    assert full.count("\n") == 200001 and "\n123456\n" in full
    read = {x.name: x for x in core.CORE_TOOLS}["artifact.read"]
    page = await call_handler(read, read.parse({"handle": "call_s-shell", "offset": full.index("150000\n")}),
                              ToolContext("s1", None, tmp_path))  # type: ignore[arg-type]
    assert page.startswith("150000\n150001")


async def test_shell_chinese_output_budgeted(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path / "home"))
    t = tool(tmp_path)
    out = await run(t, {"command": "for i in $(seq 1 2000); do echo \"第$i行 中文输出内容\"; done"}, tmp_path)
    # ~33K chars: the old 20K-char cut would have let ~20K Chinese chars (~12K tokens) through
    assert estimate_tokens(out) < 8000 and "第2000行" in out and "artifact" in out


async def test_shell_capture_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path / "home"))
    t = tool(tmp_path, max_capture_bytes=10_000, tail_bytes=2_000)
    out = await run(t, {"command": "seq 1 100000"}, tmp_path)
    # 10 KB head + 2 KB tail fit inline; the dropped middle is reported, not silently lost
    assert len(out) < 13_000 and "bytes dropped: output exceeded the 10000-byte capture" in out
    assert out.startswith("exit code 0\n1\n2\n") and out.rstrip().endswith("100000")


async def test_shell_strips_ansi(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path / "home"))
    t = tool(tmp_path)
    out = await run(t, {"command": r"printf '\033[31mred\033[0m plain\n'"}, tmp_path)
    assert out == "exit code 0\nred plain\n"


async def test_shell_timeout_returns_partial_output(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path / "home"))
    t = tool(tmp_path, timeout=10)
    t0 = time.monotonic()
    with anyio.fail_after(5):
        with pytest.raises(ToolError, match="timed out after 0.5s") as ei:
            await run(t, {"command": "echo started; echo step2; sleep 30", "timeout": 0.5}, tmp_path)
    assert time.monotonic() - t0 < 4
    assert ei.value.untrusted and "started\nstep2" in ei.value.untrusted
    with pytest.raises(ToolError, match="no output"):
        await run(t, {"command": "sleep 30", "timeout": 0.2}, tmp_path)


async def test_shell_cancel_kills_process_group(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path / "home"))
    t = tool(tmp_path, timeout=30)
    pidfile = tmp_path / "wd" / "pid"
    with anyio.move_on_after(0.5):
        await run(t, {"command": f"sleep 60 & echo $! > {pidfile}; wait"}, tmp_path)
    pid = int(pidfile.read_text())

    def alive() -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False).stdout
        return bool(state.strip()) and not state.strip().startswith("Z")   # a zombie is dead
    for _ in range(40):
        if not alive():
            break
        await anyio.sleep(0.05)
    assert not alive()
