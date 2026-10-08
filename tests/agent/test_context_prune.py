"""Cache-friendly context pruning (ContextBuilder.maybe_prune) and head + tail
previews of spilled tool results."""
from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from ventri_agent.session import SessionLog

from .harness import Env, call
from .test_context_cache import is_prefix

pytestmark = pytest.mark.anyio
ROOT = Path(__file__).resolve().parents[2]


def long_session(n: int = 30, seed: int = 3) -> list:
    rnd = random.Random(seed)
    words = ["build", "error", "warning", "numpy", "cython", "gcc", "setup", "module", "link", "compile"]

    def text(k: int) -> str:
        return " ".join(rnd.choice(words) for _ in range(k // 6))
    steps = [{"reasoning": text(2400), "tool_calls": [call("t.web", {"text": f"r{i} " + text(3000)}, id=f"c{i}")]}
             for i in range(n)]
    return [*steps, {"content": "done"}]


def tool_msgs(s):
    return [m for m in s.loop.builder.history if m.role == "tool"]


async def run(tmp_path, prune: int, keep: int = 4):
    env = Env(tmp_path, long_session(), agents={"lean": {"prune_tokens": prune, "prune_keep": keep,
                                                          "time_notes": False}})
    async with env:
        s = await env.open(agent="lean")
        r = await s.turn("build it")
        recs = env.log_records(s.id)
        history = list(s.loop.builder.history)
        replayed = SessionLog.replay(s.info.log_path).history
        return env, s, r, recs, history, replayed


async def test_pruning_cuts_prompt_tokens_in_rare_large_steps(tmp_path):
    _, _, off, recs_off, _, _ = await run(tmp_path / "off", 0)
    env, s, on, recs, history, replayed = await run(tmp_path / "on", 24_000)
    assert not any(r["t"] == "prune" for r in recs_off)
    prunes = [r for r in recs if r["t"] == "prune"]
    assert 1 <= len(prunes) <= 6                              # rare, large steps
    assert all(p["saved"] >= 4_000 for p in prunes)
    assert on.usage.prompt_tokens < 0.75 * off.usage.prompt_tokens
    # between prunes every request extends the previous one (cache keeps hitting)
    reqs = env.provider.requests
    breaks = sum(1 for a, b in itertools.pairwise(reqs) if not is_prefix(a, b))
    assert breaks == len(prunes)
    assert on.usage.hit_rate > 0.6
    # the newest prune_keep results are intact; older big ones are stubs with a working pointer
    tools = [m for m in history if m.role == "tool"]
    assert all("pruned" not in (m.content or "") for m in tools[-4:])
    stub = tools[0].content or ""
    assert stub.startswith("[older t.web result pruned") and "artifact.read(handle='c0-full')" in stub
    assert stub.rstrip().endswith("</tool-output>")           # still fenced as untrusted data
    full = (s.info.dir / "artifacts" / "c0-full.txt").read_text()
    assert full.startswith('<tool-output tool="t.web"') and len(full) > 2_500
    first_asst = next(m for m in history if m.role == "assistant")
    assert "chars elided" in first_asst.tool_calls[0].arguments
    json.loads(first_asst.tool_calls[0].arguments)            # still valid JSON
    assert "earlier reasoning shortened" in (first_asst.reasoning_content or "")
    # resume rebuilds exactly the pruned bytes
    def wire(ms):
        return [json.dumps(m.to_api(), sort_keys=True) for m in ms]
    assert wire(history) == wire(replayed)


async def test_pruning_off_by_default(tmp_path):
    async with Env(tmp_path, long_session(8)) as env:
        s = await env.open()
        await s.turn("go")
        assert not any(r["t"] == "prune" for r in env.log_records(s.id))


async def test_spilled_result_keeps_head_and_tail(tmp_path):
    script = [{"tool_calls": [call("t.big", id="call_big")]}, {"content": "ok"}]
    async with Env(tmp_path, script) as env:
        s = await env.open()
        await s.turn("go")
        out = tool_msgs(s)[0].content or ""
        assert out.startswith("line 00000") and "line 00799" in out      # the end survives
        assert "chars omitted" in out and "artifact 'call_big'" in out and len(out) < 3_200


def test_context_replay_benchmark_synthetic():
    out = subprocess.run([sys.executable, str(ROOT / "benchmarks" / "context_replay.py"), "--synthetic",
                          "--prune", "32000", "--json"], capture_output=True, text=True, check=True, timeout=120)
    off, on = json.loads(out.stdout)
    assert on["prompt_tokens"] < off["prompt_tokens"] and on["prunes"] >= 1 and off["prunes"] == 0
