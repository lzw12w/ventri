"""Replay a long agent session offline to measure context pruning.

    uv run python benchmarks/context_replay.py SESSION.jsonl [--prune 32000 48000] [--json]
    uv run python benchmarks/context_replay.py --synthetic        # no log needed

The recorded assistant messages (reasoning, content, tool calls) are fed back
through the real AgentLoop / ContextBuilder with the scripted FakeProvider, and
every tool call returns its recorded result, so the only variable is the
context policy. FakeProvider simulates DeepSeek's prefix disk cache (a request
hits the longest persisted prefix unit it extends) and estimates tokens with
``ventri_agent.tokens.estimate_tokens``; cost uses Ventri's DeepSeek price
table at peak time. Absolute numbers differ from the API's tokenizer; the
ratios between policies are the point.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio

import ventri
from ventri import Kernel
from ventri_agent.messages import Message, Usage
from ventri_agent.permission import permission
from ventri_agent.providers.base import ModelProvider
from ventri_agent.providers.fake import FakeProvider, Step
from ventri_agent.sessions import SessionManager, session_manager
from ventri_agent.tools.registry import Risk, Tool, ToolRegistry
from ventri_agent.tools.registry import plugin as registry_plugin

PEAK = datetime(2026, 10, 8, 2, 0, tzinfo=UTC)   # 10:00 Beijing, a weekday: peak prices


def load_session(path: Path) -> tuple[str, str, list[Message]]:
    system, user, msgs = "", "", []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["t"] == "prefix" and not system:
            system = rec["system"]
        elif rec["t"] == "msg":
            m = Message.from_json(rec["m"])
            if m.role == "user" and not user:
                user = m.content or ""
            elif m.role in ("assistant", "tool"):
                msgs.append(m)
    return system, user, msgs


def synthetic(steps: int = 60, seed: int = 7) -> tuple[str, str, list[Message]]:
    """A build-and-debug session shaped like Terminal-Bench build-cython-ext:
    ~60 steps, tool results of 0.2-7 KB, a few long file writes, ~850 chars of
    reasoning per step."""
    rnd = random.Random(seed)
    words = ["import", "numpy", "cython", "build", "error", "warning", "setup.py", "extension", "compile", "gcc", "linking", "module"]

    def text(n: int) -> str:
        return " ".join(rnd.choice(words) for _ in range(n // 7))
    msgs: list[Message] = []
    for i in range(steps):
        cid = f"call_{i:03d}"
        if i % 9 == 4:
            name, args = "fs.write", {"path": f"/app/f{i}.py", "content": text(rnd.randint(2000, 6000))}
        else:
            name, args = "shell.run", {"command": f"cd /app && step {i} " + text(80)}
        msgs.append(Message.assistant(text(rnd.randint(40, 200)) if i % 5 == 0 else None,
                                      reasoning=text(rnd.randint(300, 1500)),
                                      tool_calls=[_tc(cid, name, args)]))
        size = rnd.choice([200, 600, 1500, 3000, 4500, 7000])
        body = text(size)
        msgs.append(Message.tool(cid, f'<tool-output tool="{name}" trust="untrusted">\nexit code 0\n{body}\n'
                                 '</tool-output>'))
    msgs.append(Message.assistant("Done: the extension builds and the tests pass.", reasoning=text(400)))
    return "You are a coding agent in a container.", "Build the Cython extensions and verify.", msgs


def _tc(cid: str, name: str, args: dict[str, Any]) -> Any:
    from ventri_agent.messages import ToolCall
    return ToolCall(cid, name, json.dumps(args))


def script_and_tools(msgs: list[Message]) -> tuple[list[Step], dict[str, str], set[str]]:
    steps: list[Step] = []
    results: dict[str, str] = {}
    names: set[str] = set()
    for m in msgs:
        if m.role == "assistant":
            steps.append({"reasoning": m.reasoning_content or "", "content": m.content,
                          "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                                         for tc in m.tool_calls]})
            names |= {tc.name.replace("__", ".") for tc in m.tool_calls}
        elif m.role == "tool" and m.tool_call_id:
            results[m.tool_call_id] = m.content or ""
    return steps, results, names


async def replay(system: str, user: str, msgs: list[Message], prune: int, keep: int = 6) -> dict[str, Any]:
    steps, results, names = script_and_tools(msgs)
    prov = FakeProvider(list(steps), echo=False)

    @ventri.plugin(name="provider:replay", provides={"llm": ModelProvider})
    def provider(ctx: Any, config: Any) -> None:
        ctx.provide(ModelProvider, prov)

    @ventri.plugin(name="replay-tools")
    def tools(ctx: Any, config: Any, registry: ToolRegistry) -> None:
        def make(name: str) -> Tool:
            def handler(a: Any, tc: Any) -> str:
                return results.get(tc.call_id, "ok")
            return Tool(name, f"recorded {name}", handler, None, Risk.READ, timeout=None)
        for n in sorted(names | {"artifact.read"}):
            registry.register(ctx, make(n))

    with tempfile.TemporaryDirectory() as tmp:
        async with Kernel() as k:
            await k.plugin(provider)
            await k.plugin(registry_plugin)
            await k.plugin(permission, {"rules": [{"tool": "*", "action": "allow"}], "audit": None})
            await k.plugin(tools)
            await k.plugin(session_manager, {
                "dir": tmp, "extract_memory": False,
                "budget": {"max_steps": 10_000, "max_tool_calls": 10_000, "max_tokens": 10**12,
                           "max_cost_cny": 10**6, "wall_s": 10**6},
                "agents": {"replay": {"system_prompt": system, "time_notes": False,
                                      "prune_tokens": prune, "prune_keep": keep}}})
            mgr = k.get(SessionManager)
            s = await mgr.open(agent="replay")
            r = await s.turn(user)
            reqs = prov.requests
            usage = Usage()
            peak_ctx = 0
            for u in [_u for _u in _usages(Path(tmp), s.id)]:
                usage = usage + u
                peak_ctx = max(peak_ctx, u.prompt_tokens)
            money = prov.prices.price(usage, PEAK, "deepseek-flash")
            prunes = sum(1 for line in (Path(tmp) / f"{s.id}.jsonl").read_text().splitlines()
                         if '"t": "prune"' in line)
            await s.end(extract=False)
    return {"prune_tokens": prune, "status": r.status, "steps": len(reqs), "prompt_tokens": usage.prompt_tokens,
            "cache_hit": usage.cache_hit, "cache_miss": usage.cache_miss, "hit_rate": round(usage.hit_rate, 4),
            "max_context": peak_ctx, "prunes": prunes, "cost_usd": round(money.usd, 5)}


def _usages(d: Path, sid: str) -> list[Usage]:
    out = []
    for line in (d / f"{sid}.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["t"] == "usage":
            out.append(Usage.from_json(rec["usage"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--prune", type=int, nargs="*", default=[24_000, 32_000, 48_000])
    ap.add_argument("--keep", type=int, default=6)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.session is None and not a.synthetic:
        ap.error("give a session log or --synthetic")
    system, user, msgs = synthetic() if a.session is None else load_session(a.session)
    rows = [anyio.run(replay, system, user, msgs, p, a.keep, backend="asyncio") for p in [0, *a.prune]]
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    base = rows[0]
    print(f"{'prune_tokens':>12} {'steps':>5} {'prompt':>10} {'vs off':>7} {'hit':>6} {'miss':>9} "
          f"{'max ctx':>8} {'prunes':>6} {'cost $':>8}")
    for r in rows:
        rel = r["prompt_tokens"] / base["prompt_tokens"] - 1 if base["prompt_tokens"] else 0
        print(f"{r['prune_tokens'] or 'off':>12} {r['steps']:>5} {r['prompt_tokens']:>10} {rel:>+7.0%} "
              f"{r['hit_rate']:>6.1%} {r['cache_miss']:>9} {r['max_context']:>8} {r['prunes']:>6} "
              f"{r['cost_usd']:>8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
