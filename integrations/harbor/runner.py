"""Runs ONE task with Ventri Agent in headless mode, inside the task container.

    <bundled python> -I runner.py --instruction-file F --out DIR --wall SECONDS [...]

Started by ``ventri_harbor_agent.VentriAgent``. The DeepSeek key is read from
the file named by ``$VENTRI_DS_KEY_FILE`` (never argv, never printed); the file
and its private directory are deleted as soon as the key is in memory, so the
agent's own shell cannot read it later.

Ventri runs unchanged: Kernel + DeepSeek provider + ToolRegistry + permission
engine + SessionManager / AgentLoop, with tools core (time/artifact/work), fs
and shell. The session is opened **headless**: no channel, approvals decided by
``permission.unattended`` (here ``ask: allow`` because the task container is
the sandbox; irreversible tools stay denied -- none are loaded anyway), every
decision audited to ``DIR/audit.jsonl``. The agent's shell gets the container's
original environment (``--base-env-file``) with Ventri's own runtime stripped
from it, keeps cwd/env between calls, and background jobs are left running at
the end (a server the task asked for must still be up for the verifier).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import anyio
import httpx

import ventri
from ventri import Kernel
from ventri_agent.context import HEADLESS_SYSTEM
from ventri_agent.permission import permission
from ventri_agent.providers.base import ModelProvider
from ventri_agent.providers.deepseek import DEFAULT_ROUTES, DeepSeekProvider
from ventri_agent.sessions import SessionManager, session_manager
from ventri_agent.tools import core, fs, shell
from ventri_agent.tools.registry import plugin as registry_plugin

ENVIRONMENT = """## Environment
- You are working inside a Linux container on a task given by the user message; the task is graded \
automatically after you finish, by tests you cannot see. Your working directory is {cwd}.
- shell.run uses /bin/sh. The working directory and exported variables carry over between shell.run \
calls. A foreground command is killed after its timeout (at most {tmax}s); start servers, daemons and \
long builds with shell.run(background=true) and check them with shell.output (it can wait for a regex).
- Background jobs keep running after you finish, so a service the task asks for stays up for grading.
- You have about {minutes} minutes in total. Work efficiently; avoid re-reading large files."""


def system_prompt(cwd: str, tmax: float, wall: float) -> str:
    return (HEADLESS_SYSTEM + "\n\n"
            + ENVIRONMENT.format(cwd=cwd, tmax=int(tmax), minutes=max(1, int(wall // 60))))


def read_key() -> str:
    path = Path(os.environ.pop("VENTRI_DS_KEY_FILE"))
    key = path.read_text().strip()
    for p in (path, path.parent):          # best effort: the agent shell must not find it later
        try:
            p.unlink() if p.is_file() else p.rmdir()
        except OSError:
            pass
    return key


async def amain(a: argparse.Namespace) -> int:
    out = Path(a.out).resolve()   # Ventri resolves relative paths against its home
    out.mkdir(parents=True, exist_ok=True)
    key = read_key()
    instruction = Path(a.instruction_file).read_text()
    base_env = str(Path(a.base_env_file).resolve()) if a.base_env_file else None
    cwd = os.getcwd()
    routes = json.loads(json.dumps(DEFAULT_ROUTES))
    routes["default"].update({"effort": a.effort, "max_tokens": a.max_tokens})
    shell_timeout = max(30.0, min(a.shell_timeout, a.wall))
    t0 = time.monotonic()
    result: dict[str, Any] = {"status": "crashed"}
    events: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=20)) as http:
        provider = DeepSeekProvider(http, api_key=key, routes=routes, max_retries=6)
        del key

        @ventri.plugin(name="provider:deepseek(harbor)", provides={"llm": ModelProvider})
        def prov(ctx: Any, config: Any) -> None:
            ctx.provide(ModelProvider, provider)

        async with Kernel() as k:
            await k.plugin(prov)
            await k.plugin(registry_plugin)
            await k.plugin(permission, {"audit": str(out / "audit.jsonl"), "rules": [],
                                        "unattended": {"ask": "allow", "irreversible": "deny"}})
            await k.plugin(core.plugin)
            await k.plugin(fs.plugin, {"roots": [cwd, "/"], "write": "ask"})
            await k.plugin(shell.plugin, {
                "cwd": cwd, "policy": "ask", "timeout": shell_timeout, "persist": True,
                "jobs_on_dispose": "keep", "hide_paths": [sys.prefix], "preview_tokens": 4000,
                "base_env_file": base_env})
            await k.plugin(session_manager, {
                "dir": str(out / "sessions"), "extract_memory": False,
                "budget": {"max_steps": a.max_steps, "max_tool_calls": a.max_steps * 2,
                           "max_tokens": 200_000_000, "max_cost_cny": a.max_cost_usd * 7.2,
                           "usd_to_cny": 7.2, "wall_s": a.wall},
                "agents": {"harbor": {
                    "system_prompt": system_prompt(cwd, shell_timeout, a.wall),
                    "tools": ["time.now", "artifact.read", "work.*", "fs.*", "shell.*"],
                    "route": "default", "mode": "headless",
                    "prune_tokens": a.prune_tokens, "prune_keep": 6}}})
            mgr = k.get(SessionManager)
            s = await mgr.open(agent="harbor", channel="harbor", headless=True)

            def sink(ev: Any) -> None:
                if ev.kind in ("tool.start", "tool.end", "notice", "error", "turn.end"):
                    events.append({"t": round(time.monotonic() - t0, 2), "kind": ev.kind,
                                   "text": (ev.text or "")[:500],
                                   "data": {k2: v for k2, v in ev.data.items() if k2 != "result"}})
            try:
                with anyio.fail_after(a.wall + a.grace):
                    r = await s.turn(instruction, sink)
                result = {"status": r.status, "reason": r.reason, "text": r.text[:4000], "steps": r.steps,
                          "tool_calls": r.tool_calls}
            except TimeoutError:
                result = {"status": "timeout"}
            except Exception as e:  # noqa: BLE001 - recorded in result.json
                result = {"status": "crashed", "error": repr(e), "tb": traceback.format_exc()[-3000:]}
            loop = s.loop
            u = loop.totals
            result.update({"session": s.id, "model_calls": loop.calls, "prompt_tokens": u.prompt_tokens,
                           "completion_tokens": u.completion_tokens, "cache_hit": u.cache_hit,
                           "cache_miss": u.cache_miss, "reasoning_tokens": u.reasoning_tokens,
                           "cost_usd_ventri": loop.cost_usd, "seconds": round(time.monotonic() - t0, 1),
                           "wall": a.wall})
            await mgr.end(s.id, extract=False)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (out / "events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events))
    print(json.dumps({k3: v for k3, v in result.items() if k3 not in ("text", "tb")}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--max-cost-usd", type=float, default=0.5)
    ap.add_argument("--shell-timeout", type=float, default=600)
    ap.add_argument("--wall", type=float, required=True, help="Ventri's own time budget (seconds)")
    ap.add_argument("--grace", type=float, default=20, help="hard stop after wall + grace")
    ap.add_argument("--prune-tokens", type=int, default=40_000)
    ap.add_argument("--base-env-file", default=None)
    return anyio.run(amain, ap.parse_args(argv), backend="asyncio")


if __name__ == "__main__":
    sys.exit(main())
