"""Personal task eval harness (DESIGN.md 8, M2: 30 tasks, pass rate >= 80%).

    uv run python -m evals.agent.run                 # real DeepSeek (needs DEEPSEEK_API_KEY)
    uv run python -m evals.agent.run --only n01,f02  # a subset
    uv run python -m evals.agent.run --json out.json

Every task runs in a fresh fixture home (notes vault, workspace, mock web pages)
and a fresh session of the real agent runtime (ventri_agent plugins). The
simulated user approves only the tool globs listed in ``approve``; any other
approval request is denied. A task passes when all its checks pass.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import anyio
import httpx
import yaml

import ventri
from ventri import Kernel
from ventri_agent.memory import LongTermMemory, memory_sqlite
from ventri_agent.permission import ApprovalBroker, ApprovalRequest, permission
from ventri_agent.providers.base import ModelProvider
from ventri_agent.providers.pricing import PeakSchedule
from ventri_agent.sessions import SessionManager, session_manager
from ventri_agent.tools import core, fs, inspect, notes, shell
from ventri_agent.tools import memory as memory_tools
from ventri_agent.tools import web as web_tools
from ventri_agent.tools.registry import ToolRegistry
from ventri_agent.tools.registry import plugin as registry_plugin

HERE = Path(__file__).parent
WEEKDAYS = {0: ["monday", "星期一", "周一"], 1: ["tuesday", "星期二", "周二"], 2: ["wednesday", "星期三", "周三"],
            3: ["thursday", "星期四", "周四"], 4: ["friday", "星期五", "周五"], 5: ["saturday", "星期六", "周六"],
            6: ["sunday", "星期日", "星期天", "周日"]}


@dataclass
class TaskResult:
    id: str
    category: str
    passed: bool
    failures: list[str]
    answer: str
    tools: list[str]
    approvals: list[str]
    calls: int = 0
    prompt_tokens: int = 0
    cache_hit: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    status: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def load_tasks(path: Path = HERE / "tasks.yaml") -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_fixture(base: Path, fixture: dict[str, Any]) -> None:
    for rel, content in (fixture.get("files") or {}).items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    (base / "notes").mkdir(exist_ok=True)
    (base / "workspace").mkdir(exist_ok=True)


def persona(base: Path) -> str:
    return (f"You are Ventri Agent, Jeff's personal assistant on his own computer.\n"
            f"Jeff's files: notes vault at {base}/notes (notes.* tools, names relative to the vault), "
            f"workspace at {base}/workspace. fs.* paths may be absolute or relative to {base}.\n"
            "Be concise. Use tools to look things up instead of guessing.")


def page_transport(pages: dict[str, str]) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if url in pages:
            return httpx.Response(200, text=pages[url], headers={"content-type": "text/html; charset=utf-8"})
        return httpx.Response(404, text="not found")
    return httpx.MockTransport(handler)


async def mock_dns(host: str, port: int) -> list[str]:
    """The mock pages live on fictional domains; give them a public (example.com's)
    address so web.fetch's SSRF pre-flight check passes without real DNS."""
    return ["93.184.216.34"]


def check(spec: dict[str, Any], *, answer: str, base: Path, tools: list[str], executed: list[str],
          mem: LongTermMemory) -> str | None:
    """None when the check passes, else a failure description."""
    (kind, arg), = spec.items()
    low = answer.lower()
    if kind == "answer_any":
        return None if any(s.lower() in low for s in arg) else f"answer lacks any of {arg}"
    if kind == "answer_all":
        missing = [s for s in arg if s.lower() not in low]
        return None if not missing else f"answer lacks {missing}"
    if kind in ("file_contains", "file_not_contains"):
        p = base / arg["path"]
        if not p.exists():
            return f"{arg['path']} does not exist" if kind == "file_contains" else None
        text = p.read_text(encoding="utf-8")
        if kind == "file_contains":
            if "all" in arg and not all(s in text for s in arg["all"]):
                return f"{arg['path']} lacks {[s for s in arg['all'] if s not in text]}"
            if "any" in arg and not any(s in text for s in arg["any"]):
                return f"{arg['path']} lacks any of {arg['any']}"
            return None
        bad = [s for s in arg["any"] if s in text]
        return f"{arg['path']} still contains {bad}" if bad else None
    if kind == "file_absent":
        return f"{arg} exists" if (base / arg).exists() else None
    if kind == "json_equals":
        try:
            data = json.loads((base / arg["path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return f"{arg['path']}: {e}"
        diff = {k: data.get(k) for k, v in arg["values"].items() if data.get(k) != v}
        return f"{arg['path']} differs: {diff}" if diff else None
    if kind == "tool_called":
        return None if arg in tools else f"{arg} was not called (called: {tools})"
    if kind == "not_executed":
        bad = [t for t in executed if t in arg]
        return f"executed {bad}" if bad else None
    if kind == "memory_any":
        texts = " ".join(m.text for m in mem.list(status=None)).lower()
        return None if any(s.lower() in texts for s in arg) else f"no memory with any of {arg}"
    if kind == "answer_weekday":
        wd = datetime.now(ZoneInfo(arg)).weekday()
        return None if any(s in low for s in WEEKDAYS[wd]) else f"answer lacks today's weekday {WEEKDAYS[wd][0]}"
    if kind == "answer_peak":
        peak = PeakSchedule().is_peak(datetime.now(UTC))
        yes = any(s in low for s in ("yes", "是的", "是峰", "高峰", "处于峰")) and "not peak" not in low
        no = any(s in low for s in ("no,", "no.", "not peak", "off-peak", "谷时", "不是", "非高峰", "低谷"))
        if peak:
            return None if yes and not low.startswith("no") else f"expected 'peak' (it is peak), got: {answer[:120]}"
        return None if no else f"expected 'off-peak', got: {answer[:120]}"
    return f"unknown check {kind}"


async def run_task(task: dict[str, Any], fixture: dict[str, Any], provider: Any, *, timeout: float = 240.0
                   ) -> TaskResult:
    with tempfile.TemporaryDirectory(prefix=f"va-eval-{task['id']}-") as d:
        base = Path(d).resolve()
        build_fixture(base, fixture)
        approve = list(task.get("approve") or [])
        approvals: list[str] = []
        transport = page_transport(fixture.get("pages") or {})

        @ventri.plugin(name="provider:eval", provides={"llm": ModelProvider})
        def prov(ctx: Any, config: Any) -> None:
            ctx.provide(ModelProvider, provider)

        @ventri.plugin(name="tool:web(mock)")
        async def web_mock(ctx: Any, config: Any, registry: ToolRegistry) -> None:
            http = await ctx.enter(httpx.AsyncClient(transport=transport))
            registry.register(ctx, web_tools.make_tool(web_tools.WebConfig(), http, resolver=mock_dns))

        t0 = time.monotonic()
        async with Kernel() as k:
            await k.plugin(prov)
            await k.plugin(registry_plugin)
            await k.plugin(permission, {"audit": str(base / "audit.jsonl"), "approval_timeout": 30})
            await k.plugin(memory_sqlite, {"path": str(base / "memory.db")})
            mem = k.get(LongTermMemory)
            for m in fixture.get("memories") or []:
                mem.add(m["text"], m.get("kind", "fact"), confirmed=True)
            await k.plugin(core.plugin)
            await k.plugin(fs.plugin, {"roots": [str(base)], "write": "ask"})
            await k.plugin(notes.plugin, {"vault": str(base / "notes")})
            await k.plugin(shell.plugin, {"cwd": str(base / "workspace"), "timeout": 30})
            await k.plugin(web_mock)
            await k.plugin(memory_tools.plugin)
            await k.plugin(inspect.plugin)
            await k.plugin(session_manager, {
                "dir": str(base / "sessions"), "extract_memory": False,
                "agents": {"eval": {"persona": persona(base), "tools": ["*"], "route": "default"}}})
            mgr = k.get(SessionManager)
            s = await mgr.open(agent="eval", channel="eval")

            async def ask(req: ApprovalRequest) -> Any:
                ok = any(fnmatch.fnmatchcase(req.tool, p) for p in approve)
                approvals.append(f"{req.tool}:{'once' if ok else 'deny'}")
                return "once" if ok else "deny"
            k.get(ApprovalBroker).bind(s.id, "eval", ask)
            tools: list[str] = []
            executed: list[str] = []

            def sink(ev: Any) -> None:
                if ev.kind == "tool.start":
                    tools.append(ev.data["tool"])
                if ev.kind == "tool.end" and not ev.data.get("denied") and ev.data.get("ok"):
                    executed.append(ev.data["tool"])
            try:
                with anyio.fail_after(timeout):
                    r = await s.turn(task["prompt"], sink)
                answer, status = r.text, r.status
            except TimeoutError:
                answer, status = "", "timeout"
                r = None
            failures = [f for c in task.get("checks") or []
                        if (f := check(c, answer=answer, base=base, tools=tools, executed=executed, mem=mem))]
            if status != "ok":
                failures.insert(0, f"turn status {status}: {getattr(r, 'reason', '')}")
            loop = s.loop
            res = TaskResult(task["id"], task.get("category", ""), not failures, failures, answer, tools, approvals,
                             loop.calls, loop.totals.prompt_tokens, loop.totals.cache_hit, loop.cost_usd,
                             round(time.monotonic() - t0, 1), status)
            await mgr.end(s.id, extract=False)
            return res


async def run_all(tasks: list[dict[str, Any]], fixture: dict[str, Any], provider: Any,
                  progress: Any = None) -> list[TaskResult]:
    out = []
    for t in tasks:
        r = await run_task(t, fixture, provider)
        out.append(r)
        if progress:
            progress(r)
    return out


def summary(results: list[TaskResult]) -> dict[str, Any]:
    n = len(results)
    passed = sum(r.passed for r in results)
    by_cat: dict[str, list[int]] = {}
    for r in results:
        c = by_cat.setdefault(r.category, [0, 0])
        c[0] += r.passed
        c[1] += 1
    prompt = sum(r.prompt_tokens for r in results)
    return {"tasks": n, "passed": passed, "pass_rate": round(passed / n, 3) if n else 0.0,
            "by_category": {k: f"{a}/{b}" for k, (a, b) in by_cat.items()},
            "model_calls": sum(r.calls for r in results), "prompt_tokens": prompt,
            "cache_hit_rate": round(sum(r.cache_hit for r in results) / prompt, 3) if prompt else 0.0,
            "cost_usd": round(sum(r.cost_usd for r in results), 5),
            "failed": {r.id: r.failures for r in results if not r.passed}}


async def amain(args: argparse.Namespace) -> int:
    data = load_tasks(Path(args.tasks))
    tasks = data["tasks"]
    if args.only:
        keep = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in keep]
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("DEEPSEEK_API_KEY is not set", file=sys.stderr)
        return 2
    from ventri_agent.providers.deepseek import DEFAULT_ROUTES, DeepSeekProvider
    routes = json.loads(json.dumps(DEFAULT_ROUTES))
    routes["default"]["effort"] = args.effort
    routes["default"]["max_tokens"] = args.max_tokens
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15)) as http:
        provider = DeepSeekProvider(http, api_key=key, routes=routes)

        def progress(r: TaskResult) -> None:
            mark = "PASS" if r.passed else "FAIL"
            print(f"{mark} {r.id:<4} {r.category:<9} {r.calls} calls {r.seconds:>5.1f}s ${r.cost_usd:.4f} "
                  f"tools={','.join(r.tools) or '-'}" + ("" if r.passed else f"  <- {'; '.join(r.failures)}"),
                  flush=True)
        results = await run_all(tasks, data["fixture"], provider, progress)
    s = summary(results)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": s, "results": [r.__dict__ for r in results]},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if s["pass_rate"] >= args.threshold else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Ventri Agent personal task eval (M2)")
    ap.add_argument("--tasks", default=str(HERE / "tasks.yaml"))
    ap.add_argument("--only", default="")
    ap.add_argument("--effort", default="high", choices=["low", "high", "max"])
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--json", default="")
    return anyio.run(amain, ap.parse_args(), backend="asyncio")


if __name__ == "__main__":
    sys.exit(main())
