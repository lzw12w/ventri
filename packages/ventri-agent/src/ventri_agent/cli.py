"""``va`` -- the Ventri Agent command line (DESIGN.md 7).

M2 commands: ``init``, ``chat``, ``run``, ``sessions``, ``cost``, ``memory``,
``tree``, ``doctor``. ``serve`` (M4), ``propose`` / ``history`` / ``rollback`` (M3) are
reserved and exit with status 2.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio

from ventri import Kernel

from . import __version__
from .paths import home

LATER = {"serve": "M4 (local Web UI + Feishu daemon)", "propose": "M3 (evolution manager)",
         "history": "M3 (evolution ledger)", "rollback": "M3 (evolution ledger)",
         "reload": "M3 (dev-mode code reload)"}

CONFIG_TEMPLATE = """\
# Ventri Agent configuration (DESIGN.md 4.7). Edits are hot-applied by `va chat` as one transaction.
version: 1
plugins:
  - use: ventri_std.trace.jsonl
    config: {{ path: ~/.ventri/trace/ }}
  - use: {provider}
    id: ds
    config:
{provider_config}
  - use: ventri_agent.memory
    config: {{ path: ~/.ventri/memory.db }}
  - use: ventri_agent.permission
    config:
      audit: ~/.ventri/audit.jsonl
      rules: []                  # e.g. - {{ tool: "fs.write", when: {{ path: "~/notes/*" }}, action: allow }}
  - use: ventri_agent.tools.registry
  - group: tools
    plugins:
      - use: ventri_agent.tools.core
      - use: ventri_agent.tools.fs
        config: {{ roots: [~/notes, ~/.ventri/workspace], write: ask }}
      - use: ventri_agent.tools.notes
        config: {{ vault: ~/notes }}
      - use: ventri_agent.tools.shell
        config: {{ cwd: ~/.ventri/workspace, policy: ask }}
      - use: ventri_agent.tools.web
        config: {{ allow_domains: [] }}
      - use: ventri_agent.tools.memory
      - use: ventri_agent.tools.inspect
  - use: ventri_agent.sessions
    config: {{ idle_timeout: 1800, retention_days: 7 }}
agents:
  default: {{ persona: personas/default.md, tools: ["*"], route: default }}
"""

DEEPSEEK_CONFIG = """\
      {api_key}routes:
        default: {{ model: deepseek-flash,  thinking: true, effort: high }}
        plan:    {{ model: deepseek-v4-pro, thinking: true, effort: max }}
        cheap:   {{ model: deepseek-flash,  thinking: false }}"""

PERSONA = """你是 Ventri Agent，用户的个人助理，运行在用户自己的电脑上，由 DeepSeek 模型驱动。
- 默认用中文回答，简洁、具体；涉及文件和命令时给出路径。
- 你可以读写用户授权目录里的笔记和文件、抓取用户给出的网页、记住用户的偏好。
"""

DATA_POLICY = ("Data policy: conversations, tool results and memories are sent to the DeepSeek API "
               "(DESIGN.md D3). Local state lives in {home}.")


def config_text(*, fake_script: str | None = None, api_key_ref: str | None = None) -> str:
    return _config_text(fake_script, api_key_ref).replace("path: ~/.ventri/trace/", f"path: {home() / 'trace'}/")


def _config_text(fake_script: str | None, api_key_ref: str | None) -> str:
    if fake_script is not None:  # offline demo: keep every path inside the Ventri home
        cfg = f"      script_file: {json.dumps(fake_script)}\n      chunk_delay: 0.0"
        return (CONFIG_TEMPLATE.format(provider="ventri_agent.providers.fake", provider_config=cfg)
                .replace("roots: [~/notes, ~/.ventri/workspace]", "roots: [~/.ventri/workspace]")
                .replace("vault: ~/notes", "vault: ~/.ventri/notes"))
    key = f"api_key: \"{api_key_ref}\"\n      " if api_key_ref else ""
    return CONFIG_TEMPLATE.format(provider="ventri_agent.providers.deepseek",
                                  provider_config=DEEPSEEK_CONFIG.format(api_key=key))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="va", description="Ventri Agent (DeepSeek-first personal agent)")
    ap.add_argument("--version", action="version", version=f"va {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init", help="create ~/.ventri (config, persona, workspace) and store the API key")
    i.add_argument("--force", action="store_true", help="overwrite an existing ventri.yml")
    i.add_argument("--no-key", action="store_true", help="do not ask for the DeepSeek API key")
    c = sub.add_parser("chat", help="interactive chat in the terminal")
    c.add_argument("--config", type=Path, default=None)
    c.add_argument("--profile", action="append", dest="profiles", default=None)
    g = c.add_mutually_exclusive_group()
    g.add_argument("--session", "-s", help="resume (or create) this session id")
    g.add_argument("--continue", "-c", dest="cont", action="store_true", help="resume the most recent session")
    c.add_argument("--agent", "-a", default=None, help="agent preset (agents: in ventri.yml)")
    c.add_argument("--fake", metavar="SCRIPT", help="use the offline scripted model (JSON/YAML steps file)")
    c.add_argument("--show-thinking", action="store_true")
    c.add_argument("--no-watch", action="store_true", help="do not hot-apply config edits")
    r = sub.add_parser("run", help="run one task non-interactively (headless only with --headless)",
                       description="Run one task and exit. Without --headless nobody can approve anything: "
                       "calls that need approval are denied (fail closed). --headless is the explicit opt-in "
                       "to an unattended run: approvals follow permission.unattended in the config (default "
                       "deny), the system prompt is the headless one (or the preset's system_prompt), time "
                       "notes are off, and every decision is audited.")
    r.add_argument("task", nargs="?", help="task text (default: --task-file, else stdin)")
    r.add_argument("--task-file", type=Path, default=None)
    r.add_argument("--config", type=Path, default=None)
    r.add_argument("--profile", action="append", dest="profiles", default=None)
    r.add_argument("--agent", "-a", default=None, help="agent preset (agents: in ventri.yml)")
    r.add_argument("--session", "-s", default=None, help="session id (default: a new one)")
    r.add_argument("--headless", action="store_true", help="unattended mode (see above)")
    r.add_argument("--json", action="store_true", help="print the result as JSON on stdout")
    r.add_argument("--quiet", "-q", action="store_true", help="no progress on stderr")
    r.add_argument("--remember", action="store_true", help="extract long-term memories when the run ends")
    s = sub.add_parser("sessions", help="list sessions")
    s.add_argument("--json", action="store_true")
    co = sub.add_parser("cost", help="daily cost / cache report from the session logs")
    co.add_argument("--days", type=int, default=14)
    co.add_argument("--json", action="store_true")
    m = sub.add_parser("memory", help="long-term memory: list/pending/search/confirm/forget/export/import")
    m.add_argument("action", choices=["list", "pending", "search", "confirm", "forget", "export", "import"])
    m.add_argument("arg", nargs="?")
    m.add_argument("--out", "-o", type=Path)
    t = sub.add_parser("tree", help="plugin tree the configuration produces (dry run)")
    t.add_argument("--config", type=Path, default=None)
    d = sub.add_parser("doctor", help="check the environment and configuration")
    d.add_argument("--config", type=Path, default=None)
    for name, when in LATER.items():
        sub.add_parser(name, help=f"(not in M2: {when})")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd in LATER:
        print(f"va {args.cmd}: not available in 0.2 (planned for {LATER[args.cmd]})", file=sys.stderr)
        return 2
    fn = {"init": cmd_init, "chat": cmd_chat, "run": cmd_run, "sessions": cmd_sessions, "cost": cmd_cost,
          "memory": cmd_memory, "tree": cmd_tree, "doctor": cmd_doctor}[args.cmd]
    from ventri_std.config import ConfigError

    try:
        return int(fn(args) or 0)
    except ConfigError as e:
        print("configuration error:", file=sys.stderr)
        for err in e.errors:
            print(f"  - {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


# --------------------------------------------------------------------- init
def cmd_init(args: argparse.Namespace) -> int:
    h = home()
    for sub in ("profiles", "personas", "skills", "plugins", "generated", "sessions", "trace", "workspace"):
        (h / sub).mkdir(parents=True, exist_ok=True)
    cfg = h / "ventri.yml"
    key_ref = None
    if not args.no_key and sys.platform == "darwin" and sys.stdin.isatty():
        key = getpass.getpass("DeepSeek API key (stored in the macOS keychain, service 'ventri'; empty to skip): ")
        if key.strip():
            subprocess.run(["/usr/bin/security", "add-generic-password", "-U", "-s", "ventri", "-a", "deepseek",
                            "-w", key.strip()], check=True, capture_output=True)
            key_ref = "${secret:deepseek}"
    if cfg.exists() and not args.force:
        print(f"{cfg} exists (use --force to overwrite)")
    else:
        cfg.write_text(config_text(api_key_ref=key_ref), encoding="utf-8")
        print(f"wrote {cfg}")
    persona = h / "personas" / "default.md"
    if not persona.exists():
        persona.write_text(PERSONA, encoding="utf-8")
    gi = h / ".gitignore"
    if not gi.exists():
        gi.write_text("sessions/\nmemory.db*\naudit.jsonl\ntrace/\nworkspace/\n", encoding="utf-8")
    if shutil.which("git") and not (h / ".git").exists():
        subprocess.run(["git", "init", "-q", str(h)], check=False)
    if key_ref is None:
        print("API key: set DEEPSEEK_API_KEY (or VENTRI_SECRET_DEEPSEEK), or on macOS rerun `va init` "
              "to store it in the keychain and reference it as ${secret:deepseek}.")
    print(DATA_POLICY.format(home=h))
    print("next: va chat")
    return 0


# --------------------------------------------------------------------- chat
def _config_path(p: Path | None) -> Path:
    return p.expanduser() if p else home() / "ventri.yml"


def cmd_chat(args: argparse.Namespace) -> int:
    return anyio.run(chat, args, backend="asyncio")


async def chat(args: argparse.Namespace, *, terminal: Any = None) -> int:
    from ventri_std.config import Loader, load_document

    from .channels.cli import CliChannel
    from .channels.cli import plugin as cli_plugin

    cfg_path = _config_path(args.config)
    tmp: Path | None = None
    if args.fake:
        tmp = home() / ".va-fake.yml"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(config_text(fake_script=str(Path(args.fake).expanduser().resolve())), encoding="utf-8")
        cfg_path = tmp
    elif not cfg_path.exists():
        print(f"no configuration at {cfg_path}; run `va init` first (or `va chat --fake script.json`)",
              file=sys.stderr)
        return 2
    load_document(cfg_path, profiles=args.profiles)  # fail fast with a ConfigError
    async with Kernel() as app:
        loader = Loader(app.fiber, cfg_path, profiles=args.profiles)
        app.provide("config.loader", loader)
        if terminal is not None:
            app.provide("cli.terminal", terminal)
        res = await loader.apply(reason="va chat")
        if not res.ok:
            print(str(res), file=sys.stderr)
            return 1
        ch = app.get(CliChannel, None)
        if ch is None:
            await app.plugin(cli_plugin, {"session": args.session, "agent": args.agent,
                                          "resume_last": args.cont, "show_thinking": args.show_thinking})
            ch = app.get(CliChannel)

        def report(r: Any) -> None:
            ch.term.write(f"\n  · config {'applied' if getattr(r, 'ok', False) else 'NOT applied'}: "
                          f"{str(r).splitlines()[0] if str(r) else ''}\n")
        async with anyio.create_task_group() as tg:
            if not args.no_watch and not args.fake:
                tg.start_soon(lambda: loader.watch(on_result=report))
            await ch.done.wait()
            tg.cancel_scope.cancel()
    if tmp is not None:
        tmp.unlink(missing_ok=True)
    return 0


# ---------------------------------------------------------------------- run
def cmd_run(args: argparse.Namespace) -> int:
    if args.task is not None:
        task = args.task
    elif args.task_file is not None:
        task = args.task_file.expanduser().read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        task = sys.stdin.read()
    else:
        print("va run: give the task as an argument, with --task-file, or on stdin", file=sys.stderr)
        return 2
    if not task.strip():
        print("va run: empty task", file=sys.stderr)
        return 2
    return anyio.run(run_task, args, task, backend="asyncio")


async def run_task(args: argparse.Namespace, task: str) -> int:
    """One non-interactive turn. No channel is bound, so an ``ask`` is denied
    unless the session is headless and ``permission.unattended`` allows it."""
    import time

    from ventri_std.config import Loader, load_document

    from .sessions import SessionError, SessionManager

    cfg_path = _config_path(args.config)
    if not cfg_path.exists():
        print(f"no configuration at {cfg_path}; run `va init` first", file=sys.stderr)
        return 2
    load_document(cfg_path, profiles=args.profiles)
    t0 = time.monotonic()

    def say(line: str) -> None:
        if not args.quiet:
            print(line, file=sys.stderr, flush=True)

    async def sink(ev: Any) -> None:
        if ev.kind == "tool.start":
            say(f"  -> {ev.text[:200]}")
        elif ev.kind == "tool.end":
            say(f"  <- {(ev.text.splitlines() or [''])[0][:200]}")
        elif ev.kind in ("notice", "error"):
            say(f"  !! {ev.text}")

    async with Kernel() as app:
        loader = Loader(app.fiber, cfg_path, profiles=args.profiles)
        app.provide("config.loader", loader)
        res = await loader.apply(reason="va run")
        if not res.ok:
            print(str(res), file=sys.stderr)
            return 1
        mgr = app.get(SessionManager, None)
        if mgr is None:
            print("the configuration has no ventri_agent.sessions plugin", file=sys.stderr)
            return 2
        try:
            s = await mgr.open(args.session, agent=args.agent, channel="run", headless=args.headless)
        except SessionError as e:
            print(f"va run: {e}", file=sys.stderr)
            return 2
        say(f"session {s.id} (agent {s.info.agent.name}{', headless' if args.headless else ''})")
        try:
            r = await s.turn(task, sink)
        finally:
            await s.end(extract=args.remember)
    out = {"session": s.id, "status": r.status, "reason": r.reason, "text": r.text, "steps": r.steps,
           "tool_calls": r.tool_calls, "prompt_tokens": r.usage.prompt_tokens, "cache_hit": r.usage.cache_hit,
           "completion_tokens": r.usage.completion_tokens, "cost_usd": round(r.cost_usd, 6),
           "seconds": round(time.monotonic() - t0, 1), "headless": bool(args.headless)}
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(r.text)
        say(f"[{r.status}{': ' + r.reason if r.reason else ''}] {r.steps} steps, {r.tool_calls} tool calls, "
            f"${r.cost_usd:.4f}, cache {r.usage.hit_rate:.0%}")
    return 0 if r.status == "ok" else 1


# ---------------------------------------------------------------- reports
def cmd_sessions(args: argparse.Namespace) -> int:
    from .providers.pricing import BEIJING
    from .session import list_sessions

    items = list_sessions(home() / "sessions")
    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return 0
    for it in items:
        when = datetime.fromtimestamp(it["updated"], BEIJING).strftime("%m-%d %H:%M")
        print(f"{it['id']}  {when}  {it['state']:<9} {it['turns']:>3} turns  cache {it['hit_rate']:>4.0%}  "
              f"${it['cost_usd']:.4f}  {it['title']}")
    if not items:
        print("(no sessions)")
    return 0


def daily_report(records: list[dict[str, Any]], days: int = 14, rate: float = 7.2) -> list[dict[str, Any]]:
    """Aggregate ``usage`` records per Beijing-time day."""
    from .messages import Usage
    from .providers.pricing import BEIJING

    agg: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "usage": Usage(), "cost_usd": 0.0,
                                                           "peak_calls": 0, "sessions": set()})
    for r in records:
        day = datetime.fromtimestamp(float(r["ts"]), BEIJING).strftime("%Y-%m-%d")
        a = agg[day]
        a["calls"] += 1
        a["usage"] = a["usage"] + Usage.from_json(r.get("usage") or {})
        a["cost_usd"] += float(r.get("cost_usd") or 0)
        a["peak_calls"] += bool(r.get("peak"))
        a["sessions"].add(r.get("session"))
    out = []
    for day in sorted(agg)[-days:]:
        a = agg[day]
        u = a["usage"]
        out.append({"day": day, "sessions": len(a["sessions"]), "calls": a["calls"],
                    "prompt_tokens": u.prompt_tokens, "cache_hit": u.cache_hit, "cache_miss": u.cache_miss,
                    "hit_rate": round(u.hit_rate, 4), "completion_tokens": u.completion_tokens,
                    "reasoning_tokens": u.reasoning_tokens, "peak_calls": a["peak_calls"],
                    "cost_usd": round(a["cost_usd"], 6), "cost_cny": round(a["cost_usd"] * rate, 4)})
    return out


def cmd_cost(args: argparse.Namespace) -> int:
    from .session import read_usage

    rows = daily_report(read_usage(home() / "sessions"), args.days)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'day':<11}{'sess':>5}{'calls':>6}{'prompt':>10}{'cache':>7}{'output':>9}{'peak':>6}"
          f"{'USD':>10}{'CNY':>9}")
    for r in rows:
        print(f"{r['day']:<11}{r['sessions']:>5}{r['calls']:>6}{r['prompt_tokens']:>10}{r['hit_rate']:>7.1%}"
              f"{r['completion_tokens']:>9}{r['peak_calls']:>6}{r['cost_usd']:>10.4f}{r['cost_cny']:>9.3f}")
    if not rows:
        print("(no usage recorded)")
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    from .memory import LongTermMemory

    mem = LongTermMemory(home() / "memory.db")
    try:
        a = args.action
        if a in ("list", "pending"):
            for m in mem.list(status="active" if a == "list" else "pending"):
                print(m.line())
        elif a == "search":
            for m in mem.search(args.arg or ""):
                print(m.line())
        elif a in ("confirm", "forget"):
            n = int((args.arg or "0").lstrip("#"))
            ok = mem.confirm(n) if a == "confirm" else mem.forget(n)
            print("done" if ok else f"no such memory #{n}")
            return 0 if ok else 1
        elif a == "export":
            text = mem.export_markdown()
            if args.out:
                args.out.write_text(text, encoding="utf-8")
                print(f"wrote {args.out}")
            else:
                print(text)
        elif a == "import":
            n = mem.import_markdown(Path(args.arg or "").read_text(encoding="utf-8"))
            print(f"imported {n} new memories")
    finally:
        mem.close()
    return 0


def cmd_tree(args: argparse.Namespace) -> int:
    from ventri_std import cli as vcli

    return vcli.main(["tree", str(_config_path(args.config))])


def cmd_doctor(args: argparse.Namespace) -> int:
    import sqlite3

    from ventri_std import cli as vcli

    problems = 0
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
        print("  ok    SQLite FTS5 with trigram tokenizer")
    except sqlite3.Error as e:
        problems += 1
        print(f"  FAIL  SQLite FTS5 trigram unavailable: {e}")
    has_key = bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("VENTRI_SECRET_DEEPSEEK"))
    print(("  ok    " if has_key else "  warn  ") + "DeepSeek API key in the environment"
          + ("" if has_key else " (fine if ventri.yml references the keychain)"))
    rc = vcli.main(["doctor", str(_config_path(args.config))])
    return 1 if problems or rc else 0


if __name__ == "__main__":
    sys.exit(main())
