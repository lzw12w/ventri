"""CLI channel (DESIGN.md 5.8): streaming output, folded thinking, inline
approvals and slash commands (``/memory /tree /cost /think`` ...).

The channel is a plugin (``use: ventri_agent.channels.cli``, ``exclusive``:
it owns the terminal) whose REPL runs as a task of its fiber. It looks the
``SessionManager`` / ``ApprovalBroker`` up *lazily* instead of injecting them,
so a hot config change that restarts the agent runtime does not restart the
terminal; the session is transparently resumed from its log instead.
Approval prompts are answered from terminal input only (the broker mints the
decision token), so nothing the model writes can approve anything.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

import anyio
import anyio.lowlevel
from pydantic import BaseModel

import ventri

from ..loop import TurnEvent, TurnResult
from ..memory import LongTermMemory
from ..messages import Usage
from ..permission import ApprovalBroker, ApprovalRequest, Choice
from ..sessions import Session, SessionError, SessionManager

HELP = """commands:
  /help                     this help
  /think [low|high|max|off|on|show|hide]
                            effort of the main route; 'max' plans the next request with the plan
                            route (Pro, max effort) in a separate sub-call; off/on: thinking mode;
                            show/hide: display the reasoning stream
  /cost                     tokens, cache hit rate and cost of this session
  /tree                     live plugin tree
  /memory [list|pending|search Q|confirm ID|forget ID]
  /epoch                    start a new context epoch (adopt the current tool set)
  /compact                  summarise older turns now
  /retry                    re-run the model after an error
  /sessions                 recent sessions
  /end                      end this session (extract memories) and start a new one
  /suspend                  leave without ending (resume with `va chat --continue`)
  /exit                     end the session and quit (also Ctrl-D)"""


class Terminal(Protocol):
    async def readline(self, prompt: str) -> str | None: ...
    def write(self, text: str) -> None: ...


class StdTerminal:
    """stdin/stdout; ``input`` runs in a worker thread so the event loop keeps going."""

    def __init__(self) -> None:
        self.tty = sys.stdin.isatty() and sys.stdout.isatty()
        if self.tty:
            try:
                import readline  # noqa: F401 - line editing for input()
            except ImportError:  # pragma: no cover
                pass

    async def readline(self, prompt: str) -> str | None:
        def read() -> str | None:
            try:
                return input(prompt)
            except EOFError:
                return None
        line = await anyio.to_thread.run_sync(read, abandon_on_cancel=True)
        if line is not None and not self.tty:
            sys.stdout.write(line + "\n")  # echo piped input (input() already printed the prompt)
        return line

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()


@dataclass
class ScriptedTerminal:
    """For tests: answers prompts from ``inputs``; collects output."""

    inputs: list[str]
    out: list[str] = field(default_factory=list)

    async def readline(self, prompt: str) -> str | None:
        await anyio.lowlevel.checkpoint()
        if not self.inputs:
            return None
        line = self.inputs.pop(0)
        self.out.append(prompt + line + "\n")
        return line

    def write(self, text: str) -> None:
        self.out.append(text)

    @property
    def text(self) -> str:
        return "".join(self.out)


def fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def usage_line(u: Usage, cost_usd: float, rate: float = 7.2) -> str:
    return (f"in {fmt_tokens(u.prompt_tokens)} (cache {u.hit_rate:.0%}) · out {fmt_tokens(u.completion_tokens)}"
            f" · ${cost_usd:.4f} / ¥{cost_usd * rate:.3f}")


class CliConfig(BaseModel):
    session: str | None = None
    agent: str | None = None
    resume_last: bool = False
    show_thinking: bool = False
    wait_for_runtime: float = 30.0


class CliChannel:
    name = "cli"

    def __init__(self, ctx: Any, cfg: CliConfig, term: Terminal) -> None:
        self.ctx = ctx
        self.cfg = cfg
        self.term = term
        self.show_thinking = cfg.show_thinking
        self.session: Session | None = None
        self.session_id = cfg.session
        self.done = anyio.Event()
        self._unbind: Any = None
        self._plan_next = False
        self._in_reasoning = False
        self._reasoning_chars = 0
        self._line_open = False
        self.results: list[TurnResult] = []

    # -------------------------------------------------------------- lookup
    async def manager(self) -> SessionManager:
        with anyio.fail_after(self.cfg.wait_for_runtime):
            while True:
                m = self.ctx.get(SessionManager, None)
                if m is not None:
                    return m
                await anyio.sleep(0.05)

    async def ensure_session(self) -> Session:
        s = self.session
        if s is not None and s.alive:
            return s
        mgr = await self.manager()
        sid = self.session_id
        if sid is None and self.cfg.resume_last:
            sid = mgr.last_id()
        resumed = sid is not None and (mgr.dir / f"{sid}.jsonl").exists()
        s = await mgr.open(sid, agent=self.cfg.agent, channel=self.name)
        self.session, self.session_id = s, s.id
        broker = self.ctx.get(ApprovalBroker, None)
        if self._unbind:
            self._unbind()
        self._unbind = broker.bind(s.id, self.name, self.ask) if broker else None
        if broker is None:
            self.term.write("  ! no permission engine loaded: every consequential tool call will be refused\n")
        loop = s.loop
        route = loop.route
        hist = len(loop.builder.history)
        self.term.write(f"── session {s.id} ({'resumed, ' + str(hist) + ' messages' if resumed else 'new'}) · "
                        f"agent {s.info.agent.name} · {route.model}"
                        f"{' thinking ' + (route.effort or 'default') if route.thinking is not False else ''}"
                        f" · {len(loop.builder.epoch.tool_names)} tools ──\n")
        return s

    # -------------------------------------------------------------- render
    def _close_line(self) -> None:
        if self._in_reasoning and not self.show_thinking:
            cr = "\r" if getattr(self.term, "tty", False) else ""
            self.term.write(f"{cr}  ▸ thought ({self._reasoning_chars} chars)\n")
        elif self._in_reasoning or self._line_open:
            self.term.write("\n")
        self._in_reasoning = False
        self._line_open = False

    async def render(self, ev: TurnEvent) -> None:
        k = ev.kind
        if k == "reasoning":
            if not self._in_reasoning:
                self._close_line()
                self._in_reasoning = True
                self._reasoning_chars = 0
                if self.show_thinking:
                    self.term.write("  ▸ ")
                elif getattr(self.term, "tty", False):
                    self.term.write("  ▸ thinking…")
            self._reasoning_chars += len(ev.text)
            if self.show_thinking:
                self.term.write(ev.text.replace("\n", "\n    "))
        elif k == "content":
            if self._in_reasoning:
                self._close_line()
            self._line_open = True
            self.term.write(ev.text)
        elif k == "tool.start":
            self._close_line()
            self.term.write(f"  ⚙ {ev.text}\n")
        elif k == "tool.end":
            mark = "✓" if ev.data.get("ok") else "✗"
            first = " ".join(ev.text.split())
            self.term.write(f"    {mark} {first[:160]}\n")
        elif k == "notice":
            self._close_line()
            self.term.write(f"  · {ev.text}\n")
        elif k == "error":
            self._close_line()
            self.term.write(f"  ! {ev.text}\n")
        elif k == "turn.end":
            self._close_line()
            r: TurnResult = ev.data["result"]
            rate = self._rate()
            tail = f" · {r.status}: {r.reason}" if r.status != "ok" else ""
            self.term.write(f"  [turn {r.n} · {r.steps} steps · {r.tool_calls} tools · "
                            f"{usage_line(r.usage, r.cost_usd, rate)}{tail}]\n")

    def _rate(self) -> float:
        s = self.session
        return s.loop.budget.limits.usd_to_cny if s and s.alive else 7.2

    # ------------------------------------------------------------ approvals
    async def ask(self, req: ApprovalRequest) -> Choice:
        self._close_line()
        opts = "[y] allow once  " + ("[s] allow for this session  " if req.grantable else "") + "[n] deny"
        self.term.write(f"  [approval] {req.summary}  ({req.risk})\n    args: {req.args_preview}\n    {opts}\n")
        while True:
            line = await self.term.readline("    approve? ")
            ans = (line or "n").strip().lower()
            if ans in ("y", "yes", "1", "once"):
                return "once"
            if ans in ("s", "session") and req.grantable:
                return "session"
            if ans in ("n", "no", "", "deny") or line is None:
                return "deny"
            self.term.write("    please answer y / " + ("s / " if req.grantable else "") + "n\n")

    # ----------------------------------------------------------------- REPL
    async def run(self) -> None:
        try:
            await self.ensure_session()
            await self._show_new_memories()
            while True:
                line = await self.term.readline("> ")
                if line is None:
                    await self._end(quit=True)
                    return
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    if await self.command(line) == "quit":
                        return
                    continue
                await self._turn(line)
        except SessionError as e:
            self.term.write(f"error: {e}\n")
        finally:
            if self._unbind:
                self._unbind()
            self.done.set()

    async def _turn(self, text: str) -> None:
        s = await self.ensure_session()
        plan, self._plan_next = self._plan_next, False
        try:
            r = await s.turn(text, self.render, plan=plan)
            self.results.append(r)
            if r.status == "error":
                self.term.write("  (the session is intact; /retry to try again)\n")
        except SessionError:
            s = await self.ensure_session()  # runtime restarted under us: resume and retry once
            self.results.append(await s.retry(self.render) if s.loop.builder.history else
                                await s.turn(text, self.render, plan=plan))

    async def _end(self, *, quit: bool) -> None:
        s = self.session
        if s is None or not s.alive:
            return
        self._close_line()
        loop = s.loop
        self.term.write(f"── ending session {s.id}: {loop.calls} model calls this run · "
                        f"{usage_line(loop.totals, loop.cost_usd, self._rate())} (session total)\n")
        created = await s.end()
        for m in created:
            self.term.write(f"  remembered: {m.line()}\n")
        self.session = None
        if not quit:
            self.session_id = None

    async def _show_new_memories(self) -> None:
        mem = self.ctx.get(LongTermMemory, None)
        if mem is None:
            return
        pend = mem.list(status="pending")
        if pend:
            self.term.write(f"  · {len(pend)} memories await confirmation: /memory pending\n")

    # ------------------------------------------------------------- commands
    async def command(self, line: str) -> str | None:
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        s = self.session
        if cmd in ("exit", "quit", "q"):
            await self._end(quit=True)
            return "quit"
        if cmd == "suspend":
            if s is not None and s.alive:
                await s.suspend()
                self.term.write(f"suspended {s.id}; resume with: va chat --session {s.id}\n")
            return "quit"
        if cmd == "help":
            self.term.write(HELP + "\n")
            return None
        if cmd == "end":
            await self._end(quit=False)
            await self.ensure_session()
            return None
        s = await self.ensure_session()
        loop = s.loop
        if cmd == "think":
            if arg in ("low", "high"):
                loop.effort, loop.thinking = arg, True
                self.term.write(f"thinking effort: {arg}\n")
            elif arg == "max":
                self._plan_next = True
                self.term.write("next message is planned first by the plan route (separate sub-call)\n")
            elif arg in ("off", "on"):
                loop.thinking = arg == "on"
                self.term.write(f"thinking {arg}\n")
            elif arg in ("show", "hide"):
                self.show_thinking = arg == "show"
                self.term.write(f"reasoning display: {arg}\n")
            else:
                r = loop.route
                self.term.write(f"route {r.name}: {r.model} thinking={r.thinking} effort={r.effort}\n")
        elif cmd == "cost":
            u = loop.totals
            self.term.write(f"session {s.id}: {loop.turns} turns · prompt {fmt_tokens(u.prompt_tokens)} "
                            f"(hit {fmt_tokens(u.cache_hit)}, miss {fmt_tokens(u.cache_miss)}, "
                            f"rate {u.hit_rate:.1%}) · output {fmt_tokens(u.completion_tokens)} "
                            f"(reasoning {fmt_tokens(u.reasoning_tokens)}) · ${loop.cost_usd:.4f} / "
                            f"¥{loop.cost_usd * self._rate():.3f}\n")
        elif cmd == "tree":
            self.term.write(self.ctx.kernel.tree() + "\n")
        elif cmd == "memory":
            self._memory(arg)
        elif cmd == "epoch":
            e = loop.builder.new_epoch(refresh_memory=True)
            self.term.write(f"epoch {e.n}: {len(e.tool_names)} tools ({', '.join(e.tool_names)})\n")
        elif cmd == "compact":
            ok = await loop.compact(self.render)
            self.term.write("compacted\n" if ok else "nothing to compact yet\n")
        elif cmd == "retry":
            self.results.append(await s.retry(self.render))
        elif cmd == "sessions":
            for it in (await self.manager()).list()[:15]:
                self.term.write(f"  {it['id']}  {it['state']:<9} {it['turns']:>3} turns  "
                                f"${it['cost_usd']:.4f}  {it['title']}\n")
        else:
            self.term.write(f"unknown command /{cmd} (try /help)\n")
        return None

    def _memory(self, arg: str) -> None:
        mem = self.ctx.get(LongTermMemory, None)
        if mem is None:
            self.term.write("no long-term memory plugin loaded\n")
            return
        sub, _, rest = arg.partition(" ")
        if sub in ("", "list"):
            items = mem.list()
            self.term.write("\n".join(m.line() for m in items) + "\n" if items else "(no memories)\n")
        elif sub == "pending":
            items = mem.list(status="pending")
            self.term.write("\n".join(m.line() for m in items) + "\n" if items else "(nothing pending)\n")
        elif sub == "search":
            self.term.write("\n".join(m.line() for m in mem.search(rest)) + "\n")
        elif sub in ("confirm", "forget") and rest.strip().lstrip("#").isdigit():
            n = int(rest.strip().lstrip("#"))
            ok = mem.confirm(n) if sub == "confirm" else mem.forget(n)
            self.term.write(("done" if ok else f"no such memory #{n}") + "\n")
        else:
            self.term.write("usage: /memory [list|pending|search Q|confirm ID|forget ID]\n")


@ventri.plugin(name="channel:cli", config=CliConfig, exclusive=True, provides={"cli": CliChannel})
def cli(ctx: Any, cfg: CliConfig) -> None:
    """``use: ventri_agent.channels.cli`` -- the terminal REPL. A ``"cli.terminal"``
    service (tests) replaces stdin/stdout."""
    term = ctx.get("cli.terminal", None) or StdTerminal()
    ch = CliChannel(ctx, cfg, term)
    ctx.provide(CliChannel, ch)
    ctx.spawn(ch.run, name="repl")


plugin = cli
