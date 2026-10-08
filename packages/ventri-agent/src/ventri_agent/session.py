"""Per-session services (session realm): SessionInfo, SessionLog, Budget.

``SessionLog`` is the session's append-only JSONL file
(``~/.ventri/sessions/<id>.jsonl``). Record types (``t``):

* ``meta``     -- id, agent preset, channel, created
* ``prefix``   -- the frozen stable prefix of an epoch (system text, memory
  snapshot, tool names/specs): resuming rebuilds it byte-for-byte, so the
  disk cache keeps hitting after a restart
* ``msg``      -- one history message (user / assistant incl. reasoning_content / tool / system tail)
* ``compact``  -- ``drop`` leading history messages replaced by ``summary``
* ``prune``    -- written only by 0.2.0a1 development builds (removed context
  pruning); ignored on replay, so such a log loads with the full messages
* ``usage``    -- one model call: model, route, usage, cost, peak flag
* ``turn``     -- turn boundary (status, steps)
* ``work``     -- working-memory snapshot
* ``state``    -- lifecycle: suspended / ended / memory extracted up to
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .messages import Message, Usage, now_ts


@dataclass
class AgentPreset:
    """``agents: {name: {persona, tools, route, loop, ...}}`` in ventri.yml."""

    name: str = "default"
    persona: str = ""                      # file path (relative to ~/.ventri) or inline text
    tools: list[str] = field(default_factory=lambda: ["*"])
    route: str = "default"
    loop: str | None = None                # alternative agent-loop plugin (``use`` string)
    memory_k: int = 12
    system_prompt: str = ""                # file or inline text; replaces persona + built-in rules entirely
    time_notes: bool | None = None         # current-time / peak-pricing notes (None: on, off when headless)
    mode: str = "interactive"              # "headless": only opened by unattended runs (`va run`)
    inline_tokens: int = 8_000             # a tool result above this is stored as an artifact (head + tail inline)


@dataclass
class SessionInfo:
    id: str
    agent: AgentPreset
    dir: Path                              # ~/.ventri/sessions/<id>/ (artifacts)
    log_path: Path
    channel: str = "cli"
    origin: str = "user"
    created: float = field(default_factory=now_ts)
    resumed: bool = False
    headless: bool = False                 # unattended run: no human; approvals by the configured policy


@dataclass
class Replay:
    """State reconstructed from a session log (no model call)."""

    meta: dict[str, Any] = field(default_factory=dict)
    prefix: dict[str, Any] | None = None
    history: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    calls: int = 0
    turns: int = 0
    work: str = ""
    state: str = "active"
    extracted_upto: int = 0                # count of history messages already mined for memory
    last_prompt_tokens: int = 0
    compactions: int = 0
    total_messages: int = 0                # messages ever appended (incl. compacted ones)


class SessionLog:
    """Append-only JSONL writer + replay."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8")

    def append(self, t: str, **data: Any) -> None:
        rec = {"t": t, "ts": round(now_ts(), 3), **data}
        self._f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def message(self, m: Message) -> None:
        self.append("msg", m=m.to_json())

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    @staticmethod
    def replay(path: Path) -> Replay:
        r = Replay()
        if not path.exists():
            return r
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line after a crash: skip it
                t = rec.get("t")
                if t == "meta":
                    r.meta = rec
                elif t == "prefix":
                    r.prefix = rec
                elif t == "msg":
                    r.history.append(Message.from_json(rec["m"]))
                    r.total_messages += 1
                elif t == "compact":
                    drop = int(rec["drop"])
                    r.history = [Message.system(rec["summary"], compacted=drop), *r.history[drop:]]
                    r.compactions += 1
                elif t == "prune":
                    pass  # removed feature (0.2.0a1 dev builds): keep the full, unedited messages
                elif t == "usage":
                    u = Usage.from_json(rec.get("usage") or {})
                    r.usage = r.usage + u
                    r.cost_usd += float(rec.get("cost_usd") or 0)
                    r.calls += 1
                    r.last_prompt_tokens = u.prompt_tokens
                elif t == "turn":
                    r.turns = max(r.turns, int(rec.get("n") or 0))
                elif t == "work":
                    r.work = rec.get("text") or ""
                elif t == "state":
                    r.state = rec.get("state") or r.state
                    if "extracted_upto" in rec:
                        r.extracted_upto = int(rec["extracted_upto"])
        return r


def read_usage(sessions_dir: Path) -> list[dict[str, Any]]:
    """Every ``usage`` record of every session log (for ``va cost``)."""
    out: list[dict[str, Any]] = []
    if not sessions_dir.exists():
        return out
    for p in sorted(sessions_dir.glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                if '"t": "usage"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["session"] = p.stem
                out.append(rec)
    return out


@dataclass
class BudgetLimits:
    max_steps: int = 40
    max_tool_calls: int = 120
    max_tokens: int = 2_000_000            # prompt + completion per turn
    max_cost_cny: float = 2.0              # per turn (DESIGN 5.1: ¥2 equivalent)
    usd_to_cny: float = 7.2
    wall_s: float = 900.0


class Budget:
    """Session-realm service: per-turn limits (steps, tool calls, tokens, cost, wall time)."""

    def __init__(self, limits: BudgetLimits | None = None) -> None:
        self.limits = limits or BudgetLimits()
        self.start_turn()

    def start_turn(self) -> None:
        self.steps = 0
        self.tool_calls = 0
        self.tokens = 0
        self.cost_usd = 0.0
        self.started = time.monotonic()

    def charge(self, usage: Usage | None = None, cost_usd: float = 0.0, *, step: bool = False,
               tool_calls: int = 0) -> None:
        if usage is not None:
            self.tokens += usage.prompt_tokens + usage.completion_tokens
        self.cost_usd += cost_usd
        self.steps += int(step)
        self.tool_calls += tool_calls

    def exhausted(self) -> str | None:
        L = self.limits
        if self.steps >= L.max_steps:
            return f"step limit ({L.max_steps})"
        if self.tool_calls >= L.max_tool_calls:
            return f"tool-call limit ({L.max_tool_calls})"
        if self.tokens >= L.max_tokens:
            return f"token limit ({L.max_tokens})"
        if self.cost_usd * L.usd_to_cny >= L.max_cost_cny:
            return f"cost limit (¥{L.max_cost_cny:.2f})"
        if time.monotonic() - self.started >= L.wall_s:
            return f"time limit ({L.wall_s:.0f}s)"
        return None

    @property
    def can_summarize(self) -> bool:
        """Step/tool limits still allow one tool-less wrap-up call; money/time limits do not."""
        r = self.exhausted() or ""
        return r.startswith(("step", "tool-call"))


def session_paths(sessions_dir: Path, sid: str) -> tuple[Path, Path]:
    return sessions_dir / f"{sid}.jsonl", sessions_dir / sid


def list_sessions(sessions_dir: Path) -> list[dict[str, Any]]:
    out = []
    if not sessions_dir.exists():
        return out
    for p in sorted(sessions_dir.glob("*.jsonl"), key=os.path.getmtime, reverse=True):
        r = SessionLog.replay(p)
        first = next((m.content for m in r.history if m.role == "user"), "") or ""
        out.append({"id": p.stem, "agent": r.meta.get("agent", "default"), "state": r.state,
                    "turns": r.turns, "messages": r.total_messages, "cost_usd": r.cost_usd,
                    "hit_rate": r.usage.hit_rate, "updated": p.stat().st_mtime,
                    "title": first.replace("\n", " ")[:60]})
    return out
