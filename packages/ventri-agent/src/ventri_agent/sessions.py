"""SessionManager: sessions are kernel scopes (DESIGN.md 5.7).

``open()`` creates ``scope session:<id>`` under the manager's fiber, isolating
the session services (``SessionInfo``, ``SessionLog``, ``Budget``,
``WorkingMemory``, ``Grants``, ``ContextBuilder``, ``AgentLoop``, ``Replay``,
``FileState``, ``ShellJobs``),
and loads into it: session-core -> permission-gate -> context-builder ->
agent-loop (or the preset's ``loop``). Lifecycle:

* **Active** -- turns run; the JSONL log is written as things happen.
* **Suspended** -- idle timeout (default 30 min): the scope is disposed, only
  the log remains. Resuming = new scope + log replay (no model call): the
  history, epoch prefix, totals and working memory come back exactly.
* **Ending** -- user ``/end``, channel closed, or retention expiry (7 days):
  memory extraction (cheap route, JSON) -> long-term memory -> dispose.
  Disposing the scope revokes the session's grants and reclaims everything.

A process crash is a suspension without the log record: ``open(id)`` resumes it.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
from pydantic import BaseModel, Field

import ventri
from ventri import Fiber, State

from .context import ContextBuilder, context_builder
from .loop import AgentLoop, Sink, TurnResult, agent_loop
from .memory import LongTermMemory, MemoryItem, WorkingMemory
from .paths import expand
from .permission import Grants, gate
from .providers.base import ModelProvider
from .providers.pricing import BEIJING
from .session import (
    AgentPreset,
    Budget,
    BudgetLimits,
    Replay,
    SessionInfo,
    SessionLog,
    list_sessions,
    session_paths,
)
from .tools._hermes_fs.state import FileState
from .tools.registry import ToolRegistry
from .tools.shell import ShellJobs

SESSION_KEYS = (SessionInfo, SessionLog, Budget, WorkingMemory, Grants, ContextBuilder, AgentLoop, Replay,
                FileState, ShellJobs)


class SessionError(Exception):
    pass


def new_session_id() -> str:
    import secrets

    return datetime.now(BEIJING).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def _core(info: SessionInfo, replay: Replay, limits: BudgetLimits) -> Any:
    def session_core(ctx: Any, config: Any) -> None:
        log = ctx.provide(SessionLog, SessionLog(info.log_path))
        ctx.on_dispose(log.close)
        ctx.provide(SessionInfo, info)
        ctx.provide(Replay, replay)
        ctx.provide(Budget, Budget(limits))
        ctx.provide(WorkingMemory, WorkingMemory(replay.work))
        # read-before-write / stale-file state of the file tools (fs.*, notes.*),
        # scoped to this session and dropped when the session scope is disposed
        files = ctx.provide(FileState, FileState(info.id))
        ctx.on_dispose(files.close)
        # background shell jobs + persisted shell cwd/env; disposing the session kills its jobs
        jobs = ctx.provide(ShellJobs, ShellJobs(info.id))
        ctx.on_dispose(jobs.close)
        if not info.resumed:
            log.append("meta", id=info.id, agent=info.agent.name, channel=info.channel, origin=info.origin,
                       **({"headless": True} if info.headless else {}))
        else:
            log.append("state", state="resumed", **({"headless": True} if info.headless else {}))
    session_core.name = "session-core"  # type: ignore[attr-defined]
    return session_core


class Session:
    """Handle to an open session (its scope fiber and services)."""

    def __init__(self, manager: SessionManager, scope: Fiber, info: SessionInfo) -> None:
        self.manager = manager
        self.scope = scope
        self.info = info
        self.last_active = time.monotonic()
        self.ended = False

    @property
    def id(self) -> str:
        return self.info.id

    @property
    def ctx(self) -> Any:
        return self.scope.ctx

    @property
    def loop(self) -> AgentLoop:
        return self.scope.ctx.get(AgentLoop)

    @property
    def alive(self) -> bool:
        return self.scope.state is State.ACTIVE and not self.ended

    def _check(self) -> AgentLoop:
        if not self.alive:
            raise SessionError(f"session {self.id} is not active ({self.scope.state.value})")
        loop = self.scope.ctx.get(AgentLoop, None)
        if loop is None:
            raise SessionError(f"session {self.id}: agent loop unavailable")
        return loop

    async def turn(self, text: str, sink: Sink | None = None, *, plan: bool = False) -> TurnResult:
        loop = self._check()
        self.last_active = time.monotonic()
        try:
            return await loop.turn(text, sink, plan=plan)
        finally:
            self.last_active = time.monotonic()

    async def retry(self, sink: Sink | None = None) -> TurnResult:
        loop = self._check()
        self.last_active = time.monotonic()
        return await loop.retry(sink)

    async def suspend(self) -> None:
        await self.manager.suspend(self.id)

    async def end(self, *, extract: bool = True) -> list[MemoryItem]:
        return await self.manager.end(self.id, extract=extract)


class SessionsConfig(BaseModel):
    dir: str = "~/.ventri/sessions"
    idle_timeout: float = 1800.0
    retention_days: float = 7.0
    budget: dict[str, Any] = Field(default_factory=dict)
    agents: dict[str, dict[str, Any]] = Field(default_factory=dict)
    extract_memory: bool = True
    sweep_interval: float = 60.0


class SessionManager:
    """Service (root realm)."""

    def __init__(self, ctx: Any, cfg: SessionsConfig, provider: ModelProvider, registry: ToolRegistry,
                 memory: LongTermMemory | None = None) -> None:
        self.ctx = ctx
        self.cfg = cfg
        self.provider = provider
        self.registry = registry
        self.memory = memory
        self.dir = expand(cfg.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, Session] = {}
        self._lock = anyio.Lock()

    # ------------------------------------------------------------ presets
    def presets(self) -> dict[str, AgentPreset]:
        raw: dict[str, Any] = {}
        loader = self.ctx.get("config.loader", None)
        doc = getattr(loader, "document", None)
        if doc is not None and isinstance(doc.data.get("agents"), dict):
            raw.update(doc.data["agents"])
        raw.update(self.cfg.agents)
        out = {"default": AgentPreset()}
        for name, spec in raw.items():
            spec = dict(spec or {})
            tools = spec.get("tools", ["*"])
            mode = str(spec.get("mode", "interactive"))
            if mode not in ("interactive", "headless"):
                raise SessionError(f"agent preset {name!r}: mode must be interactive or headless, not {mode!r}")
            tn = spec.get("time_notes")
            out[name] = AgentPreset(name=name, persona=str(spec.get("persona", "")),
                                    tools=[tools] if isinstance(tools, str) else list(tools),
                                    route=str(spec.get("route", "default")), loop=spec.get("loop"),
                                    memory_k=int(spec.get("memory_k", 12)),
                                    system_prompt=str(spec.get("system_prompt", "") or ""),
                                    time_notes=None if tn is None else bool(tn), mode=mode,
                                    inline_tokens=int(spec.get("inline_tokens", 8_000)),
                                    compact_at=float(spec.get("compact_at", 0.6)),
                                    compact_keep_turns=int(spec.get("compact_keep_turns", 4)),
                                    compact_keep_steps=int(spec.get("compact_keep_steps", 6)))
            p = out[name]
            if not 0.1 <= p.compact_at <= 0.95 or p.compact_keep_turns < 1 or p.compact_keep_steps < 1:
                raise SessionError(f"agent preset {name!r}: compact_at must be in [0.1, 0.95] and "
                                   "compact_keep_turns / compact_keep_steps >= 1")
        return out

    # --------------------------------------------------------------- open
    async def open(self, session_id: str | None = None, *, agent: str | None = None,
                   channel: str = "cli", origin: str = "user", headless: bool = False) -> Session:
        """Open (or resume) a session. ``headless=True`` is the explicit opt-in to
        an unattended run: approvals are decided by the permission config's
        ``unattended`` policy instead of a channel, the system prompt defaults
        to the headless one and time notes are off. It is never inherited from
        the log -- every open has to ask for it again."""
        async with self._lock:
            sid = session_id or new_session_id()
            if sid in self.sessions and self.sessions[sid].alive:
                return self.sessions[sid]
            log_path, sdir = session_paths(self.dir, sid)
            replay = SessionLog.replay(log_path) if log_path.exists() else Replay()
            resumed = log_path.exists()
            name = agent or replay.meta.get("agent") or "default"
            presets = self.presets()
            if name not in presets:
                raise SessionError(f"unknown agent preset {name!r} (have: {', '.join(sorted(presets))})")
            if presets[name].mode == "headless" and not headless:
                raise SessionError(f"agent preset {name!r} is headless-only (mode: headless); "
                                   "run it unattended with `va run --headless`")
            info = SessionInfo(sid, presets[name], sdir, log_path, channel, origin, resumed=resumed,
                               headless=headless)
            limits = BudgetLimits(**self.cfg.budget)
            scope = await self.ctx.scope(f"session:{sid}", isolate=SESSION_KEYS, meta={"id": f"session:{sid}"})
            try:
                await scope.ctx.plugin(_core(info, replay, limits))
                await scope.ctx.plugin(gate)
                await scope.ctx.plugin(context_builder)
                loop_plugin: Any = agent_loop
                if info.agent.loop:
                    from ventri_std.config import resolve_use

                    loop_plugin = resolve_use(info.agent.loop)
                f = await scope.ctx.plugin(loop_plugin)
                bad = [c for c in scope.children if c.state is not State.ACTIVE]
                if bad or f.state is not State.ACTIVE:
                    why = "; ".join(f"{c.label}: {c.pending_reason or c.error!r}" for c in bad)
                    raise SessionError(f"session {sid} failed to start: {why}")
            except BaseException:
                with anyio.CancelScope(shield=True):
                    await scope.dispose()
                raise
            s = Session(self, scope, info)
            self.sessions[sid] = s
            self.ctx.trace("session.open", session=sid, agent=name, resumed=resumed,
                           history=len(replay.history), headless=headless)
            return s

    def get(self, sid: str) -> Session | None:
        s = self.sessions.get(sid)
        return s if s is not None and s.alive else None

    async def suspend(self, sid: str) -> None:
        s = self.sessions.pop(sid, None)
        if s is None or s.scope.state is State.DISPOSED:
            return
        s.ctx.get(SessionLog).append("state", state="suspended")
        await s.scope.dispose()
        self.ctx.trace("session.suspend", session=sid)

    async def end(self, sid: str, *, extract: bool = True) -> list[MemoryItem]:
        s = self.sessions.get(sid)
        if s is None or not s.alive:
            return []
        created: list[MemoryItem] = []
        try:
            if extract and self.cfg.extract_memory and self.memory is not None:
                rep = SessionLog.replay(s.info.log_path)
                created = await s.loop.extract_memories(rep.extracted_upto)
            s.ctx.get(SessionLog).append("state", state="ended")
        finally:
            s.ended = True
            self.sessions.pop(sid, None)
            with anyio.CancelScope(shield=True):
                await s.scope.dispose()
        self.ctx.trace("session.end", session=sid, memories=len(created))
        return created

    def list(self) -> list[dict[str, Any]]:
        return list_sessions(self.dir)

    def last_id(self) -> str | None:
        items = self.list()
        return items[0]["id"] if items else None

    # -------------------------------------------------------------- sweeps
    async def sweep_idle(self) -> list[str]:
        now = time.monotonic()
        idle = [sid for sid, s in self.sessions.items()
                if s.alive and now - s.last_active >= self.cfg.idle_timeout and not s.loop._lock.locked()]
        for sid in idle:
            await self.suspend(sid)
        return idle

    async def sweep_retention(self) -> list[str]:
        """End suspended sessions past the retention period (memory extraction)."""
        cutoff = time.time() - self.cfg.retention_days * 86400
        done = []
        for item in self.list():
            if item["state"] in ("ended",) or item["id"] in self.sessions or item["updated"] > cutoff:
                continue
            if item["messages"] == 0:
                continue
            try:
                s = await self.open(item["id"], channel="system", origin="routine")
                await s.end()
                done.append(item["id"])
            except Exception as e:  # noqa: BLE001 - one bad log must not stop the sweep
                self.ctx.trace("session.error", session=item["id"], error=repr(e))
        return done

    async def run_sweeper(self) -> None:
        while True:
            await anyio.sleep(self.cfg.sweep_interval)
            await self.sweep_idle()


@ventri.plugin(name="session-manager", config=SessionsConfig, provides={"sessions": SessionManager})
def session_manager(ctx: Any, cfg: SessionsConfig, provider: ModelProvider, registry: ToolRegistry,
                    memory: LongTermMemory | None = None) -> None:
    """``use: ventri_agent.sessions`` -- provides ``SessionManager``."""
    mgr = ctx.provide(SessionManager, SessionManager(ctx, cfg, provider, registry, memory))
    ctx.spawn(mgr.run_sweeper, name="idle-sweeper")


plugin = session_manager


def sessions_dir() -> Path:
    return expand("~/.ventri/sessions")
