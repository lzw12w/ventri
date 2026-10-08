"""Shared test harness: a kernel with the fake provider, the tool registry,
the permission engine, memory and the session manager, plus test tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self

import anyio
from pydantic import BaseModel

import ventri
from ventri import Kernel
from ventri_agent.memory import memory_sqlite
from ventri_agent.permission import ApprovalBroker, ApprovalRequest, AuditLog, permission
from ventri_agent.providers.base import ModelProvider
from ventri_agent.providers.fake import FakeProvider
from ventri_agent.sessions import Session, SessionManager, session_manager
from ventri_agent.tools.registry import Risk, Tool, ToolContext, ToolError, ToolRegistry
from ventri_agent.tools.registry import plugin as registry_plugin


class Probe:
    """Records side effects of the test tools."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.active = 0
        self.max_active = 0


class TextArgs(BaseModel):
    text: str = ""


class PathArgs(BaseModel):
    path: str
    content: str = ""


def make_test_tools(probe: Probe) -> list[Tool]:
    def echo(a: TextArgs, tc: ToolContext) -> str:
        probe.calls.append(("t.echo", a.text))
        return f"echo: {a.text}"

    async def slow(a: TextArgs, tc: ToolContext) -> str:
        probe.active += 1
        probe.max_active = max(probe.max_active, probe.active)
        try:
            await anyio.sleep(0.05)
        finally:
            probe.active -= 1
        probe.calls.append(("t.slow", a.text))
        return f"slow: {a.text}"

    def crash(a: TextArgs, tc: ToolContext) -> str:
        raise RuntimeError("boom")

    def fail(a: TextArgs, tc: ToolContext) -> str:
        raise ToolError("clean failure")

    async def hang(a: TextArgs, tc: ToolContext) -> str:
        await anyio.sleep(10)
        return "never"

    def big(a: TextArgs, tc: ToolContext) -> str:
        if a.text == "zh":   # 15K Chinese chars: ~9000 tokens (len/4 would have said 3750)
            return "中文工具结果" * 2500
        return "".join(f"line {i:05d} " + "x" * 60 + "\n" for i in range(800))  # ~56k chars

    def web(a: TextArgs, tc: ToolContext) -> str:
        probe.calls.append(("t.web", a.text))
        return a.text  # content chosen by the test (e.g. an injected page)

    def write(a: PathArgs, tc: ToolContext) -> str:
        probe.calls.append(("t.write", a.path))
        return f"wrote {a.path}"

    def send(a: TextArgs, tc: ToolContext) -> str:
        probe.calls.append(("t.send", a.text))
        return "sent"

    def subj(a: Any) -> dict[str, str]:
        return {"path": getattr(a, "path", "")}

    return [
        Tool("t.echo", "Echo text back.", echo, TextArgs, Risk.READ, parallel_safe=True),
        Tool("t.slow", "Slow read.", slow, TextArgs, Risk.READ, parallel_safe=True),
        Tool("t.crash", "Always crashes.", crash, TextArgs, Risk.READ),
        Tool("t.fail", "Raises a ToolError.", fail, TextArgs, Risk.READ),
        Tool("t.hang", "Never returns.", hang, TextArgs, Risk.READ, timeout=0.05),
        Tool("t.big", "Huge output.", big, TextArgs, Risk.READ),
        Tool("t.web", "Fetch a (fake) page.", web, TextArgs, Risk.READ, parallel_safe=True, untrusted=True),
        Tool("t.write", "Write a file.", write, PathArgs, Risk.WRITE_LOCAL, subject=subj),
        Tool("t.send", "Send a message (irreversible).", send, TextArgs, Risk.IRREVERSIBLE),
    ]


def tools_plugin(probe: Probe) -> Any:
    @ventri.plugin(name="test-tools")
    def test_tools(ctx: Any, config: Any, registry: ToolRegistry) -> None:
        for t in make_test_tools(probe):
            registry.register(ctx, t)
    return test_tools


def provider_plugin(prov: Any) -> Any:
    @ventri.plugin(name="provider:test", provides={"llm": ModelProvider})
    def provider(ctx: Any, config: Any) -> None:
        ctx.provide(ModelProvider, prov)
    return provider


def call(name: str, args: dict[str, Any] | None = None, id: str | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {"name": name, "arguments": args or {}}
    if id:
        d["id"] = id
    return d


class Env:
    """``async with Env(tmp_path, script) as env:`` -- a full agent runtime."""

    def __init__(self, tmp: Path, script: list[Any] | None = None, *, rules: list[dict[str, Any]] | None = None,
                 budget: dict[str, Any] | None = None, agents: dict[str, Any] | None = None,
                 provider: Any = None, memory: bool = True, approval_timeout: float = 5.0,
                 extract_memory: bool = True, idle_timeout: float = 1800.0) -> None:
        self.tmp = tmp
        self.provider = provider or FakeProvider(script or [])
        self.rules = rules or []
        self.budget = budget or {}
        self.agents = agents or {}
        self.with_memory = memory
        self.approval_timeout = approval_timeout
        self.extract_memory = extract_memory
        self.idle_timeout = idle_timeout
        self.probe = Probe()
        self.choices: list[str] = []
        self.asked: list[ApprovalRequest] = []

    async def __aenter__(self) -> Self:
        self.kernel = Kernel()
        await self.kernel.__aenter__()
        k = self.kernel
        await k.plugin(provider_plugin(self.provider))
        await k.plugin(registry_plugin)
        await k.plugin(permission, {"rules": self.rules, "audit": str(self.tmp / "audit.jsonl"),
                                    "approval_timeout": self.approval_timeout})
        if self.with_memory:
            await k.plugin(memory_sqlite, {"path": str(self.tmp / "memory.db")})
        self.tools_fiber = await k.plugin(tools_plugin(self.probe))
        self.mgr_fiber = await k.plugin(session_manager, {
            "dir": str(self.tmp / "sessions"), "budget": self.budget, "agents": self.agents,
            "extract_memory": self.extract_memory, "idle_timeout": self.idle_timeout})
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.kernel.__aexit__(*exc)  # type: ignore[arg-type]

    @property
    def mgr(self) -> SessionManager:
        return self.kernel.get(SessionManager)

    @property
    def broker(self) -> ApprovalBroker:
        return self.kernel.get(ApprovalBroker)

    @property
    def audit(self) -> AuditLog:
        return self.kernel.get(AuditLog)

    @property
    def registry(self) -> ToolRegistry:
        return self.kernel.get(ToolRegistry)

    async def open(self, sid: str | None = None, *, bind: bool = True, **kw: Any) -> Session:
        s = await self.mgr.open(sid, **kw)
        if bind:
            self.bind(s.id)
        return s

    def bind(self, sid: str) -> None:
        async def ask(req: ApprovalRequest) -> Any:
            self.asked.append(req)
            return self.choices.pop(0) if self.choices else "deny"
        self.broker.bind(sid, "test", ask)

    def log_records(self, sid: str) -> list[dict[str, Any]]:
        p = self.tmp / "sessions" / f"{sid}.jsonl"
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
