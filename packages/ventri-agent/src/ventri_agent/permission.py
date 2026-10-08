"""Permission engine, approvals and audit (DESIGN.md 5.6).

* Every tool declares a :class:`Risk`. The root ``Policy`` (config of
  ``use: ventri_agent.permission``) holds ordered ``allow / ask / deny`` rules
  matched on tool-name glob x argument predicates (path / domain / ... globs) x
  scope (agent preset, session) x origin (user / routine / evolution).
  Default when no rule matches: the tool's own default (``fs`` ``write: ask``),
  else ``read`` -> allow (tools enforce their roots) and everything else ->
  ask. ``irreversible`` / ``spend`` are **always asked**: a rule may deny them
  but never allow them, and they cannot be granted for a session.
* Each session scope runs a ``permission-gate`` plugin: it provides the
  session's ``Grants`` (isolated in the session realm -- disposing the session
  revokes them) and intercepts ``ToolCheck``. ``ask`` goes to the
  ``ApprovalBroker``, which routes an ``ApprovalRequest`` to the channel bound
  to the session; no channel or timeout (default 120 s) = deny.
* Hard rules: (1) model output never constitutes approval -- a decision is only
  accepted with a token minted by the broker from a channel's human-input
  callback (``verify``); (2) the policy is configuration only (no runtime API,
  no tool, no proposal can change it); (3) ``audit.jsonl`` is append-only and
  records every request, decision and decider.
* Fail closed: the agent loop refuses a tool call that no gate stamped
  (``ToolRequest.approved_by``).
* Unattended (headless) sessions -- opened explicitly with ``headless=True``
  (``va run --headless``), never by ``va chat`` -- have no human to ask: a
  request that would be *asked* is decided by the ``unattended`` config
  (``ask: deny|allow``, default deny), and ``irreversible`` / ``spend`` tools
  by ``unattended.irreversible`` (default deny; ``allow`` must be written
  explicitly). Rules and tool defaults still apply first (a ``deny`` stays a
  deny), and every decision is audited with ``decided_by: unattended-policy``.
"""
from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import BaseModel, Field

import ventri
from ventri import Deny, Event

from .messages import new_id, now_ts
from .paths import expand
from .session import SessionInfo
from .tools.registry import Action, Risk, Tool

Choice = Literal["once", "session", "deny"]


@dataclass
class ToolRequest:
    """The value passed through ``ctx.check(ToolCheck, request)``."""

    call_id: str
    tool: Tool
    args: Any
    session_id: str
    agent: str = "default"
    origin: str = "user"
    subject: dict[str, str] = field(default_factory=dict)
    approved_by: str | None = None   # set by a permission gate; None = refused by the loop

    @property
    def summary(self) -> str:
        subj = " ".join(f"{k}={v}" for k, v in self.subject.items())
        return f"{self.tool.name} {subj}".strip()


ToolCheck = Event[ToolRequest]("tool.check")


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    session_id: str
    tool: str
    risk: str
    subject: dict[str, str]
    summary: str
    grantable: bool
    args_preview: str
    timeout: float


@dataclass(frozen=True)
class ApprovalDecision:
    request_id: str
    choice: Choice
    decided_by: str
    token: str


ApprovalRequested = Event[ApprovalRequest]("approval.request")


# --------------------------------------------------------------------- rules
class Rule(BaseModel):
    tool: str = "*"
    action: Action
    risk: str | None = None              # match a risk class ("read", "write-local", ...)
    when: dict[str, str] = Field(default_factory=dict)  # subject attribute -> glob
    origin: str | None = None
    agent: str | None = None
    session: str | None = None

    def matches(self, req: ToolRequest) -> bool:
        if not fnmatch.fnmatchcase(req.tool.name, self.tool):
            return False
        if self.risk is not None and Risk.parse(self.risk) != req.tool.risk:
            return False
        for attr, pat in (("origin", self.origin), ("agent", self.agent), ("session_id", self.session)):
            if pat is not None and not fnmatch.fnmatchcase(getattr(req, attr), pat):
                return False
        for k, pat in self.when.items():
            v = req.subject.get(k)
            if v is None or not fnmatch.fnmatchcase(v, os.path.expanduser(pat)):
                return False
        return True


class Unattended(BaseModel):
    """How a headless session resolves what would otherwise be asked."""

    ask: Literal["deny", "allow"] = "deny"            # write-local / external calls that need approval
    irreversible: Literal["deny", "allow"] = "deny"   # irreversible / spend tools


class Policy:
    """Service (root realm): ordered rules -> ``(action, reason)``."""

    def __init__(self, rules: list[Rule] | None = None, *, approval_timeout: float = 120.0,
                 unattended: Unattended | None = None) -> None:
        self.rules = tuple(rules or ())
        self.approval_timeout = approval_timeout
        self.unattended = unattended or Unattended()

    def decide_unattended(self, req: ToolRequest) -> tuple[Action, str]:
        """The decision for an ``ask`` in a headless session (no human)."""
        if req.tool.risk >= Risk.IRREVERSIBLE:
            return self.unattended.irreversible, f"unattended.irreversible: {self.unattended.irreversible}"
        return self.unattended.ask, f"unattended.ask: {self.unattended.ask}"

    def decide(self, req: ToolRequest) -> tuple[Action, str]:
        tool = req.tool
        action: Action
        for i, r in enumerate(self.rules):
            if r.matches(req):
                action, reason = r.action, f"rule {i} ({r.tool} -> {r.action})"
                break
        else:
            allow_hit = next((k for k, pats in tool.default_allow.items()
                              if any(fnmatch.fnmatchcase(req.subject.get(k, "\0"), p) for p in pats)), None)
            if allow_hit is not None:
                action, reason = "allow", f"{tool.name} default allow ({allow_hit})"
            elif tool.default_action is not None:
                action, reason = tool.default_action, f"{tool.name} default"
            elif tool.risk == Risk.READ:
                action, reason = "allow", "read-only tool"
            else:
                action, reason = "ask", f"{tool.risk.label} needs approval"
        if action == "allow" and tool.risk >= Risk.IRREVERSIBLE:
            action, reason = "ask", f"{tool.risk.label} is always asked"
        return action, reason


class Grants:
    """Session-realm service: "allow for this session" grants (tool names)."""

    def __init__(self) -> None:
        self._tools: set[str] = set()

    def allows(self, req: ToolRequest) -> bool:
        return (req.tool.name in self._tools and req.tool.grantable
                and req.tool.risk < Risk.IRREVERSIBLE)

    def grant(self, tool: str) -> None:
        self._tools.add(tool)

    def revoke(self, tool: str | None = None) -> None:
        if tool is None:
            self._tools.clear()
        else:
            self._tools.discard(tool)

    def __iter__(self) -> Any:
        return iter(sorted(self._tools))


# --------------------------------------------------------------------- audit
class AuditLog:
    """Append-only JSONL (``audit.jsonl``)."""

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path).expanduser() if path else None
        self.records: list[dict[str, Any]] = []  # in-memory tail (tests, /audit)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **rec: Any) -> None:
        rec = {"ts": round(now_ts(), 3), **rec}
        self.records.append(rec)
        del self.records[:-1000]
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# -------------------------------------------------------------------- broker
AskHuman = Callable[[ApprovalRequest], Awaitable[Choice]]


class ApprovalBroker:
    """Routes approval requests to the channel bound to a session and mints
    decision tokens. Only :meth:`request` creates verifiable decisions, and it
    only does so from a channel's ``ask`` callback (human input)."""

    def __init__(self, *, timeout: float = 120.0) -> None:
        self._key = secrets.token_bytes(32)
        self._channels: dict[str, tuple[str, AskHuman]] = {}  # session id -> (channel, ask)
        self.timeout = timeout

    def bind(self, session_id: str, channel: str, ask: AskHuman) -> Callable[[], None]:
        self._channels[session_id] = (channel, ask)

        def unbind() -> None:
            if self._channels.get(session_id, (None,))[0] == channel:
                self._channels.pop(session_id, None)
        return unbind

    def channel_of(self, session_id: str) -> str | None:
        b = self._channels.get(session_id)
        return b[0] if b else None

    def _token(self, request_id: str, choice: str, decided_by: str) -> str:
        return hmac.new(self._key, f"{request_id}|{choice}|{decided_by}".encode(), hashlib.sha256).hexdigest()

    def verify(self, d: ApprovalDecision) -> bool:
        return hmac.compare_digest(d.token, self._token(d.request_id, d.choice, d.decided_by))

    async def request(self, req: ApprovalRequest) -> ApprovalDecision:
        bound = self._channels.get(req.session_id)
        if bound is None:
            return ApprovalDecision(req.id, "deny", "system:no-channel",
                                    self._token(req.id, "deny", "system:no-channel"))
        channel, ask = bound
        choice: Choice = "deny"
        who = f"human:{channel}"
        try:
            with anyio.fail_after(req.timeout):
                choice = await ask(req)
        except TimeoutError:
            who = "system:timeout"
        if choice not in ("once", "session", "deny"):
            choice, who = "deny", "system:invalid-choice"
        if choice == "session" and not req.grantable:
            choice = "once"
        return ApprovalDecision(req.id, choice, who, self._token(req.id, choice, who))


# --------------------------------------------------------------------- gate
class PermissionGate:
    """The per-session interceptor (fiber ``permission-gate`` in the session scope)."""

    def __init__(self, ctx: Any, policy: Policy, broker: ApprovalBroker, grants: Grants,
                 audit: AuditLog, *, headless: bool = False) -> None:
        self.ctx = ctx
        self.policy = policy
        self.broker = broker
        self.grants = grants
        self.audit = audit
        self.headless = headless

    async def __call__(self, req: ToolRequest) -> Deny | None:
        action, reason = self.policy.decide(req)
        base = {"session": req.session_id, "call": req.call_id, "tool": req.tool.name,
                "risk": req.tool.risk.label, "subject": req.subject, "origin": req.origin}
        if action == "deny":
            self.audit.write(**base, action="deny", decided_by="policy", reason=reason)
            return Deny(f"denied by policy ({reason})")
        if action == "allow":
            req.approved_by = "policy"
            self.audit.write(**base, action="allow", decided_by="policy", reason=reason)
            return None
        if self.grants.allows(req):
            req.approved_by = "grant:session"
            self.audit.write(**base, action="allow", decided_by="grant:session", reason=reason)
            return None
        if self.headless:
            action, why = self.policy.decide_unattended(req)
            self.audit.write(**base, action=action, decided_by="unattended-policy", reason=f"{reason}; {why}",
                             headless=True)
            self.ctx.trace("approval.unattended", tool=req.tool.name, risk=req.tool.risk.label, action=action)
            if action != "allow":
                return Deny(f"not approved: unattended run, {why} (no human to ask)")
            req.approved_by = "unattended-policy"
            return None
        ar = ApprovalRequest(new_id("apr_"), req.session_id, req.tool.name, req.tool.risk.label,
                             dict(req.subject), req.summary,
                             req.tool.grantable and req.tool.risk < Risk.IRREVERSIBLE,
                             _preview(req.args), self.policy.approval_timeout)
        self.audit.write(**base, action="ask", decided_by="policy", reason=reason, request=ar.id)
        self.ctx.trace("approval.request", id=ar.id, tool=req.tool.name, risk=ar.risk)
        await self.ctx.emit(ApprovalRequested, ar)
        d = await self.broker.request(ar)
        ok = self.broker.verify(d) and d.request_id == ar.id
        self.ctx.trace("approval.decision", id=ar.id, choice=d.choice, by=d.decided_by, verified=ok)
        self.audit.write(**base, action="allow" if ok and d.choice != "deny" else "deny",
                         decided_by=d.decided_by, choice=d.choice, request=ar.id, verified=ok)
        if not ok or d.choice == "deny":
            why = {"system:timeout": "approval timed out", "system:no-channel":
                   "no interactive channel to ask"}.get(d.decided_by, "the user denied it")
            return Deny(f"not approved: {why}")
        if d.choice == "session":
            self.grants.grant(req.tool.name)
        req.approved_by = d.decided_by
        return None


def _preview(args: Any, limit: int = 600) -> str:
    if isinstance(args, BaseModel):
        args = args.model_dump(mode="json")
    s = json.dumps(args, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "..."


# ------------------------------------------------------------------- plugins
GATE_PRIORITY = -1_000_000

class PermissionConfig(BaseModel):
    rules: list[Rule] = Field(default_factory=list)
    approval_timeout: float = 120.0
    audit: str | None = "~/.ventri/audit.jsonl"
    unattended: Unattended = Field(default_factory=Unattended)   # headless sessions only


@ventri.plugin(name="permission", config=PermissionConfig,
               provides={"policy": Policy, "approvals": ApprovalBroker, "audit": AuditLog})
def permission(ctx: Any, cfg: PermissionConfig) -> None:
    """``use: ventri_agent.permission`` -- Policy + ApprovalBroker + AuditLog (root realm)."""
    ctx.provide(Policy, Policy(cfg.rules, approval_timeout=cfg.approval_timeout, unattended=cfg.unattended))
    ctx.provide(ApprovalBroker, ApprovalBroker(timeout=cfg.approval_timeout))
    ctx.provide(AuditLog, AuditLog(expand(cfg.audit) if cfg.audit else None))


@ventri.plugin(name="permission-gate")
def gate(ctx: Any, config: Any, policy: Policy, broker: ApprovalBroker, audit: AuditLog,
         info: SessionInfo | None = None) -> None:
    """Session-scope plugin: provides ``Grants`` and intercepts ``ToolCheck``."""
    grants = ctx.provide(Grants, Grants())
    # Lowest priority: the gate decides on the *final* request, after any other
    # interceptor rewrote it, so an approval always covers exactly what runs.
    headless = bool(info is not None and info.headless)
    ctx.intercept(ToolCheck, PermissionGate(ctx, policy, broker, grants, audit, headless=headless),
                  priority=GATE_PRIORITY)
    ctx.on_dispose(grants.revoke)


plugin = permission
