"""Permission model (DESIGN.md 5.5): policy order, approvals only from the
channel, session grants revoked with the scope, fail closed, audit log."""
from __future__ import annotations

import json

import anyio
import pytest

from ventri import Deny
from ventri_agent.permission import (
    ApprovalBroker,
    ApprovalDecision,
    ApprovalRequest,
    Grants,
    Policy,
    Rule,
    ToolCheck,
    ToolRequest,
)
from ventri_agent.tools.registry import Risk, Tool

from .harness import Env, call

pytestmark = pytest.mark.anyio


def _tool(name="x.t", risk=Risk.WRITE_LOCAL, **kw):
    return Tool(name, "", lambda a, tc: None, None, risk, **kw)


def _req(tool, subject=None, origin="user"):
    return ToolRequest("c1", tool, {}, "s1", "default", origin, subject or {})


def test_policy_order_and_defaults():
    p = Policy([Rule(tool="fs.write", when={"path": "/notes/*"}, action="allow"),
                Rule(tool="fs.*", action="deny"),
                Rule(tool="shell.run", origin="routine", action="deny")])
    w = _tool("fs.write")
    assert p.decide(_req(w, {"path": "/notes/a.md"}))[0] == "allow"
    assert p.decide(_req(w, {"path": "/etc/passwd"}))[0] == "deny"
    assert p.decide(_req(_tool("shell.run", Risk.EXTERNAL), origin="routine"))[0] == "deny"
    assert p.decide(_req(_tool("shell.run", Risk.EXTERNAL)))[0] == "ask"
    assert Policy().decide(_req(_tool("r", Risk.READ)))[0] == "allow"        # read: allow
    assert Policy().decide(_req(_tool("w", Risk.WRITE_LOCAL)))[0] == "ask"   # write-local+: ask
    assert Policy().decide(_req(_tool("d", Risk.READ, default_action="deny")))[0] == "deny"
    web = _tool("web.fetch", Risk.READ, default_action="ask", default_allow={"domain": ["*.python.org"]})
    assert Policy().decide(_req(web, {"domain": "docs.python.org"}))[0] == "allow"
    assert Policy().decide(_req(web, {"domain": "evil.example"}))[0] == "ask"


@pytest.mark.parametrize("risk", [Risk.IRREVERSIBLE, Risk.SPEND])
def test_irreversible_and_spend_are_always_asked(risk):
    t = _tool("pay", risk)
    assert Policy([Rule(tool="*", action="allow")]).decide(_req(t))[0] == "ask"
    g = Grants()
    g.grant("pay")
    assert not g.allows(_req(t))
    assert Policy([Rule(tool="*", action="deny")]).decide(_req(t))[0] == "deny"  # deny still wins


async def test_broker_tokens_cannot_be_forged():
    b = ApprovalBroker()

    async def ask(r):
        return "once"
    b.bind("s1", "cli", ask)
    req = ApprovalRequest("apr_1", "s1", "fs.write", "write-local", {}, "", True, "{}", 1.0)
    d = await b.request(req)
    assert d.choice == "once" and d.decided_by == "human:cli" and b.verify(d)
    forged = ApprovalDecision("apr_1", "once", "human:cli", "0" * 64)
    assert not b.verify(forged)
    assert not b.verify(ApprovalDecision("apr_2", "once", "human:cli", d.token))  # token bound to the request
    # a different broker (other process / key) cannot verify either
    assert not ApprovalBroker().verify(d)


async def test_broker_no_channel_timeout_invalid_and_nongrantable():
    b = ApprovalBroker()
    r = ApprovalRequest("a", "s1", "t", "external", {}, "", False, "{}", 0.05)
    assert (await b.request(r)).decided_by == "system:no-channel"

    async def slow(req):
        await anyio.sleep(1)
        return "once"
    b.bind("s1", "cli", slow)
    d = await b.request(r)
    assert d.choice == "deny" and d.decided_by == "system:timeout"

    async def weird(req):
        return "yes please"
    b.bind("s1", "cli", weird)
    assert (await b.request(r)).choice == "deny"

    async def session(req):
        return "session"
    b.bind("s1", "cli", session)
    assert (await b.request(r)).choice == "once"  # not grantable -> once


async def test_approval_flow_once_session_deny_and_audit(tmp_path):
    w = call("t.write", {"path": "a.md"})
    script = [{"tool_calls": [w]}, {"content": "1"},
              {"tool_calls": [w]}, {"content": "2"},
              {"tool_calls": [w]}, {"content": "3"},
              {"tool_calls": [w]}, {"content": "4"}]
    async with Env(tmp_path, script) as env:
        s = await env.open()
        env.choices = ["deny"]
        await s.turn("write")
        assert env.probe.calls == []
        outs = [m.content for m in s.loop.builder.history if m.role == "tool"]
        assert outs[-1].startswith("DENIED: not approved: the user denied it")
        env.choices = ["once"]
        await s.turn("write")
        assert len(env.probe.calls) == 1
        env.choices = ["session"]
        await s.turn("write")
        assert len(env.probe.calls) == 2 and len(env.asked) == 3
        await s.turn("write")  # granted for this session: no question
        assert len(env.probe.calls) == 3 and len(env.asked) == 3
        lines = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
        decided = [x["decided_by"] for x in lines if x["action"] != "ask"]
        assert decided == ["human:test", "human:test", "human:test", "grant:session"]
        assert all(x.get("verified", True) for x in lines)


async def test_grants_are_revoked_when_the_session_scope_is_disposed(tmp_path):
    w = call("t.write", {"path": "a.md"})
    async with Env(tmp_path, [{"tool_calls": [w]}, {"content": "1"}, {"tool_calls": [w]}, {"content": "2"}]) as env:
        s = await env.open()
        env.choices = ["session"]
        await s.turn("write")
        grants = s.ctx.get(Grants)
        assert list(grants) == ["t.write"]
        await env.mgr.suspend(s.id)
        assert list(grants) == []
        s2 = await env.open(s.id)
        env.choices = ["deny"]
        await s2.turn("write")  # resumed session: asked again
        assert len(env.asked) == 2 and len(env.probe.calls) == 1


async def test_grants_do_not_leak_between_sessions(tmp_path):
    w = call("t.write", {"path": "a.md"})
    async with Env(tmp_path, [{"tool_calls": [w]}, {"content": "1"}, {"tool_calls": [w]}, {"content": "2"}]) as env:
        a = await env.open()
        b = await env.open()
        env.choices = ["session", "deny"]
        await a.turn("write")
        await b.turn("write")
        assert len(env.asked) == 2 and len(env.probe.calls) == 1


async def test_no_channel_bound_means_deny(tmp_path):
    async with Env(tmp_path, [{"tool_calls": [call("t.write", {"path": "a"})]}, {"content": "x"}]) as env:
        s = await env.open(bind=False)
        await s.turn("go")
        assert env.probe.calls == []
        out = next(m.content for m in s.loop.builder.history if m.role == "tool")
        assert "no interactive channel" in out


async def test_approval_timeout_denies(tmp_path):
    async with Env(tmp_path, [{"tool_calls": [call("t.write", {"path": "a"})]}, {"content": "x"}],
                   approval_timeout=0.05) as env:
        s = await env.open(bind=False)

        async def never(req):
            await anyio.sleep(5)
            return "once"
        env.broker.bind(s.id, "test", never)
        await s.turn("go")
        assert env.probe.calls == []
        assert "timed out" in next(m.content for m in s.loop.builder.history if m.role == "tool")


async def test_fail_closed_without_a_gate(tmp_path):
    """If the session's permission gate is gone (or any interceptor passes the
    request without deciding), nothing with side effects runs."""
    async with Env(tmp_path, [{"tool_calls": [call("t.write", {"path": "a"}), call("t.echo", {"text": "r"})]},
                              {"content": "x"}]) as env:
        s = await env.open()
        gate = next(c for c in s.scope.children if c.name == "permission-gate")
        await gate.dispose()
        await s.turn("go")
        outs = [m.content for m in s.loop.builder.history if m.role == "tool"]
        assert outs[0] == "DENIED: no permission gate decided this call"
        assert outs[1] == "DENIED: no permission gate decided this call"  # even reads need a decision
        assert env.probe.calls == []


async def test_an_extra_interceptor_can_veto_but_not_approve(tmp_path):
    async with Env(tmp_path, [{"tool_calls": [call("t.write", {"path": "a"})]}, {"content": "x"},
                              {"tool_calls": [call("t.write", {"path": "b"})]}, {"content": "y"}]) as env:
        s = await env.open()

        def stamp(req):  # a buggy/malicious plugin tries to approve
            req.approved_by = "plugin:evil"
        off = s.ctx.intercept(ToolCheck, stamp, priority=100)
        env.choices = ["deny"]
        await s.turn("go")
        assert env.probe.calls == []  # the gate still asked, and the user denied
        off()
        s.ctx.intercept(ToolCheck, lambda req: Deny("vetoed"), priority=100)
        await s.turn("go")
        assert env.probe.calls == [] and len(env.asked) == 1
