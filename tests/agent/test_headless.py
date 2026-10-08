"""Headless (unattended) mode: explicit opt-in, approvals by the configured
``unattended`` policy (fail closed by default, audited), replaceable system
prompt, no time notes, and ``va run``."""
from __future__ import annotations

import json

import pytest

from ventri_agent import cli as va
from ventri_agent.context import HEADLESS_SYSTEM, RULES
from ventri_agent.sessions import SessionError

from .harness import Env, call

pytestmark = pytest.mark.anyio


def write_then_send():
    return [{"tool_calls": [call("t.write", {"path": "a.md"})]},
            {"tool_calls": [call("t.send", {"text": "hi"})]},
            {"content": "done"}]


def decisions(env: Env) -> list[tuple[str, str, str]]:
    return [(r["tool"], r["action"], r["decided_by"]) for r in env.audit.records]


async def test_headless_default_denies_without_asking(tmp_path):
    async with Env(tmp_path, write_then_send()) as env:
        s = await env.open(headless=True)          # a channel is even bound: it must not be asked
        r = await s.turn("go")
        assert r.status == "ok" and env.asked == [] and env.probe.calls == []
        assert decisions(env) == [("t.write", "deny", "unattended-policy"), ("t.send", "deny", "unattended-policy")]
        tool_msgs = [m.content for m in s.loop.builder.history if m.role == "tool"]
        assert all("not approved: unattended run" in (c or "") for c in tool_msgs)
        assert env.audit.records[0]["headless"] is True


async def test_headless_allow_but_irreversible_needs_its_own_opt_in(tmp_path):
    async with Env(tmp_path, write_then_send(), unattended={"ask": "allow"}) as env:
        s = await env.open(headless=True)
        await s.turn("go")
        assert env.probe.calls == [("t.write", "a.md")]
        assert decisions(env) == [("t.write", "allow", "unattended-policy"), ("t.send", "deny", "unattended-policy")]
    async with Env(tmp_path, write_then_send(), unattended={"ask": "allow", "irreversible": "allow"}) as env:
        s = await env.open(headless=True)
        await s.turn("go")
        assert [c[0] for c in env.probe.calls] == ["t.write", "t.send"]


async def test_headless_rules_still_apply_first(tmp_path):
    rules = [{"tool": "t.write", "action": "deny"}]
    async with Env(tmp_path, write_then_send(), rules=rules, unattended={"ask": "allow", "irreversible": "allow"}) as env:
        s = await env.open(headless=True)
        await s.turn("go")
        assert decisions(env)[0] == ("t.write", "deny", "policy") and env.probe.calls == [("t.send", "hi")]


async def test_unattended_policy_never_applies_to_interactive_sessions(tmp_path):
    async with Env(tmp_path, write_then_send(), unattended={"ask": "allow", "irreversible": "allow"}) as env:
        env.choices = ["deny", "deny"]
        s = await env.open()                        # not headless: the human is asked
        await s.turn("go")
        assert [a.tool for a in env.asked] == ["t.write", "t.send"] and env.probe.calls == []
        assert not any(r["decided_by"] == "unattended-policy" for r in env.audit.records)
    async with Env(tmp_path, write_then_send(), unattended={"ask": "allow"}) as env:
        s = await env.open(bind=False)              # non-interactive but not headless: fail closed
        await s.turn("go")
        assert env.probe.calls == [] and {r["decided_by"] for r in env.audit.records if r["action"] == "deny"} \
            == {"system:no-channel"}


async def test_headless_prompt_language_and_no_time_notes(tmp_path):
    async with Env(tmp_path, [{"content": "ok"}, {"content": "ok"}]) as env:
        s = await env.open(headless=True)
        await s.turn("Fix the build")
        b = s.loop.builder
        assert b.epoch.system == HEADLESS_SYSTEM and "中文" not in b.epoch.system
        assert "language of the task" in b.epoch.system and "harness's own" in b.epoch.system
        assert "leave the working deliverable in place" in b.epoch.system
        assert not any(m.meta.get("tail") == "time" for m in b.history)
        meta = env.log_records(s.id)[0]
        assert meta["t"] == "meta" and meta["headless"] is True
        sid = s.id
        await s.suspend()
        s2 = await env.open(sid)                    # headless is never inherited from the log
        assert s2.info.headless is False
        await s2.turn("again")
        assert env.log_records(sid)[-1]["t"] in ("turn",)


async def test_interactive_rules_keep_deliverable_guidance(tmp_path):
    assert "leave the working deliverable in place" in RULES and "background=true" in RULES
    async with Env(tmp_path, [{"content": "ok"}]) as env:
        s = await env.open()
        await s.turn("hi")
        assert s.loop.builder.epoch.system.endswith(RULES)
        assert any(m.meta.get("tail") == "time" for m in s.loop.builder.history)


async def test_system_prompt_replaces_persona_and_rules(tmp_path):
    f = tmp_path / "sys.md"
    f.write_text("You are a build bot. Answer in English.\n")
    agents = {"inline": {"system_prompt": "Custom line one\nline two", "time_notes": False},
              "file": {"system_prompt": str(f), "time_notes": True},
              "batch": {"mode": "headless"}}
    async with Env(tmp_path, [{"content": "ok"}] * 3, agents=agents) as env:
        s = await env.open(agent="inline")
        await s.turn("hi")
        assert s.loop.builder.epoch.system == "Custom line one\nline two"
        assert not any(m.meta.get("tail") == "time" for m in s.loop.builder.history)
        s = await env.open(agent="file", headless=True)
        await s.turn("hi")
        assert s.loop.builder.epoch.system == "You are a build bot. Answer in English."
        assert any(m.meta.get("tail") == "time" for m in s.loop.builder.history)   # explicitly on
        with pytest.raises(SessionError, match="headless-only"):
            await env.open(agent="batch")
        s = await env.open(agent="batch", headless=True)
        assert s.info.headless


# ------------------------------------------------------------------ va run
def _config(vhome, tmp_path, steps, extra_perm: str = "", agents: str = "") -> None:
    script = tmp_path / "s.json"
    script.write_text(json.dumps(steps))
    vhome.mkdir(parents=True, exist_ok=True)
    text = va.config_text(fake_script=str(script))
    if extra_perm:
        text = text.replace("      rules: []", "      rules: []\n" + extra_perm)
    if agents:
        text = text.replace("agents:\n", "agents:\n" + agents)
    (vhome / "ventri.yml").write_text(text)


@pytest.fixture
def vhome(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("VENTRI_HOME", str(h))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return h


WRITE = [{"tool_calls": [{"name": "fs.write", "arguments": {"path": "x.md", "content": "hi"}}]},
         {"content": "finished"}]


def test_va_run_without_headless_fails_closed(vhome, tmp_path, capsys):
    _config(vhome, tmp_path, WRITE, "      unattended: {ask: allow}")
    assert va.main(["run", "--json", "-q", "write x.md"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok" and out["headless"] is False and out["text"] == "finished"
    assert not (vhome / "workspace" / "x.md").exists()
    audit = [json.loads(x) for x in (vhome / "audit.jsonl").read_text().splitlines()]
    assert audit[-1]["decided_by"] == "system:no-channel"


def test_va_run_headless_uses_unattended_policy(vhome, tmp_path, capsys):
    _config(vhome, tmp_path, WRITE, "      unattended: {ask: allow}")
    task = tmp_path / "task.txt"
    task.write_text("write x.md")
    assert va.main(["run", "--headless", "--task-file", str(task)]) == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == "finished" and "headless" in cap.err and "fs.write" in cap.err
    assert (vhome / "workspace" / "x.md").read_text() == "hi"
    audit = [json.loads(x) for x in (vhome / "audit.jsonl").read_text().splitlines()]
    assert audit[-1]["decided_by"] == "unattended-policy" and audit[-1]["action"] == "allow"


def test_va_run_headless_default_policy_denies(vhome, tmp_path, capsys):
    _config(vhome, tmp_path, WRITE)
    assert va.main(["run", "--headless", "-q", "write x.md"]) == 0
    assert not (vhome / "workspace" / "x.md").exists()


def test_va_run_headless_only_preset_needs_flag(vhome, tmp_path, capsys):
    _config(vhome, tmp_path, [{"content": "x"}], agents="  bot: { mode: headless }\n")
    assert va.main(["run", "-a", "bot", "-q", "hello"]) == 2
    assert "headless-only" in capsys.readouterr().err
    assert va.main(["run", "   "]) == 2 and "empty task" in capsys.readouterr().err
