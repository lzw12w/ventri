"""The CLI channel (ScriptedTerminal) and the ``va`` command."""
from __future__ import annotations

import argparse
import json

import pytest

from ventri_agent import cli as va
from ventri_agent.channels.cli import CliChannel, CliConfig, ScriptedTerminal
from ventri_agent.memory import LongTermMemory

from .harness import Env, call

pytestmark = pytest.mark.anyio


@pytest.fixture
def vhome(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("VENTRI_HOME", str(h))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return h


async def run_channel(env: Env, inputs: list[str], **cfg) -> tuple[CliChannel, str]:
    term = ScriptedTerminal(list(inputs))
    ch = CliChannel(env.kernel, CliConfig(**cfg), term)
    await ch.run()
    return ch, term.text


async def test_channel_turn_thinking_fold_tools_and_approval(tmp_path):
    script = [{"reasoning": "I should write the file.", "tool_calls": [call("t.write", {"path": "a.md"})]},
              {"content": "Done writing."}]
    async with Env(tmp_path, script, extract_memory=False) as env:
        ch, out = await run_channel(env, ["please write a.md", "y", "/cost", "/exit"])
        assert "▸ thought (24 chars)" in out and "I should write" not in out  # folded
        assert "[approval] t.write" in out and "approve? y" in out
        assert "✓ wrote a.md" in out and "Done writing." in out
        assert "[turn 1 · 2 steps · 1 tools" in out and "cache" in out
        assert "rate" in out and "¥" in out  # /cost
        assert env.probe.calls == [("t.write", "a.md")]
        assert ch.results[0].status == "ok" and ch.done.is_set()


async def test_channel_show_thinking_deny_and_session_grant(tmp_path):
    w = call("t.write", {"path": "b.md"})
    script = [{"reasoning": "visible reasoning", "tool_calls": [w]}, {"content": "ok1"},
              {"tool_calls": [w]}, {"content": "ok2"}, {"tool_calls": [w]}, {"content": "ok3"}]
    async with Env(tmp_path, script, extract_memory=False) as env:
        _, out = await run_channel(env, ["/think show", "one", "maybe", "n", "two", "s", "three", "/exit"])
        assert "visible reasoning" in out
        assert "please answer y / s / n" in out
        assert "✗ DENIED" in out
        assert out.count("[approval]") == 2  # third call covered by the session grant
        assert env.probe.calls == [("t.write", "b.md"), ("t.write", "b.md")]


async def test_channel_irreversible_offers_no_session_option(tmp_path):
    async with Env(tmp_path, [{"tool_calls": [call("t.send", {"text": "hi"})]}, {"content": "x"}],
                   extract_memory=False) as env:
        _, out = await run_channel(env, ["send it", "s", "y", "/exit"])
        assert "[s] allow for this session" not in out and "please answer y / n" in out
        assert env.probe.calls == [("t.send", "hi")]


async def test_channel_error_retry_think_and_commands(tmp_path):
    script = [{"error": 503}, {"content": "recovered"}, {"content": "planned"}, {"content": "after plan"}]
    async with Env(tmp_path, script, extract_memory=False) as env:
        env.kernel.get(LongTermMemory).add("Jeff likes tea", "preference")
        _ch, out = await run_channel(env, [
            "hello", "/retry", "/think low", "/think", "/think max", "do it", "/memory", "/memory search tea",
            "/epoch", "/tree", "/sessions", "/help", "/nonsense", "/exit"])
        assert "model call failed" in out and "/retry to try again" in out
        assert "recovered" in out
        assert "thinking effort: low" in out and "effort=low" in out
        assert "planning with deepseek-v4-pro" in out and "after plan" in out
        assert "#1 [preference] Jeff likes tea" in out
        assert "epoch 2:" in out and "session-manager" in out
        assert "unknown command /nonsense" in out and "/compact" in out
        low_req = env.provider.requests[2]
        assert low_req.model == "deepseek-v4-pro"  # the plan sub-call
        assert env.provider.requests[3].effort == "low"


async def test_channel_suspend_and_continue(tmp_path):
    async with Env(tmp_path, [{"content": "a"}, {"content": "b"}], extract_memory=False) as env:
        ch, out = await run_channel(env, ["one", "/suspend"])
        sid = ch.session_id
        assert f"va chat --session {sid}" in out
        ch2, out2 = await run_channel(env, ["two", "/exit"], resume_last=True)
        assert ch2.session_id == sid and "resumed, " in out2
        from ventri_agent.session import SessionLog
        rep = SessionLog.replay(tmp_path / "sessions" / f"{sid}.jsonl")
        assert [m.content for m in rep.history if m.role == "user"] == ["one", "two"]
        assert rep.state == "ended"


async def test_channel_eof_ends_session_and_extracts(tmp_path):
    ext = {"json": {"memories": [{"kind": "preference", "text": "Jeff drinks green tea"}]}}
    async with Env(tmp_path, [{"content": "noted"}, ext]) as env:
        _, out = await run_channel(env, ["I drink green tea"])
        assert "ending session" in out and "remembered: #1 [preference] Jeff drinks green tea" in out


# --------------------------------------------------------------------- va
def test_va_reserved_commands_exit_2(capsys):
    for cmd in ("propose", "history", "rollback"):
        assert va.main([cmd]) == 2
    assert "not available in 0.2" in capsys.readouterr().err


def test_va_init_creates_home(vhome, capsys):
    assert va.main(["init", "--no-key"]) == 0
    cfg = (vhome / "ventri.yml").read_text()
    assert "ventri_agent.providers.deepseek" in cfg and "deepseek-v4-pro" in cfg
    assert (vhome / "personas" / "default.md").exists() and (vhome / "workspace").is_dir()
    out = capsys.readouterr().out
    assert "DEEPSEEK_API_KEY" in out and "Data policy" in out
    assert va.main(["init", "--no-key"]) == 0
    assert "exists" in capsys.readouterr().out
    from ventri_std.config import load_document
    doc = load_document(vhome / "ventri.yml")
    assert doc.data["agents"]["default"]["route"] == "default"


async def test_va_chat_fake_end_to_end(vhome, tmp_path):
    script = tmp_path / "s.json"
    script.write_text(json.dumps([
        {"reasoning": "time first", "tool_calls": [{"name": "time.now", "arguments": {}}]},
        {"content": "It is now."},
        {"tool_calls": [{"name": "fs.write", "arguments": {"path": "x.md", "content": "hi"}}]},
        {"content": "wrote it"}]))
    term = ScriptedTerminal(["what time is it?", "write x.md", "y", "/cost", "/exit"])
    args = va.build_parser().parse_args(["chat", "--fake", str(script)])
    assert await va.chat(args, terminal=term) == 0
    out = term.text
    assert "agent default · deepseek-flash" in out and "⚙ time.now" in out and "It is now." in out
    assert "[approval] fs.write" in out and (vhome / "workspace" / "x.md").read_text() == "hi"
    assert (vhome / "audit.jsonl").exists() and list((vhome / "sessions").glob("*.jsonl"))
    assert not (vhome / ".va-fake.yml").exists()


def test_va_reports(vhome, capsys):
    from ventri_agent.session import SessionLog
    sdir = vhome / "sessions"
    log = SessionLog(sdir / "20261008-100000-abcd.jsonl")
    log.append("meta", id="20261008-100000-abcd", agent="default")
    from ventri_agent.messages import Message
    log.message(Message.user("hello there"))
    for hit, miss, peak in ((0, 1000, True), (900, 100, False)):
        log.append("usage", model="deepseek-flash", route="default", peak=peak, cost_usd=0.001,
                   usage={"prompt_tokens": hit + miss, "completion_tokens": 50, "cache_hit": hit,
                          "cache_miss": miss, "reasoning_tokens": 10})
    log.append("turn", n=1, status="ok")
    log.close()
    assert va.main(["sessions"]) == 0
    out = capsys.readouterr().out
    assert "20261008-100000-abcd" in out and "hello there" in out and "45%" in out
    assert va.main(["cost", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["calls"] == 2 and rows[0]["hit_rate"] == 0.45 and rows[0]["peak_calls"] == 1
    assert rows[0]["cost_cny"] == pytest.approx(0.0144)
    assert va.main(["cost"]) == 0 and "45.0%" in capsys.readouterr().out


def test_va_memory_commands(vhome, tmp_path, capsys):
    vhome.mkdir(parents=True)
    m = LongTermMemory(vhome / "memory.db")
    m.add("Jeff prefers dark mode", "preference")
    m.add("my password is hunter2", "fact")
    m.close()
    assert va.main(["memory", "list"]) == 0 and "dark mode" in capsys.readouterr().out
    assert va.main(["memory", "pending"]) == 0 and "(pending)" in capsys.readouterr().out
    assert va.main(["memory", "confirm", "2"]) == 0
    assert va.main(["memory", "search", "password"]) == 0 and "hunter2" in capsys.readouterr().out
    out = tmp_path / "mem.md"
    assert va.main(["memory", "export", "-o", str(out)]) == 0 and "## preference" in out.read_text()
    assert va.main(["memory", "forget", "#1"]) == 0
    assert va.main(["memory", "forget", "1"]) == 1
    assert va.main(["memory", "import", str(out)]) == 0 and "imported 1" in capsys.readouterr().out


def test_va_tree_and_doctor_on_fake_config(vhome, tmp_path, capsys):
    script = tmp_path / "s.json"
    script.write_text("[]")
    vhome.mkdir(parents=True)
    cfg = vhome / "ventri.yml"
    cfg.write_text(va.config_text(fake_script=str(script)))
    assert va.main(["tree", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "session-manager" in out and "tool:fs" in out
    assert va.main(["doctor", "--config", str(cfg)]) == 0
    assert "FTS5" in capsys.readouterr().out


def test_va_chat_without_config(vhome, capsys):
    assert va.main(["chat"]) == 2
    assert "va init" in capsys.readouterr().err


def test_default_config_dry_run(vhome, capsys, monkeypatch):
    """The `va init` template resolves and loads (dry run) with a placeholder key."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder-not-a-key")
    vhome.mkdir(parents=True)
    p = vhome / "ventri.yml"
    # probe: false keeps the dry run off the network
    text = va.config_text().replace("    id: ds\n    config:\n", "    id: ds\n    config:\n      probe: false\n")
    assert "probe: false" in text
    p.write_text(text)
    assert va.main(["tree", "--config", str(p)]) == 0
    out = capsys.readouterr().out
    assert "provider:deepseek" in out and "placeholder-not-a-key" not in out


def test_args_parse():
    a = va.build_parser().parse_args(["chat", "-c", "--agent", "x", "--show-thinking"])
    assert isinstance(a, argparse.Namespace) and a.cont and a.agent == "x" and a.show_thinking
