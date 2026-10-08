"""Harbor adapter helpers that do not need Harbor: the ATIF converter and the
task-timeout resolution (integrations/harbor)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# integrations/harbor is on the pytest / pyright path (pyproject.toml)
from ventri_atif import convert_dir, read_records, session_to_atif
from ventri_timeouts import agent_timeout_sec, ventri_wall

from tests.agent.harness import Env, call

AGENT_ONLY = ("model_name", "reasoning_effort", "reasoning_content", "tool_calls", "metrics")


def check_atif(d: dict) -> None:
    """The invariants Harbor's Trajectory model validates."""
    assert d["schema_version"].startswith("ATIF-v1.") and d["agent"]["name"] and d["agent"]["version"]
    assert [s["step_id"] for s in d["steps"]] == list(range(1, len(d["steps"]) + 1))
    for s in d["steps"]:
        assert s["source"] in ("system", "user", "agent")
        if s["source"] != "agent":
            assert not any(k in s for k in AGENT_ONLY)
        ids = {tc["tool_call_id"] for tc in s.get("tool_calls", [])}
        for r in (s.get("observation") or {}).get("results", []):
            assert r["source_call_id"] is None or r["source_call_id"] in ids
        if s.get("llm_call_count") == 0:
            assert "metrics" not in s and "reasoning_content" not in s
        for tc in s.get("tool_calls", []):
            assert isinstance(tc["arguments"], dict)
    json.dumps(d)


@pytest.mark.anyio
async def test_session_log_to_atif(tmp_path):
    script = [{"reasoning": "look first", "tool_calls": [call("t.echo", {"text": "a"}, id="c1"),
                                                          call("t.web", {"text": "page"}, id="c2")]},
              {"tool_calls": [call("t.echo", {"text": "b"}, id="c3")]}]
    async with Env(tmp_path, script, budget={"max_steps": 2}) as env:
        s = await env.open(headless=True)
        r = await s.turn("do the task")
        assert r.status == "budget"
        sid = s.id
    d = session_to_atif(read_records(tmp_path / "sessions" / f"{sid}.jsonl"), agent_version="test")
    assert d is not None
    check_atif(d)
    srcs = [st["source"] for st in d["steps"]]
    assert srcs[0] == "system" and "user" in srcs
    first = next(st for st in d["steps"] if st["source"] == "agent")
    assert first["reasoning_content"] == "look first" and first["llm_call_count"] == 1
    assert [tc["function_name"] for tc in first["tool_calls"]] == ["t.echo", "t.web"]
    assert [x["source_call_id"] for x in first["observation"]["results"]] == ["c1", "c2"]
    assert first["metrics"]["prompt_tokens"] > 0 and "cached_tokens" in first["metrics"]
    assert d["agent"]["tool_definitions"] and d["session_id"] == sid
    fm = d["final_metrics"]
    assert fm["total_steps"] == len(d["steps"]) and fm["extra"]["headless"] is True
    assert fm["total_prompt_tokens"] == sum(st.get("metrics", {}).get("prompt_tokens", 0) for st in d["steps"])
    assert convert_dir(tmp_path / "sessions") is not None


def test_atif_validates_with_harbor_if_installed(tmp_path):
    traj = pytest.importorskip("harbor.models.trajectories")
    log = tmp_path / "s.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in [
        {"t": "meta", "ts": 1.0, "id": "s"}, {"t": "prefix", "ts": 1.0, "system": "sys", "tools": []},
        {"t": "msg", "ts": 2.0, "m": {"role": "user", "content": "hi"}},
        {"t": "usage", "ts": 3.0, "model": "m", "usage": {"prompt_tokens": 5}, "cost_usd": 0.1},
        {"t": "msg", "ts": 3.0, "m": {"role": "assistant", "content": "hello", "reasoning_content": ""}}]))
    traj.Trajectory.model_validate(session_to_atif(read_records(log)))


def trial(tmp_path: Path, timeout: float | None, **cfg) -> Path:
    task = tmp_path / "tasks" / "t1"
    task.mkdir(parents=True)
    agent = f"[agent]\ntimeout_sec = {timeout}\n" if timeout is not None else ""
    (task / "task.toml").write_text(f"[verifier]\ntimeout_sec = 900.0\n\n{agent}")
    d = tmp_path / "jobs" / "j" / "t1__abc"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"task": {"path": "tasks/t1"}, "agent": cfg.pop("agent", {}),
                                               **cfg}))
    return d


def test_agent_timeout_follows_the_task(tmp_path):
    assert agent_timeout_sec(trial(tmp_path / "a", 750.0), cwd=tmp_path / "a") == 750.0
    assert agent_timeout_sec(trial(tmp_path / "b", 900.0, timeout_multiplier=2.0), cwd=tmp_path / "b") == 1800.0
    assert agent_timeout_sec(trial(tmp_path / "c", 900.0, agent={"override_timeout_sec": 300},
                                   agent_timeout_multiplier=1.5, timeout_multiplier=9), cwd=tmp_path / "c") == 450.0
    assert agent_timeout_sec(trial(tmp_path / "d", 900.0, agent={"max_timeout_sec": 600}),
                             cwd=tmp_path / "d") == 600.0
    assert agent_timeout_sec(trial(tmp_path / "e", None), cwd=tmp_path / "e") is None
    assert agent_timeout_sec(tmp_path / "missing") is None
    assert ventri_wall(750.0, 870) == 690.0          # 8% margin, at least 45 s
    assert ventri_wall(300.0, 870) == 255.0
    assert ventri_wall(None, 870) == 870
    assert ventri_wall(50.0, 870) == 60.0
