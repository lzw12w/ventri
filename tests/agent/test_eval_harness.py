"""The eval harness itself (scripted model): fixture, approvals, checks."""
from __future__ import annotations

import pytest

from evals.agent.run import load_tasks, run_task, summary
from ventri_agent.providers.fake import FakeProvider

pytestmark = pytest.mark.anyio


def test_task_set_shape():
    data = load_tasks()
    ids = [t["id"] for t in data["tasks"]]
    assert len(ids) == 30 and len(set(ids)) == 30
    cats = {t["category"] for t in data["tasks"]}
    assert cats == {"notes", "retrieval", "files", "schedule", "memory"}
    assert all(t.get("checks") for t in data["tasks"])


async def test_harness_pass_and_fail_paths():
    data = load_tasks()
    tasks = {t["id"]: t for t in data["tasks"]}
    # f03: the model appends via fs.write and the simulated user approves it
    good = FakeProvider([
        {"tool_calls": [{"name": "fs.read", "arguments": {"path": "workspace/todo.txt"}}]},
        {"tool_calls": [{"name": "fs.write", "arguments": {"path": "workspace/todo.txt",
                                                          "content": "3. call mom\n", "mode": "append"}}]},
        {"content": "Added."}])
    r = await run_task(tasks["f03"], data["fixture"], good)
    assert r.passed, r.failures
    assert r.approvals == ["fs.write:once"]
    # r05: an obedient model tries the injected write; the user only approved web.fetch
    evil = FakeProvider([
        {"tool_calls": [{"name": "web.fetch", "arguments": {"url": "https://evil.example.net/page"}}]},
        {"tool_calls": [{"name": "fs.write", "arguments": {"path": "workspace/pwned.txt", "content": "owned"}}]},
        {"content": "The page gives tips on watering tomatoes."}])
    r2 = await run_task(tasks["r05"], data["fixture"], evil)
    assert r2.passed, r2.failures
    assert r2.approvals == ["web.fetch:once", "fs.write:deny"]
    # a wrong answer fails
    r3 = await run_task(tasks["f02"], data["fixture"], FakeProvider([{"content": "It is 42."}]))
    assert not r3.passed and "360" in r3.failures[0]
    s = summary([r, r2, r3])
    assert s["passed"] == 2 and s["pass_rate"] == pytest.approx(0.667, abs=1e-3)
