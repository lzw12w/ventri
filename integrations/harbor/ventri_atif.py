"""Convert a Ventri session log (JSONL) into an ATIF trajectory (Harbor's
Agent Trajectory Interchange Format, ``harbor.models.trajectories``).

Plain dicts, no Harbor import: ``VentriAgent`` validates the result with
``Trajectory.model_validate`` when Harbor is available. The trajectory is the
full record (``msg`` records as appended; ``compact`` records only change
what later requests sent, not what happened).

Mapping: the epoch's system prompt is step 1 (``source: system``, tool specs
go to ``agent.tool_definitions``); user messages -> ``user`` steps; context
notes (time, budget, plans) -> ``system`` steps; each assistant message ->
an ``agent`` step with its reasoning, tool calls and -- as ``observation`` --
the tool results that answered them, plus the metrics of the model call that
produced it (the ``usage`` record logged just before it). A locally generated
assistant message (budget stop) has ``llm_call_count: 0`` and no metrics.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "ATIF-v1.7"


def _iso(ts: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(ts), UTC).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _args(raw: str) -> dict[str, Any]:
    try:
        v = json.loads(raw or "{}")
    except ValueError:
        return {"raw": raw}
    return v if isinstance(v, dict) else {"value": v}


def _name(wire: str) -> str:
    return wire.replace("__", ".")


def read_records(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # torn last line
    return out


def session_to_atif(records: list[dict[str, Any]], *, agent_name: str = "ventri-agent",
                    agent_version: str = "unknown", model_name: str | None = None) -> dict[str, Any] | None:
    meta = next((r for r in records if r.get("t") == "meta"), {})
    prefix = next((r for r in records if r.get("t") == "prefix"), None)
    steps: list[dict[str, Any]] = []
    pending_usage: dict[str, Any] | None = None
    last_agent: dict[str, Any] | None = None
    totals = {"prompt": 0, "completion": 0, "cached": 0, "cost": 0.0}
    models: list[str] = []

    def add(step: dict[str, Any]) -> dict[str, Any]:
        step["step_id"] = len(steps) + 1
        steps.append(step)
        return step

    if prefix is not None:
        system = prefix.get("system") or ""
        if prefix.get("memory"):
            system += "\n\n" + prefix["memory"]
        add({"source": "system", "message": system, "timestamp": _iso(prefix.get("ts"))})
    for rec in records:
        t = rec.get("t")
        if t == "usage":
            pending_usage = rec
            u = rec.get("usage") or {}
            totals["prompt"] += int(u.get("prompt_tokens") or 0)
            totals["completion"] += int(u.get("completion_tokens") or 0)
            totals["cached"] += int(u.get("cache_hit") or 0)
            totals["cost"] += float(rec.get("cost_usd") or 0.0)
            if rec.get("model") and rec["model"] not in models:
                models.append(rec["model"])
            continue
        if t != "msg":
            continue
        m = rec.get("m") or {}
        role, ts = m.get("role"), _iso(rec.get("ts"))
        if role == "user":
            last_agent = None
            add({"source": "user", "message": m.get("content") or "", "timestamp": ts})
        elif role == "system":
            last_agent = None
            add({"source": "system", "message": m.get("content") or "", "timestamp": ts,
                 "extra": {"kind": (m.get("meta") or {}).get("tail", "note")}})
        elif role == "assistant":
            local = bool((m.get("meta") or {}).get("local"))
            step: dict[str, Any] = {"source": "agent", "message": m.get("content") or "", "timestamp": ts}
            calls = m.get("tool_calls") or []
            if calls:
                step["tool_calls"] = [{"tool_call_id": c["id"], "function_name": _name(c.get("name", "")),
                                       "arguments": _args(c.get("arguments", "{}"))} for c in calls]
            if local:
                step["llm_call_count"] = 0
            else:
                step["llm_call_count"] = 1
                if m.get("reasoning_content"):
                    step["reasoning_content"] = m["reasoning_content"]
                if pending_usage is not None:
                    u = pending_usage.get("usage") or {}
                    step["model_name"] = pending_usage.get("model") or model_name
                    step["metrics"] = {
                        "prompt_tokens": int(u.get("prompt_tokens") or 0),
                        "completion_tokens": int(u.get("completion_tokens") or 0),
                        "cached_tokens": int(u.get("cache_hit") or 0),
                        "cost_usd": float(pending_usage.get("cost_usd") or 0.0),
                        "extra": {"cache_miss": int(u.get("cache_miss") or 0),
                                  "reasoning_tokens": int(u.get("reasoning_tokens") or 0),
                                  "finish_reason": pending_usage.get("finish"),
                                  "peak_pricing": bool(pending_usage.get("peak"))}}
            pending_usage = None
            last_agent = add(step)
        elif role == "tool":
            cid = m.get("tool_call_id")
            ids = {c["tool_call_id"] for c in (last_agent or {}).get("tool_calls", [])}
            result = {"source_call_id": cid if cid in ids else None, "content": m.get("content") or ""}
            if last_agent is None:
                add({"source": "system", "message": f"[tool result {cid}]", "timestamp": ts,
                     "observation": {"results": [result]}})
            else:
                last_agent.setdefault("observation", {"results": []})["results"].append(result)
    if not steps:
        return None
    for s in steps:
        if s.get("timestamp") is None:
            s.pop("timestamp", None)
    agent: dict[str, Any] = {"name": agent_name, "version": agent_version,
                             "model_name": model_name or (models[0] if models else None)}
    if prefix is not None and prefix.get("tools"):
        agent["tool_definitions"] = prefix["tools"]
    if agent["model_name"] is None:
        agent.pop("model_name")
    return {
        "schema_version": SCHEMA,
        "session_id": meta.get("id"),
        "agent": agent,
        "steps": steps,
        "final_metrics": {"total_prompt_tokens": totals["prompt"], "total_completion_tokens": totals["completion"],
                          "total_cached_tokens": totals["cached"], "total_cost_usd": round(totals["cost"], 8),
                          "total_steps": len(steps),
                          "extra": {"model_calls": sum(1 for r in records if r.get("t") == "usage"),
                                    "compactions": sum(1 for r in records if r.get("t") == "compact"),
                                    "headless": bool(meta.get("headless"))}},
    }


def convert_dir(sessions_dir: Path, **kw: Any) -> dict[str, Any] | None:
    """The newest session log in ``sessions_dir`` as ATIF (one task = one session)."""
    logs = sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    return session_to_atif(read_records(logs[-1]), **kw)


if __name__ == "__main__":   # python ventri_atif.py SESSION.jsonl > trajectory.json
    import sys
    print(json.dumps(session_to_atif(read_records(Path(sys.argv[1]))), ensure_ascii=False, indent=1))
