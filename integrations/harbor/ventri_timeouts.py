"""Time budget for a Harbor trial: Ventri's wall clock follows the task's own
agent timeout instead of a fixed number (no Harbor import, unit-tested)."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path


def agent_timeout_sec(trial_dir: Path, cwd: Path | None = None) -> float | None:
    """The trial's effective agent timeout, resolved like Harbor's Trial does:
    ``min(override or task [agent].timeout_sec, max_timeout_sec) * multiplier``."""
    try:
        cfg = json.loads((trial_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    agent = cfg.get("agent") or {}
    base = agent.get("override_timeout_sec")
    if base is None:
        task_path = Path((cfg.get("task") or {}).get("path") or "")
        if not task_path.is_absolute():
            task_path = (cwd or Path.cwd()) / task_path
        try:
            base = tomllib.loads((task_path / "task.toml").read_text()).get("agent", {}).get("timeout_sec")
        except (OSError, tomllib.TOMLDecodeError):
            return None
    if base is None:
        return None
    mult = cfg.get("agent_timeout_multiplier")
    if mult is None:
        mult = cfg.get("timeout_multiplier", 1.0)
    cap = agent.get("max_timeout_sec") or float("inf")
    return min(float(base), float(cap)) * float(mult or 1.0)


def ventri_wall(timeout: float | None, fallback: float) -> float:
    """Ventri's own budget: stop (and write logs) well before Harbor kills the run."""
    if timeout is None:
        return fallback
    margin = max(45.0, 0.08 * timeout)
    return max(60.0, timeout - margin)
