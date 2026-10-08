"""Snapshot / tree dump for debugging and tests."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .plugin import keyname

if TYPE_CHECKING:  # pragma: no cover
    from .fiber import Fiber
    from .kernel import Kernel


def _bindings(kernel: Kernel) -> list:
    out = list(kernel._services.values())
    for f in kernel._walk():
        tx = f.tx
        if tx is not None:
            out += [b for b in tx._overlay.values() if b is not None and b not in out]
    return out


_SECRET = ("key", "token", "secret", "password")


def _redact(config: Any) -> Any:
    if isinstance(config, dict):
        return {k: "***" if any(w in str(k).lower() for w in _SECRET) else v
                for k, v in config.items()}
    return config


def snapshot(kernel: Kernel) -> dict[str, Any]:
    bindings = _bindings(kernel)

    def node(f: Fiber) -> dict[str, Any]:
        return {
            "id": f.id,
            "name": f.name,
            "state": f.state.value,
            "staged": f.parent is not None and f.tx is not None,
            "config": _redact(f.raw_config),
            "inject": [keyname(k) for k in f.inject],
            "provides": [keyname(b.key) for b in bindings if b.owner is f],
            "tasks": sum(1 for e in f._effects if e.label.startswith("task:")),
            "effects": len(f._effects),
            "error": repr(f.error) if f.error else None,
            "children": [node(c) for c in f.children],
        }

    return {
        "fibers": node(kernel.fiber),
        "services": {keyname(k): b.owner.label for k, b in kernel._services.items()},
    }


def render_tree(kernel: Kernel) -> str:
    snap = snapshot(kernel)
    lines: list[str] = []

    def fmt(n: dict) -> str:
        s = f"{n['name']}#{n['id']} [{n['state']}]"
        if n["staged"]:
            s += " (staged)"
        if n["config"] is not None:
            s += f" config={n['config']!r}"
        if n["inject"]:
            s += f" inject={n['inject']}"
        if n["provides"]:
            s += f" provides={n['provides']}"
        if n["tasks"]:
            s += f" tasks={n['tasks']}"
        if n["error"]:
            s += f" error={n['error']}"
        return s

    def walk(n: dict, prefix: str, last: bool, top: bool) -> None:
        lines.append(fmt(n) if top else f"{prefix}{'└── ' if last else '├── '}{fmt(n)}")
        kids = n["children"]
        for i, c in enumerate(kids):
            walk(c, "" if top else prefix + ("    " if last else "│   "), i == len(kids) - 1, False)

    walk(snap["fibers"], "", True, True)
    lines.append("services: " + (", ".join(f"{k} -> {v}" for k, v in snap["services"].items()) or "(none)"))
    return "\n".join(lines)
