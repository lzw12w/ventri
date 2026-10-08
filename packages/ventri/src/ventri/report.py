"""TxReport: what a transaction did (or, for a dry run, would do)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TxReport:
    """Outcome of a transaction (``tx.report``).

    * ``added`` -- staged fibers (label -> state at commit time);
    * ``removed`` -- live fibers removed (incl. replaced ones and their subtrees);
    * ``replaced`` -- new label -> old label;
    * ``restarted`` -- live fibers that restart because a binding they use changes;
    * ``activated`` -- live PENDING fibers whose required deps become available;
    * ``services`` -- ``added`` / ``removed`` / ``replaced`` keys (``realm:Key`` for
      scope realms);
    * ``failures`` / ``pending`` -- staged fibers that FAILED / stayed PENDING (with
      ``pending_reason``); ``skipped`` -- work not attempted (stop-first in a dry run);
    * ``probes`` -- name -> ``{"ok", "value", "error"}``;
    * ``outcome`` -- ``committed`` / ``rolled_back`` / ``dry_run``; ``degraded`` -- a
      stop-first rollback restarted old instances (their memory state was lost).
    """

    tx: int
    origin: str | None = None
    reason: str | None = None
    dry_run: bool = False
    outcome: str = "open"
    degraded: bool = False
    error: str | None = None
    added: dict[str, str] = field(default_factory=dict)
    removed: list[str] = field(default_factory=list)
    replaced: dict[str, str] = field(default_factory=dict)
    restarted: list[str] = field(default_factory=list)
    activated: list[str] = field(default_factory=list)
    services: dict[str, list[str]] = field(
        default_factory=lambda: {"added": [], "removed": [], "replaced": []})
    failures: dict[str, str] = field(default_factory=dict)
    pending: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    probes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """No error, no failed staged fiber and every probe passed."""
        return (self.error is None and not self.failures
                and all(p["ok"] for p in self.probes.values()))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ok"] = self.ok
        return out

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), default=repr, ensure_ascii=False, **kw)

    def __str__(self) -> str:
        head = f"tx #{self.tx} {self.outcome}{' (dry run)' if self.dry_run else ''}"
        if self.origin:
            head += f" origin={self.origin}"
        lines = [head + (" [OK]" if self.ok else " [FAILED]")]
        if self.reason:
            lines.append(f"  reason: {self.reason}")
        if self.error:
            lines.append(f"  error: {self.error}")
        if self.degraded:
            lines.append("  degraded rollback: old instances restarted (memory state lost)")
        for title, items in (("added", [f"{k} [{v}]" for k, v in self.added.items()]),
                             ("removed", self.removed),
                             ("replaced", [f"{o} -> {n}" for n, o in self.replaced.items()]),
                             ("restarted", self.restarted),
                             ("activated", self.activated),
                             ("failed", [f"{k}: {v}" for k, v in self.failures.items()]),
                             ("pending", [f"{k}: {v}" for k, v in self.pending.items()]),
                             ("skipped", [f"{k}: {v}" for k, v in self.skipped.items()])):
            if items:
                lines.append(f"  {title}:")
                lines += [f"    - {i}" for i in items]
        svc = self.services
        if any(svc.values()):
            lines.append("  services: " + ", ".join(
                f"{op} {', '.join(keys)}" for op, keys in svc.items() if keys))
        for name, p in self.probes.items():
            lines.append(f"  probe {name}: " + ("ok" if p["ok"] else f"FAILED {p['error']}"))
        return "\n".join(lines)
