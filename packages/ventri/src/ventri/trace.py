"""Trace schema v1 (DESIGN.md 4.10) -- frozen in M1.

Every :class:`~ventri.kernel.TraceEvent` exports (``event.to_dict()``) as::

    {"v": 1, "seq": 17, "ts": 1791427200.123, "kind": "fiber.state",
     "fiber": "root/tools/github", "scope": "session:a1" | null, "tx": 3 | null,
     "attrs": {...}}

* ``v`` -- schema version (int, always 1 for this schema);
* ``seq`` -- per-kernel sequence number, strictly increasing from 1;
* ``ts`` -- wall-clock time, Unix epoch seconds (float, UTC);
* ``kind`` -- event kind (see :data:`KERNEL_KINDS`; upper layers add their own
  dotted kinds through ``ctx.trace``);
* ``fiber`` -- path of the fiber from the root, one segment per fiber: its config
  id (``meta["id"]``) if it has one, else its plugin name; ``null`` for
  kernel-level events;
* ``scope`` -- name of the nearest enclosing scope, ``null`` at root level;
* ``tx`` -- id of the transaction the event belongs to (staged fiber, or a
  ``tx.*`` event), else ``null``;
* ``attrs`` -- kind-specific attributes, redacted (``ventri.redact``); events with
  a fiber carry ``attrs.label`` (``name#id``, unique within a process).

Compatibility rule: v1 consumers must ignore unknown attrs and unknown kinds;
adding attrs or kinds is not a schema change, renaming/removing top-level
fields is (it bumps ``v``).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: Kinds the kernel emits. ``kernel.start/stop`` and ``fiber.retry`` extend the
#: list in DESIGN.md 4.10 (see the M1 implementation notes there).
KERNEL_KINDS = frozenset({
    "kernel.start", "kernel.stop",
    "fiber.state", "fiber.retry",
    "service.bind", "service.unbind",
    "task.spawn", "task.error",
    "effect.error", "event.error",
    "tx.begin", "tx.commit", "tx.rollback",
    "dep.cycle",
})

#: Kind-specific attrs the kernel guarantees (others may appear).
KERNEL_ATTRS: dict[str, tuple[str, ...]] = {
    "fiber.state": ("old", "new"),
    "fiber.retry": ("attempt",),
    "service.bind": ("key", "staged"),
    "service.unbind": ("key", "staged"),
    "task.spawn": ("task",),
    "task.error": ("task", "error"),
    "effect.error": ("effect", "error"),
    "event.error": ("event", "error"),
    "tx.begin": ("tx",),
    "tx.commit": ("tx", "services", "added", "removed"),
    "tx.rollback": ("tx", "error"),
    "dep.cycle": ("cycle",),
}

class TraceSchemaError(ValueError):
    """A record does not conform to trace schema v1."""


_FIELDS = {"v": int, "seq": int, "ts": (int, float), "kind": str, "attrs": dict}
_NULLABLE = {"fiber": str, "scope": str, "tx": int}


def validate(record: Any) -> dict[str, Any]:
    """Check one exported record against schema v1; return it or raise TraceSchemaError."""
    if not isinstance(record, dict):
        raise TraceSchemaError(f"trace record must be an object, got {type(record).__name__}")
    if record.get("v") != SCHEMA_VERSION:
        raise TraceSchemaError(f"unsupported trace schema version {record.get('v')!r}")
    for name, typ in _FIELDS.items():
        if name not in record or not isinstance(record[name], typ) or isinstance(record[name], bool):
            raise TraceSchemaError(f"field {name!r} missing or not {typ}")
    for name, typ in _NULLABLE.items():
        if name not in record or not (record[name] is None or isinstance(record[name], typ)):
            raise TraceSchemaError(f"field {name!r} missing or not {typ} | null")
    if record["seq"] < 1 or "." not in record["kind"]:
        raise TraceSchemaError("bad seq or kind")
    for attr in KERNEL_ATTRS.get(record["kind"], ()):
        if attr not in record["attrs"]:
            raise TraceSchemaError(f"{record['kind']} record lacks attrs.{attr}")
    return record


def dumps(record: dict[str, Any]) -> str:
    """One JSONL line (non-JSON attr values are rendered with ``repr``)."""
    return json.dumps(record, default=repr, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate validated records of a JSONL trace file."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield validate(json.loads(line))
