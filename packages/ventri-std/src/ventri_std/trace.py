"""JSONL trace sink (DESIGN.md 4.10): ``use: ventri_std.trace.jsonl``.

Writes every kernel trace record as one schema-v1 JSON line (see
``docs/trace-schema.md``) to ``<path>/trace.jsonl``, rotating at ``rotate_mb``
into ``trace-<UTC timestamp, µs>-<first seq>.jsonl`` (names sort chronologically) and keeping the newest ``keep``
rotated files. Records are buffered in memory and flushed every
``flush_interval`` seconds and on unload, so the hot path never touches disk.

Guarantees: records are written in ``seq`` order, each exactly once, redacted
(``ventri.redact``); unloading the sink flushes everything it received.
With ``backfill`` (default) the kernel's in-memory ring buffer is written first,
so records from before the sink loaded (``kernel.start``, earlier plugins) are
not lost if they are still in the buffer.
Non-guarantees: records emitted after the sink is disposed (e.g. ``kernel.stop``
at shutdown, its own ``disposed`` transition) are not written; a process crash
loses at most ``flush_interval`` seconds of records; a write error is reported
once as a ``trace.sink_error`` record and the batch is dropped.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio

from ventri import TraceEvent, plugin
from ventri.trace import dumps

CURRENT = "trace.jsonl"


@dataclass
class JsonlConfig:
    path: str = "~/.ventri/trace/"
    rotate_mb: float = 64
    keep: int = 10
    flush_interval: float = 0.5
    backfill: bool = True


class JsonlWriter:
    """Synchronous rotating writer; one per sink fiber."""

    def __init__(self, path: str | os.PathLike, rotate_mb: float = 64, keep: int = 10) -> None:
        p = Path(path).expanduser()
        if p.suffix != ".jsonl":
            p = p / CURRENT
        self.file = p
        self.dir = p.parent
        self.rotate_bytes = max(1, int(rotate_mb * 1024 * 1024))
        self.keep = keep
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh: Any = None
        self._first: int | None = None

    def _open(self) -> Any:
        if self._fh is None:
            self._fh = open(self.file, "a", encoding="utf-8")  # noqa: SIM115 - long-lived handle
        return self._fh

    def write(self, lines: list[tuple[int, str]]) -> None:
        for seq, line in lines:
            fh = self._open()
            if fh.tell() > 0 and fh.tell() + len(line) + 1 > self.rotate_bytes:
                self.rotate()
                fh = self._open()
            if self._first is None:
                self._first = seq
            fh.write(line + "\n")
        if self._fh is not None:
            self._fh.flush()

    def rotate(self) -> None:
        self.close()
        now = time.time()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f".{int(now % 1 * 1e6):06d}Z"
        self.file.rename(self.dir / f"{self.file.stem}-{stamp}-{self._first or 0}.jsonl")
        self._first = None
        old = sorted(self.rotated(), key=lambda q: q.name)
        for q in old[: max(0, len(old) - self.keep)]:
            q.unlink(missing_ok=True)

    def rotated(self) -> list[Path]:
        return [q for q in self.dir.glob(f"{self.file.stem}-*.jsonl") if q.is_file()]

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@plugin(name="trace.jsonl", config=JsonlConfig)
def jsonl(ctx: Any, config: JsonlConfig) -> None:
    writer = JsonlWriter(config.path, config.rotate_mb, config.keep)
    kernel = ctx.kernel
    buffer: list[tuple[int, str]] = []
    state = {"last": 0, "failed": False}

    def record(ev: TraceEvent) -> None:
        if ev.seq > state["last"]:
            state["last"] = ev.seq
            buffer.append((ev.seq, dumps(ev.to_dict())))

    def flush() -> None:
        if not buffer:
            return
        batch = buffer[:]
        buffer.clear()
        try:
            writer.write(batch)
        except OSError as e:
            if not state["failed"]:
                state["failed"] = True
                ctx.trace("trace.sink_error", error=repr(e), dropped=len(batch))

    if config.backfill:
        for ev in list(kernel.trace_log):
            record(ev)

    # effects run LIFO: unsubscribe first, then the final flush, then close
    ctx.on_dispose(writer.close)
    ctx.on_dispose(flush)
    ctx.on_dispose(kernel.on_trace(record))

    async def flusher() -> None:
        while True:
            await anyio.sleep(config.flush_interval)
            flush()

    ctx.spawn(flusher, name="flush")
