"""Read-before-write / stale-file state for the file tools.

Portions adapted from Hermes Agent (https://github.com/NousResearch/hermes-agent),
Copyright (c) 2025 Nous Research, MIT License. Sources: ``tools/file_state.py``
(``FileStateRegistry``: read stamps, last writer, per-path locks, ``check_stale``)
and ``tools/file_tools_read_tracking.py`` (full-content baselines, paged-read
coverage, blind patches, patch-failure counts).

Ventri adaptation: Hermes keys this state by ``task_id`` in module globals.
Here one process-wide :class:`FileStateRegistry` holds it keyed by *session id*
(a Ventri session is a kernel scope), and each session scope is given a
:class:`FileState` handle (``sessions.py`` provides it next to ``WorkingMemory``
and forgets the session's state when the scope is disposed). The last-writer
map stays process-wide on purpose: a write by another session is exactly the
"sibling agent" case Hermes guards against. Hermes's read dedup stubs and
consecutive-read loop blocking are not ported (see ``fs.py``).
"""
from __future__ import annotations

import hashlib
import os
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# (dev, ino, size, mtime_ns, ctime_ns) and the same + sha256 digest
Meta = tuple[int, int, int, int, int]
Version = tuple[int, int, int, int, int, bytes]

_MAX_PATHS = 4096
_MAX_WRITERS = 4096
_COVERAGE_RANGES_CAP = 256
_PATCH_FAILURE_PATHS_CAP = 64


def _evict_oldest(container: dict[Any, Any], cap: int) -> None:
    for _ in range(len(container) - cap):
        try:
            container.pop(next(iter(container)))
        except (StopIteration, KeyError):
            break


def file_metadata(resolved: str) -> Meta | None:
    try:
        st = os.stat(resolved)
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    except OSError:
        return None


def file_version(resolved: str) -> Version | None:
    """A byte snapshot (metadata + sha256), not just mtime: editors and copy
    tools can preserve mtime. None when the file changed while hashing."""
    try:
        if not stat.S_ISREG(os.stat(resolved).st_mode):
            return None
        fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                return None
            digest = hashlib.file_digest(stream, "sha256").digest()
            after = os.stat(resolved)
        meta: Meta = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if meta == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            return (*meta, digest)
        return None
    except OSError:
        return None


def _mtime_ns(resolved: str) -> int | None:
    try:
        return os.stat(resolved).st_mtime_ns
    except OSError:
        return None


def _fmt_ts(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


@dataclass
class _Coverage:
    version: Version
    ranges: list[tuple[int, int]] = field(default_factory=lambda: list[tuple[int, int]]())
    redacted: bool = False


@dataclass
class _SessionFiles:
    # resolved -> (mtime_ns, read wall time, partial)
    reads: dict[str, tuple[int, float, bool]] = field(default_factory=lambda: dict[str, tuple[int, float, bool]]())
    baselines: dict[str, Version] = field(default_factory=lambda: dict[str, Version]())
    coverage: dict[str, _Coverage] = field(default_factory=lambda: dict[str, _Coverage]())
    blind_patches: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    patch_failures: dict[str, int] = field(default_factory=lambda: dict[str, int]())


class FileStateRegistry:
    """Process-wide coordinator: per-session read state, global last writer,
    per-path locks. Thread-safe (file tools run in worker threads)."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionFiles] = {}
        self._last_writer: dict[str, tuple[str, float]] = {}
        self._path_locks: dict[str, threading.Lock] = {}
        self._path_lock_users: dict[str, int] = {}
        self._meta_lock = threading.Lock()
        self._state_lock = threading.Lock()

    # ------------------------------------------------------------ locks
    @contextmanager
    def lock_path(self, resolved: str) -> Iterator[None]:
        """Writers of the same path serialize; different paths stay parallel."""
        with self._meta_lock:
            lock = self._path_locks.setdefault(resolved, threading.Lock())
            self._path_lock_users[resolved] = self._path_lock_users.get(resolved, 0) + 1
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._meta_lock:
                users = self._path_lock_users[resolved] - 1
                if users:
                    self._path_lock_users[resolved] = users
                else:
                    self._path_lock_users.pop(resolved, None)
                    self._path_locks.pop(resolved, None)

    def _session(self, sid: str) -> _SessionFiles:
        s = self._sessions.get(sid)
        if s is None:
            s = self._sessions[sid] = _SessionFiles()
        return s

    def _stamp(self, sid: str, resolved: str, mtime_ns: int, now: float, partial: bool) -> None:
        reads = self._session(sid).reads
        reads[resolved] = (mtime_ns, now, partial)
        _evict_oldest(reads, _MAX_PATHS)

    # ------------------------------------------------------------ reads
    def record_read(self, sid: str, resolved: str, *, version_before: Meta | None,
                    snapshot: Version | None, partial: bool, offset: int, end_line: int | None,
                    total_lines: int | None, redacted: bool = False) -> bool:
        """Bookkeeping after a real read (``_record_successful_read``). Returns
        whether the session now holds the file's whole current content: one
        full page, or contiguous pages that reach the last line, at one version,
        none of them line-clamped."""
        version = snapshot or file_version(resolved)
        stable = version is not None and version[:-1] == version_before == file_metadata(resolved)
        complete = False
        with self._state_lock:
            s = self._session(sid)
            if stable and version is not None:
                complete = s.baselines.get(resolved) == version
                if not complete:
                    complete = not partial
                    if partial and end_line is not None:
                        complete, redacted = self._note_coverage(s, resolved, version, offset, end_line,
                                                                 total_lines, redacted)
                    complete = complete and not redacted
                if complete:
                    s.baselines[resolved] = version
                    _evict_oldest(s.baselines, _MAX_PATHS)
                    s.blind_patches.pop(resolved, None)
            if not complete:
                s.baselines.pop(resolved, None)
            mtime = _mtime_ns(resolved)
            if mtime is not None:
                self._stamp(sid, resolved, mtime, time.time(), not complete)
        return complete

    @staticmethod
    def _note_coverage(s: _SessionFiles, resolved: str, version: Version, start: int, end: int,
                       total_lines: int | None, redacted: bool) -> tuple[bool, bool]:
        """Merge page ``start..end`` into the coverage of ``resolved`` at ``version``."""
        entry = s.coverage.get(resolved)
        if entry is None or entry.version != version or len(entry.ranges) > _COVERAGE_RANGES_CAP:
            entry = s.coverage[resolved] = _Coverage(version)
            _evict_oldest(s.coverage, _MAX_PATHS)
        entry.redacted = entry.redacted or redacted
        merged: list[tuple[int, int]] = []
        for a, b in sorted([*entry.ranges, (start, end)]):
            if merged and a <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        entry.ranges = merged
        complete = (isinstance(total_lines, int) and total_lines > 0
                    and merged[0][0] <= 1 and merged[0][1] >= total_lines)
        return complete, entry.redacted

    # ------------------------------------------------------------ writes
    def note_write(self, sid: str, resolved: str) -> None:
        """A successful write: global last writer and this session's own stamp
        (a write is an implicit read of what it wrote)."""
        mtime = _mtime_ns(resolved)
        if mtime is None:
            return
        now = time.time()
        with self._state_lock:
            self._last_writer[resolved] = (sid, now)
            _evict_oldest(self._last_writer, _MAX_WRITERS)
            self._stamp(sid, resolved, mtime, now, False)
            s = self._session(sid)
            s.coverage.pop(resolved, None)

    def mark_full_baseline(self, sid: str, resolved: str, expected_sha256: str | None = None) -> None:
        version = file_version(resolved)
        if version is None or (expected_sha256 is not None and version[-1].hex() != expected_sha256):
            return
        with self._state_lock:
            s = self._session(sid)
            s.baselines[resolved] = version
            s.blind_patches.pop(resolved, None)

    def has_full_baseline(self, sid: str, resolved: str) -> bool:
        with self._state_lock:
            baseline = self._session(sid).baselines.get(resolved)
        return baseline is not None and file_version(resolved) == baseline

    def known_full_sha256(self, sid: str, resolved: str) -> str | None:
        """sha256 of the bytes the session knows in full, if they are still on disk."""
        with self._state_lock:
            baseline = self._session(sid).baselines.get(resolved)
        if baseline is not None and file_version(resolved) == baseline:
            return baseline[-1].hex()
        return None

    def carry_baseline(self, sid: str, resolved: str, known_sha: str | None, read_sha: str | None,
                       written_sha: str | None) -> None:
        """After the session's own edit (``_carry_full_write_baselines``): keep a
        whole-file baseline when the edit read bytes the session knew in full (or
        created the file: ``read_sha == ""``); otherwise remember a blind patch."""
        if written_sha is None:
            return
        vouched = read_sha == "" or (read_sha is not None and read_sha == known_sha)
        if vouched:
            self.mark_full_baseline(sid, resolved, written_sha)
            return
        with self._state_lock:
            s = self._session(sid)
            s.baselines.pop(resolved, None)
            s.blind_patches[resolved] = written_sha
            _evict_oldest(s.blind_patches, _MAX_PATHS)

    def is_own_blind_patch(self, sid: str, resolved: str) -> bool:
        with self._state_lock:
            written = self._session(sid).blind_patches.get(resolved)
        if written is None:
            return False
        version = file_version(resolved)
        return version is not None and version[-1].hex() == written

    # ------------------------------------------------------------ staleness
    def check_stale(self, sid: str, resolved: str) -> str | None:
        """Model-facing reason a write would be stale, else None. Severity:
        another session wrote after our read > on-disk drift / partial read."""
        with self._state_lock:
            stamp = self._session(sid).reads.get(resolved)
            last_writer = self._last_writer.get(resolved)
        if stamp is None and last_writer is None:  # net-new file / first touch
            return None
        current = _mtime_ns(resolved)
        if current is None:
            return None  # the write creates it; not stale
        if last_writer is not None:
            writer, writer_ts = last_writer
            if writer != sid:
                if stamp is None:
                    return (f"{resolved} was modified by another session ({writer!r}) but this "
                            "session never read it. Read the file before writing to avoid "
                            "overwriting those changes.")
                if writer_ts > stamp[1]:
                    return (f"{resolved} was modified by another session ({writer!r}) at "
                            f"{_fmt_ts(writer_ts)} -- after this session's last read at "
                            f"{_fmt_ts(stamp[1])}. Re-read the file before writing.")
        if stamp is not None:
            read_mtime, _ts, partial = stamp
            if current != read_mtime:
                return (f"{resolved} was modified since you last read it (external edit, shell "
                        "command or another writer). Re-read the file before writing.")
            if partial:
                return (f"{resolved} was last read with offset/limit pagination (partial view). "
                        "Read the remaining pages, or use fs.edit, before overwriting it.")
            return None
        return f"{resolved} was not read by this session. Read the file first so you can write an informed edit."

    def stale_overwrite_blocker(self, sid: str, resolved: str) -> str | None:
        """Why a whole-file overwrite must NOT replace the existing file, else
        None (``_stale_overwrite_blocker``). Refused before any disk mutation."""
        stale = self.check_stale(sid, resolved)
        if stale:
            return stale
        if self.has_full_baseline(sid, resolved):
            return None
        try:
            exists = os.path.lexists(resolved)
        except OSError:
            return None
        if not exists:
            return None
        if self.is_own_blind_patch(sid, resolved):
            return ("Your edit changed this file without a full view of its current content. Read the "
                    "current file in full before a whole-file overwrite, or continue with targeted edits.")
        return (f"{resolved} exists but this session has not seen its full current content (never "
                "read, edited without a prior full read, or only a partial view). Read the file -- "
                "every page of it, if it needs offset/limit -- or use fs.edit for a targeted edit; a "
                "stale conversation copy must not overwrite the current disk content.")

    # ------------------------------------------------------------ patch failures
    def record_patch_failure(self, sid: str, resolved: str) -> int:
        with self._state_lock:
            failures = self._session(sid).patch_failures
            if resolved not in failures:
                _evict_oldest(failures, _PATCH_FAILURE_PATHS_CAP - 1)
            failures[resolved] = failures.get(resolved, 0) + 1
            return failures[resolved]

    def reset_patch_failures(self, sid: str, resolved: str) -> None:
        with self._state_lock:
            self._session(sid).patch_failures.pop(resolved, None)

    # ------------------------------------------------------------ lifecycle
    def forget(self, sid: str) -> None:
        """Drop a finished session's stamps and writer claims (a finished
        session is no longer a concurrent writer)."""
        with self._state_lock:
            self._sessions.pop(sid, None)
            for p in [p for p, (w, _ts) in self._last_writer.items() if w == sid]:
                del self._last_writer[p]

    def clear(self) -> None:
        with self._state_lock:
            self._sessions.clear()
            self._last_writer.clear()


_REGISTRY = FileStateRegistry()


def get_registry() -> FileStateRegistry:
    return _REGISTRY


class FileState:
    """Session-scoped handle onto the registry (one per session scope)."""

    def __init__(self, session_id: str, registry: FileStateRegistry | None = None) -> None:
        self.session_id = session_id
        self.registry = registry or _REGISTRY

    def __repr__(self) -> str:
        return f"<FileState session={self.session_id}>"

    def close(self) -> None:
        self.registry.forget(self.session_id)

    def lock(self, resolved: str) -> Any:
        return self.registry.lock_path(resolved)

    def record_read(self, resolved: str, **kw: Any) -> bool:
        return self.registry.record_read(self.session_id, resolved, **kw)

    def note_write(self, resolved: str) -> None:
        self.registry.note_write(self.session_id, resolved)

    def mark_full_baseline(self, resolved: str, expected_sha256: str | None = None) -> None:
        self.registry.mark_full_baseline(self.session_id, resolved, expected_sha256)

    def known_full_sha256(self, resolved: str) -> str | None:
        return self.registry.known_full_sha256(self.session_id, resolved)

    def carry_baseline(self, resolved: str, known_sha: str | None, read_sha: str | None,
                       written_sha: str | None) -> None:
        self.registry.carry_baseline(self.session_id, resolved, known_sha, read_sha, written_sha)

    def check_stale(self, resolved: str) -> str | None:
        return self.registry.check_stale(self.session_id, resolved)

    def stale_overwrite_blocker(self, resolved: str) -> str | None:
        return self.registry.stale_overwrite_blocker(self.session_id, resolved)

    def record_patch_failure(self, resolved: str) -> int:
        return self.registry.record_patch_failure(self.session_id, resolved)

    def reset_patch_failures(self, resolved: str) -> None:
        self.registry.reset_patch_failures(self.session_id, resolved)
