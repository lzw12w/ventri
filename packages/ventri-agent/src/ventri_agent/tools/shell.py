"""``shell.*`` -- run commands in a confined working directory, in the
foreground or as background jobs.

``shell.run`` has risk ``external`` (a command can do anything the user can) and
is never grantable for the whole session: with the default ``policy: ask`` every
run is approved individually, including ``background: true`` starts.
Secrets-looking environment variables (``*KEY*``, ``*TOKEN*``, ``*SECRET*``,
``*PASSWORD*``), ``VENTRI_*`` variables and Ventri's own Python runtime (its
virtualenv ``bin`` on ``PATH``, ``PYTHONHOME``/``PYTHONPATH`` into it, plus any
``hide_paths``) are removed from the child's environment. This is *not* a
sandbox.

Foreground runs: stdout + stderr go to an unlinked spool file, not a pipe, and
the call returns as soon as the foreground shell exits -- a ``cmd &`` /
``nohup cmd &`` child that inherited the output keeps running without holding
the call open. On timeout or cancellation the whole process group is killed.
Memory and disk stay bounded (the first ``max_capture_bytes`` plus a rolling
``tail_bytes``); ANSI escapes and control characters are stripped and a result
over ~6000 tokens comes back as head + tail with the full capture in an artifact
(``artifact.read``). Output is marked untrusted (it can echo anything).

Background jobs (``shell.run(background=true)``) get an id; their output goes to
the artifact ``job-<id>`` (a ring capped at ``job_log_bytes``) and is read with
``shell.output`` (read-only, can wait for exit or a regex). ``shell.jobs`` lists
them and ``shell.kill`` signals a job's process group -- only jobs this session
started. Jobs are owned by the session (:class:`ShellJobs`, a session-scoped
service) and are killed when the session scope is disposed
(``jobs_on_dispose: kill``; ``keep`` leaves them running, e.g. a server that
must outlive an unattended run). A daemon that calls ``setsid`` itself leaves
the job's process group and is not tracked.

``persist: true`` carries the working directory and exported environment from
one foreground ``shell.run`` to the next (like a terminal), re-filtered through
the same secret/runtime stripping on every call.
"""
from __future__ import annotations

import asyncio
import codecs
import contextlib
import fcntl
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import IO, Any, Literal

from pydantic import BaseModel, Field

import ventri
from ventri.secret import sensitive

from ..paths import expand
from .output import preview, strip_ansi
from .registry import Risk, Tool, ToolContext, ToolError, ToolRegistry


class RunArgs(BaseModel):
    command: str = Field(description="Shell command (/bin/sh -c)")
    cwd: str = Field(".", description="Working directory, relative to the configured cwd")
    timeout: float = Field(60.0, description="Seconds before the command is killed (foreground only)")
    background: bool = Field(False, description=(
        "Start as a background job and return its id at once (servers, long builds); "
        "read its output with shell.output, stop it with shell.kill"))


class JobArgs(BaseModel):
    job: str = Field(description="Job id returned by shell.run(background=true)")


class OutputArgs(BaseModel):
    job: str = Field(description="Job id returned by shell.run(background=true)")
    wait: float = Field(0.0, description="Seconds to wait for the job to exit (or for `pattern`) before returning")
    pattern: str | None = Field(None, description="Return as soon as new output matches this regex")
    since: Literal["last", "start"] = Field("last", description="'last': output since the previous shell.output; "
                                                                 "'start': the whole log")


class KillArgs(BaseModel):
    job: str = Field(description="Job id returned by shell.run(background=true)")
    signal: Literal["TERM", "INT", "KILL", "HUP"] = Field("TERM", description="Signal for the job's process group")


class ShellConfig(BaseModel):
    cwd: str = "~/.ventri/workspace"
    policy: Literal["ask", "allow", "deny"] = "ask"
    timeout: float = 120.0
    max_capture_bytes: int = 4_000_000   # kept in full (artifact); beyond it only a tail survives
    tail_bytes: int = 256_000            # rolling tail kept once the capture cap is hit
    preview_tokens: int = 6_000          # inline result budget (head + tail)
    max_output: int | None = None        # deprecated (pre-0.3 char cap); ignored
    persist: bool = False                # carry cwd + exported env across foreground runs
    max_jobs: int = 16                   # running background jobs per session
    job_log_bytes: int = 8_000_000       # a job log over this keeps only its last job_keep_bytes
    job_keep_bytes: int = 2_000_000
    jobs_on_dispose: Literal["kill", "keep"] = "kill"
    hide_runtime: bool = True            # strip Ventri's own Python runtime from the child env
    hide_paths: list[str] = Field(default_factory=list)  # extra dirs to strip from PATH (bundled runtimes)
    base_env_file: str | None = None     # child env base (KEY=VALUE lines or NUL-separated) instead of ours


# ----------------------------------------------------------------- environment
_RUNTIME_VARS = ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "PYTHONEXECUTABLE")
_SHELL_VARS = {"PWD", "OLDPWD", "SHLVL", "_"}


def _under(path: str, dirs: Iterable[str]) -> bool:
    if not path:
        return False
    try:
        p = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    return any(p == d or p.startswith(d.rstrip(os.sep) + os.sep) for d in dirs)


def runtime_dirs(extra: Iterable[str] = ()) -> list[str]:
    """Directories that belong to Ventri's own runtime: its virtualenv (when it
    runs from one) and any explicitly configured dirs (a bundled interpreter)."""
    out = [os.path.realpath(os.path.expanduser(d)) for d in extra if d]
    if sys.prefix != sys.base_prefix:
        out.append(os.path.realpath(sys.prefix))
    return out


def clean_env(base: Mapping[str, str] | None = None, *, hide_runtime: bool = True,
              hide_paths: Iterable[str] = ()) -> dict[str, str]:
    """The child environment: no secrets, no ``VENTRI_*`` internals, and (with
    ``hide_runtime``) none of Ventri's own interpreter on ``PATH``."""
    src = os.environ if base is None else base
    hidden = runtime_dirs(hide_paths) if hide_runtime else []
    py_dirs = [*hidden, os.path.realpath(sys.prefix)] if hide_runtime else []
    out: dict[str, str] = {}
    for k, v in src.items():
        if sensitive(k) or k.startswith("VENTRI_"):
            continue
        if k in _RUNTIME_VARS and py_dirs and any(_under(p, py_dirs) for p in v.split(os.pathsep) if p):
            continue
        out[k] = v
    if hidden and "PATH" in out:
        out["PATH"] = os.pathsep.join(p for p in out["PATH"].split(os.pathsep) if p and not _under(p, hidden))
    return out


def read_env_file(path: str | Path) -> dict[str, str]:
    """``KEY=VALUE`` per line, or NUL-separated (``env -0``)."""
    raw = Path(expand(str(path))).read_bytes().decode("utf-8", "replace")
    items = raw.split("\0") if "\0" in raw else raw.splitlines()
    out: dict[str, str] = {}
    for item in items:
        k, sep, v = item.partition("=")
        if sep and k and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            out[k] = v
    return out


# --------------------------------------------------------------- process utils
def _kill_group(proc: Any, sig: int = signal.SIGKILL) -> None:
    """Signal the command and everything it started (it runs in its own session)."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, sig)
        else:  # pragma: no cover - non-POSIX
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


class _Capture:
    """Bounded output: a head up to ``cap`` bytes plus a rolling tail."""

    def __init__(self, cap: int, tail: int) -> None:
        self.cap, self.tail_cap = cap, tail
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0

    def feed(self, chunk: bytes) -> None:
        room = self.cap - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self.tail += chunk
            if len(self.tail) > self.tail_cap:
                cut = len(self.tail) - self.tail_cap
                self.dropped += cut
                del self.tail[:cut]

    def text(self) -> str:
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        out = dec.decode(bytes(self.head), final=not self.tail)
        if self.dropped:
            out += f"\n[... {self.dropped} bytes dropped: output exceeded the {self.cap}-byte capture ...]\n"
            dec = codecs.getincrementaldecoder("utf-8")("replace")  # tail starts mid-stream
        if self.tail:
            out += dec.decode(bytes(self.tail), final=True)
        return strip_ansi(out)


def _set_append(f: IO[bytes]) -> None:
    fd = f.fileno()
    fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_APPEND)


def _pread_all(fd: int, start: int, end: int) -> bytes:
    out = bytearray()
    while start < end:
        chunk = os.pread(fd, min(1 << 20, end - start), start)
        if not chunk:
            break
        out += chunk
        start += len(chunk)
    return bytes(out)


class _Spool:
    """The foreground command's output file (unlinked, ``O_APPEND``). Disk use is
    bounded: past ``cap + tail + slack`` the head is saved once and the file is
    truncated to its tail; the writer appends at the new end."""

    SLACK: int = 8_000_000

    def __init__(self, cap: int, tail: int) -> None:
        self.f: IO[bytes] = tempfile.TemporaryFile()  # noqa: SIM115 - closed by close()
        _set_append(self.f)
        self.cap, self.tail = cap, tail
        self.head: bytes | None = None
        self.base = 0                       # logical offset of the file's first byte

    def check(self) -> None:
        fd = self.f.fileno()
        size = os.fstat(fd).st_size
        if size <= self.cap + self.tail + self.SLACK:
            return
        if self.head is None:
            self.head = _pread_all(fd, 0, self.cap)
        keep = _pread_all(fd, size - self.tail, size)
        os.ftruncate(fd, 0)
        os.write(fd, keep)
        self.base += size - len(keep)

    def capture(self) -> _Capture:
        fd = self.f.fileno()
        size = os.fstat(fd).st_size
        cap = _Capture(self.cap, self.tail)
        if self.head is None:
            pos = 0
            while pos < size:
                chunk = os.pread(fd, 1 << 20, pos)
                if not chunk:
                    break
                cap.feed(chunk)
                pos += len(chunk)
            return cap
        rest = _pread_all(fd, max(0, size - self.tail), size)
        cap.head = bytearray(self.head)
        cap.tail = bytearray(rest)
        cap.dropped = max(0, self.base + size - len(self.head) - len(rest))
        return cap

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.f.close()


# ------------------------------------------------------------------- jobs
class Job:
    def __init__(self, jid: str, command: str, cwd: Path, proc: subprocess.Popen[bytes], log: Path) -> None:
        self.id, self.command, self.cwd, self.proc, self.log = jid, command, cwd, proc, log
        self.started = time.monotonic()
        self.ended: float | None = None
        self.base = 0          # logical offset of the log file's first byte (ring compaction)
        self.cursor = 0        # logical offset shell.output(since=last) continues from
        self.dropped = 0

    @property
    def handle(self) -> str:
        return f"job-{self.id}"

    def poll(self) -> int | None:
        rc = self.proc.poll()
        if rc is not None and self.ended is None:
            self.ended = time.monotonic()
        return rc

    def size(self) -> int:
        try:
            return self.log.stat().st_size
        except OSError:
            return 0

    def compact(self, limit: int, keep: int) -> None:
        size = self.size()
        if size <= limit:
            return
        with open(self.log, "r+b") as f:
            f.seek(max(0, size - keep))
            tail = f.read()
            marker = f"[... earlier output dropped: job log exceeded {limit} bytes ...]\n".encode()
            f.seek(0)
            f.truncate(0)
            f.write(marker + tail)
        self.dropped += size - len(tail)
        self.base += size - len(tail) - len(marker)

    def read(self, since: int) -> tuple[str, int, bool]:
        """Text from logical offset ``since``; (text, new cursor, gap?)."""
        size = self.size()
        start = max(0, since - self.base)
        gap = since < self.base
        with open(self.log, "rb") as f:
            f.seek(start)
            data = f.read(max(0, size - start))
        return strip_ansi(data.decode("utf-8", "replace")), self.base + start + len(data), gap

    def status(self) -> str:
        rc = self.poll()
        took = (self.ended or time.monotonic()) - self.started
        if rc is None:
            return f"running for {took:.0f}s"
        if rc < 0:
            return f"killed by signal {-rc} after {took:.0f}s"
        return f"exited with code {rc} after {took:.0f}s"


class ShellJobs:
    """Session service: this session's background jobs and persisted shell
    state. ``close()`` (on session dispose) kills running jobs' process groups
    unless ``keep`` is set."""

    def __init__(self, session_id: str, *, on_dispose: Literal["kill", "keep"] = "kill") -> None:
        self.session_id = session_id
        self.on_dispose = on_dispose
        self.jobs: dict[str, Job] = {}
        self.cwd: str | None = None             # persisted working directory
        self.env: dict[str, str] | None = None  # persisted exported environment
        self._n = 0
        self._monitor: asyncio.Task[None] | None = None
        self.limits: tuple[int, int] = (8_000_000, 2_000_000)

    def __repr__(self) -> str:
        return f"<ShellJobs session={self.session_id} jobs={len(self.jobs)}>"

    def next_id(self) -> str:
        self._n += 1
        return str(self._n)

    def running(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.poll() is None]

    def get(self, jid: str) -> Job:
        job = self.jobs.get(jid.removeprefix("job-"))
        if job is None:
            have = ", ".join(self.jobs) or "none"
            raise ToolError(f"no job {jid!r} in this session (jobs: {have})")
        return job

    def ensure_monitor(self) -> None:
        if self._monitor is not None and not self._monitor.done():
            return
        with contextlib.suppress(RuntimeError):
            self._monitor = asyncio.get_running_loop().create_task(self._watch())

    async def _watch(self) -> None:
        while self.running():
            for j in list(self.jobs.values()):
                with contextlib.suppress(OSError):
                    j.compact(*self.limits)
            await asyncio.sleep(1.0)

    def close(self) -> None:
        if self._monitor is not None:
            self._monitor.cancel()
            self._monitor = None
        if self.on_dispose == "keep":
            for j in self.jobs.values():
                j.poll()
            return
        live = [j for j in self.jobs.values() if j.poll() is None]
        for j in live:
            _kill_group(j.proc, signal.SIGTERM)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and any(j.poll() is None for j in live):
            time.sleep(0.02)
        for j in live:
            _kill_group(j.proc, signal.SIGKILL)   # also any stragglers left in the group
            with contextlib.suppress(subprocess.TimeoutExpired):
                j.proc.wait(timeout=1.0)
            j.poll()


_FALLBACK: dict[str, ShellJobs] = {}


def shell_jobs(tc: ToolContext, on_dispose: Literal["kill", "keep"] = "kill") -> ShellJobs:
    """The session's :class:`ShellJobs` (a per-session-id fallback when the
    tool runs outside a session scope, e.g. in tests)."""
    st = None
    if tc.ctx is not None:
        try:
            st = tc.get(ShellJobs)
        except Exception:  # noqa: BLE001 - no session scope
            st = None
    if isinstance(st, ShellJobs):
        return st
    return _FALLBACK.setdefault(tc.session_id, ShellJobs(tc.session_id, on_dispose=on_dispose))


# ----------------------------------------------------------------- persistence
_DUMP_ENV = "import json,os,sys;json.dump(dict(os.environ),open(sys.argv[1],'w'))"


def _persist_wrap(command: str, state: Path) -> str:
    st = shlex.quote(str(state))
    py = shlex.quote(sys.executable)
    trap = (f'__ventri_rc=$?; pwd -P > {st}.cwd 2>/dev/null; '
            f'{py} -I -c {shlex.quote(_DUMP_ENV)} {st}.env 2>/dev/null; exit $__ventri_rc')
    return f"trap {shlex.quote(trap)} EXIT\n{command}\n"


def _load_persisted(state: Path, jobs: ShellJobs) -> None:
    with contextlib.suppress(OSError, ValueError):
        cwd = Path(f"{state}.cwd").read_text(encoding="utf-8").strip()
        if cwd:
            jobs.cwd = cwd
    with contextlib.suppress(OSError, ValueError):
        env = json.loads(Path(f"{state}.env").read_text(encoding="utf-8"))
        if isinstance(env, dict):
            jobs.env = {str(k): str(v) for k, v in env.items() if k not in _SHELL_VARS}
    for suffix in (".cwd", ".env"):
        with contextlib.suppress(OSError):
            Path(f"{state}{suffix}").unlink()


def _spawn(command: str, wd: Path, env: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
    """Start a background job in its own session, output appended to ``log``
    (``O_APPEND``: the ring compaction can truncate under the writer)."""
    with open(log, "ab") as out:
        return subprocess.Popen(command, shell=True, cwd=wd, env=env, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)


# ------------------------------------------------------------------- tools
def make_tools(cfg: ShellConfig) -> list[Tool]:
    base = Path(os.path.realpath(expand(cfg.cwd)))

    def env_for(jobs: ShellJobs | None) -> dict[str, str]:
        src: Mapping[str, str] | None = None
        if jobs is not None and jobs.env is not None:
            src = jobs.env
        elif cfg.base_env_file:
            try:
                src = read_env_file(cfg.base_env_file)
            except OSError:
                src = None
        return clean_env(src, hide_runtime=cfg.hide_runtime, hide_paths=cfg.hide_paths)

    def workdir(a: RunArgs, jobs: ShellJobs | None) -> Path:
        if jobs is not None and a.cwd in (".", "") and jobs.cwd and os.path.isdir(jobs.cwd):
            return Path(jobs.cwd)
        wd = Path(os.path.realpath(base / a.cwd))
        if wd != base and base not in wd.parents:
            raise ToolError(f"cwd {a.cwd!r} escapes {base}")
        wd.mkdir(parents=True, exist_ok=True)
        return wd

    def jobs_of(tc: ToolContext) -> ShellJobs:
        j = shell_jobs(tc, cfg.jobs_on_dispose)
        j.on_dispose = cfg.jobs_on_dispose
        j.limits = (cfg.job_log_bytes, cfg.job_keep_bytes)
        return j

    async def run(a: RunArgs, tc: ToolContext) -> str:
        jobs = jobs_of(tc)
        if a.background:
            return await start_job(a, tc, jobs)
        persisted = jobs if cfg.persist else None
        wd = workdir(a, persisted)
        env = env_for(persisted)
        limit = min(a.timeout, cfg.timeout)
        command = a.command
        state: Path | None = None
        if cfg.persist:
            state_dir = tc.workdir / ".shell"
            state_dir.mkdir(parents=True, exist_ok=True)
            state = state_dir / f"state-{secrets.token_hex(4)}"
            command = _persist_wrap(a.command, state)
        spool = _Spool(cfg.max_capture_bytes, cfg.tail_bytes)
        try:
            proc = await asyncio.create_subprocess_shell(
                command, cwd=wd, env=env, stdin=asyncio.subprocess.DEVNULL,
                stdout=spool.f, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            # wait for the foreground shell only: background children may keep the spool open
            waiter = asyncio.ensure_future(proc.wait())
            try:
                deadline = time.monotonic() + limit
                while not waiter.done():
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError
                    await asyncio.wait({waiter}, timeout=min(0.25, left))
                    spool.check()
            except TimeoutError:
                _kill_group(proc)
                await proc.wait()
                partial = spool.capture().text()
                raise ToolError(f"command timed out after {limit}s and was killed"
                                + ("; partial output follows" if partial.strip() else " (no output)"),
                                untrusted=preview(tc, partial, "shell", budget=cfg.preview_tokens) or None) from None
            except BaseException:   # cancelled (session stop / loop timeout): never leak the process group
                _kill_group(proc)   # no await here: a cancelled scope would re-cancel it; asyncio reaps the child
                raise
            text = spool.capture().text()
        finally:
            spool.close()
            if state is not None:
                _load_persisted(state, jobs)
        return f"exit code {proc.returncode}\n" + preview(tc, text, "shell", budget=cfg.preview_tokens)

    async def start_job(a: RunArgs, tc: ToolContext, jobs: ShellJobs) -> str:
        if len(jobs.running()) >= cfg.max_jobs:
            raise ToolError(f"{cfg.max_jobs} background jobs are already running; "
                            "stop one with shell.kill or wait for it with shell.output")
        persisted = jobs if cfg.persist else None
        wd = workdir(a, persisted)
        env = env_for(persisted)
        jid = jobs.next_id()
        art = tc.workdir / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        log = art / f"job-{jid}.txt"
        proc = _spawn(a.command, wd, env, log)
        job = Job(jid, a.command, wd, proc, log)
        jobs.jobs[jid] = job
        jobs.ensure_monitor()
        for _ in range(5):              # a typo or instant failure is reported right away
            if job.poll() is not None:
                break
            await asyncio.sleep(0.05)
        text, job.cursor, _ = job.read(0)
        head = (f"background job {jid} started (pid {proc.pid}, cwd {wd}); {job.status()}. "
                f"Output goes to artifact {job.handle!r}; read it with shell.output(job={jid!r}), "
                f"stop it with shell.kill(job={jid!r}).")
        return head + (("\n" + preview(tc, text, "job", budget=cfg.preview_tokens)) if text.strip() else "")

    async def output(a: OutputArgs, tc: ToolContext) -> str:
        jobs = jobs_of(tc)
        job = jobs.get(a.job)
        rx = None
        if a.pattern:
            try:
                rx = re.compile(a.pattern, re.MULTILINE)
            except re.error as e:
                raise ToolError(f"bad pattern: {e}") from None
        start = 0 if a.since == "start" else job.cursor
        deadline = time.monotonic() + max(0.0, min(a.wait, cfg.timeout))
        matched = False
        while True:
            with contextlib.suppress(OSError):
                job.compact(cfg.job_log_bytes, cfg.job_keep_bytes)
            if job.poll() is not None:
                break
            if rx is not None:
                text, _, _ = job.read(start)
                if rx.search(text):
                    matched = True
                    break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.2)
        text, job.cursor, gap = job.read(start)
        note = " (pattern matched)" if matched else ""
        head = f"job {job.id}: {job.status()}{note}; log artifact {job.handle!r}\n"
        if gap:
            head += "[... part of the output was dropped by the job-log cap ...]\n"
        if not text:
            return head + ("(no new output)" if a.since == "last" else "(no output)")
        return head + preview(tc, text, "job", budget=cfg.preview_tokens)

    def list_jobs(a: Any, tc: ToolContext) -> str:
        jobs = jobs_of(tc)
        if not jobs.jobs:
            return "no background jobs in this session"
        lines = []
        for j in jobs.jobs.values():
            cmd = j.command if len(j.command) <= 120 else j.command[:117] + "..."
            lines.append(f"{j.id}: {j.status()}; pid {j.proc.pid}; cwd {j.cwd}; {cmd}")
        return "\n".join(lines)

    async def kill(a: KillArgs, tc: ToolContext) -> str:
        jobs = jobs_of(tc)
        job = jobs.get(a.job)
        if job.poll() is not None:
            return f"job {job.id} already {job.status()}"
        sig = getattr(signal, f"SIG{a.signal}")
        _kill_group(job.proc, sig)
        for _ in range(50):
            if job.poll() is not None:
                break
            await asyncio.sleep(0.05)
        if job.poll() is None and a.signal != "KILL":
            return f"sent SIG{a.signal} to job {job.id}; still {job.status()} (use signal KILL to force)"
        return f"job {job.id}: {job.status()}"

    common: dict[str, Any] = {"timeout": cfg.timeout + 5}
    return [
        Tool("shell.run", f"Run a shell command (cwd {base}); background=true starts a job.", run, RunArgs,
             risk=Risk.EXTERNAL, idempotent=False, default_action=cfg.policy, grantable=False,
             subject=lambda a: {"command": a.command, "cwd": a.cwd,
                                **({"background": "true"} if a.background else {})},
             untrusted=True, **common),
        Tool("shell.output", "Read a background job's output (optionally wait for exit or a regex).", output,
             OutputArgs, risk=Risk.READ, untrusted=True, subject=lambda a: {"job": a.job}, **common),
        Tool("shell.jobs", "List this session's background jobs.", list_jobs, None, risk=Risk.READ,
             parallel_safe=True),
        Tool("shell.kill", "Signal a background job started by this session (its whole process group).",
             kill, KillArgs, risk=Risk.WRITE_LOCAL, idempotent=False, default_action="allow",
             subject=lambda a: {"job": a.job, "signal": a.signal}),
    ]


def make_tool(cfg: ShellConfig) -> Tool:
    """``shell.run`` alone (kept for callers/tests that only need it)."""
    return make_tools(cfg)[0]


@ventri.plugin(name="tool:shell", config=ShellConfig)
def shell(ctx: Any, cfg: ShellConfig, registry: ToolRegistry) -> None:
    """``use: ventri_agent.tools.shell`` (config ``cwd``, ``policy``, ``timeout``, ``persist``...)."""
    expand(cfg.cwd).mkdir(parents=True, exist_ok=True)
    for t in make_tools(cfg):
        registry.register(ctx, t)


plugin = shell
