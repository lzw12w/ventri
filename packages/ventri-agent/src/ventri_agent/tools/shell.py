"""``shell.run`` -- run a command in a confined working directory.

Risk ``external`` (a command can do anything the user can), never grantable for
the whole session: with the default ``policy: ask`` every run is approved
individually. Secrets-looking environment variables (``*KEY*``, ``*TOKEN*``,
``*SECRET*``, ``*PASSWORD*``) are removed from the child's environment; the
command is killed at the timeout. This is *not* a sandbox.

Output (stdout + stderr) is streamed with bounded memory: the first
``max_capture_bytes`` are kept plus a rolling tail, ANSI escapes and control
characters are stripped, and a result over ~6000 tokens is returned as head +
tail with the full capture in an artifact (``artifact.read``). A timed-out
command returns its partial output. The output is marked untrusted (it can
echo anything, e.g. a fetched web page)."""
from __future__ import annotations

import asyncio
import codecs
import os
import signal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

import ventri
from ventri.secret import sensitive

from ..paths import expand
from .output import preview, strip_ansi
from .registry import Risk, Tool, ToolContext, ToolError, ToolRegistry


class RunArgs(BaseModel):
    command: str = Field(description="Shell command (/bin/sh -c)")
    cwd: str = Field(".", description="Working directory, relative to the configured cwd")
    timeout: float = Field(60.0, description="Seconds before the command is killed")


class ShellConfig(BaseModel):
    cwd: str = "~/.ventri/workspace"
    policy: Literal["ask", "allow", "deny"] = "ask"
    timeout: float = 120.0
    max_capture_bytes: int = 4_000_000   # kept in full (artifact); beyond it only a tail survives
    tail_bytes: int = 256_000            # rolling tail kept once the capture cap is hit
    preview_tokens: int = 6_000          # inline result budget (head + tail)
    max_output: int | None = None        # deprecated (pre-0.3 char cap); ignored


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not sensitive(k)}


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the command and everything it started (it runs in its own session)."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - non-POSIX
            proc.kill()
    except ProcessLookupError:
        pass


class _Capture:
    """Bounded output buffer: a head up to ``cap`` bytes plus a rolling tail."""

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


def make_tool(cfg: ShellConfig) -> Tool:
    base = Path(os.path.realpath(expand(cfg.cwd)))

    async def run(a: RunArgs, tc: ToolContext) -> str:
        wd = Path(os.path.realpath(base / a.cwd))
        if wd != base and base not in wd.parents:
            raise ToolError(f"cwd {a.cwd!r} escapes {base}")
        wd.mkdir(parents=True, exist_ok=True)
        limit = min(a.timeout, cfg.timeout)
        proc = await asyncio.create_subprocess_shell(
            a.command, cwd=wd, env=clean_env(), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        cap = _Capture(cfg.max_capture_bytes, cfg.tail_bytes)

        async def pump() -> None:
            assert proc.stdout is not None
            while chunk := await proc.stdout.read(65536):
                cap.feed(chunk)
            await proc.wait()

        try:
            await asyncio.wait_for(pump(), timeout=limit)
        except TimeoutError:
            _kill_group(proc)
            await proc.wait()
            partial = cap.text()
            raise ToolError(f"command timed out after {limit}s and was killed"
                            + ("; partial output follows" if partial.strip() else " (no output)"),
                            untrusted=preview(tc, partial, "shell", budget=cfg.preview_tokens) or None) from None
        except BaseException:   # cancelled (session stop / loop timeout): never leak the process group
            _kill_group(proc)   # no await here: a cancelled scope would re-cancel it; asyncio reaps the child
            raise
        return f"exit code {proc.returncode}\n" + preview(tc, cap.text(), "shell", budget=cfg.preview_tokens)

    return Tool("shell.run", f"Run a shell command (cwd {base}).", run, RunArgs, risk=Risk.EXTERNAL,
                idempotent=False, default_action=cfg.policy, grantable=False,
                subject=lambda a: {"command": a.command, "cwd": a.cwd}, untrusted=True,
                timeout=cfg.timeout + 5)


@ventri.plugin(name="tool:shell", config=ShellConfig)
def shell(ctx: Any, cfg: ShellConfig, registry: ToolRegistry) -> None:
    """``use: ventri_agent.tools.shell`` (config ``cwd``, ``policy``, ``timeout``)."""
    expand(cfg.cwd).mkdir(parents=True, exist_ok=True)
    registry.register(ctx, make_tool(cfg))


plugin = shell
