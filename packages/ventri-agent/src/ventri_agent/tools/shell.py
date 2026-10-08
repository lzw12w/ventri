"""``shell.run`` -- run a command in a confined working directory.

Risk ``external`` (a command can do anything the user can), never grantable for
the whole session: with the default ``policy: ask`` every run is approved
individually. Secrets-looking environment variables (``*KEY*``, ``*TOKEN*``,
``*SECRET*``, ``*PASSWORD*``) are removed from the child's environment; the
command is killed at the timeout. This is *not* a sandbox."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

import ventri
from ventri.secret import sensitive

from ..paths import expand
from .registry import Risk, Tool, ToolContext, ToolError, ToolRegistry


class RunArgs(BaseModel):
    command: str = Field(description="Shell command (/bin/sh -c)")
    cwd: str = Field(".", description="Working directory, relative to the configured cwd")
    timeout: float = Field(60.0, description="Seconds before the command is killed")


class ShellConfig(BaseModel):
    cwd: str = "~/.ventri/workspace"
    policy: Literal["ask", "allow", "deny"] = "ask"
    timeout: float = 120.0
    max_output: int = 20_000


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not sensitive(k)}


def make_tool(cfg: ShellConfig) -> Tool:
    base = Path(os.path.realpath(expand(cfg.cwd)))

    async def run(a: RunArgs, tc: ToolContext) -> str:
        wd = Path(os.path.realpath(base / a.cwd))
        if wd != base and base not in wd.parents:
            raise ToolError(f"cwd {a.cwd!r} escapes {base}")
        wd.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_shell(
            a.command, cwd=wd, env=clean_env(), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=min(a.timeout, cfg.timeout))
        except (TimeoutError, asyncio.CancelledError):
            proc.kill()
            await proc.wait()
            raise ToolError(f"command timed out after {min(a.timeout, cfg.timeout)}s") from None
        text = out.decode("utf-8", "replace")
        if len(text) > cfg.max_output:
            text = text[:cfg.max_output] + f"\n[... {len(text) - cfg.max_output} more chars]"
        return f"exit code {proc.returncode}\n{text}"

    return Tool("shell.run", f"Run a shell command (cwd {base}).", run, RunArgs, risk=Risk.EXTERNAL,
                idempotent=False, default_action=cfg.policy, grantable=False,
                subject=lambda a: {"command": a.command, "cwd": a.cwd}, timeout=cfg.timeout + 5)


@ventri.plugin(name="tool:shell", config=ShellConfig)
def shell(ctx: Any, cfg: ShellConfig, registry: ToolRegistry) -> None:
    """``use: ventri_agent.tools.shell`` (config ``cwd``, ``policy``, ``timeout``)."""
    expand(cfg.cwd).mkdir(parents=True, exist_ok=True)
    registry.register(ctx, make_tool(cfg))


plugin = shell
