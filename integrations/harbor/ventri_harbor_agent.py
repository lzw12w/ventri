"""Harbor installed-agent adapter: Ventri Agent + DeepSeek (e.g. Terminal-Bench).

    PYTHONPATH=integrations/harbor harbor run -a ventri_harbor_agent:VentriAgent \\
        -m deepseek/deepseek-flash --allow-agent-host api.deepseek.com ...

* **Install** uploads a self-contained bundle (``build_bundle.sh``: a
  standalone CPython 3.12 + the ventri wheels + ``runner.py``) and unpacks it
  under a hidden, per-trial path (``/usr/local/lib/.vrt-<hex>``). It is never
  put on ``PATH``; the agent's shell does not see it (Ventri strips its own
  runtime from the child environment and the headless prompt says not to use
  it).
* **Run** starts ``runner.py`` with Ventri in **headless** mode (see the runner
  docstring for the permission policy). The time budget follows the task: the
  trial's agent timeout (task ``[agent].timeout_sec`` with Harbor's override /
  max / multiplier applied) minus a margin, not a fixed number. The agent shell
  gets the container's original environment, captured before Ventri starts.
* **Key**: ``DEEPSEEK_API_KEY`` is copied from the Harbor process into a 0600
  file in a private 0700 directory owned by the agent user; the runner deletes
  both once the key is in memory (the adapter deletes them again afterwards).
  It is never on a command line or in a log.
* **Results**: ``populate_context_post_run`` fills token / cost counts and
  writes ``trajectory.json`` in ATIF (validated with Harbor's model).
"""
from __future__ import annotations

import json
import os
import secrets
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Trajectory
from ventri_atif import convert_dir
from ventri_timeouts import agent_timeout_sec, ventri_wall

HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get("VENTRI_HARBOR_BUNDLE", HERE / "dist" / "ventri_bundle.tgz"))
VERSION_FILE = "VERSION"          # inside the bundle: "ventri-agent <version>@<commit>"


class VentriAgent(BaseInstalledAgent):
    SUPPORTS_ATIF = True

    def __init__(self, *args: Any, effort: str = "high", max_steps: int = 250, max_cost_usd: float = 0.5,
                 shell_timeout: int = 600, wall_sec: int = 870,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.effort = effort
        self.max_steps = int(max_steps)
        self.max_cost_usd = float(max_cost_usd)
        self.shell_timeout = int(shell_timeout)
        self.wall_fallback = float(wall_sec)       # only when the task timeout cannot be resolved
        self.rt = f"/usr/local/lib/.vrt-{secrets.token_hex(4)}"
        self._version: str | None = None

    @staticmethod
    def name() -> str:
        return "ventri"

    def version(self) -> str | None:
        if self._version is None:
            try:
                import tarfile
                with tarfile.open(BUNDLE) as tf:
                    f = tf.extractfile(VERSION_FILE)
                    self._version = f.read().decode().strip() if f else "ventri-agent"
            except (OSError, KeyError, tarfile.TarError):
                self._version = "ventri-agent"
        return self._version

    @property
    def remote_session_logs_dir(self) -> PurePosixPath | None:
        return PurePosixPath(str(self.environment_logs_dir)) / "ventri" / "sessions"

    def convert_trajectory(self, logs_dir: Path) -> Trajectory | None:
        d = convert_dir(logs_dir / "sessions", agent_version=self.version() or "", model_name=self._model())
        return Trajectory.model_validate(d) if d else None

    def _model(self) -> str | None:
        return (self.model_name or "").split("/")[-1] or None

    async def install(self, environment: BaseEnvironment) -> None:
        rt = shlex.quote(self.rt)
        await environment.upload_file(BUNDLE, "/installed-agent/vb.tgz")
        await self.exec_as_root(
            environment,
            command=f"mkdir -p {rt} && tar xzf /installed-agent/vb.tgz -C {rt} && rm -f /installed-agent/vb.tgz "
                    f"&& chmod -R a+rX {rt} && {rt}/rt/bin/python3.12 -I -c 'import ventri_agent'")

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY not set in the harbor process")
        timeout = agent_timeout_sec(self.logs_dir.parent)
        wall = ventri_wall(timeout, self.wall_fallback)
        kdir = f"/installed-agent/.k-{secrets.token_hex(4)}"
        kpath = f"{kdir}/k"
        envf = "/installed-agent/.env0"
        owner = environment.default_user
        chown = f"chown {shlex.quote(str(owner))} {kdir} {envf} && " if owner is not None else ""
        await self.exec_as_root(environment, command=f"mkdir -p {kdir} && touch {envf} && {chown}"
                                                     f"chmod 700 {kdir} && chmod 600 {envf}")
        # the container's own environment, as the agent user sees it, before Ventri adds anything
        await self.exec_as_agent(environment, command=f"(env -0 > {envf} || env > {envf}) 2>/dev/null; true")
        with tempfile.TemporaryDirectory() as d:
            kp = Path(d) / "k"
            kp.write_text(key)
            kp.chmod(0o600)
            await environment.upload_file(kp, kpath)
            ip = Path(d) / "instruction.md"
            ip.write_text(instruction)
            await environment.upload_file(ip, "/installed-agent/instruction.md")
        files = f"{kpath} /installed-agent/instruction.md"
        if owner is not None:
            await self.exec_as_root(environment, command=f"chown {shlex.quote(str(owner))} {files}")
        await self.exec_as_root(environment, command=f"chmod 600 {kpath}")
        rt = shlex.quote(self.rt)
        out = PurePosixPath(str(self.environment_logs_dir))
        self.logger.info(f"ventri: agent timeout {timeout}s -> ventri wall {wall:.0f}s")
        try:
            await self.exec_as_agent(
                environment,
                command=(f"{rt}/rt/bin/python3.12 -I {rt}/runner.py "
                         f"--instruction-file /installed-agent/instruction.md --out {out / 'ventri'} "
                         f"--effort {shlex.quote(self.effort)} --max-steps {self.max_steps} "
                         f"--max-cost-usd {self.max_cost_usd} --shell-timeout {self.shell_timeout} "
                         f"--wall {wall:.0f} --base-env-file {envf} "
                         f"2>&1 | tail -c 20000 > {out / 'ventri-stdout.txt'}"),
                env={"VENTRI_DS_KEY_FILE": kpath, "PYTHONUNBUFFERED": "1"},
            )
        finally:
            await environment.exec(command=f"rm -rf {kdir}", user="root")

    def populate_context_post_run(self, context: AgentContext) -> None:
        p = self.logs_dir / "ventri" / "result.json"
        if p.exists():
            r = json.loads(p.read_text())
            context.n_input_tokens = r.get("prompt_tokens") or 0
            context.n_cache_tokens = r.get("cache_hit") or 0
            context.n_output_tokens = r.get("completion_tokens") or 0
            context.cost_usd = r.get("cost_usd_ventri")
        try:
            traj = self.convert_trajectory(self.logs_dir / "ventri")
        except Exception as e:  # noqa: BLE001 - a bad log must not fail the trial
            self.logger.warning(f"ventri: ATIF conversion failed: {e}")
            return
        if traj is not None:
            (self.logs_dir / "trajectory.json").write_text(json.dumps(traj.to_json_dict(), ensure_ascii=False,
                                                                      indent=1))
