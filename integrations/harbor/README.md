# Ventri Agent on Harbor (Terminal-Bench)

A [Harbor](https://github.com/laude-institute/harbor) installed-agent adapter that runs Ventri Agent
with DeepSeek inside each task container, in **headless** mode.

| file | role |
|---|---|
| `ventri_harbor_agent.py` | `VentriAgent(BaseInstalledAgent)`: install, run, token/cost counts, ATIF `trajectory.json` |
| `runner.py` | runs one task in the container: Kernel + DeepSeek + fs/shell tools, `SessionManager.open(headless=True)` |
| `ventri_atif.py` | Ventri session log → ATIF (Harbor's Agent Trajectory Interchange Format) |
| `ventri_timeouts.py` | Ventri's wall clock from the trial's agent timeout (task `[agent].timeout_sec`, override/max/multiplier) |
| `build_bundle.sh` | builds `dist/ventri_bundle.tgz`: standalone CPython 3.12 + ventri wheels + runner (Linux x86_64) |

## Run

```bash
integrations/harbor/build_bundle.sh                       # -> integrations/harbor/dist/ventri_bundle.tgz
export DEEPSEEK_API_KEY=...                               # read by the adapter, never logged
PYTHONPATH=$PWD/integrations/harbor harbor run -p datasets/terminal-bench-2-1 -i build-cython-ext \
  -a ventri_harbor_agent:VentriAgent -m deepseek/deepseek-flash \
  --allow-agent-host api.deepseek.com -o jobs --job-name ventri -n 1 -y
```

Agent kwargs (`--agent-kwarg k=v`): `effort` (high), `max_steps` (250), `max_cost_usd` (0.5 per trial),
`shell_timeout` (600 s per foreground command), `prune_tokens` (40000; 0 disables context pruning),
`wall_sec` (870, only used when the task timeout cannot be resolved).

## What runs in the container

* The bundle is unpacked to a hidden per-trial path (`/usr/local/lib/.vrt-<hex>`), never put on `PATH`;
  Ventri strips its own interpreter from the agent shell's environment, which is the container's original
  environment captured before Ventri starts. The headless prompt tells the model not to use the harness runtime.
* Headless session: approvals are decided by `permission.unattended` = `{ask: allow, irreversible: deny}`
  (the container is the sandbox), audited to `/logs/agent/ventri/audit.jsonl` with
  `decided_by: unattended-policy`.
* Shell: cwd/env persist between calls; servers go to `shell.run(background=true)`; background jobs are left
  running when the run ends (`jobs_on_dispose: keep`) so a requested service is still up for the verifier.
* Time: Ventri stops at `agent timeout − max(45 s, 8%)` and is hard-stopped 20 s later, before Harbor kills it,
  so logs and the trajectory are always written.
* Key: copied into a 0600 file in a private 0700 directory owned by the agent user; the runner deletes it as
  soon as it has read it (the adapter deletes it again afterwards).

Outputs per trial (`jobs/<job>/<trial>/agent/`): `trajectory.json` (ATIF), `ventri/result.json`,
`ventri/events.jsonl`, `ventri/sessions/*.jsonl` (full Ventri session log), `ventri/audit.jsonl`.
