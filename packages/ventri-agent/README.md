# ventri-agent

Ventri Agent: a DeepSeek-first personal agent built on the [Ventri](../../README.md) plugin kernel.
Command: `va` (`va init`, `va chat`, `va sessions`, `va cost`, `va memory ...`, `va tree`, `va doctor`).

Everything is a plugin: the model provider (`ventri_agent.providers.deepseek`, `.openai_compat`, `.fake`),
the tool registry and built-in tools (`ventri_agent.tools.*`), the permission engine
(`ventri_agent.permission`), long-term memory (`ventri_agent.memory`), the session manager
(`ventri_agent.sessions`, one kernel scope per session) and the CLI channel (`ventri_agent.channels.cli`).

See the repository README and `docs/DESIGN.md` section 5 for the design, and `docs/api.md` for the API.
MIT licensed. Python 3.12+, asyncio only.
