"""An alternative agent-loop plugin (agents: {alt: {loop: ...}}) for tests."""
from typing import Any

import ventri
from ventri_agent.loop import AgentLoop, TurnResult
from ventri_agent.messages import Usage
from ventri_agent.session import SessionInfo


class EchoLoop:
    def __init__(self) -> None:
        self.turns = 0

    async def turn(self, text: str, sink: Any = None, *, plan: bool = False) -> TurnResult:
        self.turns += 1
        return TurnResult(self.turns, "ok", f"ALT: {text}", Usage(), 0.0, 0, 0)


@ventri.plugin(name="alt-loop")
def alt(ctx: Any, config: Any, info: SessionInfo) -> None:
    ctx.provide(AgentLoop, EchoLoop())


plugin = alt
