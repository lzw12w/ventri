"""Plugins addressed by ``use:`` in the config-loader tests."""
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

from ventri import Secret, plugin

INSTANCES: list[Any] = []


@dataclass
class LLMConfig:
    model: str = "flash"
    api_key: Secret[str] | str | None = None
    fail: bool = False


class LLM:
    Config = LLMConfig
    provides: ClassVar = {"llm": object}

    def __init__(self, ctx: Any, cfg: LLMConfig) -> None:
        if cfg.fail:
            raise RuntimeError("bad llm config")
        self.cfg = cfg
        INSTANCES.append(self)
        ctx.provide("llm", self)


@plugin(name="agent")
def agent(ctx: Any, config: Any, llm: Annotated[Any, "llm"]) -> None:
    ctx.provide("agent", {"llm": llm})


@dataclass
class ToolConfig:
    name: str = "tool"


@plugin(name="tool", config=ToolConfig)
def tool(ctx: Any, config: ToolConfig) -> None:
    pass


@plugin(name="needs_missing")
def needs_missing(ctx: Any, config: Any, x: Annotated[Any, "nobody-provides-this"]) -> None:
    pass


FLAKY = {"fail": True}


@plugin(name="flaky")
def flaky(ctx: Any, config: Any) -> None:
    if FLAKY["fail"]:
        raise RuntimeError("flaky")


PORTS: dict[int, Any] = {}


@plugin(name="server", exclusive=True)
def server(ctx: Any, config: Any) -> None:
    port = config["port"]
    if port in PORTS or config.get("fail"):
        raise RuntimeError(f"port {port} busy")
    PORTS[port] = config
    ctx.on_dispose(lambda: PORTS.pop(port))
    ctx.provide("server", config)
