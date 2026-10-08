"""Demo: fake LLM service + tool plugin + background task + failing transaction rollback.

Run:  uv sync && uv run python examples/demo.py
"""
from dataclasses import dataclass

import anyio

from ventri import Kernel, PluginError, plugin


class FakeLLM:
    def __init__(self, model: str) -> None:
        self.model = model

    async def complete(self, prompt: str) -> str:
        await anyio.sleep(0)
        return f"[{self.model}] echo: {prompt}"


class LLMPlugin:
    """Class plugin: typed config, provides the FakeLLM service."""
    name = "llm"

    @dataclass
    class Config:
        model: str = "deepseek-chat"
        api_key: str = ""

    def __init__(self, ctx, config: "LLMPlugin.Config") -> None:
        if not config.api_key:
            raise ValueError("api_key is required")
        ctx.provide(FakeLLM, FakeLLM(config.model), name="llm")


@plugin(name="tool", inject=[FakeLLM])
def tool_plugin(ctx, config):
    llm = ctx.get(FakeLLM)  # typed access -> FakeLLM

    async def call(prompt):
        return await llm.complete(prompt)

    ctx.on("tool:call", call)

    async def heartbeat():
        try:
            while True:
                await anyio.sleep(0.05)
        finally:
            print(f"  [heartbeat of tool bound to {llm.model}] cancelled")

    ctx.spawn(heartbeat)


@plugin(name="broken")
def broken_plugin(ctx):
    ctx.on_dispose(lambda: print("  [broken] cleanup ran during rollback"))
    raise RuntimeError("broken plugin refuses to start")


async def main() -> None:
    async with Kernel() as app:
        app.on_trace(lambda e: print("  trace:", e) if e.kind.startswith("tx.") else None)

        await app.plugin(tool_plugin)  # PENDING until FakeLLM exists
        print("== tool loaded before its dependency ==")
        print(app.tree())

        llm = await app.plugin(LLMPlugin, {"model": "deepseek-chat", "api_key": "sk-demo"})
        print("\n== after llm loaded ==")
        print(app.tree())
        print("tool:call ->", await app.serial("tool:call", "hello"))

        print("\n== failing transaction: replace llm config + load broken plugin ==")
        before = app.snapshot()
        try:
            async with app.transaction() as tx:
                await tx.replace(llm, config={"model": "deepseek-reasoner", "api_key": "sk-demo"})
                print("  inside tx (staged fibers visible, live world unchanged):")
                print("  " + app.tree().replace("\n", "\n  "))
                await tx.plugin(broken_plugin)
        except PluginError as e:
            print(f"  rolled back: {e}")
        print(app.tree())
        print("state identical to before:", app.snapshot() == before)
        print("tool:call ->", await app.serial("tool:call", "still works"))

        print("\n== successful hot swap via ctx.replace ==")
        await app.replace(llm, {"model": "deepseek-reasoner", "api_key": "sk-demo"})
        print(app.tree())
        print("tool:call ->", await app.serial("tool:call", "after swap"))

        print("\n== failing hot swap (missing api_key) ==")
        new_llm = next(c for c in app.fiber.children if c.name == "llm")
        try:
            await app.replace(new_llm, {"model": "deepseek-v9"})
        except PluginError as e:
            print(f"  rolled back: {e}")
        print(app.tree())
        print("\n== shutdown ==")
    print("kernel closed")


if __name__ == "__main__":
    anyio.run(main)
