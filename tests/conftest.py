import anyio
import pytest

from pykernel import plugin


@pytest.fixture(params=["asyncio", "trio"])
def anyio_backend(request):
    return request.param


async def wait_for(pred, timeout=2.0):
    with anyio.fail_after(timeout):
        while not pred():
            await anyio.sleep(0.001)


class LLM:
    def __init__(self, model: str) -> None:
        self.model = model
        self.closed = False


@plugin(name="llm")
def llm_plugin(ctx, config):
    config = config or {}
    if config.get("fail"):
        raise ValueError("bad llm config")
    svc = LLM(config.get("model", "v1"))
    ctx.provide(LLM, svc, name="llm")
    ctx.on_dispose(lambda: setattr(svc, "closed", True))


def make_tool(log: list, name: str = "tool"):
    @plugin(name=name, inject=[LLM])
    def tool(ctx):
        llm = ctx.get(LLM)
        log.append(("apply", name, llm.model))
        # cleanup runs while the dependency is still available
        ctx.on_dispose(lambda: log.append(("dispose", name, ctx.get(LLM).model)))
    return tool
