"""Signature injection and optional dependencies (DESIGN.md 4.2)."""
from typing import Annotated

import pytest

from ventri import Kernel, State, plugin

from .conftest import LLM, llm_plugin

pytestmark = pytest.mark.anyio


class Calendar:
    def __init__(self, name: str = "cal") -> None:
        self.name = name


def calendar_plugin(ctx, config):
    ctx.provide(Calendar, Calendar((config or {}).get("name", "cal")))


async def test_required_and_optional_dependencies_from_signature():
    calls = []

    @plugin(timeout=10)
    async def daily_brief(ctx, cfg, llm: LLM, cal: Calendar | None):
        assert ctx.get(LLM) is llm  # same per-fiber view as the injected argument
        calls.append((llm.model, cal.name if cal else None))
        ctx.on_dispose(lambda: calls.append(("dispose", cal.name if cal else None)))

    async with Kernel() as app:
        f = await app.plugin(daily_brief, {"hour": 8})
        assert f.state is State.PENDING and calls == []  # LLM is required
        assert f.inject == (LLM,) and f.optional == (Calendar,)
        await app.plugin(llm_plugin, {"model": "v1"})
        assert f.state is State.ACTIVE and calls == [("v1", None)]  # Calendar does not block

        cal = await app.plugin(calendar_plugin, {"name": "work"})
        # an optional dependency appearing restarts the fiber so it can bind it
        assert calls[-2:] == [("dispose", None), ("v1", "work")]
        await cal.dispose()
        # ...and disappearing restarts it with None; teardown still saw the calendar
        assert calls[-2:] == [("dispose", "work"), ("v1", None)]
        assert f.state is State.ACTIVE
        snap = app.snapshot()["fibers"]["children"][0]
        assert snap["inject"] == ["LLM"] and snap["optional"] == ["Calendar"]


async def test_class_plugin_keyword_only_and_annotated_string_keys():
    got = {}

    class Agent:
        def __init__(self, ctx, config, llm: LLM, *, store: Annotated[dict, "store"],
                     extra: int = 3, other: int | str = "x", cal: Calendar | None = None):
            got.update(llm=llm.model, store=store, extra=extra, cal=cal)

    async with Kernel() as app:
        await app.plugin(llm_plugin)
        f = await app.plugin(Agent)
        assert f.state is State.PENDING and f.inject == (LLM, "store")
        app.provide("store", {"a": 1})
        await app.settle()
        assert f.state is State.ACTIVE and f.optional == (Calendar,)
        assert got == {"llm": "v1", "store": {"a": 1}, "extra": 3, "cal": None}


async def test_explicit_inject_merges_with_signature():
    @plugin(inject=["svc"])
    def p(ctx, config, llm: LLM):
        pass

    async with Kernel() as app:
        f = await app.plugin(p)
        assert f.inject == ("svc", LLM)


async def test_bad_signatures_are_rejected_at_load():
    def unannotated(ctx, config, llm):
        pass

    def two_types(ctx, config, x: LLM | Calendar):
        pass

    ns: dict = {}
    exec("def forward(ctx, config, x: 'Nope'): pass", ns)  # noqa: S102

    async with Kernel() as app:
        with pytest.raises(TypeError, match="needs a type annotation"):
            await app.plugin(unannotated)
        with pytest.raises(TypeError, match="only `X | None`"):
            await app.plugin(two_types)
        with pytest.raises(TypeError, match="cannot resolve annotation"):
            await app.plugin(ns["forward"])
        assert app.fiber.children == []


async def test_plain_ctx_config_plugins_unchanged():
    async with Kernel() as app:
        a = await app.plugin(lambda ctx: None)
        b = await app.plugin(lambda ctx, config: None, {"x": 1})
        c = await app.plugin(lambda: None)
        assert {a.state, b.state, c.state} == {State.ACTIVE}
