"""Event priority, typed events and interceptors (DESIGN.md 4.8)."""
from dataclasses import dataclass

import pytest

from ventri import Kernel
from ventri.events import Deny, Event, Rewrite

pytestmark = pytest.mark.anyio


@dataclass
class ChannelMessage:
    text: str


@dataclass
class ToolCall:
    tool: str
    path: str = ""


MessageIn = Event[ChannelMessage]("message.in")
ToolCallEv = Event[ToolCall]("tool.call")


async def test_priority_orders_listeners_across_fibers():
    order = []
    async with Kernel() as app:
        async def a(ctx):
            ctx.on("e", lambda: order.append("a0"))
            ctx.on("e", lambda: order.append("a10"), priority=10)

        async def b(ctx):
            ctx.on("e", lambda: order.append("b0"))
            ctx.on("e", lambda: order.append("b-5"), priority=-5)
            ctx.on("e", lambda: order.append("b10"), priority=10)

        await app.plugin(a)
        fb = await app.plugin(b)
        await app.emit("e")
        assert order == ["a10", "b10", "a0", "b0", "b-5"]  # ties keep registration order
        order.clear()
        assert app.bail("e") is None and order == ["a10", "b10", "a0", "b0", "b-5"]
        await fb.dispose()
        order.clear()
        await app.emit("e")
        assert order == ["a10", "a0"]


async def test_serial_respects_priority():
    async with Kernel() as app:
        app.on("q", lambda x: "low")
        app.on("q", lambda x: "high", priority=5)
        assert await app.serial("q", 1) == "high"


async def test_typed_event_shares_the_string_channel():
    got = []
    async with Kernel() as app:
        def p(ctx):
            ctx.on(MessageIn, lambda m: got.append(("typed", m.text)))
            ctx.on("message.in", lambda m: got.append(("str", m.text)))
        await app.plugin(p)
        await app.emit(MessageIn, ChannelMessage("hi"))
        await app.parallel("message.in", ChannelMessage("yo"))
        assert got[:2] == [("typed", "hi"), ("str", "hi")]
        assert sorted(got[2:]) == [("str", "yo"), ("typed", "yo")]
        assert Event("message.in") == MessageIn and hash(Event("message.in")) == hash(MessageIn)
        with pytest.raises(TypeError):
            await app.emit(42)


async def test_interceptors_deny_rewrite_pass_in_priority_order():
    audit = []

    def permission(ctx):
        def policy(call: ToolCall):
            if call.tool == "shell.run":
                return Deny("shell needs approval")
            if call.path.startswith("~"):
                return Rewrite(ToolCall(call.tool, "/home/jeff" + call.path[1:]))
            return None
        ctx.intercept(ToolCallEv, policy, priority=100)

    def auditor(ctx):
        ctx.intercept(ToolCallEv, lambda call: audit.append(call.path))  # sees the rewritten value

    async with Kernel() as app:
        await app.plugin(auditor)
        perm = await app.plugin(permission)
        assert await app.check(ToolCallEv, ToolCall("shell.run")) == Deny("shell needs approval")
        assert audit == []  # Deny stops the chain
        out = await app.check(ToolCallEv, ToolCall("fs.read", "~/notes"))
        assert out == ToolCall("fs.read", "/home/jeff/notes") and audit == ["/home/jeff/notes"]
        heard = []
        app.on("tool.call", heard.append)
        await app.emit(ToolCallEv, ToolCall("fs.read"))
        assert len(heard) == 1 and audit == ["/home/jeff/notes"]  # emit never runs interceptors
        await perm.dispose()
        assert await app.check(ToolCallEv, ToolCall("shell.run")) == ToolCall("shell.run")


async def test_interceptors_follow_scope_filter_and_fail_closed():
    async with Kernel() as app:
        app.intercept(ToolCallEv, lambda c: Deny("root policy") if c.tool == "x" else None)
        a = await app.scope("a")
        b = await app.scope("b")
        await b.ctx.plugin(lambda ctx: ctx.intercept(ToolCallEv, lambda c: Deny("b only")))
        # the root policy applies inside session a; session b's interceptor does not
        assert await a.ctx.check(ToolCallEv, ToolCall("x")) == Deny("root policy")
        assert await a.ctx.check(ToolCallEv, ToolCall("y")) == ToolCall("y")
        assert await b.ctx.check(ToolCallEv, ToolCall("y")) == Deny("b only")

        await a.ctx.plugin(lambda ctx: ctx.intercept("tool.call", lambda c: 1 / 0))
        with pytest.raises(ZeroDivisionError):
            await a.ctx.check(ToolCallEv, ToolCall("y"))
        c = await app.scope("c")
        await c.ctx.plugin(lambda ctx: ctx.intercept("tool.call", lambda v: "yes"))
        with pytest.raises(TypeError, match="expected Deny, Rewrite or None"):
            await c.ctx.check("tool.call", ToolCall("y"))
