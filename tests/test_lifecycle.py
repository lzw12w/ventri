from dataclasses import dataclass

import anyio
import pytest

from ventri import Kernel, ServiceConflict, ServiceNotFound, State, plugin

from .conftest import LLM, llm_plugin, make_tool, wait_for

pytestmark = pytest.mark.anyio


async def test_inject_pending_then_active():
    log = []
    async with Kernel() as app:
        tool = await app.plugin(make_tool(log))
        assert tool.state is State.PENDING and log == []
        llm = await app.plugin(llm_plugin, {"model": "v1"})
        assert llm.state is State.ACTIVE
        assert tool.state is State.ACTIVE
        assert log == [("apply", "tool", "v1")]


async def test_service_removal_unloads_dependents_and_they_return():
    log = []
    async with Kernel() as app:
        llm = await app.plugin(llm_plugin, {"model": "v1"})
        tool = await app.plugin(make_tool(log))
        await llm.dispose()
        assert llm.state is State.DISPOSED
        assert tool.state is State.PENDING
        assert not app.has(LLM)
        # dependent was torn down *before* the service disappeared
        assert log == [("apply", "tool", "v1"), ("dispose", "tool", "v1")]
        await app.plugin(llm_plugin, {"model": "v2"})
        assert tool.state is State.ACTIVE
        assert log[-1] == ("apply", "tool", "v2")


async def test_root_provide_then_settle():
    log = []
    async with Kernel() as app:
        tool = await app.plugin(make_tool(log))
        app.provide(LLM, LLM("root"))
        await app.settle()
        assert tool.state is State.ACTIVE and log == [("apply", "tool", "root")]


async def test_cascade_disposal_order():
    log = []

    def child(name):
        @plugin(name=name)
        def p(ctx):
            ctx.on_dispose(lambda: log.append(f"{name}.e1"))
            ctx.on_dispose(lambda: log.append(f"{name}.e2"))
        return p

    @plugin(name="grand")
    def grand(ctx):
        ctx.on_dispose(lambda: log.append("grand.e1"))

    async def parent(ctx):
        ctx.on_dispose(lambda: log.append("parent.e1"))
        a = await ctx.plugin(child("a"))
        await a.ctx.plugin(grand)
        await ctx.plugin(child("b"))
        ctx.on_dispose(lambda: log.append("parent.e2"))

    async with Kernel() as app:
        f = await app.plugin(parent)
        assert [c.name for c in f.children] == ["a", "b"]
        await f.dispose()
        assert log == ["b.e2", "b.e1", "grand.e1", "a.e2", "a.e1", "parent.e2", "parent.e1"]
        disposed = [e.fiber for e in app.trace_log
                    if e.kind == "fiber.state" and e.data["new"] == "disposed"]
        assert [d.split("#")[0] for d in disposed] == ["b", "grand", "a", "parent"]
        assert app.fiber.children == []


async def test_spawned_tasks_cancelled_on_dispose():
    events = []

    async def worker(tag):
        events.append(f"start:{tag}")
        try:
            await anyio.sleep_forever()
        finally:
            events.append(f"cancelled:{tag}")

    async def p(ctx):
        ctx.spawn(worker, 1)
        ctx.spawn(worker, 2)
        ctx.on_dispose(lambda: events.append("effect"))

    async with Kernel() as app:
        f = await app.plugin(p)
        await wait_for(lambda: len(events) == 2)
        await f.dispose()
        # strict LIFO: the effect registered last runs first, then task 2, then task 1
        assert events[2:] == ["effect", "cancelled:2", "cancelled:1"]


async def test_kernel_exit_cancels_everything():
    events = []

    async def worker():
        try:
            await anyio.sleep_forever()
        finally:
            events.append("cancelled")

    async def p(ctx):
        ctx.spawn(worker)
        await ctx.plugin(lambda c: c.spawn(worker))

    async with Kernel() as app:
        await app.plugin(p)
        await anyio.sleep(0.01)
    assert events == ["cancelled", "cancelled"]


async def test_task_crash_fails_owner_only():
    async def bad():
        await anyio.sleep(0.01)
        raise RuntimeError("boom")

    cleaned = []

    async def p(ctx):
        ctx.spawn(bad)
        ctx.spawn(anyio.sleep_forever)
        ctx.on_dispose(lambda: cleaned.append(True))

    async with Kernel() as app:
        f = await app.plugin(p)
        other = await app.plugin(lambda ctx: None)
        await wait_for(lambda: f.state is State.FAILED)
        assert isinstance(f.error, RuntimeError) and cleaned == [True]
        assert other.state is State.ACTIVE
        assert any(e.kind == "task.error" for e in app.trace_log)


async def test_dispose_while_loading():
    started = anyio.Event()
    log = []

    async def slow(ctx):
        ctx.on_dispose(lambda: log.append("cleanup"))
        ctx.spawn(anyio.sleep_forever)
        started.set()
        await anyio.sleep_forever()

    async with Kernel() as app:
        result = {}

        async def load():
            result["f"] = await app.plugin(slow)

        async with anyio.create_task_group() as tg:
            tg.start_soon(load)
            await started.wait()
            f = app.fiber.children[0]
            assert f.state is State.LOADING
            await f.dispose()
            assert f.state is State.DISPOSED
        assert result["f"] is f
        assert log == ["cleanup"] and f._effects == [] and f._tg is None
        assert app.fiber.children == []


async def test_self_dispose_inside_apply_and_from_own_task():
    async def selfkill(ctx):
        ctx.on_dispose(lambda: None)
        await ctx.fiber.dispose()
        await anyio.sleep(10)  # cancelled

    async def via_task(ctx):
        async def job():
            await anyio.sleep(0.005)
            await ctx.fiber.dispose()
        ctx.spawn(job)

    async with Kernel() as app:
        a = await app.plugin(selfkill)
        assert a.state is State.DISPOSED
        b = await app.plugin(via_task)
        assert b.state is State.ACTIVE
        await wait_for(lambda: b.state is State.DISPOSED)
        assert app.fiber.children == []


async def test_cancellation_mid_load_leaks_nothing():
    cancelled = []

    async def bg():
        try:
            await anyio.sleep_forever()
        finally:
            cancelled.append("bg")

    async def child(ctx):
        ctx.provide("child-svc", 1)

    async def slow(ctx):
        ctx.provide("svc", object())
        ctx.on("evt", lambda: None)
        ctx.spawn(bg)
        await ctx.plugin(child)
        await anyio.sleep_forever()

    async with Kernel() as app:
        dep = await app.plugin(plugin(lambda ctx: None, inject=["svc"]))
        with anyio.move_on_after(0.05) as scope:
            await app.plugin(slow)
        assert scope.cancelled_caught
        assert not app.has("svc") and not app.has("child-svc")
        assert app._listeners.get("evt") == []
        assert cancelled == ["bg"]
        assert dep.state is State.PENDING
        assert [c.name for c in app.fiber.children] == ["<lambda>"]
        states = {e.fiber.split("#")[0]: e.data["new"] for e in app.trace_log if e.kind == "fiber.state"}
        assert states["slow"] == "disposed" and states["child"] == "disposed"


async def test_typed_access_attr_and_conflicts():
    @dataclass
    class Cfg:
        greeting: str = "hi"

    class Greeter:
        name = "greeter"
        Config = Cfg

        def __init__(self, ctx, config):
            self.ctx, self.config, self.stopped = ctx, config, False

        async def start(self):
            self.ctx.provide(Greeter, self, name="greeter")

        def stop(self):
            self.stopped = True

    async with Kernel() as app:
        f = await app.plugin(Greeter, {"greeting": "你好"})
        g = app.get(Greeter)
        assert isinstance(g, Greeter) and g.config.greeting == "你好"
        assert app.greeter is g and f.instance is g
        with pytest.raises(ServiceNotFound):
            app.get("nope")
        assert app.get("nope", 42) == 42
        with pytest.raises(AttributeError):
            _ = app.nope
        bad = await app.plugin(lambda ctx: ctx.provide(Greeter, object()))
        assert bad.state is State.FAILED and isinstance(bad.error, ServiceConflict)
        await f.dispose()
        assert g.stopped and not app.has(Greeter)
        await bad.restart()  # recovers now that the key is free
        assert bad.state is State.ACTIVE


async def test_events():
    async with Kernel() as app:
        got = []

        async def p(ctx):
            ctx.on("e", lambda x: got.append(("a", x)))

            async def slow(x):
                await anyio.sleep(0)
                got.append(("b", x))
            ctx.on("e", slow)
            ctx.on("e", lambda x: 1 / 0)
            ctx.on("q", lambda x: None)
            ctx.on("q", lambda x: x * 2)
            ctx.on("q", lambda x: x * 3)

        f = await app.plugin(p)
        await app.emit("e", 1)
        assert got == [("a", 1), ("b", 1)]
        await app.parallel("e", 2)
        assert sorted(got[2:]) == [("a", 2), ("b", 2)]
        assert sum(e.kind == "event.error" for e in app.trace_log) == 2
        assert await app.serial("q", 5) == 10
        assert app.bail("q", 5) == 10
        with pytest.raises(TypeError):
            app.bail("e", 3)
        await f.dispose()
        await app.emit("e", 9)
        assert not any(x == 9 for _, x in got)


async def test_snapshot_tree_and_trace_hook():
    seen = []
    async with Kernel() as app:
        app.on_trace(seen.append)
        await app.plugin(llm_plugin, {"model": "v1"})
        await app.plugin(make_tool([]))
        snap = app.snapshot()
        names = [c["name"] for c in snap["fibers"]["children"]]
        assert names == ["llm", "tool"]
        assert snap["services"] == {"LLM": "llm#1"}
        tree = app.tree()
        assert "llm#1 [active]" in tree and "inject=['LLM']" in tree
        assert {"fiber.state", "service.bind"} <= {e.kind for e in seen}


async def test_concurrent_load_dispose_chaos_leaves_consistent_state():
    import random

    rnd = random.Random(int(__import__("os").environ.get("PYK_SEED", "1234")))
    live_tasks = []

    async def bg():
        live_tasks.append(1)
        try:
            await anyio.sleep_forever()
        finally:
            live_tasks.pop()

    def make(i):
        async def p(ctx):
            ctx.provide(f"svc{i}", i)
            ctx.on("tick", lambda: None)
            ctx.spawn(bg)
            await anyio.sleep(rnd.random() / 200)
            if i % 3 == 0:
                await ctx.plugin(plugin(lambda c: c.spawn(bg), inject=[f"svc{i - 1}"] if i else []))
        p.__name__ = f"p{i}"
        return p

    async with Kernel() as app:
        fibers = []

        async def load(i):
            fibers.append(await app.plugin(make(i)))

        async def disposer():
            for _ in range(30):
                await anyio.sleep(rnd.random() / 300)
                live = [c for c in app.fiber.children if c.state is not State.DISPOSED]
                if live:
                    await rnd.choice(live).dispose()

        async with anyio.create_task_group() as tg:
            for i in range(30):
                tg.start_soon(load, i)
            tg.start_soon(disposer)
        await app.settle()
        alive = [f for f in app._walk()]
        assert all(f.state in (State.ACTIVE, State.PENDING) for f in alive)
        # every service belongs to an ACTIVE fiber, every listener to a live fiber
        assert all(b.owner.state is State.ACTIVE for b in app._services.values())
        assert all(lst.fiber.state is State.ACTIVE for lst in app._listeners.get("tick", []))
        assert len(live_tasks) == sum(1 for f in alive if f.state is State.ACTIVE)
    assert live_tasks == [] and app._services == {} and app._listeners.get("tick") == []
