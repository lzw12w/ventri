"""Scopes / realms (DESIGN.md 4.6): per-session service isolation and reclamation."""
import anyio
import pytest

from ventri import Kernel, ServiceConflict, State, plugin

from .conftest import LLM, llm_plugin, wait_for

pytestmark = pytest.mark.anyio


class SessionLog:
    def __init__(self, sid: str) -> None:
        self.sid = sid
        self.closed = False


class Grants:
    pass


def session_log(ctx, config):
    log = SessionLog(config["sid"])
    ctx.provide(SessionLog, log, name="log")
    ctx.on_dispose(lambda: setattr(log, "closed", True))


def make_agent(seen: list, running: list):
    @plugin(name="agent", inject=[SessionLog, LLM])
    def agent(ctx):
        seen.append((ctx.get(SessionLog).sid, ctx.get(LLM).model))

        async def loop():
            running.append(ctx.get(SessionLog).sid)
            try:
                await anyio.sleep_forever()
            finally:
                running.remove(ctx.get(SessionLog).sid)
        ctx.spawn(loop)
    return agent


async def test_two_sessions_are_isolated_and_reclaimed_to_baseline():
    seen, running = [], []
    agent = make_agent(seen, running)
    async with Kernel() as app:
        await app.plugin(llm_plugin, {"model": "v1"})
        baseline = app.snapshot()

        a = await app.scope("session:a", isolate={SessionLog, Grants})
        b = await app.scope("session:b", isolate={SessionLog, Grants})
        await a.ctx.plugin(session_log, {"sid": "a"})
        await b.ctx.plugin(session_log, {"sid": "b"})
        await a.ctx.plugin(agent)
        await b.ctx.plugin(agent)
        await wait_for(lambda: len(running) == 2)

        # each session sees its own SessionLog, both share the root LLM
        assert sorted(seen) == [("a", "v1"), ("b", "v1")]
        assert a.ctx.get(SessionLog).sid == "a" and b.ctx.get(SessionLog).sid == "b"
        assert a.ctx.log.sid == "a"  # name-based attribute access is realm-aware too
        # the root realm does not see session services
        assert not app.has(SessionLog)
        snap = app.snapshot()
        assert snap["services"] == {"LLM": "llm#1"}
        assert snap["realms"] == {
            a.label: {"isolate": ["Grants", "SessionLog"], "services": {"SessionLog": "session_log#4"}},
            b.label: {"isolate": ["Grants", "SessionLog"], "services": {"SessionLog": "session_log#5"}},
        }

        log_a = a.ctx.get(SessionLog)
        await a.dispose()
        assert log_a.closed and running == ["b"] and a.state is State.DISPOSED
        assert b.ctx.get(SessionLog).sid == "b"  # the other session is untouched
        await b.dispose()
        assert running == []
        assert app.snapshot() == baseline


async def test_same_isolated_key_in_two_sessions_does_not_conflict():
    async with Kernel() as app:
        a = await app.scope("a", isolate={SessionLog})
        b = await app.scope("b", isolate={SessionLog})
        a.ctx.provide(SessionLog, SessionLog("a"))
        b.ctx.provide(SessionLog, SessionLog("b"))  # no ServiceConflict
        f = await a.ctx.plugin(lambda ctx: ctx.provide(SessionLog, SessionLog("dup")))
        assert f.state is State.FAILED and isinstance(f.error, ServiceConflict)


async def test_non_isolated_keys_go_to_the_root_realm():
    async with Kernel() as app:
        s = await app.scope("s", isolate={SessionLog})
        await s.ctx.plugin(llm_plugin, {"model": "from-session"})
        # LLM is not isolated by the scope -> registered globally
        assert app.get(LLM).model == "from-session"
        await s.dispose()
        assert not app.has(LLM)


async def test_isolation_has_no_fallback_to_outer_realm():
    """Documented semantics: an isolated key never falls back to an outer realm, so a
    session plugin waits for the session's own provider (it is PENDING, not wired to
    the root's instance)."""
    seen, running = [], []
    async with Kernel() as app:
        await app.plugin(llm_plugin)
        app.provide(SessionLog, SessionLog("root"))
        s = await app.scope("s", isolate={SessionLog})
        ag = await s.ctx.plugin(make_agent(seen, running))
        assert ag.state is State.PENDING and seen == []
        assert ag.pending_reason == "missing: SessionLog (no plugin provides it)"
        await s.ctx.plugin(session_log, {"sid": "s"})
        assert ag.state is State.ACTIVE and seen == [("s", "v1")]


async def test_nested_scopes_resolve_to_nearest_isolating_ancestor():
    async with Kernel() as app:
        outer = await app.scope("outer", isolate={SessionLog, Grants})
        inner = await outer.ctx.scope("inner", isolate={SessionLog})
        outer.ctx.provide(SessionLog, SessionLog("outer"))
        outer.ctx.provide(Grants, Grants())
        inner.ctx.provide(SessionLog, SessionLog("inner"))
        leaf = await inner.ctx.plugin(lambda ctx: None)
        assert leaf.ctx.get(SessionLog).sid == "inner"
        assert leaf.ctx.get(Grants) is outer.ctx.get(Grants)  # inner reads outer's realm
        assert outer.ctx.get(SessionLog).sid == "outer"
        await inner.dispose()
        assert outer.ctx.get(SessionLog).sid == "outer"


async def test_dependents_inside_a_scope_follow_scope_services():
    seen, running = [], []
    async with Kernel() as app:
        await app.plugin(llm_plugin)
        s = await app.scope("s", isolate={SessionLog})
        log = await s.ctx.plugin(session_log, {"sid": "v1"})
        ag = await s.ctx.plugin(make_agent(seen, running))
        new = await s.ctx.replace(log, {"sid": "v2"})
        assert new.state is State.ACTIVE and ag.state is State.ACTIVE
        assert seen == [("v1", "v1"), ("v2", "v1")]
        await new.dispose()
        assert ag.state is State.PENDING


async def test_session_change_only_reconciles_the_session_subtree(monkeypatch):
    async with Kernel() as app:
        outside = [await app.plugin(plugin(lambda ctx: None, inject=["never"])) for _ in range(3)]
        s = await app.scope("s", isolate={"k"})
        walked = []
        orig = Kernel._region

        def spy(self, realms):
            out = orig(self, realms)
            walked.extend(out)
            return out
        monkeypatch.setattr(Kernel, "_region", spy)
        dep = await s.ctx.plugin(plugin(lambda ctx: None, inject=["k"]))
        await s.ctx.plugin(lambda ctx: ctx.provide("k", 1))
        assert dep.state is State.ACTIVE
        assert walked and not any(f in walked for f in outside)


async def test_events_do_not_cross_sibling_scopes():
    heard = []

    def listener(tag):
        def p(ctx):
            ctx.on("evt", lambda src: heard.append((tag, src)))
        return p

    async with Kernel() as app:
        await app.plugin(listener("root"))
        a = await app.scope("a")
        b = await app.scope("b")
        nested = await a.ctx.scope("a.n")
        await a.ctx.plugin(listener("a"))
        await b.ctx.plugin(listener("b"))
        await nested.ctx.plugin(listener("a.n"))

        await a.ctx.emit("evt", "a")
        assert sorted(heard) == [("a", "a"), ("a.n", "a"), ("root", "a")]
        heard.clear()
        await nested.ctx.emit("evt", "a.n")
        assert sorted(heard) == [("a", "a.n"), ("a.n", "a.n"), ("root", "a.n")]
        heard.clear()
        await app.emit("evt", "root")  # root-level events reach every scope
        assert sorted(heard) == [("a", "root"), ("a.n", "root"), ("b", "root"), ("root", "root")]


async def test_transaction_inside_scope_rolls_back_realm_exactly():
    async with Kernel() as app:
        s = await app.scope("s", isolate={SessionLog})
        log = await s.ctx.plugin(session_log, {"sid": "v1"})
        before = app.snapshot()

        async def broken(ctx):
            raise RuntimeError("nope")
        with pytest.raises(Exception, match="nope"):
            async with s.ctx.transaction() as tx:
                await tx.replace(log, config={"sid": "v2"})
                assert s.ctx.get(SessionLog).sid == "v1"  # live view unchanged
                await tx.plugin(broken)
        assert app.snapshot() == before
        assert s.ctx.get(SessionLog).sid == "v1" and log.state is State.ACTIVE


async def test_scope_dispose_cancels_tasks_and_listeners_of_all_descendants():
    cancelled = []

    async def bg(tag):
        try:
            await anyio.sleep_forever()
        finally:
            cancelled.append(tag)

    async def p(ctx):
        ctx.spawn(bg, ctx.fiber.name)
        ctx.on("x", lambda: None)

    async with Kernel() as app:
        s = await app.scope("s", isolate={SessionLog})
        inner = await s.ctx.scope("inner")
        await s.ctx.plugin(p)
        await inner.ctx.plugin(p)
        await anyio.sleep(0.01)
        await s.dispose()
        assert sorted(cancelled) == ["p", "p"]
        assert app._listeners.get("x") == []
        assert app.fiber.children == []


async def test_scope_disposed_during_its_transaction_fails_the_transaction():
    """Regression (chaos seed 869): disposing a scope while a transaction has staged
    plugins under it must fail the transaction, not commit bindings of disposed
    fibers into the root realm."""
    from ventri import TransactionError

    async with Kernel() as app:
        s = await app.scope("session:x", isolate=["llm"])
        baseline = app.snapshot()
        with pytest.raises(TransactionError, match="disposed"):
            async with s.ctx.transaction() as tx:
                await tx.plugin(lambda ctx, config: ctx.provide("cache", 1))
                await s.dispose()
        assert "cache" not in app._services
        assert s.state is State.DISPOSED
        assert app.snapshot()["services"] == baseline["services"]
    assert app._services == {}


async def test_staged_child_of_disposed_staged_parent_is_not_committed():
    async with Kernel() as app:
        async with app.transaction() as tx:
            g = await tx.plugin(lambda ctx, config: None)
            await tx.plugin(lambda ctx, config: ctx.provide("k", 1), parent=g)
            await tx.dispose(g)  # drops the staged subtree from the transaction
        assert "k" not in app._services and not app.fiber.children
