"""Chaos: concurrent loads, scopes, transactions (incl. failing and stop-first ones),
disposals and events under random interleavings. Invariants: no leak, no deadlock,
no silent PENDING, nothing staged survives.

Seeds: ``VENTRI_CHAOS_SEEDS`` (default 25; the M1 exit criterion run uses 1000) starting
at ``VENTRI_CHAOS_BASE`` (default 0).
"""
import os
import random

import anyio
import pytest

from ventri import (
    Kernel,
    PluginError,
    Retry,
    ServiceConflict,
    State,
    TransactionError,
    plugin,
)

pytestmark = pytest.mark.anyio

SEEDS = int(os.environ.get("VENTRI_CHAOS_SEEDS", "25"))
BASE = int(os.environ.get("VENTRI_CHAOS_BASE", "0"))
KEYS = ["llm", "db", "cache", "tools"]
EXPECTED = (PluginError, TransactionError, ServiceConflict, RuntimeError)


def _all_realms(app):
    yield app._root_realm
    for f in app._walk():
        if f.is_scope:
            yield f.realm


@pytest.mark.parametrize("seed", range(BASE, BASE + SEEDS))
async def test_chaos(seed):
    rnd = random.Random(seed)
    live_tasks = []
    exclusive_holders = []

    async def bg():
        live_tasks.append(1)
        try:
            await anyio.sleep_forever()
        finally:
            live_tasks.pop()

    def make(n):
        key = rnd.choice(KEYS)
        dep = rnd.choice([None, *KEYS])
        mode = rnd.choices(["ok", "fail", "slow", "exclusive"], [8, 1, 1, 1])[0]
        pause = rnd.random() / 500

        async def p(ctx, config):
            if mode == "exclusive":
                if exclusive_holders:
                    raise RuntimeError("exclusive resource busy")
                exclusive_holders.append(n)
                ctx.on_dispose(lambda: exclusive_holders.remove(n))
            ctx.spawn(bg)
            ctx.on("tick", lambda *_: None, priority=rnd.randint(-2, 2))
            await anyio.sleep(pause if mode != "slow" else 1)
            if mode == "fail":
                raise RuntimeError("load failed")
            if not ctx.has(key):
                ctx.provide(key, n)

        p.__name__ = f"p{n}"
        opts = {"inject": [dep] if dep and dep != key else []}
        if mode == "exclusive":
            opts["exclusive"] = True
        return plugin(p, name=f"p{n}", **opts)

    counter = iter(range(10_000))

    def live(app, scopes_only=False):
        return [f for f in app._walk() if f.state in (State.ACTIVE, State.PENDING)
                and f.tx is None and (f.is_scope if scopes_only else not f.is_scope)]

    async def guarded(coro):
        try:
            await coro
        except EXPECTED:
            pass

    async with Kernel(load_timeout=0.05, retry=Retry(max=1, base=0.001)) as app:

        def target_ctx():
            scopes = live(app, scopes_only=True)
            return rnd.choice(scopes).ctx if scopes and rnd.random() < 0.5 else app

        async def loader():
            for _ in range(8):
                await guarded(target_ctx().plugin(make(next(counter))))
                await anyio.sleep(rnd.random() / 400)

        async def scoper():
            for i in range(4):
                await guarded(app.scope(f"session:{seed}:{i}", isolate=rnd.sample(KEYS, 2)))
                await anyio.sleep(rnd.random() / 300)
                scopes = live(app, scopes_only=True)
                if scopes and rnd.random() < 0.5:
                    await guarded(rnd.choice(scopes).dispose())

        async def transactor():
            for _ in range(4):
                await anyio.sleep(rnd.random() / 300)

                async def body():
                    ctx = target_ctx()
                    async with ctx.transaction(timeout=0.5, reason="chaos") as tx:
                        for _ in range(rnd.randint(1, 3)):
                            op = rnd.choice(["add", "dispose", "replace"])
                            cands = [f for f in live(app) if tx._in_scope(f.parent)]
                            if op == "add" or not cands:
                                await tx.plugin(make(next(counter)))
                            elif op == "dispose":
                                await tx.dispose(rnd.choice(cands))
                            else:
                                await tx.replace(rnd.choice(cands), make(next(counter)))
                        if rnd.random() < 0.2:
                            raise RuntimeError("abort")
                await guarded(body())

        async def disposer():
            for _ in range(10):
                await anyio.sleep(rnd.random() / 300)
                fs = live(app)
                if fs:
                    await guarded(rnd.choice(fs).dispose())

        async def emitter():
            for _ in range(10):
                await anyio.sleep(rnd.random() / 300)
                await app.emit("tick")

        with anyio.fail_after(20):  # deadlock guard
            async with anyio.create_task_group() as tg:
                for fn in (loader, loader, scoper, transactor, transactor, disposer, emitter):
                    tg.start_soon(fn)
            await app.settle()
            # scheduled retries (max=1) may still be loading: wait for quiescence
            quiet = 0
            while quiet < 2:
                await anyio.sleep(0.01)  # > retry delay, so a scheduled retry has started
                quiet = 0 if any(f.state is State.LOADING for f in app._walk()) else quiet + 1
            await app.settle()

        fibers = list(app._walk())
        assert all(f.tx is None for f in fibers), "staged fiber survived its transaction"
        assert all(f.state in (State.ACTIVE, State.PENDING, State.FAILED) for f in fibers)
        for f in fibers:
            if f.state is State.PENDING and not f._parked:
                assert f.pending_reason, f"silent PENDING: {f!r}"
        for realm in _all_realms(app):
            for b in realm.services.values():
                assert b.owner.state is State.ACTIVE and b.tx is None
        assert all(lst.fiber.state is State.ACTIVE for lst in app._listeners.get("tick", []))
        for f in [app.fiber, *fibers]:
            if f._txlock is not None:
                assert not f._txlock._writer and f._txlock._readers == 0 and not f._txlock._queue
        assert len(exclusive_holders) <= 1
    assert live_tasks == [] and exclusive_holders == []
    assert app._services == {} and app._listeners.get("tick", []) == []
    assert all(not f._effects for f in [app.fiber])
