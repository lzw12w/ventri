"""Kernel-level load timeout and FAILED retry policy (DESIGN.md 4.3, 4.9)."""
import anyio
import pytest

from ventri import Kernel, PluginError, State, plugin
from ventri.errors import LoadTimeout
from ventri.plugin import Retry

from .conftest import wait_for

pytestmark = pytest.mark.anyio


async def test_load_timeout_fails_and_cleans_up():
    cleaned, cancelled = [], []

    async def bg():
        try:
            await anyio.sleep_forever()
        finally:
            cancelled.append(True)

    @plugin(timeout=0.05)
    async def hangs(ctx):
        ctx.provide("half", 1)
        ctx.spawn(bg)
        ctx.on_dispose(lambda: cleaned.append(True))
        await anyio.sleep_forever()

    async with Kernel() as app:
        other = await app.plugin(lambda ctx: None)
        f = await app.plugin(hangs)
        assert f.state is State.FAILED and isinstance(f.error, LoadTimeout)
        assert f.error.timeout == 0.05
        assert cleaned == [True] and cancelled == [True] and not app.has("half")
        assert other.state is State.ACTIVE
        assert [e.data["new"] for e in app.trace_log if e.fiber == f.label and e.kind == "fiber.state"] \
            == ["loading", "failed"]


async def test_timeout_defaults_and_overrides():
    async def slow(ctx):
        await anyio.sleep(0.1)

    async with Kernel(load_timeout=0.02) as app:
        a = await app.plugin(slow)
        assert isinstance(a.error, LoadTimeout)          # kernel default applies
        b = await app.plugin(slow, timeout=None)          # disabled per fiber
        assert b.state is State.ACTIVE
        c = await app.plugin(plugin(slow, timeout=1))     # plugin metadata wins over default
        assert c.state is State.ACTIVE and c.load_timeout == 1
        d = await app.plugin(plugin(slow, timeout=1), timeout=0.01)  # fiber override wins
        assert isinstance(d.error, LoadTimeout)


async def test_timeout_covers_loading_only_not_running_tasks():
    @plugin(timeout=0.02)
    def p(ctx):
        ctx.spawn(anyio.sleep_forever)

    async with Kernel() as app:
        f = await app.plugin(p)
        await anyio.sleep(0.05)
        assert f.state is State.ACTIVE


async def test_timeout_inside_transaction_rolls_back():
    async with Kernel() as app:
        before = app.snapshot()
        with pytest.raises(PluginError) as ei:
            async with app.transaction() as tx:
                await tx.plugin(lambda ctx: ctx.provide("ok", 1))
                await tx.plugin(plugin(anyio.sleep_forever, timeout=0.02))
        assert isinstance(ei.value.__cause__, LoadTimeout)
        assert app.snapshot() == before


async def test_no_retry_by_default():
    async with Kernel() as app:
        f = await app.plugin(lambda ctx: 1 / 0)
        await anyio.sleep(0.05)
        assert f.state is State.FAILED
        assert not any(e.kind == "fiber.retry" for e in app.trace_log)


async def test_retry_recovers_a_flaky_plugin():
    attempts = []

    @plugin(retry={"max": 3, "base": 0.005})
    def flaky(ctx):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("not yet")
        ctx.provide("svc", "up")

    async with Kernel() as app:
        f = await app.plugin(flaky)
        assert f.state is State.FAILED
        await wait_for(lambda: f.state is State.ACTIVE)
        assert app.get("svc") == "up" and len(attempts) == 3
        retries = [e.data["attempt"] for e in app.trace_log if e.kind == "fiber.retry"]
        assert retries == [1, 2]


async def test_retry_gives_up_after_max_with_exponential_backoff():
    assert [Retry(base=1, cap=5).delay(n) for n in (1, 2, 3, 4)] == [1, 2, 4, 5]
    assert Retry(backoff="fixed", base=2).delay(7) == 2
    attempts = []

    def always(ctx):
        attempts.append(anyio.current_time())
        raise RuntimeError("down")

    async with Kernel(retry=Retry(max=2, base=0.01)) as app:  # kernel-wide default policy
        f = await app.plugin(always)
        await wait_for(lambda: any(e.data.get("gave_up") for e in app.trace_log
                                   if e.kind == "fiber.retry"))
        assert f.state is State.FAILED and len(attempts) == 3
        assert attempts[2] - attempts[1] >= attempts[1] - attempts[0]  # backoff grows
        await f.restart()  # manual restart resets the streak (fails again, retries again)
        await wait_for(lambda: len(attempts) == 6)


async def test_retry_after_task_crash_and_streak_reset():
    starts = []

    async def crash():
        await anyio.sleep(0.01)
        raise RuntimeError("crash")

    def crashing(ctx):
        starts.append(1)
        ctx.spawn(crash)

    async with Kernel() as app:
        # reset_after is long: crashes count towards one streak -> gives up after max
        f = await app.plugin(crashing, retry=Retry(max=2, base=0.001, reset_after=60))
        await wait_for(lambda: any(e.data.get("gave_up") for e in app.trace_log
                                   if e.kind == "fiber.retry" and e.fiber == f.label))
        assert len(starts) == 3 and f.state is State.FAILED
        # reset_after=0: every activation counts as healthy -> the streak never exceeds 1
        starts.clear()
        g = await app.plugin(crashing, retry=Retry(max=1, base=0.001, reset_after=0))
        await wait_for(lambda: len(starts) >= 4)
        assert g._failures == 1
        await g.dispose()


async def test_dispose_cancels_scheduled_retry():
    attempts = []

    def bad(ctx):
        attempts.append(1)
        raise RuntimeError("x")

    async with Kernel() as app:
        f = await app.plugin(bad, retry=Retry(max=3, base=0.03))
        await f.dispose()
        await anyio.sleep(0.08)
        assert attempts == [1] and f.state is State.DISPOSED


async def test_staged_fibers_are_not_retried():
    attempts = []

    @plugin(retry=Retry(max=5, base=0.001))
    def bad(ctx):
        attempts.append(1)
        raise RuntimeError("x")

    async with Kernel() as app:
        with pytest.raises(PluginError):
            async with app.transaction() as tx:
                await tx.plugin(bad)
        await anyio.sleep(0.02)
        assert attempts == [1]
