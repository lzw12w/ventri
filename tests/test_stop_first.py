"""Stop-first replacement for exclusive resources (DESIGN.md 4.5.4, N3)."""
import pytest

from ventri import Kernel, PluginError, State, TransactionError, plugin

pytestmark = pytest.mark.anyio

PORTS: dict[int, str] = {}


class Server:
    def __init__(self, port: int, tag: str) -> None:
        self.port, self.tag, self.requests = port, tag, 0


def make_server(log: list, exclusive: bool = True):
    @plugin(name="server", exclusive=exclusive, provides=[Server])
    def server(ctx, config):
        port, tag = config["port"], config["tag"]
        if config.get("fail"):
            raise RuntimeError("bad config")
        if port in PORTS:
            raise OSError(f"port {port} in use by {PORTS[port]}")
        PORTS[port] = tag
        log.append(f"start {tag}")
        ctx.provide(Server, Server(port, tag))

        def stop():
            del PORTS[port]
            log.append(f"stop {tag}")
        ctx.on_dispose(stop)
    return server


def make_client(log: list):
    @plugin(name="client")
    def client(ctx, config, srv: Server):
        log.append(f"client->{srv.tag}")
    return client


@pytest.fixture(autouse=True)
def _ports():
    PORTS.clear()
    yield
    PORTS.clear()


async def test_blue_green_cannot_replace_an_exclusive_resource():
    log = []
    async with Kernel() as app:
        old = await app.plugin(make_server(log, exclusive=False), {"port": 80, "tag": "v1"})
        with pytest.raises(PluginError, match="port 80 in use"):
            await app.replace(old, {"port": 80, "tag": "v2"})
        assert old.state is State.ACTIVE and PORTS == {80: "v1"}


async def test_stop_first_stops_old_then_starts_new():
    log = []
    async with Kernel() as app:
        old = await app.plugin(make_server(log), {"port": 80, "tag": "v1"})
        await app.plugin(make_client(log))
        async with app.transaction() as tx:
            new = await tx.replace(old, config={"port": 80, "tag": "v2"})
            assert new.state is State.PENDING and new.pending_reason.startswith("parked")
            assert log == ["start v1", "client->v1"]  # nothing happens before commit
        assert log == ["start v1", "client->v1", "stop v1", "start v2", "client->v2"]
        assert old.state is State.DISPOSED and new.state is State.ACTIVE
        assert app.get(Server).tag == "v2" and PORTS == {80: "v2"}
        r = tx.report
        assert r.replaced == {new.label: old.label} and r.services["replaced"] == ["Server"]
        assert r.restarted == ["client#2"] and not r.degraded


async def test_explicit_strategy_on_a_non_exclusive_plugin():
    log = []
    async with Kernel() as app:
        old = await app.plugin(make_server(log, exclusive=False), {"port": 80, "tag": "v1"})
        new = await app.replace(old, {"port": 80, "tag": "v2"}, strategy="stop-first")
        assert new.state is State.ACTIVE and PORTS == {80: "v2"}
        with pytest.raises(ValueError, match="unknown replace strategy"):
            async with app.transaction() as tx:
                await tx.replace(new, strategy="sideways")


async def test_failed_stop_first_is_a_degraded_rollback():
    log = []
    async with Kernel() as app:
        old = await app.plugin(make_server(log), {"port": 80, "tag": "v1"})
        client = await app.plugin(make_client(log))
        old_instance = app.get(Server)
        old_instance.requests = 7  # in-memory state
        before = app.snapshot()
        with pytest.raises(TransactionError, match="bad config"):
            async with app.transaction() as tx:
                await tx.replace(old, config={"port": 80, "tag": "v2", "fail": True})
        # configuration and service topology are restored ...
        assert app.snapshot() == before
        assert old.state is State.ACTIVE and old.raw_config == {"port": 80, "tag": "v1"}
        assert client.state is State.ACTIVE and PORTS == {80: "v1"}
        # ... but the old instance was restarted: its memory state is gone
        assert app.get(Server) is not old_instance and app.get(Server).requests == 0
        assert log == ["start v1", "client->v1", "stop v1", "start v1", "client->v1"]
        assert tx.report.degraded and tx.report.outcome == "rolled_back"
        rb = [e for e in app.trace_log if e.kind == "tx.rollback"][-1]
        assert rb.data["degraded"] is True


async def test_failure_before_the_stop_phase_is_a_clean_rollback():
    log = []

    def broken(ctx):
        raise RuntimeError("other plugin broken")

    async with Kernel() as app:
        old = await app.plugin(make_server(log), {"port": 80, "tag": "v1"})
        inst = app.get(Server)
        with pytest.raises(TransactionError):
            async with app.transaction() as tx:
                await tx.replace(old, config={"port": 80, "tag": "v2"})
                # staged after the replace; it only fails at commit preparation
                await tx.plugin(plugin(broken, inject=["late"]))
                await app.plugin(lambda ctx: ctx.provide("late", 1))
        assert app.get(Server) is inst and not tx.report.degraded  # old never stopped
        assert log == ["start v1"]


async def test_live_dependents_observe_a_gap_during_stop_first():
    """Documented non-guarantee: unlike blue-green (N4), stop-first has a window in
    which the service is absent from the live registry."""
    log, gaps = [], []
    async with Kernel() as app:
        old = await app.plugin(make_server(log), {"port": 80, "tag": "v1"})
        app.on_trace(lambda e: gaps.append(e.kind) if not app.has(Server) else None)
        await app.replace(old, {"port": 80, "tag": "v2"})
        assert gaps and app.has(Server)


async def test_stop_first_is_skipped_in_a_dry_run():
    log = []
    async with Kernel() as app:
        old = await app.plugin(make_server(log), {"port": 80, "tag": "v1"})
        before = app.snapshot()
        async with app.transaction(dry_run=True) as tx:
            new = await tx.replace(old, config={"port": 80, "tag": "v2"})
        assert tx.report.skipped == {new.label: f"stop-first replacement of {old.label} "
                                                "is not started in a dry run"}
        assert log == ["start v1"] and app.snapshot() == before
