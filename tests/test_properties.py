"""Hypothesis property tests: transaction atomicity, scope isolation, event order,
config-diff convergence."""
import os

import anyio
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ventri import Kernel, PluginError, State, TransactionError, plugin

KEYS = ["a", "b", "c", "d"]
SETTINGS = settings(max_examples=int(os.environ.get("VENTRI_HYPOTHESIS_EXAMPLES", "60")), deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


def make(key, dep, fail, tag):
    def p(ctx, config):
        if fail:
            raise RuntimeError("boom")
        if not ctx.has(key):
            ctx.provide(key, tag)
        ctx.on("ev", lambda: None)
    return plugin(p, name=f"p{tag}", inject=[dep] if dep and dep != key else [])


plugin_st = st.tuples(st.sampled_from(KEYS), st.one_of(st.none(), st.sampled_from(KEYS)))
op_st = st.one_of(
    st.tuples(st.just("add"), plugin_st, st.booleans()),
    st.tuples(st.just("dispose"), st.integers(0, 20)),
    st.tuples(st.just("replace"), st.integers(0, 20), plugin_st),
)


def _invariants(app):
    for f in app._walk():
        assert f.tx is None
    for b in app._services.values():
        assert b.owner.state is State.ACTIVE and b.tx is None
    assert all(lst.fiber.state is State.ACTIVE for lst in app._listeners.get("ev", []))


@SETTINGS
@given(base=st.lists(plugin_st, max_size=5), ops=st.lists(op_st, min_size=1, max_size=6),
       abort=st.booleans())
def test_transaction_is_atomic(base, ops, abort):
    async def main():
        tags = iter(range(1000))
        async with Kernel() as app:
            for key, dep in base:
                await app.plugin(make(key, dep, False, next(tags)))
            before = app.snapshot()
            failed = False
            try:
                async with app.transaction() as tx:
                    for op in ops:
                        live = [f for f in app.fiber.children if f.tx is None
                                and f not in tx._removed()]
                        if op[0] == "add":
                            (key, dep), fail = op[1], op[2]
                            await tx.plugin(make(key, dep, fail and not abort, next(tags)))
                        elif op[0] == "dispose" and live:
                            await tx.dispose(live[op[1] % len(live)])
                        elif op[0] == "replace" and live:
                            key, dep = op[2]
                            await tx.replace(live[op[1] % len(live)], make(key, dep, False, next(tags)))
                    if abort:
                        raise KeyError("abort")
            except (KeyError, PluginError, TransactionError):
                failed = True
            if failed:
                assert app.snapshot() == before  # rollback restores the exact pre-state
            _invariants(app)
        assert app._services == {}
    anyio.run(main, backend="asyncio")


@SETTINGS
@given(sessions=st.lists(st.sets(st.sampled_from(KEYS), min_size=1), min_size=1, max_size=4),
       root_keys=st.sets(st.sampled_from(KEYS)))
def test_scope_isolation_and_reclaim(sessions, root_keys):
    async def main():
        async with Kernel() as app:
            for k in root_keys:
                app.provide(k, "root")
            baseline = app.snapshot()
            scopes = []
            for i, iso in enumerate(sessions):
                s = await app.scope(f"s{i}", isolate=iso)
                await s.ctx.plugin(lambda ctx, config, iso=iso, i=i: [
                    ctx.provide(k, f"s{i}") for k in iso])
                scopes.append((s, iso))
            for i, (s, iso) in enumerate(scopes):
                for k in KEYS:
                    expect = f"s{i}" if k in iso else ("root" if k in root_keys else None)
                    assert s.ctx.get(k, None) == expect  # never another session's value
            for k in KEYS:
                assert app.get(k, None) == ("root" if k in root_keys else None)
            for s, _ in scopes:
                await s.dispose()
            assert app.snapshot() == baseline
    anyio.run(main, backend="asyncio")


@SETTINGS
@given(prios=st.lists(st.integers(-3, 3), min_size=1, max_size=12))
def test_event_priority_order(prios):
    async def main():
        order = []
        async with Kernel() as app:
            for i, pr in enumerate(prios):
                app.on("e", lambda i=i: order.append(i), priority=pr)
            await app.emit("e")
            expected = sorted(range(len(prios)), key=lambda i: (-prios[i], i))
            assert order == expected
            order.clear()
            await app.serial("e")
            assert order == expected
    anyio.run(main, backend="asyncio")


entry_st = st.fixed_dictionaries({"id": st.sampled_from(["x", "y", "z", "w"]),
                                  "name": st.sampled_from(["n1", "n2"]),
                                  "group": st.booleans()})


@SETTINGS
@given(a=st.lists(entry_st, max_size=4, unique_by=lambda e: e["id"]),
       b=st.lists(entry_st, max_size=4, unique_by=lambda e: e["id"]))
def test_config_diff_converges(a, b):
    from ventri_std.config import DictSecrets, apply_document, load_document, plan

    def doc(entries):
        lines = ["version: 1", "plugins:"] if entries else ["version: 1"]
        for e in entries:
            if e["group"]:
                lines += [f"  - group: {e['id']}", "    plugins:",
                          "      - use: tests.std.cfg_plugins.tool",
                          f"        config: {{name: {e['name']}}}"]
            else:
                lines += ["  - use: tests.std.cfg_plugins.tool", f"    id: {e['id']}",
                          f"    config: {{name: {e['name']}}}"]
        return load_document(text="\n".join(lines), secrets=DictSecrets({}))

    def shape(app):
        def node(f):
            return (f.meta.get("config_id"), f.name, repr(f.raw_config),
                    tuple(sorted(node(c) for c in f.children)))
        return sorted(node(f) for f in app.fiber.children)

    async def main():
        async with Kernel() as one, Kernel() as two:
            assert (await apply_document(one.fiber, doc(a))).ok
            assert (await apply_document(one.fiber, doc(b))).ok
            assert (await apply_document(two.fiber, doc(b))).ok
            assert shape(one) == shape(two)  # A -> B equals B from scratch
            assert plan(one.fiber, doc(b)).empty  # and re-applying B is a no-op
    anyio.run(main, backend="asyncio")
