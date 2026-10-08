"""ventri_std.config: YAML -> profiles -> secrets -> validation -> diff -> one transaction."""
import textwrap

import anyio
import pytest

from ventri import Kernel, Secret, State
from ventri_std import config as cfgmod
from ventri_std.config import ConfigError, DictSecrets, EnvSecrets, Loader, interpolate, load_document

from . import cfg_plugins

pytestmark = pytest.mark.anyio

P = "tests.std.cfg_plugins"
BASE = f"""
version: 1
plugins:
  - use: {P}.LLM
    id: ds
    config: {{ model: flash, api_key: "${{secret:deepseek}}" }}
  - use: {P}.agent
  - group: tools
    plugins:
      - use: {P}.tool
        config: {{ name: a }}
      - use: {P}.tool
        config: {{ name: b }}
"""
SECRETS = DictSecrets({"deepseek": "sk-live-123"})


def write(path, text):
    path.write_text(textwrap.dedent(text))


def loader(app, path, **kw):
    kw.setdefault("secrets", SECRETS)
    return Loader(app.fiber, path, **kw)


def commits(app):
    return [e for e in app.trace_log if e.kind == "tx.commit" and e.data.get("origin") == "config"]


def managed(app):
    return {f.meta["config_id"]: f for f in app.fiber.children if "config_id" in f.meta} | {
        f.meta["config_id"]: f for g in app.fiber.children for f in g.children if "config_id" in f.meta}


async def test_initial_apply_builds_tree_in_one_transaction(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE)
    async with Kernel() as app:
        res = await loader(app, cfg).apply()
        assert res.ok and res.report.outcome == "committed" and len(commits(app)) == 1
        m = managed(app)
        assert set(m) == {"ds", f"{P}.agent", "tools", f"tools/{P}.tool", f"tools/{P}.tool@2"}
        assert all(f.state is State.ACTIVE for f in m.values())
        assert app.get("agent")["llm"] is app.get("llm")
        assert app.get("llm").cfg.api_key == Secret("sk-live-123")
        assert m["ds"].path == "root/ds" and m[f"tools/{P}.tool@2"].path == f"root/tools/{P}.tool@2"
        assert [c.op for c in res.plan.changes].count("add") == 5
        blob = repr(app.snapshot()) + app.tree() + str(res) + res.report.to_json() + repr(
            [e.to_dict() for e in app.trace_log])
        assert "sk-live-123" not in blob


async def test_reapply_without_semantic_change_is_a_noop(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE)
    async with Kernel() as app:
        ld = loader(app, cfg)
        await ld.apply()
        before = app.snapshot()
        # reformatted YAML + an explicitly spelled-out default: same validated configs
        write(cfg, BASE.replace("{ model: flash, api_key", "{ fail: false, api_key").replace(
            "config: { name: a }", "config:\n          name: a"))
        res = await ld.apply()
        assert res.plan.empty and res.report is None and str(res.plan) == "no changes"
        assert app.snapshot() == before and len(commits(app)) == 1


async def test_exit_criterion_edit_applies_hot_in_one_transaction(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE)
    async with Kernel() as app:
        ld = loader(app, cfg)
        await ld.apply()
        old = managed(app)
        write(cfg, BASE.replace("model: flash", "model: pro"))
        res = await ld.apply()
        assert res.ok and len(commits(app)) == 2
        assert [(c.op, c.id) for c in res.plan.changes] == [("replace", "ds")]
        assert "config changed: model" in str(res.plan)
        new = managed(app)
        assert new["ds"] is not old["ds"] and old["ds"].state is State.DISPOSED
        assert app.get("llm").cfg.model == "pro"
        assert app.get("agent")["llm"] is app.get("llm")  # dependent restarted
        for cid in ("tools", f"tools/{P}.tool", f"tools/{P}.tool@2"):
            assert new[cid] is old[cid]  # untouched
        assert res.report.replaced == {new["ds"].label: old["ds"].label}
        assert new["ds"].meta == {"config_id": "ds", "id": "ds", "config_use": f"{P}.LLM"}


async def test_exit_criterion_failure_rolls_back_old_config_keeps_serving(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE)
    async with Kernel() as app:
        ld = loader(app, cfg)
        await ld.apply()
        llm = app.get("llm")
        before = app.snapshot()
        write(cfg, BASE.replace("model: flash", "model: pro, fail: true").replace(
            "config: { name: b }", "config: { name: c }"))
        res = await ld.apply()
        assert not res.ok and res.report.outcome == "rolled_back"
        assert "bad llm config" in res.report.error
        assert app.get("llm") is llm and llm.cfg.model == "flash"  # old config still serving
        assert app.get("agent")["llm"] is llm
        assert app.snapshot() == before  # the tool change was rolled back too
        assert ld.document is not None and ld.last is res
        assert [e.data.get("origin") for e in app.trace_log if e.kind == "tx.rollback"] == ["config"]


async def test_dry_run_reports_without_changing(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE)
    async with Kernel() as app:
        ld = loader(app, cfg)
        dry = await ld.apply(dry_run=True)
        assert dry.report.outcome == "dry_run" and dry.report.ok and len(dry.report.added) == 5
        assert not app.fiber.children
        await ld.apply()
        before = app.snapshot()
        write(cfg, BASE.replace("model: flash", "fail: true"))
        dry = await ld.apply(dry_run=True)
        assert dry.report.dry_run and not dry.ok and any("bad llm config" in v for v in
                                                          dry.report.failures.values())
        assert app.snapshot() == before and ld.last is not dry


async def test_profiles_patch_ops(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE.replace("version: 1", "version: 1\nprofiles: [work]"))
    (tmp_path / "profiles").mkdir()
    write(tmp_path / "profiles" / "work.yml", f"""
        patch:
          - {{ id: ds, config: {{ model: pro }} }}
          - {{ id: "{P}.tool@2", disabled: true }}
          - {{ add: {{ use: {P}.tool, id: extra, config: {{ name: x }} }}, under: tools }}
          - {{ remove: "{P}.agent" }}
    """)
    async with Kernel() as app:
        ld = loader(app, cfg)
        await ld.apply()
        m = managed(app)
        assert set(m) == {"ds", "tools", f"tools/{P}.tool", "tools/extra"}
        assert app.get("llm").cfg.model == "pro"
        assert app.get("llm").cfg.api_key.reveal() == "sk-live-123"  # deep merge kept the key
        assert ld.document.profiles == ["work"]
        # override the profile list (CLI --profile): back to base
        ld.profiles = []
        res = await ld.apply()
        assert res.ok and set(managed(app)) == {"ds", f"{P}.agent", "tools", f"tools/{P}.tool",
                                                f"tools/{P}.tool@2"}


async def test_profile_errors(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE)
    (tmp_path / "profiles").mkdir()
    for patch, msg in ((("- { add: { group: g2, plugins: [{use: a, id: x}] } }\n"
                         "  - { add: { use: b, id: x }, under: tools }\n"
                         "  - { id: x, disabled: true }"), "ambiguous"),
                       ("- { id: nope, disabled: true }", "no entry"),
                       ("- { frobnicate: 1 }", "needs 'id'"),
                       ("- { add: { use: x }, under: ds }", "must name a group")):
        write(tmp_path / "profiles" / "p.yml", "patch:\n  " + patch)
        with pytest.raises(ConfigError, match=msg):
            load_document(cfg, profiles=["p"], secrets=SECRETS)
    with pytest.raises(ConfigError, match="cannot read"):
        load_document(cfg, profiles=["missing"], secrets=SECRETS)


async def test_disable_group_disposes_subtree(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE)
    async with Kernel() as app:
        ld = loader(app, cfg)
        await ld.apply()
        tools = managed(app)["tools"]
        write(cfg, BASE.replace("  - group: tools", "  - group: tools\n    disabled: true"))
        res = await ld.apply()
        assert [(c.op, c.id, c.reason) for c in res.plan.changes] == [("remove", "tools", "disabled")]
        assert tools.state is State.DISPOSED and not tools.children
        assert set(managed(app)) == {"ds", f"{P}.agent"}


async def test_validation_errors_are_collected_and_change_nothing(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE)
    async with Kernel() as app:
        ld = loader(app, cfg)
        await ld.apply()
        before = app.snapshot()
        write(cfg, BASE.replace("model: flash", "modle: flash").replace(f"{P}.agent", f"{P}.nope"))
        with pytest.raises(ConfigError) as ei:
            await ld.apply()
        assert len(ei.value.errors) == 2
        assert any(e.startswith("ds: TypeError") for e in ei.value.errors)
        assert any("cannot resolve use" in e for e in ei.value.errors)
        assert app.snapshot() == before
    for text, msg in (("version: 2", "unsupported config version"), ("[1, 2]", "mapping"),
                      ("version: 1\nplugins: [{id: x}]", "needs 'use'"),
                      ("version: 1\nplugins: [{use: a, bogus: 1}]", "unknown keys"),
                      ("version: 1\nplugins: [{use: a, id: x}, {use: b, id: x}]", "duplicate id"),
                      ("version: 1\nplugins: [{use: a, id: 'x/y'}]", "invalid id"),
                      ("version: 1\nplugins: {a: 1}", "must be a list"),
                      ("version: [", "invalid YAML")):
        with pytest.raises(ConfigError, match=msg):
            load_document(text=text, secrets=SECRETS)


async def test_secrets_and_env(tmp_path, monkeypatch):
    env = {"HOME_DIR": "/home/j", "VENTRI_SECRET_DEEP_SEEK": "from-env"}
    assert interpolate({"a": ["${env:HOME_DIR}/x", "${env:NOPE:-dflt}"]}, SECRETS, env) == {
        "a": ["/home/j/x", "dflt"]}
    v = interpolate("Bearer ${secret:deepseek}", SECRETS, env)
    assert isinstance(v, Secret) and v.reveal() == "Bearer sk-live-123"
    assert EnvSecrets(environ=env).get("deep-seek") == "from-env"
    with pytest.raises(ConfigError, match="secret 'nope' not found"):
        interpolate({"k": "${secret:nope}"}, SECRETS, env)
    with pytest.raises(ConfigError, match="'NOPE' is not set"):
        interpolate("${env:NOPE}", SECRETS, env)
    if not cfgmod.sys.platform.startswith("darwin"):
        assert cfgmod.KeychainSecrets().get("x") is None  # keychain only exists on macOS

    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE)
    async with Kernel() as app:
        store = {"deepseek": "sk-1"}
        ld = loader(app, cfg, secrets=DictSecrets(store))
        await ld.apply()
        ld.secrets = DictSecrets({"deepseek": "sk-2"})  # rotated secret -> replace, never printed
        res = await ld.apply()
        assert [c.id for c in res.plan.changes] == ["ds"] and "api_key" in str(res.plan)
        assert "sk-1" not in str(res) and "sk-2" not in str(res)
        assert app.get("llm").cfg.api_key.reveal() == "sk-2"


async def test_failed_fiber_is_retried_on_apply_and_strict(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, f"version: 1\nplugins:\n  - use: {P}.flaky\n")
    async with Kernel() as app:
        ld = loader(app, cfg)
        res = await ld.apply()
        assert not res.ok and not app.fiber.children  # rolled back: nothing half-applied
        cfg_plugins.FLAKY["fail"] = False
        assert (await ld.apply()).ok
        write(cfg, f"version: 1\nplugins:\n  - use: {P}.needs_missing\n")
        lenient = await ld.apply(dry_run=True)
        assert lenient.ok and "missing: nobody-provides-this" in str(lenient.report)
        strict = await ld.apply(strict=True)
        assert not strict.ok and "strict" in strict.report.error
        assert managed(app)[f"{P}.flaky"].state is State.ACTIVE  # old tree kept


async def test_use_resolution(monkeypatch):
    assert cfgmod.resolve_use(f"{P}:LLM") is cfg_plugins.LLM
    assert cfgmod.resolve_use(f"{P}.agent") is cfg_plugins.agent
    assert cfgmod.resolve_use("ventri_std.trace.jsonl").name == "trace.jsonl"
    with pytest.raises(ConfigError, match="neither"):
        cfgmod.resolve_use("tests.std")

    class EP:
        name = "llm"

        def load(self):
            return cfg_plugins.LLM

    monkeypatch.setattr(cfgmod, "entry_points", lambda group: [EP()])
    assert cfgmod.resolve_use("llm") is cfg_plugins.LLM


async def test_watcher_debounces_into_one_transaction(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, BASE)
    results = []
    async with Kernel() as app:
        ld = loader(app, cfg)
        await ld.apply()
        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: ld.watch(poll=0.01, debounce=0.1, on_result=results.append))
            for model in ("a", "b", "c"):  # a burst of saves
                write(cfg, BASE.replace("model: flash", f"model: {model}"))
                await anyio.sleep(0.03)
            with anyio.fail_after(3):
                while not results:
                    await anyio.sleep(0.01)
            assert len(commits(app)) == 2 and app.get("llm").cfg.model == "c"
            write(cfg, "version: [")  # broken save: reported, old config keeps serving
            with anyio.fail_after(3):
                while len(results) < 2:
                    await anyio.sleep(0.01)
            assert isinstance(results[1], ConfigError) and app.get("llm").cfg.model == "c"
            assert any(e.kind == "config.error" for e in app.trace_log)
            write(cfg, BASE.replace("model: flash", "model: d"))
            with anyio.fail_after(3):
                while len(results) < 3:
                    await anyio.sleep(0.01)
            assert results[2].ok and app.get("llm").cfg.model == "d"
            tg.cancel_scope.cancel()
        assert len(results) == 3


async def test_loader_plugin(tmp_path):
    cfg = tmp_path / "ventri.yml"
    write(cfg, f"version: 1\nplugins:\n  - use: {P}.tool\n    id: t\n")
    async with Kernel() as app:
        lf = await app.plugin(cfgmod.loader, {"path": str(cfg), "poll": 0.01, "debounce": 0.05})
        ld = app.get("config.loader")
        with anyio.fail_after(3):
            while "t" not in _ids(app):
                await anyio.sleep(0.01)
        write(cfg, f"version: 1\nplugins:\n  - use: {P}.tool\n    id: u\n")
        with anyio.fail_after(3):
            while "u" not in _ids(app):
                await anyio.sleep(0.01)
        assert "t" not in _ids(app) and ld.document.entries[0].id == "u"
        await lf.dispose()
        assert "u" in _ids(app)  # managed, not owned: unloading the loader leaves the tree


def _ids(app):
    return {f.meta.get("config_id") for f in app.fiber.children}


async def test_exclusive_plugin_is_replaced_stop_first(tmp_path):
    cfg = tmp_path / "ventri.yml"
    doc = f"version: 1\nplugins:\n  - use: {P}.server\n    id: srv\n    config: {{port: 80, v: 1}}\n"
    write(cfg, doc)
    async with Kernel() as app:
        ld = loader(app, cfg)
        await ld.apply()
        write(cfg, doc.replace("v: 1", "v: 2"))  # same port: blue-green would collide
        res = await ld.apply()
        assert res.ok and app.get("server")["v"] == 2 and list(cfg_plugins.PORTS) == [80]
        write(cfg, doc.replace("v: 1", "v: 3, fail: true"))
        res = await ld.apply()
        assert not res.ok and res.report.degraded  # old config restarted
        assert app.get("server")["v"] == 2 and list(cfg_plugins.PORTS) == [80]
