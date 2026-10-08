"""ventri CLI: apply --dry-run / tree / doctor / stubgen / run."""
import argparse
import json
import textwrap

import anyio
import pytest

from ventri_std import cli

P = "tests.std.cfg_plugins"
GOOD = f"""
version: 1
plugins:
  - use: {P}.LLM
    id: ds
    config: {{ api_key: "${{secret:deepseek}}" }}
  - use: {P}.agent
"""


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_SECRET_DEEPSEEK", "sk-cli-secret")
    p = tmp_path / "ventri.yml"
    p.write_text(textwrap.dedent(GOOD))
    return p


def test_apply_requires_dry_run(cfg, capsys):
    assert cli.main(["apply", str(cfg)]) == 2
    assert "no daemon control channel" in capsys.readouterr().err


def test_apply_dry_run_text_and_json(cfg, capsys):
    assert cli.main(["apply", "--dry-run", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "+ ds" in out and "dry run" in out and "[OK]" in out and "sk-cli-secret" not in out
    assert cli.main(["apply", "--dry-run", "--json", str(cfg)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] and data["report"]["outcome"] == "dry_run" and len(data["changes"]) == 2
    assert cli.main(["apply", "--validate-only", str(cfg)]) == 0
    assert "configuration OK: 2 entries" in capsys.readouterr().out


def test_apply_failures_and_config_errors(cfg, capsys):
    cfg.write_text(textwrap.dedent(GOOD).replace("api_key", "fail: true, api_key"))
    assert cli.main(["apply", "--dry-run", str(cfg)]) == 1
    assert "bad llm config" in capsys.readouterr().out
    cfg.write_text("version: 7\n")
    assert cli.main(["apply", "--dry-run", str(cfg)]) == 2
    assert "unsupported config version" in capsys.readouterr().err
    assert cli.main(["apply", "--dry-run", str(cfg.parent / "nope.yml")]) == 2


def test_tree(cfg, capsys):
    assert cli.main(["tree", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "LLM" in out and "agent" in out


def test_doctor(cfg, capsys, monkeypatch):
    assert cli.main(["doctor", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "configuration valid (2 entries" in out and "no problems found" in out
    cfg.write_text(f"version: 1\nplugins:\n  - use: {P}.needs_missing\n")
    assert cli.main(["doctor", str(cfg)]) == 1
    assert "would stay pending: missing: nobody-provides-this" in capsys.readouterr().out
    monkeypatch.setenv("VENTRI_HOME", str(cfg.parent / "empty"))
    assert cli.main(["doctor"]) == 1
    assert "FAIL  config file" in capsys.readouterr().out


def test_stubgen(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["stubgen", "-m", P, "--no-entry-points", "--out", str(tmp_path)]) == 0
    assert "1 attributes: llm" in capsys.readouterr().out
    assert (tmp_path / "ventri_stubs.pyi").exists() and (tmp_path / "ventri_stubs.py").exists()


@pytest.mark.anyio
async def test_run_hot_applies_edits(cfg, capsys, monkeypatch):
    real_watch = cli.Loader.watch
    monkeypatch.setattr(cli.Loader, "watch",
                        lambda self, on_result=None: real_watch(self, poll=0.01, debounce=0.05,
                                                                on_result=on_result))
    args = argparse.Namespace(config=cfg, profiles=None, strict=False, no_watch=False)
    async with anyio.create_task_group() as tg:
        tg.start_soon(cli.cmd_run, args)
        await anyio.sleep(0.1)
        cfg.write_text(textwrap.dedent(GOOD).replace("api_key", "model: pro, api_key"))
        with anyio.fail_after(3):
            while "config changed: model" not in capsys.readouterr().out:
                await anyio.sleep(0.02)
        tg.cancel_scope.cancel()
