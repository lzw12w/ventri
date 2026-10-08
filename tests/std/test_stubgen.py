"""ventri stubgen: typed ctx.<service> (M1 exit criterion: pyright strict sees ctx.llm)."""
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ventri import Kernel, State, plugin
from ventri_std import stubgen

ROOT = Path(__file__).resolve().parents[2]

PLUGIN_SRC = '''
from ventri import plugin


class ModelProvider:
    def complete(self, prompt: str) -> str:
        return prompt.upper()


class Memory:
    pass


@plugin(provides={"llm": ModelProvider, "memory": Memory, "class": Memory, "get": Memory})
def deepseek(ctx, config) -> None:
    ctx.provide(ModelProvider, ModelProvider())
    ctx.provide(Memory, Memory())


def not_a_plugin() -> None:
    pass
'''

OTHER_SRC = '''
from ventri import plugin


@plugin(provides={"llm": str})
def other_llm(ctx, config) -> None:
    pass
'''

USER_SRC = '''
from typing import reveal_type

from ventri_stubs import Ctx


def good(ctx: Ctx) -> str:
    reveal_type(ctx.llm)
    reveal_type(ctx.memory)
    return ctx.memory.__class__.__name__


def bad(ctx: Ctx) -> int:
    return ctx.memory
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "myplug.py").write_text(PLUGIN_SRC)
    (tmp_path / "otherplug.py").write_text(OTHER_SRC)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("myplug", None)
    sys.modules.pop("otherplug", None)
    return tmp_path


def test_generate_stub(project):
    res = stubgen.generate(project, ["myplug", "otherplug"], use_entry_points=False)
    pyi = (project / "ventri_stubs.pyi").read_text()
    assert "class Ctx(Context):" in pyi
    assert "    llm: ModelProvider | str  # from deepseek, other_llm" in pyi
    assert "    memory: Memory  # from deepseek" in pyi
    assert "from myplug import Memory" in pyi and "from myplug import ModelProvider" in pyi
    assert "class:" not in pyi and "    get:" not in pyi
    assert any("'class'" in w for w in res.warnings) and any("shadows" in w for w in res.warnings)
    assert any("different types" in w for w in res.warnings)
    runtime = (project / "ventri_stubs.py").read_text()
    assert "from ventri import Context as Ctx" in runtime


def test_local_and_string_keys_become_any():
    class Local:
        pass

    p = plugin(lambda ctx, config: None, name="p", provides={"a": Local, "b": "string-key", "c": int})
    res = stubgen.collect([("p", p)])
    assert res.attrs == {"a": ["Any"], "b": ["Any"], "c": ["int"]}
    assert "from typing import Any" in res.pyi()
    assert "class Ctx(Context):\n" in stubgen.collect([]).pyi() and "pass" in stubgen.collect([]).pyi()


@pytest.mark.anyio
async def test_named_provides_enable_ctx_attribute_at_runtime(project):
    import myplug

    seen = {}

    def consumer(ctx, config, llm: myplug.ModelProvider) -> None:
        seen["attr"] = ctx.llm.complete("hi")

    async with Kernel() as app:
        await app.plugin(myplug.deepseek)
        f = await app.plugin(consumer)
        assert f.state is State.ACTIVE and seen["attr"] == "HI"
        assert app.memory.__class__ is myplug.Memory


@pytest.mark.skipif(shutil.which("pyright") is None and not (ROOT / ".venv/bin/pyright").exists(),
                    reason="pyright not installed")
def test_pyright_strict_types_ctx_llm(project):
    stubgen.generate(project, ["myplug"], use_entry_points=False)
    (project / "user.py").write_text(USER_SRC)
    (project / "pyrightconfig.json").write_text(json.dumps({
        "typeCheckingMode": "strict", "pythonVersion": "3.12", "include": ["user.py"],
        "extraPaths": [str(ROOT / "packages/ventri/src"), str(project)]}))
    exe = shutil.which("pyright") or str(ROOT / ".venv/bin/pyright")
    out = subprocess.run([exe, "--outputjson", "-p", str(project)], cwd=project,
                         capture_output=True, text=True, timeout=300, check=False)
    diags = json.loads(out.stdout)["generalDiagnostics"]
    infos = [d["message"] for d in diags if d["severity"] == "information"]
    errors = [d for d in diags if d["severity"] == "error"]
    assert 'Type of "ctx.llm" is "ModelProvider"' in infos
    assert 'Type of "ctx.memory" is "Memory"' in infos
    assert len(errors) == 1 and errors[0]["range"]["start"]["line"] == textwrap.dedent(
        USER_SRC).splitlines().index("    return ctx.memory"), errors
