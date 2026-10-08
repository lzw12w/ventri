"""ToolRegistry (strict schemas, lifecycle, wire names) and the built-in tools."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import anyio
import httpx
import pytest
from pydantic import BaseModel, Field

import ventri
from ventri import Kernel
from ventri_agent.permission import Policy, ToolRequest
from ventri_agent.tools import core, fs, notes, shell, web
from ventri_agent.tools.registry import (
    Risk,
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
    call_handler,
    tool_schema,
)
from ventri_agent.tools.registry import plugin as registry_plugin

pytestmark = pytest.mark.anyio


class Inner(BaseModel):
    tag: str = Field(min_length=1, max_length=10)


class Args(BaseModel):
    path: str = Field(description="file path")
    limit: int = 10
    mode: Literal["a", "b"] = "a"
    tags: list[Inner] = Field(default_factory=list, min_length=0, max_length=5)
    note: str | None = None


def test_strict_schema_is_all_required_and_closed():
    s = tool_schema(Args)
    assert s["type"] == "object" and s["additionalProperties"] is False
    assert sorted(s["required"]) == sorted(s["properties"])
    assert s["properties"]["path"] == {"type": "string", "description": "file path"}
    assert {"type": "null"} in s["properties"]["limit"]["anyOf"]  # optional -> nullable
    assert {"type": "null"} in s["properties"]["note"]["anyOf"]
    inner = s["properties"]["tags"]["anyOf"][0]["items"]
    assert inner["additionalProperties"] is False and inner["required"] == ["tag"]
    text = json.dumps(s)
    for bad in ("$ref", "$defs", "minLength", "maxLength", "minItems", "maxItems", '"default"', '"title"'):
        assert bad not in text
    loose = tool_schema(Args, strict=False)
    assert loose["required"] == ["path"] and "additionalProperties" not in loose


def test_null_for_defaulted_field_means_default():
    t = Tool("x.y", "", lambda a, tc: a, Args)
    a = t.parse({"path": "p", "limit": None, "mode": None, "tags": None, "note": None})
    assert (a.limit, a.mode, a.tags, a.note) == (10, "a", [], None)
    assert t.wire_name == "x__y" and t.spec()["function"]["name"] == "x__y"


async def test_registry_lifecycle_follows_the_plugin_fiber():
    reg_events = []

    @ventri.plugin(name="tp")
    def tp(ctx, config, registry: ToolRegistry):
        registry.register(ctx, Tool("a.b", "", lambda a, tc: "x", None))

    async with Kernel() as app:
        await app.plugin(registry_plugin)
        reg = app.get(ToolRegistry)
        reg.watch(lambda: reg_events.append(reg.version))
        f = await app.plugin(tp)
        assert "a.b" in reg and reg.get("a__b").name == "a.b" and reg.get("a.b").source == f.label
        with pytest.raises(ValueError):
            reg.register(None, Tool("a.b", "", lambda a, tc: "x", None))
        with pytest.raises(ValueError):
            reg.register(None, Tool("a__b", "", lambda a, tc: "x", None))  # wire collision
        await f.dispose()
        assert "a.b" not in reg and len(reg_events) == 2
        reg.register(None, Tool("fs.read", "", lambda a, tc: "", None))
        reg.register(None, Tool("fs.write", "", lambda a, tc: "", None))
        reg.register(None, Tool("web.fetch", "", lambda a, tc: "", None))
        assert [t.name for t in reg.select(["fs"])] == ["fs.read", "fs.write"]
        assert [t.name for t in reg.select(["*.fetch", "fs.r*"])] == ["fs.read", "web.fetch"]


def tc(tmp_path, ctx=None):
    return ToolContext("s1", ctx, tmp_path)


async def run(tool, args, tmp_path, ctx=None):
    return await call_handler(tool, tool.parse(args), tc(tmp_path, ctx))


# ---------------------------------------------------------------------- fs
async def test_fs_tools_are_confined_to_roots(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.md").write_text("hello world\nsecond line\n")
    (tmp_path / "secret.txt").write_text("top secret")
    os.symlink(tmp_path / "secret.txt", root / "link.txt")
    tools = {t.name: t for t in fs.make_tools(fs.Roots([str(root)]))}
    assert "hello world" in await run(tools["fs.read"], {"path": "a.md"}, tmp_path)
    for bad in ("../secret.txt", str(tmp_path / "secret.txt"), "link.txt"):
        with pytest.raises(ToolError, match="outside the allowed roots"):
            await run(tools["fs.read"], {"path": bad}, tmp_path)
    with pytest.raises(ToolError):
        await run(tools["fs.write"], {"path": "../x.md", "content": "x"}, tmp_path)
    assert "a.md" in await run(tools["fs.list"], {"path": "."}, tmp_path)
    assert "a.md:2: second line" in await run(tools["fs.search"], {"query": "SECOND"}, tmp_path)
    await run(tools["fs.write"], {"path": "sub/n.md", "content": "one"}, tmp_path)
    with pytest.raises(ToolError, match="exists"):
        await run(tools["fs.write"], {"path": "sub/n.md", "content": "two"}, tmp_path)
    await run(tools["fs.write"], {"path": "sub/n.md", "content": " two", "mode": "append"}, tmp_path)
    await run(tools["fs.edit"], {"path": "sub/n.md", "old": "one", "new": "1"}, tmp_path)
    assert (root / "sub/n.md").read_text() == "1 two"
    with pytest.raises(ToolError, match="occurrence"):
        await run(tools["fs.edit"], {"path": "sub/n.md", "old": "zzz", "new": "1"}, tmp_path)
    w = tools["fs.write"]
    assert w.risk == Risk.WRITE_LOCAL and w.default_action == "ask" and tools["fs.read"].untrusted
    assert w.describe_call(w.parse({"path": "x.md", "content": ""})) == {"path": str(root / "x.md")}


def test_fs_write_policy_rule_on_real_path(tmp_path):
    from ventri_agent.permission import Rule
    root = tmp_path / "notes"
    root.mkdir()
    w = next(t for t in fs.make_tools(fs.Roots([str(root)])) if t.name == "fs.write")
    p = Policy([Rule(tool="fs.write", when={"path": str(root) + "/*"}, action="allow")])

    def decide(path):
        a = w.parse({"path": path, "content": ""})
        return p.decide(ToolRequest("c", w, a, "s", "default", "user", w.describe_call(a)))[0]
    assert decide("ok.md") == "allow"
    assert decide("../escape.md") == "ask"  # rule does not match; the tool would refuse anyway


# ------------------------------------------------------------------- shell
async def test_shell_run_env_cwd_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MY_API_KEY", "should-not-leak")
    monkeypatch.setenv("HARMLESS_VAR", "visible")
    t = shell.make_tool(shell.ShellConfig(cwd=str(tmp_path / "wd"), timeout=5))
    (tmp_path / "wd").mkdir()
    out = await run(t, {"command": "echo $HARMLESS_VAR; echo key=$MY_API_KEY; pwd"}, tmp_path)
    assert out.startswith("exit code 0") and "visible" in out and "should-not-leak" not in out
    assert str((tmp_path / "wd").resolve()) in out
    with pytest.raises(ToolError, match="escapes"):
        await run(t, {"command": "true", "cwd": ".."}, tmp_path)
    with anyio.fail_after(5):
        with pytest.raises(ToolError, match="timed out"):
            await run(t, {"command": "sleep 30 & sleep 30", "timeout": 0.2}, tmp_path)
    assert t.risk == Risk.EXTERNAL and not t.grantable and t.default_action == "ask"
    assert "exit code 3" in await run(t, {"command": "exit 3"}, tmp_path)


# --------------------------------------------------------------------- web
PAGE = """<!doctype html><html><head><title>Example Page</title><script>evil()</script>
<style>.x{}</style></head><body><nav>menu</nav><h1>Heading</h1><p>Some <b>text</b> with a
<a href="/next">link</a>.</p><ul><li>one</li><li>two</li></ul><footer>foot</footer></body></html>"""


async def test_web_fetch_extracts_markdown(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/feed":
            return httpx.Response(200, text="<rss>items</rss>", headers={"content-type": "application/rss+xml"})
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html; charset=utf-8"})
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = web.make_tool(web.WebConfig(allow_domains=["*.python.org"]), http)
    out = await run(t, {"url": "https://example.com/page"}, tmp_path)
    assert "Title: Example Page" in out and "# Heading" in out and "evil()" not in out
    assert "[link](https://example.com/next)" in out and "- one" in out
    assert "<rss>items</rss>" in await run(t, {"url": "https://example.com/feed"}, tmp_path)
    with pytest.raises(ToolError, match="http"):
        await run(t, {"url": "file:///etc/passwd"}, tmp_path)
    assert t.untrusted and t.default_action == "ask"
    pol = Policy()

    def decide(url):
        a = t.parse({"url": url})
        return pol.decide(ToolRequest("c", t, a, "s", "d", "user", t.describe_call(a)))[0]
    assert decide("https://docs.python.org/3/") == "allow" and decide("https://evil.example/") == "ask"


# --------------------------------------------------------------- notes/core
async def test_notes_tools(tmp_path):
    vault = tmp_path / "vault"
    (vault / "weekly").mkdir(parents=True)
    (vault / "weekly" / "w41.md").write_text("# W41\n读书笔记：DeepSeek 缓存\n")
    tools = {t.name: t for t in notes.make_tools(fs.Roots([str(vault)]))}
    assert "weekly/w41" in await run(tools["notes.list"], {}, tmp_path)
    assert "DeepSeek" in await run(tools["notes.read"], {"name": "weekly/w41"}, tmp_path)
    assert "weekly/w41" in await run(tools["notes.search"], {"query": "缓存"}, tmp_path)
    await run(tools["notes.write"], {"name": "inbox", "content": "todo"}, tmp_path)
    assert (vault / "inbox.md").read_text() == "todo"
    with pytest.raises(ToolError):
        await run(tools["notes.read"], {"name": "../../etc/passwd"}, tmp_path)
    assert tools["notes.write"].risk == Risk.WRITE_LOCAL


async def test_core_tools(tmp_path):
    tools = {t.name: t for t in core.CORE_TOOLS}
    now = await run(tools["time.now"], {"tz": "Asia/Shanghai"}, tmp_path)
    assert now["tz"] == "Asia/Shanghai" and now["deepseek_pricing"] in ("peak", "off-peak")
    with pytest.raises(ToolError):
        await run(tools["time.now"], {"tz": "Mars/Base"}, tmp_path)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "call_1.txt").write_text("abcdefghij")
    assert (await run(tools["artifact.read"], {"handle": "call_1", "offset": 2, "limit": 3}, tmp_path)).startswith("cde")
    for bad in ("../../x", "a/b", "..", ""):
        with pytest.raises(ToolError):
            await run(tools["artifact.read"], {"handle": bad}, tmp_path)


def test_paths_honour_ventri_home(tmp_path, monkeypatch):
    from ventri_agent.paths import expand, home
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path))
    assert home() == tmp_path
    assert expand("~/.ventri/memory.db") == tmp_path / "memory.db"
    assert expand("sessions") == tmp_path / "sessions"
    assert expand("/abs/x") == Path("/abs/x")
