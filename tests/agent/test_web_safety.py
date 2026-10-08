"""web.fetch hardening: SSRF blocking, DNS rebinding, redirects, credentials,
non-text / truncation reporting, artifact hand-off."""
from __future__ import annotations

import ipaddress
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from ventri_agent.permission import AuditLog, Grants, Policy, Rule
from ventri_agent.tokens import estimate_tokens
from ventri_agent.tools import _url_safety as us
from ventri_agent.tools import core, web
from ventri_agent.tools.registry import ToolContext, ToolError, call_handler

pytestmark = pytest.mark.anyio

PUBLIC = "93.184.216.34"


def resolver_for(table: dict[str, list[str]]):
    calls: list[str] = []

    async def resolve(host: str, port: int) -> list[str]:
        calls.append(host)
        if host not in table:
            raise OSError("NXDOMAIN")
        return table[host]
    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


class FakeCtx:
    def __init__(self, **services: Any) -> None:
        self.s = {type(v): v for v in services.values()}

    def get(self, key: Any, default: Any = None) -> Any:
        return self.s.get(key, default)


async def run(tool, args, tmp_path: Path, ctx=None, call_id="call_1"):
    c: Any = ctx
    return await call_handler(tool, tool.parse(args), ToolContext("s1", c, tmp_path, call_id=call_id))


# ------------------------------------------------------------------ address classes
@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "100.64.0.1", "224.0.0.1", "240.0.0.1",
    "0.0.0.0", "::1", "fe80::1", "fc00::1", "ff02::1", "::", "::ffff:127.0.0.1", "::ffff:0:127.0.0.1",
    "::ffff:10.0.0.1"])
def test_private_classes_blocked(ip):
    assert us.ip_block_reason(ipaddress.ip_address(ip), False) == "private/internal address"
    assert us.ip_block_reason(ipaddress.ip_address(ip), True) is None   # opt-in LAN / loopback


@pytest.mark.parametrize("ip", [
    "169.254.169.254", "169.254.170.2", "169.254.1.1", "100.100.100.200", "fd00:ec2::254",
    "::ffff:169.254.169.254", "::ffff:0:169.254.169.254", "::ffff:100.100.100.200"])
def test_metadata_always_blocked(ip):
    for allow in (False, True):
        assert "metadata" in (us.ip_block_reason(ipaddress.ip_address(ip), allow) or "")


def test_public_ok():
    assert us.ip_block_reason(ipaddress.ip_address(PUBLIC), False) is None
    assert us.ip_block_reason(ipaddress.ip_address("2606:4700::1111"), False) is None


@pytest.mark.parametrize(("url", "needle"), [
    ("ftp://example.com/x", "http(s)"),
    ("file:///etc/passwd", "http(s)"),
    ("https://user:pw@example.com/", "credentials"),
    ("https://user@example.com/", "credentials"),
    ("https://example.com/cb?access_token=abc123", "access_token"),
    ("https://example.com/?Signature=xyz&x=1", "Signature"),
    ("https://example.com/k/sk-abcdefghijklmnop1234", "token"),
    ("https://example.com/?q=ghp_ABCDEFGHIJKLMNOP", "token"),
    ("http://metadata.google.internal/computeMetadata/v1/", "metadata"),
    ("http://169.254.169.254/latest/meta-data/", "metadata"),
    ("http://[fd00:ec2::254]/", "metadata"),
    ("http://[::ffff:127.0.0.1]:8080/", "private"),
    ("http://127.0.0.1/", "private"),
    ("http://example.com:99999/", "malformed"),
])
def test_static_block(url, needle):
    why = us.static_block_reason(us.normalize_url(url), allow_private=False)
    assert why and needle in why


def test_static_allows_ordinary_urls():
    for url in ("https://docs.python.org/3/?q=token+bucket", "https://example.com/search?key=value&code=1",
                "https://例子.测试/路径?q=中文"):
        assert us.static_block_reason(us.normalize_url(url), allow_private=False) is None
    assert us.normalize_url("https://例子.测试/a b").startswith("https://xn--")


async def test_checked_ips_validates_every_answer(monkeypatch):
    r = resolver_for({"mixed.test": [PUBLIC, "10.0.0.5"], "ok.test": [PUBLIC, "2606:4700::1111"],
                      "mapped.test": ["::ffff:169.254.169.254"]})
    with pytest.raises(us.UrlBlocked, match="private"):
        await us.checked_ips("mixed.test", 80, r, allow_private=False)
    with pytest.raises(us.UrlBlocked, match="metadata"):
        await us.checked_ips("mapped.test", 80, r, allow_private=True)
    assert await us.checked_ips("ok.test", 80, r, allow_private=False) == [PUBLIC, "2606:4700::1111"]
    with pytest.raises(us.DnsFailed):
        await us.checked_ips("nx.test", 80, r, allow_private=False)


# ------------------------------------------------------------------ tool, mocked transport
def mock_tool(routes: dict[str, Any], resolver: Any = None, seen: list[str] | None = None, **cfg: Any):
    seen = seen if seen is not None else []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        resp: Any = routes.get(f"{req.url.host}{req.url.path}")
        if resp is None:
            return httpx.Response(404, text="nope")
        return resp
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = resolver or resolver_for({h: [PUBLIC] for h in ("a.test", "b.test", "docs.python.org", "big.test")}
                                 | {"internal.test": ["10.0.0.7"], "meta.test": ["169.254.169.254"]})
    return web.make_tool(web.WebConfig(**cfg), http, resolver=r), seen


def redirect(to: str, code: int = 302) -> httpx.Response:
    return httpx.Response(code, headers={"location": to})


async def test_private_resolution_blocked_before_request(tmp_path):
    t, seen = mock_tool({"internal.test/": httpx.Response(200, text="secret")})
    with pytest.raises(ToolError, match="private"):
        await run(t, {"url": "http://internal.test/"}, tmp_path)
    with pytest.raises(ToolError, match="DNS resolution failed"):
        await run(t, {"url": "http://nx.test/"}, tmp_path)
    assert seen == []


async def test_dns_failure_allowed_behind_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    t, _ = mock_tool({"nx.test/": httpx.Response(200, text="via proxy")})
    assert "via proxy" in await run(t, {"url": "https://nx.test/"}, tmp_path)


async def test_redirect_to_metadata_or_private_not_followed(tmp_path):
    t, seen = mock_tool({"a.test/m": redirect("http://169.254.169.254/latest/meta-data/"),
                         "a.test/p": redirect("http://internal.test/admin"),
                         "a.test/c": redirect("https://u:p@b.test/"),
                         "a.test/f": redirect("file:///etc/passwd")}, allow_domains=["*"])
    for path, needle in (("m", "metadata"), ("p", "private"), ("c", "credentials"), ("f", "http")):
        with pytest.raises(ToolError, match=needle) as ei:
            await run(t, {"url": f"https://a.test/{path}"}, tmp_path)
        assert "not followed" in str(ei.value) and ei.value.untrusted
    assert all("a.test" in u for u in seen)


async def test_redirect_cap(tmp_path):
    routes = {f"a.test/{i}": redirect(f"/{i + 1}") for i in range(10)}
    t, seen = mock_tool(routes)
    with pytest.raises(ToolError, match="too many redirects"):
        await run(t, {"url": "https://a.test/0"}, tmp_path)
    assert len(seen) == 6     # the original request + 5 followed hops
    routes["a.test/3"] = httpx.Response(200, text="landed")
    t, _ = mock_tool(routes)
    out = await run(t, {"url": "https://a.test/0"}, tmp_path)
    assert "landed" in out and "Redirects: 302 https://a.test/0 -> https://a.test/1" in out


async def test_redirect_to_unapproved_domain_fails_closed(tmp_path):
    routes = {"a.test/go": redirect("https://b.test/landing?ref=1"), "b.test/landing": httpx.Response(200, text="B"),
              "a.test/docs": redirect("https://docs.python.org/3/"),
              "docs.python.org/3/": httpx.Response(200, text="python docs")}
    t, seen = mock_tool(routes, allow_domains=["*.python.org", "docs.python.org"])
    with pytest.raises(ToolError, match="not pre-approved") as ei:
        await run(t, {"url": "https://a.test/go"}, tmp_path)
    assert ei.value.untrusted == "https://b.test/landing?ref=1"
    assert not any("b.test" in u for u in seen)
    assert "python docs" in await run(t, {"url": "https://a.test/docs"}, tmp_path)   # allow_domains hop


async def test_redirect_uses_policy_grants_and_audit(tmp_path):
    routes = {"a.test/go": redirect("https://b.test/x"), "b.test/x": httpx.Response(200, text="B page"),
              "a.test/bad": redirect("https://docs.python.org/"),
              "docs.python.org/": httpx.Response(200, text="never")}
    t, seen = mock_tool(routes)
    audit, grants = AuditLog(None), Grants()
    deny_docs = Rule(tool="web.fetch", action="deny", when={"domain": "docs.python.org"})
    ctx = FakeCtx(p=Policy([deny_docs]), g=grants, a=audit)
    with pytest.raises(ToolError, match="not pre-approved"):
        await run(t, {"url": "https://a.test/go"}, tmp_path, ctx)
    assert audit.records[-1]["decided_by"] == "redirect:needs-approval"
    assert audit.records[-1]["redirect_from"] == "https://a.test/go"
    grants.grant("web.fetch")                     # "allow web.fetch for this session"
    assert "B page" in await run(t, {"url": "https://a.test/go"}, tmp_path, ctx)
    assert audit.records[-1]["decided_by"] == "grant:session" and audit.records[-1]["action"] == "allow"
    with pytest.raises(ToolError, match="not pre-approved"):   # a deny rule beats the grant
        await run(t, {"url": "https://a.test/bad"}, tmp_path, ctx)
    assert audit.records[-1]["action"] == "deny"
    allow_b = Rule(tool="web.fetch", action="allow", when={"domain": "b.test"})
    ctx2 = FakeCtx(p=Policy([allow_b]), g=Grants(), a=audit)
    assert "B page" in await run(t, {"url": "https://a.test/go"}, tmp_path, ctx2)
    assert not any("docs.python.org" in u for u in seen)


async def test_non_text_and_binary_refused(tmp_path):
    t, _ = mock_tool({
        "a.test/doc.pdf": httpx.Response(200, content=b"%PDF-1.7", headers={"content-type": "application/pdf",
                                                                            "content-length": "8"}),
        "a.test/img": httpx.Response(200, content=b"\x89PNG\r\n", headers={"content-type": "image/png"}),
        "a.test/sneaky": httpx.Response(200, content=b"abc\x00\x01\x02", headers={"content-type": "text/plain"}),
        "a.test/api": httpx.Response(200, json={"ok": True})})
    with pytest.raises(ToolError, match=r"non-text content \(application/pdf, 8 bytes\)"):
        await run(t, {"url": "https://a.test/doc.pdf"}, tmp_path)
    with pytest.raises(ToolError, match="image/png"):
        await run(t, {"url": "https://a.test/img"}, tmp_path)
    with pytest.raises(ToolError, match="binary"):
        await run(t, {"url": "https://a.test/sneaky"}, tmp_path)
    assert '{"ok":true}' in await run(t, {"url": "https://a.test/api"}, tmp_path)


async def test_byte_truncation_is_reported(tmp_path):
    t, _ = mock_tool({"a.test/": httpx.Response(200, text="x" * 5000, headers={"content-type": "text/plain"})},
                     max_bytes=1000)
    out = await run(t, {"url": "https://a.test/"}, tmp_path)
    assert "cut at the 1,000-byte download limit" in out and "x" * 1000 in out and "x" * 1001 not in out


async def test_large_page_goes_to_artifact(tmp_path):
    lines = "".join(f"line {i:05d} " + "lorem ipsum " * 8 + "\n" for i in range(3000))
    zh = "".join(f"第{i}段：" + "中文网页内容很长。" * 10 + "\n" for i in range(2000))
    t, _ = mock_tool({"big.test/en": httpx.Response(200, text=lines, headers={"content-type": "text/plain"}),
                      "big.test/zh": httpx.Response(200, text=zh, headers={"content-type": "text/plain"})})
    for path, text in (("en", lines), ("zh", zh)):
        out = await run(t, {"url": f"https://big.test/{path}"}, tmp_path, call_id=f"c_{path}")
        assert estimate_tokens(out) < 8000           # never re-spilled by the loop
        handle = f"c_{path}-web"
        assert f"artifact {handle!r}" in out and text.splitlines()[-1] in out   # head + tail
        assert (tmp_path / "artifacts" / f"{handle}.txt").read_text(encoding="utf-8") == text
        read = {x.name: x for x in core.CORE_TOOLS}["artifact.read"]
        page = await call_handler(read, read.parse({"handle": handle, "offset": 100}),
                                  ToolContext("s1", None, tmp_path))  # type: ignore[arg-type]
        assert text[100:200] in page
    out = await run(t, {"url": "https://big.test/en", "max_chars": 500}, tmp_path, call_id="c_m")
    assert "artifact 'c_m-web'" in out and estimate_tokens(out) < 400


# ------------------------------------------------------------------ real sockets: connect-time guard
class _Server:
    def __init__(self) -> None:
        self.hits: list[str] = []
        hits = self.hits
        port: list[int] = []

        class H(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                host = (self.headers.get("Host") or "").split(":")[0]
                hits.append(f"{host}{self.path}")
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", f"http://other.test:{port[0]}/page")
                    self.end_headers()
                    return
                body = f"hello from {host}".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_port
        port.append(self.port)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.close()


async def test_guarded_client_dials_vetted_ip(server, tmp_path):
    r = resolver_for({"local.test": ["127.0.0.1"], "other.test": ["127.0.0.1"]})
    async with us.guarded_client(r, allow_private=True) as http:
        t = web.make_tool(web.WebConfig(allow_private_urls=True), http, resolver=r)
        out = await run(t, {"url": f"http://local.test:{server.port}/x"}, tmp_path)
        assert "hello from local.test" in out          # Host header kept, IP dialed
        with pytest.raises(ToolError, match="not pre-approved"):
            await run(t, {"url": f"http://local.test:{server.port}/redirect"}, tmp_path)
    assert server.hits == ["local.test/x", "local.test/redirect"]   # other.test never contacted
    async with us.guarded_client(r) as http:      # default policy: loopback refused up front
        t = web.make_tool(web.WebConfig(), http, resolver=r)
        with pytest.raises(ToolError, match="private"):
            await run(t, {"url": f"http://127.0.0.1:{server.port}/"}, tmp_path)
        with pytest.raises(ToolError, match="private"):
            await run(t, {"url": f"http://local.test:{server.port}/"}, tmp_path)
    assert len(server.hits) == 2


async def test_dns_rebinding_blocked_at_connect(server, tmp_path):
    answers = iter([[PUBLIC], ["127.0.0.1"], ["127.0.0.1"]])

    async def rebinding(host: str, port: int) -> list[str]:
        return next(answers)          # public for the pre-flight check, loopback at connect
    async with us.guarded_client(rebinding) as http:
        t = web.make_tool(web.WebConfig(), http, resolver=rebinding)
        with pytest.raises(ToolError, match="blocked at connect"):
            await run(t, {"url": f"http://rebind.test:{server.port}/"}, tmp_path)
    assert server.hits == []


async def test_unix_socket_refused():
    backend = us.GuardedBackend(us.system_resolver, allow_private=False)
    import httpcore
    with pytest.raises(httpcore.ConnectError, match="unix"):
        await backend.connect_unix_socket("/var/run/docker.sock")
