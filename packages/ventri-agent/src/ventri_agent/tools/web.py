"""``web.fetch`` -- fetch a URL the user gave (or a feed) and return readable text.

http(s) only; size- and time-limited; HTML is reduced to Markdown-ish text
(scripts, styles, navigation dropped). Risk ``read`` but **asked** by default
unless the domain matches ``allow_domains`` -- a fetched URL can carry data out
(prompt-injection exfiltration), so it is not treated as a free read. Output is
fenced as untrusted. There is no web *search* (deferred, DESIGN D10).

Safety (see ``_url_safety``): private / loopback / link-local / CGNAT /
multicast / reserved / cloud-metadata targets are refused (``allow_private_urls``
opts into LAN and loopback; metadata stays blocked), every DNS answer is
validated before the request and again at TCP connect (DNS rebinding), URLs
with embedded credentials, credential-named query parameters or token-looking
strings are refused. Redirects are followed by hand, at most ``max_redirects``
hops, and every hop is re-checked: scheme, address, and the permission policy
for its domain. A hop to the same host as an already-approved URL, a domain the
policy allows, or one covered by a session grant is followed; anything that
would need a fresh approval stops with an error naming the target URL, so the
model has to call ``web.fetch`` again for it and the normal approval runs
(approvals stay in the loop's pre-execution phase, outside tool timeouts).
Non-text responses are refused with their content type; a body cut at
``max_bytes`` is labelled; long text is kept whole in an artifact and returned
as head + tail."""
from __future__ import annotations

import fnmatch
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel, Field

import ventri

from ..tokens import estimate_tokens
from ._url_safety import (
    DnsFailed,
    Resolver,
    UrlBlocked,
    checked_ips,
    guarded_client,
    normalize_host,
    normalize_url,
    proxy_configured,
    static_block_reason,
    system_resolver,
)
from .output import PREVIEW_TOKENS, preview, write_artifact
from .registry import Tool, ToolContext, ToolError, ToolRegistry

SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe", "template"}
BLOCK = {"p", "div", "section", "article", "main", "br", "tr", "table", "ul", "ol", "pre", "blockquote",
         "h1", "h2", "h3", "h4", "h5", "h6", "li", "hr"}


class _Extractor(HTMLParser):
    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.out: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False
        self._href: str | None = None
        self._pre = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP:
            self.skip += 1
            return
        if tag == "title":
            self._in_title = True
        if self.skip:
            return
        if tag in BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("- ")
        elif tag == "pre":
            self._pre += 1
            self.out.append("```\n")
        elif tag == "a":
            href = dict(attrs).get("href")
            if href and not href.startswith(("javascript:", "#")):
                self._href = urljoin(self.base, href)
                self.out.append("[")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if tag == "title":
            self._in_title = False
        if self.skip:
            return
        if tag == "a" and self._href:
            self.out.append(f"]({self._href})")
            self._href = None
        elif tag == "pre":
            self._pre = max(0, self._pre - 1)
            self.out.append("\n```\n")
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self.skip:
            return
        self.out.append(data if self._pre else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        t = "".join(self.out)
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


def html_to_markdown(html: str, base: str = "") -> tuple[str, str]:
    """``(title, text)``."""
    p = _Extractor(base)
    p.feed(html)
    p.close()
    return p.title.strip(), p.text()


class FetchArgs(BaseModel):
    url: str = Field(description="http(s) URL")
    max_chars: int | None = Field(None, description=(
        "Optional cap on characters returned inline; long pages are always kept whole in an artifact"))


class WebConfig(BaseModel):
    allow_domains: list[str] = Field(default_factory=list)  # e.g. ["*.python.org", "docs.deepseek.com"]
    timeout: float = 30.0
    max_bytes: int = 3_000_000
    max_redirects: int = 5
    allow_private_urls: bool = False   # loopback / LAN targets; cloud metadata is blocked regardless
    user_agent: str = "VentriAgent/0.2 (+https://github.com/lzw12w/ventri)"


_TEXT_TYPES = ("text/", "html", "xml", "json", "javascript", "ecmascript", "rss", "atom", "csv", "yaml",
               "markdown", "x-www-form-urlencoded")
_REDIRECTS = (301, 302, 303, 307, 308)


def _is_text(ctype: str) -> bool:
    return not ctype or any(t in ctype for t in _TEXT_TYPES)


def _host(url: str) -> str:
    return normalize_host(urlsplit(url).hostname)


def _subject(url: str) -> dict[str, str]:
    u = normalize_url(url)
    return {"domain": _host(u), "url": u}


def make_tool(cfg: WebConfig, http: httpx.AsyncClient, resolver: Resolver = system_resolver) -> Tool:
    ref: list[Tool] = []

    async def preflight(url: str) -> None:
        if why := static_block_reason(url, allow_private=cfg.allow_private_urls):
            raise ToolError(f"blocked: {why}")
        p = urlsplit(url)
        try:
            await checked_ips(p.hostname or "", p.port or (443 if p.scheme == "https" else 80), resolver,
                              allow_private=cfg.allow_private_urls)
        except DnsFailed as e:
            if not proxy_configured():   # behind a proxy, the proxy resolves (Hermes behaviour)
                raise ToolError(f"blocked: {e}") from None
        except UrlBlocked as e:
            raise ToolError(f"blocked: {e}") from None

    def follow_ok(tc: ToolContext, prev: str, url: str, approved: set[str]) -> tuple[bool, str]:
        """May the redirect ``prev -> url`` be followed without asking anyone?"""
        from ..permission import AuditLog, Grants, Policy, ToolRequest  # lazy: permission imports tools
        host = _host(url)
        policy = tc.get(Policy) if tc.ctx is not None else None
        if host in approved:
            ok, why, by = True, "same host as an approved URL", "redirect:same-host"
        elif policy is None:   # no permission engine (embedding / tests): allow_domains only
            ok = any(fnmatch.fnmatchcase(host, pat) for pat in cfg.allow_domains)
            why, by = ("allow_domains" if ok else "not in allow_domains"), "redirect:config"
        else:
            req = ToolRequest(tc.call_id, ref[0], FetchArgs(url=url, max_chars=None), tc.session_id, origin=tc.origin,
                              subject=_subject(url))
            action, why = policy.decide(req)
            grants = tc.get(Grants)
            ok = action == "allow" or (action == "ask" and grants is not None and grants.allows(req))
            by = "policy" if action != "ask" else ("grant:session" if ok else "redirect:needs-approval")
        audit = tc.get(AuditLog) if tc.ctx is not None else None
        if audit is not None:
            audit.write(session=tc.session_id, call=tc.call_id, tool="web.fetch", risk=ref[0].risk.label,
                        subject=_subject(url), origin=tc.origin, action="allow" if ok else "deny",
                        decided_by=by, reason=why, redirect_from=prev)
        return ok, why

    async def fetch(a: FetchArgs, tc: ToolContext) -> str:
        url = normalize_url(a.url)
        await preflight(url)
        approved = {_host(url)}       # the original call passed the permission gate
        chain: list[str] = []
        chunks: list[bytes] = []
        ctype, enc, status, cut = "", "utf-8", 0, False
        for _hop in range(cfg.max_redirects + 1):
            target = ""
            try:
                async with http.stream("GET", url, headers={"User-Agent": cfg.user_agent},
                                       follow_redirects=False, timeout=cfg.timeout) as r:
                    status = r.status_code
                    if status in _REDIRECTS and r.headers.get("location"):
                        target = normalize_url(urljoin(str(r.url), r.headers["location"]))
                    else:
                        ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                        if not _is_text(ctype):
                            size = r.headers.get("content-length")
                            raise ToolError(f"{url} returned non-text content ({ctype}"
                                            + (f", {int(size):,} bytes" if size and size.isdigit() else "")
                                            + "); web.fetch only returns text (HTML, plain text, JSON, XML, feeds)")
                        size_read = 0
                        async for b in r.aiter_bytes():
                            room = cfg.max_bytes - size_read
                            if len(b) > room:
                                chunks.append(b[:room])
                                size_read += room
                                cut = True
                                break
                            chunks.append(b)
                            size_read += len(b)
                        enc = r.charset_encoding or "utf-8"
            except httpx.HTTPError as e:
                raise ToolError(f"fetch failed: {type(e).__name__}: {e}") from e
            if not target:
                break
            if len(chain) >= cfg.max_redirects:
                raise ToolError(f"too many redirects (more than {cfg.max_redirects}); stopped at {url}",
                                untrusted="\n".join([*chain, f"{status} {url} -> {target}"]))
            try:
                await preflight(target)
            except ToolError as e:
                raise ToolError(f"{url} redirected ({status}) to a refused target -- {e}; not followed",
                                untrusted=target) from None
            ok, why = follow_ok(tc, url, target, approved)
            if not ok:
                raise ToolError(
                    f"{url} redirected ({status}) to a domain that is not pre-approved for web.fetch ({why}); "
                    "the redirect was NOT followed. If the user wants that page, call web.fetch again with the "
                    "target URL below (that call goes through approval)", untrusted=target)
            chain.append(f"{status} {url} -> {target}")
            approved.add(_host(target))
            url = target
        raw = b"".join(chunks)
        if b"\x00" in raw[:4096]:
            raise ToolError(f"{url} returned binary content ({ctype or 'no content type'}, NUL bytes in the "
                            "first 4 KB); web.fetch only returns text")
        body = raw.decode(enc, "replace") if _known(enc) else raw.decode("utf-8", "replace")
        if "html" in ctype or body.lstrip()[:15].lower().startswith(("<!doctype html", "<html")):
            title, text = html_to_markdown(body, url)
        else:
            title, text = "", body
        head = f"URL: {url}\nStatus: {status}\nContent-Type: {ctype or 'unknown'}\n"
        if chain:
            head += "Redirects: " + "; ".join(chain) + "\n"
        head += f"Title: {title}\n"
        if cut:
            head += (f"[Note: the response body was cut at the {cfg.max_bytes:,}-byte download limit; "
                     "the text below is incomplete]\n")
        if a.max_chars is not None and 0 < a.max_chars < len(text):
            handle = write_artifact(tc, text, "web")
            return (f"{head}\n{text[:a.max_chars]}\n\n[... {len(text) - a.max_chars} more chars "
                    f"(~{estimate_tokens(text)} tokens in total) in artifact {handle!r}: "
                    f"artifact.read(handle={handle!r}, offset={a.max_chars}) ...]")
        return head + "\n" + preview(tc, text, "web", budget=PREVIEW_TOKENS - estimate_tokens(head))

    tool = Tool("web.fetch", "Fetch a web page or feed by URL and return its text (no search).", fetch,
                FetchArgs, parallel_safe=True, default_action="ask",
                default_allow={"domain": list(cfg.allow_domains)} if cfg.allow_domains else {},
                subject=lambda a: _subject(a.url), untrusted=True,
                timeout=cfg.timeout * (cfg.max_redirects + 1) + 10)
    ref.append(tool)
    return tool


def _known(enc: str) -> bool:
    import codecs
    try:
        codecs.lookup(enc)
    except LookupError:
        return False
    return True


@ventri.plugin(name="tool:web", config=WebConfig)
async def web(ctx: Any, cfg: WebConfig, registry: ToolRegistry) -> None:
    """``use: ventri_agent.tools.web`` -- registers ``web.fetch`` (SSRF-guarded client)."""
    http = await ctx.enter(guarded_client(allow_private=cfg.allow_private_urls))
    registry.register(ctx, make_tool(cfg, http))


plugin = web
