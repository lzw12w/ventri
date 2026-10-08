"""``web.fetch`` -- fetch a URL the user gave (or a feed) and return readable text.

http(s) only; size- and time-limited; HTML is reduced to Markdown-ish text
(scripts, styles, navigation dropped). Risk ``read`` but **asked** by default
unless the domain matches ``allow_domains`` -- a fetched URL can carry data out
(prompt-injection exfiltration), so it is not treated as a free read. Output is
fenced as untrusted. There is no web *search* (deferred, DESIGN D10)."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel, Field

import ventri

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
    max_chars: int = Field(20_000, description="Maximum characters of text to return")


class WebConfig(BaseModel):
    allow_domains: list[str] = Field(default_factory=list)  # e.g. ["*.python.org", "docs.deepseek.com"]
    timeout: float = 30.0
    max_bytes: int = 3_000_000
    user_agent: str = "VentriAgent/0.2 (+https://github.com/lzw12w/ventri)"


def make_tool(cfg: WebConfig, http: httpx.AsyncClient) -> Tool:
    async def fetch(a: FetchArgs, tc: ToolContext) -> str:
        parts = urlsplit(a.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ToolError("only http(s) URLs can be fetched")
        try:
            async with http.stream("GET", a.url, headers={"User-Agent": cfg.user_agent},
                                   follow_redirects=True, timeout=cfg.timeout) as r:
                chunks, size = [], 0
                async for b in r.aiter_bytes():
                    size += len(b)
                    if size > cfg.max_bytes:
                        break
                    chunks.append(b)
                status, ctype, final = r.status_code, r.headers.get("content-type", ""), str(r.url)
                enc = r.charset_encoding or "utf-8"
        except httpx.HTTPError as e:
            raise ToolError(f"fetch failed: {type(e).__name__}: {e}") from e
        body = b"".join(chunks).decode(enc, "replace")
        if "html" in ctype or body.lstrip()[:15].lower().startswith(("<!doctype html", "<html")):
            title, text = html_to_markdown(body, final)
        else:
            title, text = "", body
        cut = text[:a.max_chars]
        more = f"\n[... {len(text) - len(cut)} more chars]" if len(text) > len(cut) else ""
        return f"URL: {final}\nStatus: {status}\nTitle: {title}\n\n{cut}{more}"

    return Tool("web.fetch", "Fetch a web page or feed by URL and return its text (no search).", fetch,
                FetchArgs, parallel_safe=True, default_action="ask",
                default_allow={"domain": list(cfg.allow_domains)} if cfg.allow_domains else {},
                subject=lambda a: {"domain": urlsplit(a.url).hostname or "", "url": a.url},
                untrusted=True, timeout=cfg.timeout + 10)


@ventri.plugin(name="tool:web", config=WebConfig)
async def web(ctx: Any, cfg: WebConfig, registry: ToolRegistry) -> None:
    """``use: ventri_agent.tools.web`` -- registers ``web.fetch``."""
    http = await ctx.enter(httpx.AsyncClient())
    registry.register(ctx, make_tool(cfg, http))


plugin = web
