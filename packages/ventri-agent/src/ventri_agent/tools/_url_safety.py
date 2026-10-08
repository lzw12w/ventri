"""URL safety for ``web.fetch``: SSRF blocking, connect-time DNS-rebinding guard,
credential-bearing URL refusal.

Adapted from Hermes Agent ``tools/url_safety.py`` and the secret-prefix list of
``agent/redact.py`` (https://github.com/NousResearch/hermes-agent, commit
a28a5d03), Copyright (c) 2025 Nous Research, MIT License -- see
THIRD_PARTY_NOTICES.md. Changes for Ventri: the policy is passed in
(``allow_private``) instead of read from Hermes config; the resolver is
injectable and async (tests); no trusted-private-host or fake-ip exemptions;
a DNS failure is refused unless an HTTP proxy is configured (then the proxy
resolves, as in Hermes); a narrower secret-prefix list.

Layers:

1. :func:`static_block_reason` -- scheme, embedded credentials, credential-named
   query parameters, secret-looking tokens, metadata hostnames, literal IPs.
2. :func:`check_resolved` -- resolve the host and validate *every* answer
   (private, loopback, link-local, CGNAT, multicast, reserved, unspecified,
   cloud metadata; IPv4-mapped / -translated IPv6 is classified by its IPv4).
3. :class:`GuardedBackend` -- an httpcore network backend that resolves and
   validates again at TCP connect and dials the vetted IP (Host header and TLS
   SNI keep the hostname), so a name that re-resolves to 127.0.0.1 between the
   pre-flight check and the connect (DNS rebinding) is refused. Unix sockets
   are refused. Requests through an environment proxy are not dialed by us and
   are covered by layers 1-2 only.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlsplit, urlunsplit

import httpcore
import httpx

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str, int], Awaitable[list[str]]]

HTTP_SCHEMES = frozenset({"http", "https"})
_PROXY_ENV_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
MAX_CONNECT_IPS = 8

# Unambiguously credential-bearing query parameter names (Hermes list; bare words
# like ``code``/``key``/``session`` are excluded so ordinary pages still work).
SENSITIVE_QUERY_PARAMS = frozenset({
    "access_token", "api_key", "apikey", "auth_token", "authorization", "awsaccesskeyid",
    "client_secret", "credential", "credentials", "jwt", "password", "passwd", "secret",
    "session_id", "signature", "token", "x_amz_security_token", "x_amz_signature",
    "x-amz-security-token", "x-amz-signature"})

# Vendor token prefixes (subset of Hermes agent/redact.py _PREFIX_PATTERNS) + JWTs.
_SECRET_PATTERNS = (
    r"sk-[A-Za-z0-9_-](?:\.?[A-Za-z0-9_-]){9,}",   # OpenAI / DeepSeek / Anthropic style
    r"sk_(?:live|test)_[A-Za-z0-9]{10,}", r"rk_live_[A-Za-z0-9]{10,}",
    r"gh[pousr]_[A-Za-z0-9]{10,}", r"github_pat_[A-Za-z0-9_]{10,}",
    r"glpat-[A-Za-z0-9_\-]{10,}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}", r"xapp-\d+-[A-Za-z0-9-]{10,}",
    r"AIza[A-Za-z0-9_-]{30,}", r"AKIA[A-Z0-9]{16}",
    r"SG\.[A-Za-z0-9_-]{10,}", r"hf_[A-Za-z0-9]{10,}", r"npm_[A-Za-z0-9]{10,}",
    r"pypi-[A-Za-z0-9_-]{10,}", r"xai-[A-Za-z0-9]{30,}", r"gsk_[A-Za-z0-9]{10,}",
    r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",  # JWT
)
_SECRET_RE = re.compile("|".join(_SECRET_PATTERNS))

# Cloud metadata -- always blocked, even with allow_private.
BLOCKED_HOSTNAMES = frozenset({"metadata.google.internal", "metadata.goog"})
_METADATA_V4 = ("169.254.169.254", "169.254.170.2", "169.254.169.253", "100.100.100.200")
_ALWAYS_BLOCKED_IPS = frozenset(
    {ipaddress.ip_address(ip) for ip in _METADATA_V4}
    | {ipaddress.ip_address("::ffff:" + ip) for ip in _METADATA_V4}
    | {ipaddress.ip_address("fd00:ec2::254")})
_ALWAYS_BLOCKED_NETS = tuple(ipaddress.ip_network(n) for n in ("169.254.0.0/16", "::ffff:169.254.0.0/112"))
_CGNAT = ipaddress.ip_network("100.64.0.0/10")              # neither private nor global in ipaddress
_IPV4_TRANSLATED = ipaddress.ip_network("::ffff:0:0:0/96")  # RFC 2765 ::ffff:0:a.b.c.d


class UrlBlocked(Exception):
    """The URL or one of its resolved addresses violates the fetch policy."""


class DnsFailed(UrlBlocked):
    """The host could not be resolved (refused unless a proxy will resolve it)."""


def proxy_configured() -> bool:
    return any(os.environ.get(v) for v in _PROXY_ENV_VARS)


def normalize_url(url: str) -> str:
    """ASCII-safe URL (IDNA host, percent-encoded non-ASCII path/query); other
    schemes are returned unchanged for the caller to reject."""
    raw = re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)\s+", r"\1", url.strip())
    try:
        p = urlsplit(raw)
    except ValueError:
        return raw
    if p.scheme.lower() not in HTTP_SCHEMES:
        return raw
    netloc, host = p.netloc, p.hostname
    if host:
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError:
            ascii_host = host
        if ascii_host != host:
            netloc = netloc.replace(host, ascii_host, 1)
    safe = "/%:@!$&'()*+,;="
    return urlunsplit((p.scheme, netloc, quote(p.path, safe=safe), quote(p.query, safe=safe + "?"),
                       quote(p.fragment, safe=safe + "?")))


def normalize_host(host: str | None) -> str:
    return (host or "").strip().lower().rstrip(".").strip("[]")


def parse_ip(host: str) -> IPAddress | None:
    try:
        return ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return None


def embedded_ipv4(ip: IPAddress) -> IPAddress:
    """The IPv4 address behind an IPv4-mapped or IPv4-translated IPv6 address."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip in _IPV4_TRANSLATED:
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip


def ip_block_reason(ip: IPAddress, allow_private: bool) -> str | None:
    v = embedded_ipv4(ip)
    if ip in _ALWAYS_BLOCKED_IPS or v in _ALWAYS_BLOCKED_IPS or any(v in n for n in _ALWAYS_BLOCKED_NETS):
        return "cloud metadata / link-local address"
    if not allow_private and (v.is_private or v.is_loopback or v.is_link_local or v.is_reserved
                              or v.is_multicast or v.is_unspecified or v in _CGNAT):
        return "private/internal address"
    return None


def sensitive_query_param(url: str) -> str | None:
    try:
        q = urlsplit(url).query
    except ValueError:
        return None
    return next((k for k, v in parse_qsl(q, keep_blank_values=True)
                 if v and unquote(k).lower() in SENSITIVE_QUERY_PARAMS), None)


def static_block_reason(url: str, *, allow_private: bool) -> str | None:
    """Checks that need no DNS. ``url`` should already be normalized."""
    try:
        p = urlsplit(url)
        p.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError as e:
        return f"malformed URL ({e})"
    if p.scheme.lower() not in HTTP_SCHEMES:
        return "only http(s) URLs can be fetched"
    host = normalize_host(p.hostname)
    if not host:
        return "URL has no host"
    if p.username is not None or p.password is not None:
        return "URL contains embedded credentials (user:password@); credentials must not be sent in URLs"
    if name := sensitive_query_param(url):
        return f"URL carries a credential-like query parameter ({name!r}); secrets must not be sent in URLs"
    if _SECRET_RE.search(url) or _SECRET_RE.search(unquote(url)):
        return "URL contains what appears to be an API key or token; secrets must not be sent in URLs"
    if host in BLOCKED_HOSTNAMES:
        return f"{host} is a cloud metadata hostname"
    ip = parse_ip(host)
    if ip is not None and (why := ip_block_reason(ip, allow_private)):
        return f"{host} is a {why}"
    return None


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.to_thread(socket.getaddrinfo, host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    return [str(sa[0]) for *_, sa in infos]


async def checked_ips(host: str, port: int, resolver: Resolver, *, allow_private: bool) -> list[str]:
    """Resolve ``host`` and validate every answer; return up to
    :data:`MAX_CONNECT_IPS` dialable addresses. Raises :class:`UrlBlocked`."""
    host = normalize_host(host)
    if not host:
        raise UrlBlocked("empty hostname")
    if host in BLOCKED_HOSTNAMES:
        raise UrlBlocked(f"{host} is a cloud metadata hostname")
    literal = parse_ip(host)
    if literal is not None:
        answers = [host]
    else:
        try:
            answers = await resolver(host, port)
        except OSError as e:
            raise DnsFailed(f"DNS resolution failed for {host}") from e
    safe: list[str] = []
    for raw in answers:
        ip = parse_ip(raw)
        if ip is None:
            raise UrlBlocked(f"unparseable address {raw!r} for {host}")
        if why := ip_block_reason(ip, allow_private):
            raise UrlBlocked(f"{host} resolves to a {why} ({ip})")
        s = str(ip)
        if s not in safe and len(safe) < MAX_CONNECT_IPS:
            safe.append(s)
    if not safe:
        raise UrlBlocked(f"DNS returned no addresses for {host}")
    return safe


class GuardedBackend(httpcore.AsyncNetworkBackend):
    """Connect-time guard: re-resolve, validate every answer, dial a vetted IP."""

    def __init__(self, resolver: Resolver, *, allow_private: bool) -> None:
        from httpcore._backends.auto import AutoBackend  # httpcore's default backend
        self._backend = AutoBackend()
        self._resolver = resolver
        self._allow_private = allow_private

    async def connect_tcp(self, host: str, port: int, timeout: float | None = None,
                          local_address: str | None = None, socket_options: Any = None) -> Any:
        try:
            ips = await checked_ips(host, port, self._resolver, allow_private=self._allow_private)
        except UrlBlocked as e:
            raise httpcore.ConnectError(f"blocked at connect: {e}") from e
        last: Exception | None = None
        for ip in ips:
            try:
                return await self._backend.connect_tcp(ip, port, timeout=timeout, local_address=local_address,
                                                       socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as e:
                last = e
        assert last is not None
        raise last

    async def connect_unix_socket(self, path: str, timeout: float | None = None,
                                  socket_options: Any = None) -> Any:
        raise httpcore.ConnectError("unix sockets are refused by the SSRF guard")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


def guarded_client(resolver: Resolver = system_resolver, *, allow_private: bool = False,
                   **kwargs: Any) -> httpx.AsyncClient:
    """``httpx.AsyncClient`` whose direct connections go through :class:`GuardedBackend`.
    Environment proxies (``trust_env``) are honored and are not guarded at connect."""
    client = httpx.AsyncClient(**kwargs)
    # Swap the direct transport's pool backend (as Hermes does); proxy mounts are left alone.
    pool = getattr(client._transport, "_pool", None)  # pyright: ignore[reportPrivateUsage]
    if pool is None or not hasattr(pool, "_network_backend"):
        raise RuntimeError("unsupported httpx version: cannot install the SSRF connect guard")
    pool._network_backend = GuardedBackend(resolver, allow_private=allow_private)
    return client
