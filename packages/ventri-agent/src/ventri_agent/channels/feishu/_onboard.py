"""Feishu / Lark scan-to-create ("一键创建智能体应用") registration.

Adapted from Hermes Agent ``plugins/platforms/feishu/adapter.py`` (the QR
onboarding section: ``_post_registration``, ``_init_registration``,
``_begin_registration``, ``_poll_registration``, ``probe_bot``,
``_probe_bot_http``; https://github.com/NousResearch/hermes-agent, commit
a28a5d03, Copyright (c) 2025 Nous Research, MIT License), with protocol
details aligned to the official SDK's implementation of the same flow
(``lark_oapi/scene/registration``, lark-oapi 1.7.3,
https://github.com/larksuite/oapi-sdk-python, Copyright (c) 2023 Lark
Technologies Pte. Ltd., MIT License): ``slow_down`` handling, the
``expires_in`` spelling, the ``from/tp/source`` QR parameters and the
``addons`` encoding. See THIRD_PARTY_NOTICES.md.

The flow is OAuth 2.0 device authorization (RFC 8628) against
``https://accounts.feishu.cn/oauth/v1/app/registration``:
``init`` (the environment must support ``client_secret``) -> ``begin``
(``archetype=PersonalAgent``; returns a device code and a verification URL
to show as a QR code) -> ``poll`` until the user scanned and confirmed in the
Feishu / Lark app -> ``client_id`` / ``client_secret`` (App ID / App Secret)
and the scanning user's ``open_id`` (scoped to the new app). A ``lark``
``tenant_brand`` switches polling to ``accounts.larksuite.com``.

Changes for Ventri: httpx with timeouts and an injectable transport (tests
never touch the network) instead of urllib; typed errors instead of
``None``; network errors and 5xx are retried until the deadline; no prints
(the caller reports progress); the App Secret is kept out of ``repr`` and
never logged.
"""
from __future__ import annotations

import base64
import gzip
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx

log = logging.getLogger("ventri_agent.feishu.onboard")

ACCOUNTS_URLS = {"feishu": "https://accounts.feishu.cn", "lark": "https://accounts.larksuite.com"}
OPEN_URLS = {"feishu": "https://open.feishu.cn", "lark": "https://open.larksuite.com"}
REGISTRATION_PATH = "/oauth/v1/app/registration"
REQUEST_TIMEOUT = 10.0
DEFAULT_INTERVAL = 5
DEFAULT_EXPIRE = 600

# What Ventri's channel needs, for ``--minimal`` (``addons.preset=false``: the
# app gets only these instead of the platform's agent template).
MINIMAL_ADDONS: dict[str, Any] = {
    "preset": False,
    "scopes": {"tenant": ["im:message.p2p_msg:readonly", "im:message.group_at_msg:readonly",
                          "im:message:send_as_bot", "im:message:update", "application:bot.basic_info:read"]},
    "events": {"items": {"tenant": ["im.message.receive_v1"]}},
    "callbacks": {"items": ["card.action.trigger"]},
}


class RegistrationError(Exception):
    """``code``: ``access_denied`` | ``expired_token`` | ``unsupported_auth_method`` |
    ``bad_response`` | ``network`` | another error string from the server."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Begin:
    device_code: str
    qr_url: str
    user_code: str = ""
    interval: int = DEFAULT_INTERVAL
    expire_in: int = DEFAULT_EXPIRE


@dataclass(frozen=True)
class Credentials:
    app_id: str
    app_secret: str = field(repr=False)
    domain: str = "feishu"
    open_id: str = ""
    tenant_brand: str = ""


def encode_addons(addons: dict[str, Any]) -> str:
    """gzip + URL-safe base64 of the compact JSON (lark-oapi ``_encode_addons``)."""
    payload = json.dumps(addons, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(gzip.compress(payload.encode("utf-8"), mtime=0)).decode("ascii").rstrip("=")


def _int(v: Any, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


class Registrar:
    """One registration conversation. ``transport`` (httpx) and ``sleep`` /
    ``clock`` are injectable so tests run offline and instantly."""

    def __init__(self, *, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 source: str = "ventri") -> None:
        self.client = httpx.Client(transport=transport, timeout=REQUEST_TIMEOUT,
                                   headers={"User-Agent": f"ventri-agent ({source})"})
        self.sleep = sleep
        self.clock = clock
        self.source = source

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- HTTP
    def post(self, domain: str, body: dict[str, str]) -> dict[str, Any]:
        """POST a form; parse JSON even on 4xx (``authorization_pending`` comes as 400)."""
        url = ACCOUNTS_URLS.get(domain, ACCOUNTS_URLS["feishu"]) + REGISTRATION_PATH
        try:
            r = self.client.post(url, content=urlencode(body),
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
        except httpx.HTTPError as e:
            raise RegistrationError("network", f"{type(e).__name__}: {e}") from None
        try:
            data = r.json()
        except ValueError:
            code = "network" if r.status_code >= 500 else "bad_response"
            raise RegistrationError(code, f"HTTP {r.status_code}, not JSON") from None
        if not isinstance(data, dict):
            raise RegistrationError("bad_response", f"HTTP {r.status_code}, unexpected JSON")
        if r.status_code >= 500 and not data.get("error"):
            raise RegistrationError("network", f"HTTP {r.status_code}")
        return data

    # ------------------------------------------------------------ steps
    def init(self, domain: str = "feishu") -> None:
        res = self.post(domain, {"action": "init"})
        methods = res.get("supported_auth_methods") or []
        if "client_secret" not in methods:
            raise RegistrationError("unsupported_auth_method",
                                    f"registration does not offer client_secret auth (offers {methods})")

    def begin(self, domain: str = "feishu", *, addons: dict[str, Any] | None = None) -> Begin:
        res = self.post(domain, {"action": "begin", "archetype": "PersonalAgent",
                                 "auth_method": "client_secret", "request_user_info": "open_id"})
        if res.get("error"):
            raise RegistrationError(str(res["error"]), str(res.get("error_description", "")))
        device_code = str(res.get("device_code") or "")
        uri = str(res.get("verification_uri_complete") or res.get("verification_uri") or "")
        if not device_code or not uri:
            raise RegistrationError("bad_response", "begin returned no device_code / verification URL")
        return Begin(device_code=device_code, qr_url=self.qr_url(uri, addons), user_code=str(res.get("user_code", "")),
                     interval=_int(res.get("interval"), DEFAULT_INTERVAL),
                     expire_in=_int(res.get("expire_in", res.get("expires_in")), DEFAULT_EXPIRE))

    def qr_url(self, uri: str, addons: dict[str, Any] | None = None) -> str:
        parsed = urlparse(uri)
        params = parse_qs(parsed.query)
        params.update({"from": ["sdk"], "tp": ["sdk"], "source": [f"python-sdk/{self.source}"]})
        if addons is not None:
            params["addons"] = [encode_addons(addons)]
        return urlunparse(parsed._replace(query=urlencode(params, doseq=True)))

    def poll(self, begin: Begin, *, domain: str = "feishu", timeout: float | None = None,
             on_status: Callable[[str], None] | None = None) -> Credentials:
        """Poll until the user confirms. Raises :class:`RegistrationError`
        (``access_denied`` / ``expired_token`` / ...)."""
        expire = begin.expire_in if timeout is None else min(begin.expire_in, timeout)
        deadline = self.clock() + expire
        interval = float(begin.interval)
        current, switched = domain, False
        notify = on_status or (lambda status: None)
        failures = 0
        while self.clock() < deadline:
            try:
                res = self.post(current, {"action": "poll", "device_code": begin.device_code})
            except RegistrationError as e:
                if e.code != "network":
                    raise
                failures += 1
                notify("network-retry")
                log.info("feishu registration: poll failed (%s); retrying", e.message)
                self.sleep(min(interval * (1 + min(failures, 5)), 30.0))
                continue
            failures = 0
            user = res.get("user_info") or {}
            if res.get("client_id") and res.get("client_secret"):
                brand = str(user.get("tenant_brand") or "")
                return Credentials(app_id=str(res["client_id"]), app_secret=str(res["client_secret"]),
                                   domain=brand if brand in ("feishu", "lark") else current,
                                   open_id=str(user.get("open_id") or ""), tenant_brand=brand)
            if user.get("tenant_brand") == "lark" and not switched:
                current, switched = "lark", True
                notify("domain-switched")
                continue
            error = str(res.get("error") or "")
            if error in ("", "authorization_pending"):
                notify("pending")
            elif error == "slow_down":
                interval += 5
                notify("slow-down")
            elif error in ("access_denied", "expired_token"):
                raise RegistrationError(error, str(res.get("error_description", "")))
            else:
                raise RegistrationError(error, str(res.get("error_description", "")))
            self.sleep(interval)
        raise RegistrationError("expired_token", f"not confirmed within {int(expire)} s")

    # ------------------------------------------------------------ probe
    def probe_bot(self, creds: Credentials) -> dict[str, str] | None:
        """``/open-apis/bot/v3/info`` with a tenant token -> ``{"app_name", "open_id"}``
        (the bot's own open_id), or ``None`` (best effort; never raises)."""
        base = OPEN_URLS.get(creds.domain, OPEN_URLS["feishu"])
        try:
            tok = self.client.post(f"{base}/open-apis/auth/v3/tenant_access_token/internal",
                                   json={"app_id": creds.app_id, "app_secret": creds.app_secret}).json()
            token = tok.get("tenant_access_token") if isinstance(tok, dict) else None
            if not token:
                log.info("feishu registration: no tenant token (code %s)", (tok or {}).get("code"))
                return None
            data = self.client.get(f"{base}/open-apis/bot/v3/info",
                                   headers={"Authorization": f"Bearer {token}"}).json()
        except (httpx.HTTPError, ValueError) as e:
            log.info("feishu registration: bot probe failed: %s", type(e).__name__)
            return None
        if not isinstance(data, dict) or data.get("code") != 0:
            return None
        bot = data.get("bot") or (data.get("data") or {}).get("bot") or {}
        return {"app_name": str(bot.get("app_name") or bot.get("bot_name") or ""),
                "open_id": str(bot.get("open_id") or "")}
