"""Feishu transport: the lark-oapi SDK behind two small seams.

* :class:`FeishuApi` -- the four REST calls the channel needs (send / reply,
  patch a card, bot identity). :class:`LarkApi` implements it with the SDK's
  synchronous client in worker threads (``anyio.to_thread``); tests use a fake.
* :class:`WsRunner` -- one ``lark_oapi.ws.Client`` (long connection) in a
  dedicated thread with its *own* event loop. The SDK is thread/loop-hostile
  (module-global loop, blocking ``requests`` calls inside coroutines, a
  ``start()`` that never returns), so it never touches the kernel's loop:
  events cross over as plain dataclasses via ``loop.call_soon_threadsafe``;
  card clicks wait (bounded) for the kernel loop's verdict to build the toast.
  Shutdown sends a WebSocket CLOSE frame (so Feishu stops routing to the dead
  socket), stops the worker loop and joins the thread; the channel supervisor
  restarts the runner with backoff when it exits on its own.

``lark_oapi`` is imported lazily (it is the optional ``feishu`` extra and
takes seconds to import).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import importlib.util
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import anyio

from ._hermes import REPLY_FALLBACK_CODES, MentionRef, bind_thread_loop, install_loop_proxy, ws_client_class

log = logging.getLogger("ventri_agent.feishu")

INSTALL_HINT = "the Feishu channel needs lark-oapi: pip install 'ventri-agent[feishu]' (or uv sync --extra feishu)"
RATE_LIMIT_CODES = frozenset({230020, 99991400})


def load_sdk() -> Any:
    """Import ``lark_oapi`` (raises ``RuntimeError`` with an install hint)."""
    try:
        return importlib.import_module("lark_oapi")
    except ImportError as e:
        raise RuntimeError(INSTALL_HINT) from e


def sdk_available() -> bool:
    return importlib.util.find_spec("lark_oapi") is not None


# ------------------------------------------------------------------ events
@dataclass(frozen=True)
class Inbound:
    """``im.message.receive_v1``, flattened."""

    event_id: str
    message_id: str
    chat_id: str
    chat_type: str                 # p2p | group
    sender_open_id: str
    sender_type: str               # user | bot | app
    message_type: str
    content: str                   # raw JSON string
    create_time_ms: int = 0
    thread_id: str = ""
    root_id: str = ""
    mentions: tuple[MentionRef, ...] = ()


@dataclass(frozen=True)
class CardAction:
    """``card.action.trigger`` (new-style card callback), flattened."""

    event_id: str
    operator_open_id: str
    chat_id: str
    message_id: str
    value: dict[str, Any] = field(default_factory=dict)
    token: str = ""


def _s(v: Any) -> str:
    return "" if v is None else str(v)


def inbound_from_sdk(data: Any) -> Inbound | None:
    """``P2ImMessageReceiveV1`` (or anything shaped like it) -> :class:`Inbound`."""
    ev = getattr(data, "event", None)
    msg = getattr(ev, "message", None)
    sender = getattr(ev, "sender", None)
    if msg is None or sender is None:
        return None
    sid = getattr(sender, "sender_id", None)
    mentions = []
    for m in getattr(msg, "mentions", None) or []:
        mid = getattr(m, "id", None)
        mentions.append(MentionRef(key=_s(getattr(m, "key", "")), name=_s(getattr(m, "name", "")),
                                   open_id=_s(getattr(mid, "open_id", "")) if mid is not None else ""))
    try:
        ct = int(_s(getattr(msg, "create_time", "")) or 0)
    except ValueError:
        ct = 0
    return Inbound(
        event_id=_s(getattr(getattr(data, "header", None), "event_id", "")),
        message_id=_s(getattr(msg, "message_id", "")), chat_id=_s(getattr(msg, "chat_id", "")),
        chat_type=_s(getattr(msg, "chat_type", "")) or "p2p",
        sender_open_id=_s(getattr(sid, "open_id", "")) if sid is not None else "",
        sender_type=_s(getattr(sender, "sender_type", "")) or "user",
        message_type=_s(getattr(msg, "message_type", "")), content=_s(getattr(msg, "content", "")),
        create_time_ms=ct, thread_id=_s(getattr(msg, "thread_id", "")), root_id=_s(getattr(msg, "root_id", "")),
        mentions=tuple(mentions))


def card_action_from_sdk(data: Any) -> CardAction:
    ev = getattr(data, "event", None)
    action = getattr(ev, "action", None)
    value = getattr(action, "value", None)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = {}
    ctx = getattr(ev, "context", None)
    return CardAction(
        event_id=_s(getattr(getattr(data, "header", None), "event_id", "")),
        operator_open_id=_s(getattr(getattr(ev, "operator", None), "open_id", "")),
        chat_id=_s(getattr(ctx, "open_chat_id", "")), message_id=_s(getattr(ctx, "open_message_id", "")),
        value=value if isinstance(value, dict) else {}, token=_s(getattr(ev, "token", "")))


# --------------------------------------------------------------------- REST
class FeishuApiError(Exception):
    def __init__(self, code: Any, msg: str) -> None:
        super().__init__(f"[{code}] {msg}")
        self.code = code
        self.msg = msg


class FeishuApi(Protocol):
    async def send_card(self, chat_id: str, card: str, *, reply_to: str | None = None,
                        in_thread: bool = False) -> str: ...
    async def send_text(self, chat_id: str, text: str, *, reply_to: str | None = None,
                        in_thread: bool = False) -> str: ...
    async def patch_card(self, message_id: str, card: str) -> None: ...
    async def bot_info(self) -> dict[str, Any]: ...


class LarkApi:
    """:class:`FeishuApi` over ``lark_oapi.Client`` (blocking calls in threads)."""

    def __init__(self, sdk: Any, *, app_id: str, app_secret: str, domain: str, timeout: float = 30.0) -> None:
        self.sdk = sdk
        const = importlib.import_module("lark_oapi.core.const")
        self.im = importlib.import_module("lark_oapi.api.im.v1")
        self.client = (sdk.Client.builder().app_id(app_id).app_secret(app_secret)
                       .domain(const.LARK_DOMAIN if domain == "lark" else const.FEISHU_DOMAIN)
                       .timeout(timeout).log_level(sdk.LogLevel.WARNING).build())
        self.limiter = anyio.CapacityLimiter(8)

    async def _call(self, fn: Callable[[], Any]) -> Any:
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = await anyio.to_thread.run_sync(fn, limiter=self.limiter)
            except Exception as e:  # noqa: BLE001 - network errors: retry with backoff
                last = e
                await anyio.sleep(0.5 * 2 ** attempt)
                continue
            if resp.success():
                return resp
            code = getattr(resp, "code", None)
            if code in RATE_LIMIT_CODES and attempt < 2:
                await anyio.sleep(1.0 * 2 ** attempt)
                continue
            raise FeishuApiError(code, _s(getattr(resp, "msg", "")))
        raise FeishuApiError("network", repr(last))

    async def _send(self, chat_id: str, msg_type: str, content: str, reply_to: str | None,
                    in_thread: bool) -> str:
        im = self.im
        if reply_to:
            body = (im.ReplyMessageRequestBody.builder().content(content).msg_type(msg_type)
                    .reply_in_thread(in_thread).uuid(str(uuid.uuid4())).build())
            req = im.ReplyMessageRequest.builder().message_id(reply_to).request_body(body).build()
            try:
                resp = await self._call(lambda: self.client.im.v1.message.reply(req))
                return _s(resp.data.message_id)
            except FeishuApiError as e:
                if e.code not in REPLY_FALLBACK_CODES:
                    raise
        body = (im.CreateMessageRequestBody.builder().receive_id(chat_id).msg_type(msg_type)
                .content(content).uuid(str(uuid.uuid4())).build())
        req = im.CreateMessageRequest.builder().receive_id_type("chat_id").request_body(body).build()
        resp = await self._call(lambda: self.client.im.v1.message.create(req))
        return _s(resp.data.message_id)

    async def send_card(self, chat_id: str, card: str, *, reply_to: str | None = None,
                        in_thread: bool = False) -> str:
        return await self._send(chat_id, "interactive", card, reply_to, in_thread)

    async def send_text(self, chat_id: str, text: str, *, reply_to: str | None = None,
                        in_thread: bool = False) -> str:
        return await self._send(chat_id, "text", json.dumps({"text": text}, ensure_ascii=False),
                                reply_to, in_thread)

    async def patch_card(self, message_id: str, card: str) -> None:
        im = self.im
        req = (im.PatchMessageRequest.builder().message_id(message_id)
               .request_body(im.PatchMessageRequestBody.builder().content(card).build()).build())
        await self._call(lambda: self.client.im.v1.message.patch(req))

    async def bot_info(self) -> dict[str, Any]:
        """``GET /open-apis/bot/v3/info`` (tenant token, no extra scope)."""
        core = importlib.import_module("lark_oapi.core")
        model = importlib.import_module("lark_oapi.core.model")
        req = (model.BaseRequest.builder().http_method(core.HttpMethod.GET).uri("/open-apis/bot/v3/info")
               .token_types({core.AccessTokenType.TENANT}).build())
        resp = await anyio.to_thread.run_sync(lambda: self.client.request(req), limiter=self.limiter)
        raw = getattr(getattr(resp, "raw", None), "content", b"") or b"{}"
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        if data.get("code", 0) != 0:
            raise FeishuApiError(data.get("code"), _s(data.get("msg")))
        return dict(data.get("bot") or {})


# ---------------------------------------------------------------- WS runner
@dataclass
class WsCallbacks:
    """Thread-safe callbacks into the kernel loop (all non-blocking except
    ``card_action``, which returns the toast within a bound)."""

    message: Callable[[Inbound], None]
    card_action: Callable[[CardAction], tuple[str, str]]
    link_up: Callable[[], None] = lambda: None
    link_lost: Callable[[str], None] = lambda reason: None


class WsRunner:
    """One long-connection client in its own thread and event loop."""

    def __init__(self, *, app_id: str, app_secret: str, domain: str, callbacks: WsCallbacks,
                 on_exit: Callable[[BaseException | None], None]) -> None:
        self.app_id = app_id
        self._secret = app_secret
        self.domain = domain
        self.cb = callbacks
        self.on_exit = on_exit
        self._lock = threading.Lock()
        self._stopping = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None
        self.thread = threading.Thread(target=self._main, name=f"feishu-ws-{app_id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    # ------------------------------------------------------------- thread
    def _handler(self, sdk: Any) -> Any:
        resp_mod = importlib.import_module("lark_oapi.event.callback.model.p2_card_action_trigger")

        def on_message(data: Any) -> None:
            try:
                ib = inbound_from_sdk(data)
                if ib is not None:
                    self.cb.message(ib)
            except Exception:
                log.exception("feishu: dropping a malformed message event")

        def on_card(data: Any) -> Any:
            kind, text = "error", "处理失败"
            try:
                kind, text = self.cb.card_action(card_action_from_sdk(data))
            except Exception:
                log.exception("feishu: card action failed")
            return resp_mod.P2CardActionTriggerResponse({"toast": {"type": kind, "content": text}})

        def ignore(data: Any) -> None:
            return None

        b = sdk.EventDispatcherHandler.builder("", "", sdk.LogLevel.WARNING)
        b = b.register_p2_im_message_receive_v1(on_message).register_p2_card_action_trigger(on_card)
        for name in ("register_p2_im_message_message_read_v1",
                     "register_p2_im_chat_access_event_bot_p2p_chat_entered_v1"):
            reg = getattr(b, name, None)
            if reg is not None:  # harmless if subscribed: no "processor not found" 500s
                b = reg(ignore)
        return b.build()

    def _main(self) -> None:
        loop = asyncio.new_event_loop()
        err: BaseException | None = None
        try:
            asyncio.set_event_loop(loop)
            with self._lock:
                if self._stopping:
                    return
                self._loop = loop
            sdk = load_sdk()
            ws_module = importlib.import_module("lark_oapi.ws.client")
            const = importlib.import_module("lark_oapi.core.const")
            install_loop_proxy(ws_module)
            bind_thread_loop(loop)
            cls = ws_client_class(ws_module, on_link_up=self.cb.link_up, on_dead=self._dead)
            client = cls(self.app_id, self._secret, log_level=sdk.LogLevel.WARNING, event_handler=self._handler(sdk),
                         domain=const.LARK_DOMAIN if self.domain == "lark" else const.FEISHU_DOMAIN,
                         auto_reconnect=True)
            client.on_reconnecting = lambda: self.cb.link_lost("reconnecting")
            with self._lock:
                if self._stopping:
                    return
                self._client = client
            client.start()   # returns only when the loop is stopped (or the first connect fails hard)
        except BaseException as e:  # noqa: BLE001 - reported to the supervisor
            err = e
        finally:
            bind_thread_loop(None)
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception as e:  # noqa: BLE001 - best effort
                log.debug("feishu: websocket loop teardown: %r", e)
            loop.close()
            with self._lock:
                self._loop = None
                stopping = self._stopping
            self.on_exit(None if stopping else (err or RuntimeError("long connection ended")))

    def _dead(self, e: BaseException) -> None:
        if not self._stopping:
            log.warning("feishu: long connection lost for good (%r); rebuilding", e)
            self.cb.link_lost(repr(e))

    # --------------------------------------------------------------- stop
    def stop(self, timeout: float = 10.0) -> bool:
        """Blocking; call from a worker thread. True if the thread ended."""
        with self._lock:
            self._stopping = True
            loop, client = self._loop, self._client
        if client is not None:
            client._auto_reconnect = False
        if loop is not None and not loop.is_closed():
            if client is not None and loop.is_running():
                try:  # CLOSE frame first, or Feishu keeps routing to the stale socket for a while
                    asyncio.run_coroutine_threadsafe(client._disconnect(), loop).result(timeout=5)
                except Exception as e:  # noqa: BLE001 - the socket may already be gone
                    log.debug("feishu: close frame not sent: %r", e)
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout)
        alive = self.thread.is_alive()
        if alive:
            log.warning("feishu: websocket thread did not exit within %.0fs", timeout)
        return not alive


def bridge_card_action(loop: asyncio.AbstractEventLoop, fn: Callable[[CardAction], tuple[str, str]],
                       act: CardAction, timeout: float = 2.0) -> tuple[str, str]:
    """Run ``fn(act)`` on ``loop`` from another thread and wait (Feishu wants
    the callback answered within 3 s)."""
    fut: concurrent.futures.Future[tuple[str, str]] = concurrent.futures.Future()

    def run() -> None:
        try:
            fut.set_result(fn(act))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)

    t0 = time.monotonic()
    try:
        loop.call_soon_threadsafe(run)
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        log.warning("feishu: card action not handled within %.1fs", time.monotonic() - t0)
        return "info", "处理中…"
    except RuntimeError:
        return "error", "服务正在关闭"
