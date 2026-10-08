"""The Feishu / Lark channel (``use: ventri_agent.channels.feishu``).

A stop-first channel plugin like the CLI one, but a daemon: it holds one long
connection to Feishu (see :mod:`.transport`), maps chats to Ventri sessions
and turns each allowed message into a session turn whose progress is shown on
a card that is patched in place. Approval requests become cards with
buttons; only the allowlisted user whose message started the turn can click
them (checked against the card callback's operator and an HMAC over the
request id / choice / owner / chat that the buttons carry), a timeout denies,
and the broker still mints and the gate still verifies the decision token,
so nothing written in a chat -- by a person or by the model -- can approve
anything. Allowlists fail closed: an empty ``allow_users`` lets nobody in;
unknown users get a one-line reply with their open_id so they can be added.

It only connects under ``va serve`` (``run: serve``, the default) so a
``va chat`` on the same configuration does not open a second connection that
would take half the events (Feishu delivers each event to one connection).
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import BaseModel, Field

import ventri
from ventri import Secret

from ...loop import TurnEvent, TurnResult
from ...paths import expand
from ...permission import ApprovalRequest, AuditLog, Choice
from ...serve import SERVE_KEY
from ...session import SessionLog, session_paths
from ...sessions import Session, SessionError
from ..cli import CliChannel, CliConfig, usage_line
from . import render as R
from ._hermes import (
    MentionRef,
    load_content,
    mentions_map,
    normalize_text,
    parse_post,
    strip_edge_self_mentions,
)
from .transport import (
    INSTALL_HINT,
    CardAction,
    FeishuApi,
    FeishuApiError,
    Inbound,
    LarkApi,
    WsCallbacks,
    WsRunner,
    bridge_card_action,
    load_sdk,
    sdk_available,
)

log = logging.getLogger("ventri_agent.feishu")

FEISHU_HELP = """**可用命令**
- `/help` 本帮助
- `/new`（或 `/end`、`/exit`）结束当前会话（提取记忆），下一条消息开启新会话
- `/suspend` 挂起会话（下一条消息自动恢复）
- `/retry` 出错后重试上一轮
- `/think [low|high|max|off|on|show|hide]` 思考强度 / 显示思考过程
- `/cost` 本会话的 token、缓存命中率与费用
- `/compact` 立即压缩较早的对话
- `/epoch` 开启新的上下文纪元（采用当前工具集）
- `/memory [list|pending|search 关键词|confirm ID|forget ID]` 长期记忆
- `/sessions` 最近的会话 · `/tree` 插件树 · `/id` 你的 open_id 与本会话 chat_id

审批只能通过卡片按钮完成，聊天文字（包括模型输出）不能批准任何操作。"""

UNKNOWN_HINT_INTERVAL = 3600.0
SEEN_TTL = 24 * 3600.0
SEEN_CAP = 2048
SUPPORTED_TYPES = ("text", "post")


class FeishuConfig(BaseModel):
    app_id: str = ""
    app_secret: Secret[str] | None = None        # None: $FEISHU_APP_SECRET
    domain: Literal["feishu", "lark"] = "feishu"
    allow_users: list[str] = Field(default_factory=list)   # open_ids (ou_...); empty: nobody
    allow_chats: list[str] = Field(default_factory=list)   # group chat_ids (oc_...); empty: no groups
    require_mention: bool = True                 # groups: only messages that @ the bot
    group_session: Literal["chat", "thread"] = "chat"
    reply_unknown: bool = True                   # tell unknown users their open_id (at most hourly)
    agent: str | None = None
    run: Literal["serve", "always"] = "serve"
    progress_interval: float = Field(default=1.5, ge=0.5)
    max_message_age: float = Field(default=600.0, gt=0)
    show_thinking: bool = False
    state_dir: str = "~/.ventri/feishu"
    bot_open_id: str = ""                        # normally fetched from /bot/v3/info
    wait_for_runtime: float = 30.0


def _now() -> float:
    return time.time()


# ------------------------------------------------------------------ helpers
class BufferTerminal:
    """The CLI channel's terminal, collected into a buffer we send as a card."""

    tty = False

    def __init__(self) -> None:
        self.parts: list[str] = []

    async def readline(self, prompt: str) -> str | None:  # approvals use cards, never text
        return None

    def write(self, text: str) -> None:
        self.parts.append(text)

    def take(self) -> str:
        s = "".join(self.parts)
        self.parts.clear()
        return s


class JsonState:
    """A small JSON file written atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        self.data: dict[str, Any] = data if isinstance(data, dict) else {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            log.warning("feishu: cannot save %s: %s", self.path, e)


@dataclass
class Pending:
    req: ApprovalRequest
    owner: str
    chat_id: str
    event: anyio.Event = field(default_factory=anyio.Event)
    message_id: str | None = None
    choice: Choice = "deny"
    decider: str = ""
    started: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class Incoming:
    text: str
    ib: Inbound


# ------------------------------------------------------------- conversation
class Conversation(CliChannel):
    """One chat (or thread) <-> one session. Reuses the CLI channel's session
    handling and slash commands; output goes to cards instead of a terminal."""

    name = "feishu"

    def __init__(self, ch: FeishuChannel, key: str, ib: Inbound) -> None:
        cfg = ch.cfg
        super().__init__(ch.ctx, CliConfig(agent=cfg.agent, show_thinking=cfg.show_thinking,
                                           wait_for_runtime=cfg.wait_for_runtime), BufferTerminal())
        self.ch = ch
        self.key = key
        self.chat_id = ib.chat_id
        self.chat_type = ib.chat_type
        self.requester = ""
        self.reply_to: str | None = None
        self.in_thread = False
        self.progress: R.Progress | None = None
        self.tx, self.rx = anyio.create_memory_object_stream[Incoming](32)

    @property
    def buf(self) -> BufferTerminal:
        assert isinstance(self.term, BufferTerminal)
        return self.term

    def reply_target(self) -> str | None:
        return self.reply_to if (self.chat_type != "p2p" or self.in_thread) else None

    # ------------------------------------------------------------ session
    async def ensure_session(self) -> Session:
        s = self.session
        if s is not None and s.alive:
            return s
        mgr = await self.manager()
        sid = self.ch.sessions.data.get(self.key)
        if sid:
            path, _ = session_paths(mgr.dir, sid)
            if path.exists() and SessionLog.replay(path).state == "ended":
                sid = None
        self.session_id = sid
        s = await super().ensure_session()
        if self.ch.sessions.data.get(self.key) != s.id:
            self.ch.sessions.data[self.key] = s.id
            self.ch.sessions.save()
        return s

    def forget_session(self) -> None:
        self.session, self.session_id = None, None
        if self.ch.sessions.data.pop(self.key, None) is not None:
            self.ch.sessions.save()

    def unbind(self) -> None:
        if self._unbind:
            self._unbind()
            self._unbind = None

    # ------------------------------------------------------------- render
    async def render(self, ev: TurnEvent) -> None:
        p = self.progress
        if p is None:
            await super().render(ev)
            return
        k = ev.kind
        if k == "reasoning":
            p.thinking = True
            p.reasoning_chars += len(ev.text)
            if self.show_thinking:
                p.reasoning_tail = (p.reasoning_tail + ev.text)[-2000:]
            p.touch()
        elif k == "content":
            p.thinking = False
            p.content += ev.text
            p.touch()
        elif k == "tool.start":
            p.tool_start(ev.text)
        elif k == "tool.end":
            p.tool_end(bool(ev.data.get("ok")))
        elif k == "notice":
            p.notices.append(ev.text)
            p.touch()
        elif k == "error":
            p.notices.append("! " + ev.text)
            p.touch()

    async def ask(self, req: ApprovalRequest) -> Choice:
        return await self.ch.ask_approval(self, req)

    # ------------------------------------------------------------- inbound
    async def worker(self) -> None:
        async with self.rx:
            async for item in self.rx:
                try:
                    await self.process(item)
                except anyio.get_cancelled_exc_class():
                    raise
                except Exception as e:
                    log.exception("feishu: message %s failed", item.ib.message_id)
                    await self.ch.reply_markdown(self, f"⚠️ 出错了：`{type(e).__name__}: {e}`", template="red")

    async def process(self, item: Incoming) -> None:
        self.requester = item.ib.sender_open_id
        self.reply_to = item.ib.message_id
        self.in_thread = bool(item.ib.thread_id)
        text = item.text.strip()
        if text.startswith("/"):
            await self.slash(text)
        else:
            await self.run_turn(text)

    async def slash(self, line: str) -> None:
        cmd = line[1:].partition(" ")[0].lower()
        if cmd in ("new", "end", "exit", "quit", "q"):
            if self.session is None or not self.session.alive:
                sid = self.ch.sessions.data.get(self.key)
                if sid:
                    with contextlib.suppress(SessionError, TimeoutError):
                        await self.ensure_session()
            await self._end(quit=False)
            self.forget_session()
            out = self.buf.take().strip()
            await self.ch.reply_markdown(self, (_code(out) + "\n" if out else "") + "下一条消息将开启新会话。")
            return
        if cmd == "suspend":
            s = self.session
            if s is not None and s.alive:
                await s.suspend()
                await self.ch.reply_markdown(self, f"已挂起会话 `{s.id}`，下一条消息会自动恢复。")
            else:
                await self.ch.reply_markdown(self, "当前没有活动的会话。")
            self.session = None
            return
        if cmd == "help":
            await self.ch.reply_markdown(self, FEISHU_HELP)
            return
        if cmd == "id":
            await self.ch.reply_markdown(self, f"open_id：`{self.requester}`\nchat_id：`{self.chat_id}`\n"
                                                f"会话：`{self.ch.sessions.data.get(self.key, '（尚未开始）')}`")
            return
        if cmd == "retry":
            await self.run_turn(None)
            return
        await self.command(line)
        out = self.buf.take().strip()
        if out:
            await self.ch.reply_markdown(self, _code(out))

    # --------------------------------------------------------------- turns
    async def run_turn(self, text: str | None) -> None:
        """One turn (``text=None``: /retry) with a live progress card."""
        p = R.Progress()
        n0 = len(self.results)
        try:
            await self.ensure_session()
        except (SessionError, TimeoutError) as e:
            await self.ch.reply_markdown(self, f"⚠️ 无法打开会话：{e}", template="red")
            return
        p.header = self.buf.take().strip().strip("─").strip()
        self.progress = p
        mid = await self.ch.send_card(self, R.progress_card(p, show_thinking=self.show_thinking))
        state = {"mid": mid}
        err = ""
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self.ch.flush_progress, self, p, state)
                try:
                    if text is None:
                        s = await self.ensure_session()
                        self.results.append(await s.retry(self.render))
                    else:
                        await self._turn(text)
                except (SessionError, TimeoutError) as e:
                    err = f"会话不可用：{e}"
                except anyio.get_cancelled_exc_class():
                    raise
                except Exception as e:
                    log.exception("feishu: turn failed")
                    err = f"{type(e).__name__}: {e}"
                finally:
                    tg.cancel_scope.cancel()
        except anyio.get_cancelled_exc_class():
            with anyio.CancelScope(shield=True), anyio.move_on_after(5):
                await self.ch.finish(self, p, state.get("mid"), None, "⚠️ 服务已停止，本轮未完成")
            raise
        finally:
            self.progress = None
        r = self.results[-1] if len(self.results) > n0 else None
        extra = self.buf.take().strip()
        if r is not None and r.status == "error":
            extra = "会话完好，发送 /retry 重试。"
        await self.ch.finish(self, p, state.get("mid"), r, err, note=extra)


def _code(text: str) -> str:
    return "```\n" + text.replace("```", "'''") + "\n```"


# ------------------------------------------------------------------ channel
class FeishuChannel:
    def __init__(self, ctx: Any, cfg: FeishuConfig, *, active: bool, api: FeishuApi | None = None,
                 ws_factory: Any = None) -> None:
        self.ctx = ctx
        self.cfg = cfg
        self.active = active
        self.api: FeishuApi | None = api
        self.ws_factory = ws_factory
        self.allow_users = frozenset(cfg.allow_users)
        self.allow_chats = frozenset(cfg.allow_chats)
        self.bot_open_id = cfg.bot_open_id
        sd = expand(cfg.state_dir) / (cfg.app_id or "default")
        self.state_dir = sd
        self.sessions = JsonState(sd / "sessions.json")
        self.seen = JsonState(sd / "seen.json")
        self._seen_events: dict[str, float] = {}
        self._hinted: dict[str, float] = {}
        self.conversations: dict[str, Conversation] = {}
        self.pending: dict[str, Pending] = {}
        self.dropped: list[tuple[str, str]] = []      # (reason, message_id), for tests and /tree-style debugging
        self.link = "idle"
        self.ready = anyio.Event()
        self._key = secrets.token_bytes(32)
        self._tx, self._rx = anyio.create_memory_object_stream[Inbound](256)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tg: anyio.abc.TaskGroup | None = None
        self._runner: Any = None
        self._lock_fd: int | None = None
        self.reconnect_min = 5.0        # seconds; doubles up to 60 while the connection keeps failing
        self._prune_seen()

    # ---------------------------------------------------------- lifecycle
    def _secret(self) -> str:
        s = self.cfg.app_secret
        return s.reveal() if s is not None else os.environ.get("FEISHU_APP_SECRET", "")

    def check(self) -> None:
        if not self.cfg.app_id or not self._secret():
            raise ValueError("Feishu channel: app_id and app_secret are required "
                             "(app_secret: \"${secret:feishu_app_secret}\" or $FEISHU_APP_SECRET)")
        if self.api is None and not sdk_available():
            raise RuntimeError(INSTALL_HINT)
        if not self.allow_users:
            log.warning("feishu: allow_users is empty -- nobody can use the bot "
                        "(unknown users are told their open_id)")

    def acquire_lock(self) -> None:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - not POSIX
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.state_dir / "serve.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise RuntimeError(f"Feishu app {self.cfg.app_id} is already served by another process "
                               f"(lock {self.state_dir / 'serve.lock'}); Feishu would split the events "
                               "between the two connections") from None
        self._lock_fd = fd

    def release_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    async def run(self) -> None:
        if not self.active:
            log.info("feishu: idle (connects under `va serve`, or set run: always)")
            await anyio.sleep_forever()
        self._loop = asyncio.get_running_loop()
        try:
            async with anyio.create_task_group() as tg:
                self._tg = tg
                if self.api is None:
                    self.api = await anyio.to_thread.run_sync(self._make_api)
                tg.start_soon(self._hydrate_bot)
                tg.start_soon(self._supervise)
                self.ready.set()
                async with self._rx:
                    async for ib in self._rx:
                        try:
                            await self.handle(ib)
                        except Exception:  # one bad event must not take the channel down
                            log.exception("feishu: cannot handle message %s", ib.message_id)
        finally:
            self._tg = None
            for conv in self.conversations.values():
                conv.unbind()
            self.seen.save()

    def _make_api(self) -> FeishuApi:
        return LarkApi(load_sdk(), app_id=self.cfg.app_id, app_secret=self._secret(), domain=self.cfg.domain)

    async def _hydrate_bot(self) -> None:
        if self.bot_open_id:
            return
        delay = 2.0
        while not self.bot_open_id:
            try:
                assert self.api is not None
                info = await self.api.bot_info()
                self.bot_open_id = str(info.get("open_id") or "")
                if self.bot_open_id:
                    log.info("feishu: bot %s (%s)", info.get("app_name", ""), self.bot_open_id)
                    return
            except Exception as e:  # noqa: BLE001 - retried; group @-detection is fail-closed meanwhile
                log.warning("feishu: cannot fetch the bot identity (%s); retrying in %.0fs", e, delay)
            await anyio.sleep(delay)
            delay = min(delay * 2, 120.0)

    async def _supervise(self) -> None:
        loop = asyncio.get_running_loop()
        factory = self.ws_factory or WsRunner
        failures = 0
        while True:
            exited = anyio.Event()
            box: list[BaseException | None] = []

            def on_exit(err: BaseException | None, _box: list[BaseException | None] = box,
                        _ev: anyio.Event = exited) -> None:
                def done() -> None:
                    _box.append(err)
                    _ev.set()
                with contextlib.suppress(RuntimeError):  # loop already closed
                    loop.call_soon_threadsafe(done)

            runner = factory(app_id=self.cfg.app_id, app_secret=self._secret(), domain=self.cfg.domain,
                             callbacks=WsCallbacks(message=self.on_message, card_action=self._card_from_thread,
                                                   link_up=self._link_up, link_lost=self._link_lost),
                             on_exit=on_exit)
            self._runner = runner
            t0 = time.monotonic()
            try:
                self.link = "connecting"
                runner.start()
                await exited.wait()
            finally:
                self._runner = None
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(runner.stop)
            err = box[0] if box else None
            self.link = "down"
            failures = 0 if time.monotonic() - t0 > 120 else failures + 1
            auth = "auth" in repr(err).lower() or type(err).__name__ == "ClientException"
            wait = self.reconnect_min * (12 if auth else min(2 ** (failures - 1), 12))
            log.warning("feishu: long connection ended (%r); reconnecting in %.0fs", err, wait)
            self.ctx.trace("feishu.link", state="down", error=repr(err))
            await anyio.sleep(wait)

    def _link_up(self) -> None:
        self._call_soon(self._set_link, "up")

    def _link_lost(self, reason: str) -> None:
        self._call_soon(self._set_link, "reconnecting")

    def _set_link(self, state: str) -> None:
        if state != self.link:
            log.info("feishu: link %s", state)
            self.ctx.trace("feishu.link", state=state)
        self.link = state

    def _call_soon(self, fn: Any, *args: Any) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            if _on_loop(loop):
                fn(*args)
            else:
                loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:  # loop closed during shutdown
            pass

    async def dispose(self) -> None:
        r = self._runner
        if r is not None:
            self._runner = None
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(r.stop)
        for conv in self.conversations.values():
            conv.unbind()
        self.release_lock()

    # ------------------------------------------------------------ inbound
    def on_message(self, ib: Inbound) -> None:
        """Thread-safe: called from the websocket thread (or a test)."""
        self._call_soon(self._enqueue, ib)

    def _enqueue(self, ib: Inbound) -> None:
        try:
            self._tx.send_nowait(ib)
        except anyio.WouldBlock:
            log.warning("feishu: inbound queue full, dropping message %s", ib.message_id)
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            pass

    def _drop(self, reason: str, ib: Inbound, level: int = logging.DEBUG) -> None:
        self.dropped.append((reason, ib.message_id))
        del self.dropped[:-200]
        log.log(level, "feishu: ignored %s (%s, chat %s, sender %s)", ib.message_id, reason, ib.chat_id,
                ib.sender_open_id)

    def _prune_seen(self) -> None:
        cutoff = _now() - SEEN_TTL
        items = sorted(((k, v) for k, v in self.seen.data.items()
                        if isinstance(v, (int, float)) and v >= cutoff), key=lambda kv: kv[1])
        self.seen.data = dict(items[-SEEN_CAP:])

    def _is_duplicate(self, ib: Inbound) -> bool:
        now = _now()
        if ib.event_id:
            if ib.event_id in self._seen_events:
                return True
            self._seen_events[ib.event_id] = now
            if len(self._seen_events) > SEEN_CAP:
                for k in list(self._seen_events)[: len(self._seen_events) - SEEN_CAP]:
                    del self._seen_events[k]
        if not ib.message_id:
            return False
        if ib.message_id in self.seen.data:
            return True
        self.seen.data[ib.message_id] = now
        if len(self.seen.data) > SEEN_CAP + 64:
            self._prune_seen()
        self.seen.save()
        return False

    async def handle(self, ib: Inbound) -> None:
        """Filter one inbound message and queue it on its conversation."""
        if self._is_duplicate(ib):
            return self._drop("duplicate", ib)
        if ib.sender_type != "user":
            return self._drop("not-a-user", ib)
        if self.bot_open_id and ib.sender_open_id == self.bot_open_id:
            return self._drop("self", ib)
        if ib.create_time_ms and _now() - ib.create_time_ms / 1000 > self.cfg.max_message_age:
            return self._drop("stale", ib, logging.INFO)
        mentions = [replace(m, is_self=bool(self.bot_open_id) and m.open_id == self.bot_open_id,
                            is_all=m.key == "@_all") for m in ib.mentions]
        if ib.chat_type == "p2p":
            if ib.sender_open_id not in self.allow_users:
                await self._hint_unknown(ib, "user")
                return self._drop("user-not-allowed", ib, logging.WARNING)
        else:
            if self.cfg.require_mention:
                if not self.bot_open_id:
                    return self._drop("bot-identity-unknown", ib, logging.WARNING)
                if not any(m.is_self for m in mentions):
                    return self._drop("no-mention", ib)
            if ib.chat_id not in self.allow_chats:
                await self._hint_unknown(ib, "chat")
                return self._drop("chat-not-allowed", ib, logging.WARNING)
            if ib.sender_open_id not in self.allow_users:
                await self._hint_unknown(ib, "user")
                return self._drop("user-not-allowed", ib, logging.WARNING)
        text = self.extract_text(ib, mentions)
        if text is None:
            await self.reply_to_message(ib, f"暂不支持 `{ib.message_type}` 类型的消息，请发送文字。")
            return self._drop("unsupported-type", ib)
        if not text.strip():
            return self._drop("empty", ib)
        conv = self._conversation(ib)
        try:
            conv.tx.send_nowait(Incoming(text, ib))
        except anyio.WouldBlock:
            await self.reply_to_message(ib, "上一条消息还在处理，队列已满，请稍后再发。")
            self._drop("busy", ib, logging.WARNING)

    def extract_text(self, ib: Inbound, mentions: list[MentionRef]) -> str | None:
        if ib.message_type not in SUPPORTED_TYPES:
            return None
        payload = load_content(ib.content)
        mm = mentions_map(mentions)
        if ib.message_type == "text":
            text = normalize_text(str(payload.get("text", "")), mm)
        else:
            text = parse_post(payload, mm)
        return strip_edge_self_mentions(text, mentions)

    def session_key(self, ib: Inbound) -> str:
        if ib.chat_type == "p2p":
            return f"p2p:{ib.chat_id}"
        if self.cfg.group_session == "thread" and ib.thread_id:
            return f"group:{ib.chat_id}:{ib.thread_id}"
        return f"group:{ib.chat_id}"

    def _conversation(self, ib: Inbound) -> Conversation:
        key = self.session_key(ib)
        conv = self.conversations.get(key)
        if conv is None:
            conv = self.conversations[key] = Conversation(self, key, ib)
            assert self._tg is not None
            self._tg.start_soon(conv.worker)
        return conv

    async def _hint_unknown(self, ib: Inbound, what: str) -> None:
        ident = ib.sender_open_id if what == "user" else ib.chat_id
        log.warning("feishu: message from a %s not in the allowlist: %s", what, ident)
        self.ctx.trace("feishu.unknown", what=what, id=ident)
        if not self.cfg.reply_unknown:
            return
        now = time.monotonic()
        if now - self._hinted.get(ident, -1e9) < UNKNOWN_HINT_INTERVAL:
            return
        self._hinted[ident] = now
        if what == "user":
            text = (f"你还没有被授权使用这个机器人。你的 open_id 是：{ident}\n"
                    "请让机器人的主人把它加入 ventri.yml 里飞书渠道的 allow_users。")
        else:
            text = (f"这个群还没有被授权。群 chat_id 是：{ident}\n"
                    "请让机器人的主人把它加入 ventri.yml 里飞书渠道的 allow_chats。")
        await self.reply_to_message(ib, text, text_only=True)

    # ----------------------------------------------------------- outbound
    async def reply_to_message(self, ib: Inbound, text: str, *, text_only: bool = False) -> None:
        assert self.api is not None
        reply = ib.message_id if ib.chat_type != "p2p" else None
        try:
            if text_only:
                await self.api.send_text(ib.chat_id, text, reply_to=reply, in_thread=bool(ib.thread_id))
            else:
                await self.api.send_card(ib.chat_id, R.card_json(R.markdown_card(text)), reply_to=reply,
                                         in_thread=bool(ib.thread_id))
        except Exception as e:  # noqa: BLE001
            log.warning("feishu: reply failed: %s", e)

    async def send_card(self, conv: Conversation, card: dict[str, Any]) -> str | None:
        assert self.api is not None
        try:
            return await self.api.send_card(conv.chat_id, R.card_json(card), reply_to=conv.reply_target(),
                                            in_thread=conv.in_thread)
        except Exception as e:  # noqa: BLE001
            log.warning("feishu: sending a card failed: %s", e)
            return None

    async def send_text(self, conv: Conversation, text: str) -> None:
        assert self.api is not None
        for piece in R.chunk_markdown(text, 30_000) or [" "]:
            try:
                await self.api.send_text(conv.chat_id, piece, reply_to=conv.reply_target(), in_thread=conv.in_thread)
            except Exception as e:  # noqa: BLE001
                log.warning("feishu: sending text failed: %s", e)
                return

    async def reply_markdown(self, conv: Conversation, md: str, *, template: str = "blue",
                             title: str | None = None) -> None:
        chunks = R.chunk_markdown(md) or [" "]
        for piece in chunks:
            c = R.markdown_card(piece, title=title, template=template)
            if not R.fits(c) or await self.send_card(conv, c) is None:
                await self.send_text(conv, piece)

    async def patch(self, mid: str, card: dict[str, Any]) -> bool:
        assert self.api is not None
        try:
            await self.api.patch_card(mid, R.card_json(card))
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("feishu: updating card %s failed: %s", mid, e)
            return False

    async def flush_progress(self, conv: Conversation, p: R.Progress, state: dict[str, Any]) -> None:
        """Patch the progress card when it changed (and its timer every ~10 s)."""
        last_v, last_t = p.version, time.monotonic()
        interval = self.cfg.progress_interval
        while True:
            await anyio.sleep(interval)
            mid = state.get("mid")
            if not mid:
                return
            now = time.monotonic()
            if p.version == last_v and now - last_t < 10:
                continue
            last_v, last_t = p.version, now
            c = R.progress_card(p, show_thinking=conv.show_thinking)
            if not R.fits(c):
                continue
            try:
                assert self.api is not None
                await self.api.patch_card(mid, R.card_json(c))
                interval = self.cfg.progress_interval
            except FeishuApiError as e:
                if e.code in (230020, 99991400):
                    interval = min(interval * 2, 10.0)
                else:
                    log.warning("feishu: progress card %s not updatable (%s); final answer as a new card", mid, e)
                    state["mid"] = None
                    return
            except Exception as e:  # noqa: BLE001
                log.warning("feishu: progress update failed: %s", e)

    async def finish(self, conv: Conversation, p: R.Progress, mid: str | None, r: TurnResult | None,
                     error: str = "", *, note: str = "") -> None:
        if r is not None:
            answer = r.text or ("" if r.status == "ok" else p.content)
            status = r.status
            rate = conv._rate()
            footer = (f"turn {r.n} · {r.steps} 步 · {r.tool_calls} 次工具 · {usage_line(r.usage, r.cost_usd, rate)}")
            if r.status != "ok":
                error = error or f"{r.status}: {r.reason}"
        else:
            answer, status, footer = p.content, "error", ""
        if note:
            footer = f"{footer}\n{note}" if footer else note
        chunks = R.chunk_markdown(answer) or [""]
        first = R.final_card(p, chunks[0], status=status if not error or status != "ok" else "error",
                             footer=footer, error=error, more=len(chunks) - 1)
        placed = False
        if R.fits(first):
            if mid:
                placed = await self.patch(mid, first)
            if not placed:
                placed = await self.send_card(conv, first) is not None
        if not placed:
            await self.send_text(conv, "\n\n".join(x for x in (chunks[0], error, footer) if x))
        for piece in chunks[1:]:
            await self.reply_markdown(conv, piece)

    # ----------------------------------------------------------- approvals
    def _sig(self, rid: str, choice: str, owner: str, chat_id: str) -> str:
        return hmac.new(self._key, f"{rid}|{choice}|{owner}|{chat_id}".encode(), hashlib.sha256).hexdigest()

    def _audit(self, **rec: Any) -> None:
        audit = self.ctx.get(AuditLog, None)
        if audit is not None:
            audit.write(**rec)

    async def ask_approval(self, conv: Conversation, req: ApprovalRequest) -> Choice:
        owner = conv.requester
        base = {"event": "feishu.approval", "request": req.id, "session": req.session_id, "tool": req.tool,
                "chat": conv.chat_id, "owner": owner}
        if not owner or owner not in self.allow_users:
            self._audit(**base, outcome="deny", reason="no allowlisted requester")
            return "deny"
        choices = ["once", "session", "deny"] if req.grantable else ["once", "deny"]
        values = {c: {"va": "approval", "rid": req.id, "c": c, "sig": self._sig(req.id, c, owner, conv.chat_id)}
                  for c in choices}
        card = R.approval_card(summary=req.summary, risk=req.risk, args_preview=req.args_preview,
                               grantable=req.grantable, timeout=req.timeout, values=values)
        pend = Pending(req, owner, conv.chat_id)
        self.pending[req.id] = pend
        p = conv.progress
        if p is not None:
            p.waiting = f"{req.tool} {req.summary}"[:120]
            p.touch()
        outcome: str | None = None
        try:
            pend.message_id = await self.send_card(conv, card)
            if pend.message_id is None:
                self._audit(**base, outcome="deny", reason="approval card could not be sent")
                outcome = "deny"
                return "deny"
            await pend.event.wait()
            outcome = pend.choice
            self._audit(**base, outcome=pend.choice, operator=pend.decider)
            return pend.choice
        finally:
            self.pending.pop(req.id, None)
            if p is not None:
                p.waiting = ""
                p.touch()
            if pend.message_id:
                if outcome is None:
                    timed_out = time.monotonic() - pend.started >= req.timeout - 0.5
                    label = "超时，已拒绝" if timed_out else "已取消，已拒绝"
                    self._audit(**base, outcome="deny", reason="timeout" if timed_out else "cancelled")
                    done = R.approval_result_card(summary=req.summary, risk=req.risk, outcome=label)
                else:
                    done = R.approval_result_card(summary=req.summary, risk=req.risk, outcome=outcome)
                with anyio.CancelScope(shield=True), anyio.move_on_after(5):
                    await self.patch(pend.message_id, done)

    def _card_from_thread(self, act: CardAction) -> tuple[str, str]:
        loop = self._loop
        if loop is None:
            return "error", "服务尚未就绪"
        return bridge_card_action(loop, self.on_card_action, act)

    def on_card_action(self, act: CardAction) -> tuple[str, str]:
        """Validate a button click (runs on the kernel loop); returns the toast."""
        v = act.value
        if v.get("va") != "approval":
            return "info", "这张卡片没有可执行的操作"
        rid, choice, sig = str(v.get("rid", "")), str(v.get("c", "")), str(v.get("sig", ""))
        pend = self.pending.get(rid)
        reason = ""
        if pend is None:
            reason = "该审批已失效（已处理、超时或服务已重启）"
        elif choice not in ("once", "session", "deny") or (choice == "session" and not pend.req.grantable):
            reason = "无效的选项"
        elif not hmac.compare_digest(sig, self._sig(rid, choice, pend.owner, pend.chat_id)):
            reason = "签名无效"
        elif act.operator_open_id != pend.owner or act.operator_open_id not in self.allow_users:
            reason = "只有发起这次请求的人可以审批"
        elif act.chat_id != pend.chat_id or (pend.message_id is not None and act.message_id != pend.message_id):
            reason = "卡片与审批请求不匹配"
        elif pend.event.is_set():
            reason = "已经处理过了"
        if reason:
            self._audit(event="feishu.approval.rejected", request=rid, choice=choice,
                        operator=act.operator_open_id, chat=act.chat_id, reason=reason)
            self.ctx.trace("feishu.approval.rejected", request=rid, reason=reason)
            return "error", reason
        assert pend is not None
        pend.choice = choice  # type: ignore[assignment]
        pend.decider = act.operator_open_id
        pend.event.set()
        return "success", R.CHOICE_LABEL[choice]


def _on_loop(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


@ventri.plugin(name="channel:feishu", config=FeishuConfig, exclusive=True)
def feishu(ctx: Any, cfg: FeishuConfig) -> None:
    """``use: ventri_agent.channels.feishu`` -- Feishu / Lark bot over the long
    connection. Services ``"feishu.api"`` / ``"feishu.ws"`` (tests) replace the
    REST client and the websocket runner factory."""
    hub = ctx.get(SERVE_KEY, None)
    active = cfg.run == "always" or hub is not None
    ch = FeishuChannel(ctx, cfg, active=active, api=ctx.get("feishu.api", None),
                       ws_factory=ctx.get("feishu.ws", None))
    if active:
        ch.check()
        ch.acquire_lock()
        ctx.on_dispose(ch.release_lock)
        if hub is not None:
            ctx.on_dispose(hub.attach(ch))
        ctx.on_dispose(ch.dispose)
    ctx.spawn(ch.run, name="feishu")


plugin = feishu
