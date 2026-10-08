"""``va feishu setup``: create a Feishu / Lark bot by scanning a QR code and
wire it into ``ventri.yml``.

init -> begin -> show the QR code (terminal, ``--qr-png``, URL; ``--json``
emits JSON lines for automation) -> poll until the user confirms in the
Feishu / Lark app -> probe the bot -> store the App Secret (macOS keychain,
else a 0600 file under ``$VENTRI_HOME/secrets``) -> enable the channel in
``ventri.yml`` with ``allow_users: [<the scanner's open_id>]`` (a backup of
the file is written first; comments elsewhere are preserved). The secret is
never printed, logged or written to the configuration.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import yaml

from ._onboard import MINIMAL_ADDONS, Credentials, Registrar, RegistrationError

FEISHU_USE = "ventri_agent.channels.feishu"
SECRET_NAME = "feishu_app_secret"
SECRET_REF = "${secret:" + SECRET_NAME + "}"
_ENV_NAME = "VENTRI_SECRET_FEISHU_APP_SECRET"
_SECURITY = "/usr/bin/security"
_SAFE_SECRET = re.compile(r"^[A-Za-z0-9_.\-]{8,256}$")


class SetupError(Exception):
    pass


# ------------------------------------------------------------------- QR
def qr_matrix(url: str) -> list[list[bool]] | None:
    """Module matrix (True = dark) with a 4-module quiet zone, or None without ``qrcode``."""
    try:
        import qrcode  # type: ignore[import-not-found, import-untyped]  # optional (feishu extra)
    except ImportError:
        return None
    from qrcode.constants import ERROR_CORRECT_M  # type: ignore[import-not-found, import-untyped]

    qr = qrcode.QRCode(border=4, error_correction=ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    return [[bool(c) for c in row] for row in qr.get_matrix()]


def qr_text(m: list[list[bool]]) -> str:
    """Half-block rendering, light modules drawn (for dark terminal backgrounds)."""
    rows = [*m, [False] * len(m[0])] if len(m) % 2 else m
    glyph = {(False, False): "█", (False, True): "▀", (True, False): "▄", (True, True): " "}
    return "\n".join("".join(glyph[(top, bot)] for top, bot in zip(rows[i], rows[i + 1], strict=True))
                     for i in range(0, len(rows), 2))


def write_png(m: list[list[bool]], path: Path, scale: int = 8) -> None:
    """A grayscale PNG of the matrix (no imaging library needed). Mode 0600:
    whoever scans the code first creates the app and becomes its allowed user."""
    raw = bytearray()
    for row in m:
        line = bytes(0 if dark else 255 for dark in row for _ in range(scale))
        for _ in range(scale):
            raw += b"\x00" + line
    w, h = len(m[0]) * scale, len(m) * scale

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(png)
    os.chmod(path, 0o600)


# --------------------------------------------------------------- secrets
@dataclass(frozen=True)
class StoredSecret:
    kind: str          # keychain | file
    where: str


def _keychain_store(value: str) -> bool:
    if sys.platform != "darwin" or not Path(_SECURITY).exists():
        return False
    # `security -i` reads the command from stdin, so the secret never appears in
    # the process list (argv); store_secret() has already restricted its charset.
    cmd = f'add-generic-password -U -s ventri -a {SECRET_NAME} -w "{value}"\n'
    try:
        r = subprocess.run([_SECURITY, "-i"], input=cmd, text=True, capture_output=True, timeout=20,
                           check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if r.returncode != 0:
        return False
    from ventri_std.config import KeychainSecrets

    return KeychainSecrets().get(SECRET_NAME) == value


def store_secret(value: str, how: str = "auto") -> StoredSecret:
    """``auto``: keychain on macOS when it works, else the 0600 file."""
    from ventri_std.config import FileSecrets

    if not _SAFE_SECRET.match(value):
        raise SetupError("the App Secret returned by Feishu has an unexpected format; not stored")
    if how in ("auto", "keychain"):
        if _keychain_store(value):
            return StoredSecret("keychain", f"macOS keychain (service 'ventri', account '{SECRET_NAME}')")
        if how == "keychain":
            raise SetupError("could not store the secret in the macOS keychain (use --secret-store file)")
    p = FileSecrets().set(SECRET_NAME, value)
    return StoredSecret("file", str(p))


def secret_configured() -> bool:
    from ventri_std.config import ConfigError, default_secrets

    try:
        return default_secrets().get(SECRET_NAME) is not None
    except ConfigError:
        return False


# ---------------------------------------------------------------- config
def _feishu_items(plugins: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in plugins or []:
        if not isinstance(it, dict):
            continue
        if str(it.get("use", "")) == FEISHU_USE:
            out.append(it)
        if "group" in it:
            out.extend(_feishu_items(it.get("plugins")))
    return out


def current_feishu(path: Path) -> dict[str, Any] | None:
    """The config mapping of the (single) enabled Feishu block, if any."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    items = [it for it in _feishu_items(data.get("plugins")) if not it.get("disabled")]
    if len(items) != 1:
        return None
    cfg = items[0].get("config")
    return cfg if isinstance(cfg, dict) else {}


def _scalar(v: Any) -> str:
    if isinstance(v, str) and "${" in v:
        return json.dumps(v)
    s = yaml.safe_dump(v, default_flow_style=True, allow_unicode=True, width=10_000).strip()
    return s.removesuffix("\n...").removesuffix("...").strip()


def render_block(cfg: dict[str, Any], *, indent: str = "  ", block_id: str = "feishu") -> list[str]:
    notes = {"allow_users": "# open_ids allowed to use the bot (the account that scanned the QR code)",
             "allow_chats": "# group chat_ids (oc_...); @ the bot in a group to learn its id",
             "app_secret": "# the secret itself is in the keychain / ~/.ventri/secrets",
             "require_mention": "# groups: only messages that @ the bot"}
    lines = [f"{indent}- use: {FEISHU_USE}", f"{indent}  id: {block_id}", f"{indent}  config:"]
    for k, v in cfg.items():
        line = f"{indent}    {k}: {_scalar(v)}"
        if k in notes:
            line = f"{line:<52} {notes[k]}"
        lines.append(line)
    return lines


def merged_config(old: dict[str, Any] | None, creds: Credentials) -> dict[str, Any]:
    """New app: credentials, domain and the scanner as the only allowed user
    (open_ids are per app, old ones are meaningless); other settings kept."""
    out: dict[str, Any] = {"app_id": creds.app_id, "app_secret": SECRET_REF, "domain": creds.domain,
                           "allow_users": [creds.open_id] if creds.open_id else [],
                           "allow_chats": [], "require_mention": True}
    for k, v in (old or {}).items():
        if k not in ("app_id", "app_secret", "domain", "allow_users", "bot_open_id"):
            out[k] = v
    return out


_ACTIVE = re.compile(r"^(?P<ind>[ \t]*)- use:\s*['\"]?ventri_agent\.channels\.feishu['\"]?\s*(#.*)?$")
_COMMENTED = re.compile(r"^(?P<ind>[ \t]*)#\s?- use:\s*ventri_agent\.channels\.feishu\s*(#.*)?$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _block_end(lines: list[str], start: int) -> int:
    base = _indent(lines[start])
    end = start + 1
    while end < len(lines):
        s = lines[end].strip()
        if s and not s.startswith("#") and _indent(lines[end]) <= base:
            break
        end += 1
    while end > start + 1 and (not lines[end - 1].strip() or (lines[end - 1].strip().startswith("#")
                                                                and _indent(lines[end - 1]) <= base)):
        end -= 1
    return end


def plan_config(text: str, creds: Credentials) -> tuple[str, str]:
    """``(new_text, how)``; ``how``: replaced | uncommented | appended | rewritten."""
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise SetupError("ventri.yml is not a mapping")
    items = _feishu_items(data.get("plugins"))
    if len(items) > 1:
        raise SetupError("ventri.yml has several Feishu channel blocks; keep one and re-run")
    lines = text.splitlines()
    old = items[0] if items else None
    old_cfg = old.get("config") if old and isinstance(old.get("config"), dict) else None
    block_id = str(old.get("id", "feishu")) if old else "feishu"
    new_cfg = merged_config(old_cfg, creds)
    active = [i for i, ln in enumerate(lines) if _ACTIVE.match(ln)]
    if old is not None and len(active) == 1:
        i = active[0]
        end = _block_end(lines, i)
        ind = _ACTIVE.match(lines[i]).group("ind")  # type: ignore[union-attr]
        lines[i:end] = render_block(new_cfg, indent=ind, block_id=block_id)
        how = "replaced"
    elif old is not None:  # unusual layout (flow style, `use` not first, ...): rewrite the document
        old.clear()
        old.update({"use": FEISHU_USE, "id": block_id, "config": new_cfg})
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False), "rewritten"
    else:
        commented = [i for i, ln in enumerate(lines) if _COMMENTED.match(ln)]
        if commented:
            i = commented[0]
            ind = _COMMENTED.match(lines[i]).group("ind")  # type: ignore[union-attr]
            end = i + 1
            while end < len(lines) and re.match(rf"^{re.escape(ind)}#\s{{2,}}\S", lines[end]):
                end += 1
            lines[i:end] = render_block(new_cfg, indent=ind)
            how = "uncommented"
        else:
            top = [i for i, ln in enumerate(lines) if re.match(r"^plugins:\s*(#.*)?$", ln)]
            if len(top) != 1 or not isinstance(data.get("plugins"), list):
                raise SetupError("ventri.yml has no block-style `plugins:` list to add the channel to")
            i = top[0] + 1
            item_ind = "  "
            while i < len(lines):
                ln = lines[i]
                if ln.strip() and not ln.lstrip().startswith("#") and _indent(ln) == 0:
                    break
                if ln.lstrip().startswith("- ") and item_ind == "  " and i > top[0]:
                    item_ind = ln[: _indent(ln)] or "  "
                i += 1
            while i > top[0] + 1 and not lines[i - 1].strip():
                i -= 1
            lines[i:i] = render_block(new_cfg, indent=item_ind)
            how = "appended"
    return "\n".join(lines) + "\n", how


def write_config(path: Path, creds: Credentials) -> tuple[Path, str]:
    """Edit ``ventri.yml`` (backup first). Returns ``(backup, how)``."""
    text = path.read_text(encoding="utf-8")
    new, how = plan_config(text, creds)
    check = current_feishu_text(new)
    if check is None or check.get("app_id") != creds.app_id:
        raise SetupError("could not edit ventri.yml safely; add the Feishu block by hand (docs/feishu-setup.md)")
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    n = 1
    while backup.exists():
        n += 1
        backup = path.with_name(f"{path.name}.bak-{stamp}-{n}")
    shutil.copy2(path, backup)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(new, encoding="utf-8")
    shutil.copymode(path, tmp)
    os.replace(tmp, path)
    return backup, how


def current_feishu_text(text: str) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return None
    items = [it for it in _feishu_items(data.get("plugins")) if not it.get("disabled")]
    if len(items) != 1 or not isinstance(items[0].get("config"), dict):
        return None
    return items[0]["config"]


# ------------------------------------------------------------------ flow
@dataclass
class Options:
    config: Path
    domain: str = "feishu"
    qr_png: Path | None = None
    json: bool = False
    minimal: bool = False
    secret_store: str = "auto"
    force: bool = False
    timeout: float | None = None


class Reporter:
    def __init__(self, opts: Options, out: TextIO, err: TextIO) -> None:
        self.opts, self.out, self.err = opts, out, err

    def say(self, text: str = "") -> None:
        (self.err if self.opts.json else self.out).write(text + "\n")
        (self.err if self.opts.json else self.out).flush()

    def event(self, kind: str, **data: Any) -> None:
        if self.opts.json:
            self.out.write(json.dumps({"event": kind, **data}, ensure_ascii=False) + "\n")
            self.out.flush()


def run_setup(opts: Options, *, out: TextIO | None = None, err: TextIO | None = None,
              registrar: Registrar | None = None) -> int:
    rep = Reporter(opts, out or sys.stdout, err or sys.stderr)
    cfg_path = opts.config
    if not cfg_path.exists():
        rep.say(f"no configuration at {cfg_path}; run `va init` first")
        rep.event("error", error="no-config", message=f"no configuration at {cfg_path}")
        return 2
    existing = current_feishu(cfg_path)
    if existing and existing.get("app_id") and secret_configured() and not opts.force:
        rep.say(f"Feishu is already set up in {cfg_path} (app {existing.get('app_id')}); "
                "`va serve` runs it. Use --force to create a new bot.")
        rep.event("done", already=True, app_id=existing.get("app_id"), domain=existing.get("domain", "feishu"),
                  allow_users=existing.get("allow_users", []), config=str(cfg_path))
        return 0
    if opts.qr_png is not None and qr_matrix("probe") is None:
        rep.say("--qr-png needs the qrcode package: pip install 'ventri-agent[feishu]'")
        rep.event("error", error="no-qrcode", message="--qr-png needs the qrcode package")
        return 2
    reg = registrar or Registrar()
    try:
        return _flow(opts, rep, reg)
    finally:
        if registrar is None:
            reg.close()


def _flow(opts: Options, rep: Reporter, reg: Registrar) -> int:
    rep.say("Feishu / Lark: create a bot by scanning a QR code")
    try:
        reg.init(opts.domain)
        begin = reg.begin(opts.domain, addons=MINIMAL_ADDONS if opts.minimal else None)
    except RegistrationError as e:
        rep.say(f"  could not start the registration: {e}")
        rep.event("error", error=e.code, message=e.message)
        return 1
    matrix = qr_matrix(begin.qr_url)
    png = None
    if opts.qr_png is not None and matrix is not None:
        write_png(matrix, opts.qr_png)
        png = str(opts.qr_png.expanduser())
    rep.event("qr", url=begin.qr_url, expire_in=begin.expire_in, png=png, user_code=begin.user_code)
    if matrix is not None:
        rep.say(qr_text(matrix))
    rep.say(f"  Scan with the Feishu / Lark app (扫一扫), or open on your phone:\n  {begin.qr_url}")
    if png:
        rep.say(f"  QR code saved to {png}")
    if matrix is None:
        rep.say("  (pip install 'ventri-agent[feishu]' to show a scannable QR code here)")
    rep.say(f"  The code is single-use and expires in {begin.expire_in // 60} min. Whoever scans it creates the "
            "app and becomes its only allowed user: only scan it yourself.")
    rep.say("  Waiting for confirmation…")
    try:
        creds = reg.poll(begin, domain=opts.domain, timeout=opts.timeout,
                         on_status=lambda s: rep.event("status", status=s))
    except RegistrationError as e:
        why = {"access_denied": "the registration was declined in the app",
               "expired_token": "the QR code expired before it was confirmed"}.get(e.code, str(e))
        rep.say(f"  not completed: {why}. Run `va feishu setup` again for a new code.")
        rep.event("error", error=e.code, message=why)
        return 1
    bot = reg.probe_bot(creds) or {}
    try:
        stored = store_secret(creds.app_secret, opts.secret_store)
        backup, how = write_config(opts.config, creds)
    except SetupError as e:
        rep.say(f"  setup failed: {e}")
        rep.say(f"  The app {creds.app_id} exists; delete it in the developer console or re-run with --force.")
        rep.event("error", error="setup", message=str(e), app_id=creds.app_id)
        return 1
    rep.say(f"  ✓ app {creds.app_id} created ({creds.domain}" + (f", bot “{bot['app_name']}”" if bot.get("app_name")
                                                                 else "") + ")")
    rep.say(f"  ✓ App Secret stored in {stored.where} (not shown)")
    if os.environ.get(_ENV_NAME) and stored.kind == "file":
        rep.say(f"  ! ${_ENV_NAME} is set and takes precedence over the file; unset it")
    rep.say(f"  ✓ {opts.config}: Feishu channel {'enabled' if how != 'replaced' else 'updated'}; only "
            f"{creds.open_id or '(no open_id returned: message the bot to learn yours)'} may use it "
            f"(backup: {backup})")
    from .transport import sdk_available

    rep.say("  Next: " + ("" if sdk_available() else "uv sync --extra feishu (or pip install 'ventri-agent[feishu]'), then ")
            + "va serve   — then message the bot in Feishu.")
    rep.say("  If approval buttons fail with error 200340, enable the card callback over the long connection and "
            "publish a version in the developer console (docs/feishu-setup.md).")
    rep.event("done", already=False, app_id=creds.app_id, domain=creds.domain, open_id=creds.open_id,
              bot_name=bot.get("app_name", ""), bot_open_id=bot.get("open_id", ""), secret_store=stored.kind,
              secret_location=stored.where, config=str(opts.config), backup=str(backup), edit=how)
    return 0
