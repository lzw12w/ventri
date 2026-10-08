"""`va feishu setup` (scan-to-create), fully offline: every HTTP call goes to a
scripted httpx.MockTransport, sleeps are virtual, the keychain is faked."""
from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import stat
import struct
import zlib
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import yaml

from ventri_agent import cli
from ventri_agent.channels.feishu import _onboard, wizard
from ventri_agent.channels.feishu._onboard import Credentials, Registrar, RegistrationError
from ventri_std import config as cfgmod
from ventri_std.config import load_document

SECRET = "FAKEsecretFAKEsecretFAKEsecret42"
SECRET2 = "OTHERsecretOTHERsecretOTHER0099"


class FakeFeishu:
    """The registration endpoint + the two open-apis calls of the bot probe."""

    def __init__(self, polls: list, *, app_id: str = "cli_a1b2c3", secret: str = SECRET,
                 open_id: str = "ou_scanner", methods=("client_secret",)) -> None:
        self.polls = list(polls)
        self.app_id, self.secret, self.open_id, self.methods = app_id, secret, open_id, list(methods)
        self.calls: list[tuple[str, str, dict]] = []   # (host, action/path, body)

    def ok(self, brand: str = "feishu") -> dict:
        return {"client_id": self.app_id, "client_secret": self.secret,
                "user_info": {"open_id": self.open_id, "tenant_brand": brand}}

    def __call__(self, req: httpx.Request) -> httpx.Response:
        host = req.url.host
        if req.url.path == "/oauth/v1/app/registration":
            body = {k: v[0] for k, v in parse_qs(req.content.decode()).items()}
            self.calls.append((host, body["action"], body))
            if body["action"] == "init":
                return httpx.Response(200, json={"supported_auth_methods": self.methods})
            if body["action"] == "begin":
                return httpx.Response(200, json={
                    "device_code": "dev-123", "user_code": "ABCD",
                    "verification_uri_complete": f"https://{host}/page/cli?user_code=ABCD",
                    "interval": 5, "expire_in": 600})
            step = self.polls.pop(0) if self.polls else {"error": "authorization_pending"}
            if isinstance(step, Exception):
                raise step
            if isinstance(step, httpx.Response):
                return step
            if step == "ok":
                return httpx.Response(200, json=self.ok())
            if step == "ok-lark":
                return httpx.Response(200, json=self.ok("lark"))
            status = 400 if step.get("error") else 200
            return httpx.Response(status, json=step)
        self.calls.append((host, req.url.path, {}))
        if req.url.path.endswith("/tenant_access_token/internal"):
            sent = json.loads(req.content)
            good = sent == {"app_id": self.app_id, "app_secret": self.secret}
            return httpx.Response(200, json={"code": 0, "tenant_access_token": "t-tok"} if good else {"code": 10014})
        if req.url.path == "/open-apis/bot/v3/info":
            assert req.headers["Authorization"] == "Bearer t-tok"
            return httpx.Response(200, json={"code": 0, "bot": {"app_name": "Jeff's agent", "open_id": "ou_bot"}})
        return httpx.Response(404, json={})


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s

    def __call__(self) -> float:
        return self.t


def registrar(fake: FakeFeishu, clock: Clock | None = None) -> Registrar:
    c = clock or Clock()
    return Registrar(transport=httpx.MockTransport(fake), sleep=c.sleep, clock=c)


def run(fake, clock=None, **kw):
    reg = registrar(fake, clock)
    domain = kw.pop("domain", "feishu")
    begin = reg.begin(domain)
    return reg, begin, reg.poll(begin, domain=domain, **kw)


@pytest.fixture(autouse=True)
def no_keychain(monkeypatch):
    """Never touch a real keychain (macOS CI) in these tests."""
    monkeypatch.setattr(cfgmod.KeychainSecrets, "get", lambda self, name: None)
    monkeypatch.setattr(wizard, "_keychain_store", lambda value: False)


@pytest.fixture
def vhome(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("VENTRI_HOME", str(h))
    monkeypatch.delenv("VENTRI_SECRET_FEISHU_APP_SECRET", raising=False)
    monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
    (h / "ventri.yml").write_text(cli.config_text(), encoding="utf-8")
    return h


# ------------------------------------------------------------ protocol
def test_begin_and_poll_until_confirmed():
    fake, clock = FakeFeishu([{"error": "authorization_pending"}, {}, "ok"]), Clock()
    reg = registrar(fake, clock)
    reg.init()
    begin = reg.begin()
    assert begin.device_code == "dev-123" and begin.interval == 5 and begin.expire_in == 600
    q = parse_qs(urlparse(begin.qr_url).query)
    assert q["user_code"] == ["ABCD"] and q["from"] == ["sdk"] and q["tp"] == ["sdk"]
    assert q["source"] == ["python-sdk/ventri"] and "addons" not in q
    creds = reg.poll(begin)
    assert creds == Credentials("cli_a1b2c3", SECRET, "feishu", "ou_scanner", "feishu")
    assert SECRET not in repr(creds)
    assert clock.sleeps == [5, 5]
    actions = [(h, a) for h, a, _ in fake.calls]
    assert actions == [("accounts.feishu.cn", "init"), ("accounts.feishu.cn", "begin")] + [
        ("accounts.feishu.cn", "poll")] * 3
    b = fake.calls[1][2]
    assert b == {"action": "begin", "archetype": "PersonalAgent", "auth_method": "client_secret",
                 "request_user_info": "open_id"}
    assert fake.calls[2][2] == {"action": "poll", "device_code": "dev-123"}


def test_init_requires_client_secret_auth():
    with pytest.raises(RegistrationError) as ei:
        registrar(FakeFeishu([], methods=["private_key_jwt"])).init()
    assert ei.value.code == "unsupported_auth_method"


@pytest.mark.parametrize("error", ["access_denied", "expired_token"])
def test_denied_and_expired_stop_polling(error):
    fake = FakeFeishu([{"error": "authorization_pending"}, {"error": error, "error_description": "nope"}])
    with pytest.raises(RegistrationError) as ei:
        run(fake)
    assert ei.value.code == error
    assert [a for _, a, _ in fake.calls].count("poll") == 2


def test_deadline_expires_while_pending():
    fake, clock = FakeFeishu([]), Clock()   # pending forever
    with pytest.raises(RegistrationError) as ei:
        run(fake, clock, timeout=30)
    assert ei.value.code == "expired_token" and clock.t >= 30
    assert sum(clock.sleeps) < 40


def test_slow_down_backs_off():
    fake, clock = FakeFeishu([{"error": "slow_down"}, {"error": "authorization_pending"}, "ok"]), Clock()
    run(fake, clock)
    assert clock.sleeps == [10, 10]


def test_lark_tenant_switches_domain():
    fake, clock = FakeFeishu([{"error": "authorization_pending", "user_info": {"tenant_brand": "lark"}},
                              "ok-lark"]), Clock()
    statuses: list[str] = []
    reg, _, creds = run(fake, clock, on_status=statuses.append)
    assert creds.domain == "lark" and creds.tenant_brand == "lark"
    assert [h for h, a, _ in fake.calls if a == "poll"] == ["accounts.feishu.cn", "accounts.larksuite.com"]
    assert statuses == ["domain-switched"]
    assert reg.probe_bot(creds) == {"app_name": "Jeff's agent", "open_id": "ou_bot"}
    assert fake.calls[-1][0] == "open.larksuite.com"


def test_lark_domain_option_begins_on_larksuite():
    fake = FakeFeishu(["ok-lark"])
    _, begin, creds = run(fake, domain="lark")
    assert {h for h, _, _ in fake.calls} == {"accounts.larksuite.com"} and creds.domain == "lark"
    assert begin.qr_url.startswith("https://accounts.larksuite.com/")


def test_network_errors_and_5xx_are_retried():
    fake, clock = FakeFeishu([httpx.ConnectError("down"), httpx.ReadTimeout("slow"),
                              httpx.Response(502, text="<html>bad gateway</html>"),
                              httpx.Response(503, json={}), "ok"]), Clock()
    statuses: list[str] = []
    _, _, creds = run(fake, clock, on_status=statuses.append)
    assert creds.app_id == "cli_a1b2c3"
    assert statuses == ["network-retry"] * 4
    assert all(s <= 30 for s in clock.sleeps) and len(clock.sleeps) == 4


def test_network_down_until_deadline_is_an_expiry():
    fake, clock = FakeFeishu([httpx.ConnectError("down")] * 500), Clock()
    with pytest.raises(RegistrationError) as ei:
        run(fake, clock, timeout=120)
    assert ei.value.code == "expired_token"


def test_unknown_error_and_bad_begin():
    with pytest.raises(RegistrationError) as ei:
        run(FakeFeishu([{"error": "invalid_grant"}]))
    assert ei.value.code == "invalid_grant"

    def broken(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"interval": 5})
    with pytest.raises(RegistrationError) as ei:
        Registrar(transport=httpx.MockTransport(broken)).begin()
    assert ei.value.code == "bad_response"


def test_addons_encoding_matches_the_sdk_format():
    reg = registrar(FakeFeishu([]))
    begin = reg.begin(addons=_onboard.MINIMAL_ADDONS)
    enc = parse_qs(urlparse(begin.qr_url).query)["addons"][0]
    raw = gzip.decompress(base64.urlsafe_b64decode(enc + "=" * (-len(enc) % 4)))
    assert json.loads(raw) == _onboard.MINIMAL_ADDONS
    assert "=" not in enc and enc == _onboard.encode_addons(_onboard.MINIMAL_ADDONS)  # deterministic (mtime=0)


def test_probe_bot_failure_is_none():
    fake = FakeFeishu([])
    assert registrar(fake).probe_bot(Credentials("cli_a1b2c3", "wrong-secret")) is None

    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")
    assert Registrar(transport=httpx.MockTransport(down)).probe_bot(Credentials("a", "b")) is None


# ------------------------------------------------------------------- QR
def read_png(p: Path) -> tuple[int, int, list[bytes]]:
    data = p.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(kind + body)
        if kind == b"IHDR":
            w, h, depth, color = struct.unpack(">IIBB", body[:10])
            assert (depth, color) == (8, 0)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    rows = [raw[i * (w + 1):(i + 1) * (w + 1)] for i in range(h)]
    assert all(r[0] == 0 for r in rows)
    return w, h, [r[1:] for r in rows]


def test_qr_png_and_terminal_rendering(tmp_path):
    url = "https://accounts.feishu.cn/page/cli?user_code=ABCD&from=sdk"
    m = wizard.qr_matrix(url)
    assert m is not None and len(m) == len(m[0]) and len(m) >= 29   # version >= 1 + quiet zone
    assert not any(m[0]) and m[4][4] and m[4][10]                   # quiet zone, finder pattern corner
    out = tmp_path / "sub" / "qr.png"
    wizard.write_png(m, out, scale=4)
    w, h, rows = read_png(out)
    assert w == h == len(m) * 4
    assert all(rows[y * 4][x * 4] == (0 if m[y][x] else 255) for y in range(len(m)) for x in range(len(m)))
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    text = wizard.qr_text(m).splitlines()
    assert len(text) == (len(m) + 1) // 2 and {len(line) for line in text} == {len(m)}


# --------------------------------------------------------------- config
def creds(app_id="cli_a1b2c3", open_id="ou_scanner", domain="feishu", secret=SECRET) -> Credentials:
    return Credentials(app_id, secret, domain, open_id)


def feishu_entries(text: str) -> list[dict]:
    return wizard._feishu_items(yaml.safe_load(text)["plugins"])


def test_template_block_is_uncommented_in_place():
    text = cli.config_text()
    new, how = wizard.plan_config(text, creds())
    assert how == "uncommented"
    [item] = feishu_entries(new)
    assert item["id"] == "feishu" and item["config"] == {
        "app_id": "cli_a1b2c3", "app_secret": "${secret:feishu_app_secret}", "domain": "feishu",
        "allow_users": ["ou_scanner"], "allow_chats": [], "require_mention": True}
    # everything else, comments included, is untouched
    gone = [ln for ln in text.splitlines() if ln not in new.splitlines()]
    assert gone and all(ln.lstrip().startswith("# ") for ln in gone)
    assert "# Feishu / Lark bot: `va feishu setup`" in new
    assert new.index("ventri_agent.sessions") < new.index("ventri_agent.channels.feishu") < new.index("agents:")


def test_existing_block_is_replaced_keeping_other_settings():
    text = cli.config_text() + "\n# trailing comment kept\n"
    first, _ = wizard.plan_config(text, creds(app_id="cli_old", open_id="ou_old"))
    first = first.replace("      require_mention: true", "      require_mention: false\n      agent: helper\n"
                          "      bot_open_id: ou_oldbot   # stale")
    new, how = wizard.plan_config(first, creds(app_id="cli_new", open_id="ou_new", domain="lark"))
    assert how == "replaced"
    [item] = feishu_entries(new)
    cfg = item["config"]
    assert cfg["app_id"] == "cli_new" and cfg["domain"] == "lark" and cfg["allow_users"] == ["ou_new"]
    assert cfg["agent"] == "helper" and cfg["require_mention"] is False and "bot_open_id" not in cfg
    assert "# trailing comment kept" in new and "agents:" in new
    again, _ = wizard.plan_config(new, creds(app_id="cli_new", open_id="ou_new", domain="lark"))
    assert again == new   # idempotent


def test_block_is_appended_when_there_is_no_example():
    text = "version: 1\n# my config\nplugins:\n    - use: ventri_agent.sessions   # sessions\n\nagents: {}\n"
    new, how = wizard.plan_config(text, creds())
    assert how == "appended"
    assert new.startswith("version: 1\n# my config\nplugins:\n    - use: ventri_agent.sessions   # sessions\n"
                          "    - use: ventri_agent.channels.feishu\n")
    assert yaml.safe_load(new)["agents"] == {} and len(feishu_entries(new)) == 1


def test_unusual_layout_is_rewritten_and_duplicates_refused():
    text = "version: 1\nplugins:\n  - {use: ventri_agent.channels.feishu, config: {app_id: x, agent: a}}\n"
    new, how = wizard.plan_config(text, creds())
    assert how == "rewritten"
    [item] = feishu_entries(new)
    assert item["config"]["agent"] == "a" and item["config"]["allow_users"] == ["ou_scanner"]
    dup = "version: 1\nplugins:\n  - use: ventri_agent.channels.feishu\n  - use: ventri_agent.channels.feishu\n"
    with pytest.raises(wizard.SetupError, match="several"):
        wizard.plan_config(dup, creds())


# ----------------------------------------------------------------- flow
def opts(home: Path, **kw) -> wizard.Options:
    return wizard.Options(config=home / "ventri.yml", **kw)


def setup(home: Path, fake: FakeFeishu, **kw) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    rc = wizard.run_setup(opts(home, **kw), out=out, err=err, registrar=registrar(fake))
    return rc, out.getvalue(), err.getvalue()


def test_setup_end_to_end_secret_never_shown(vhome, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    original = (vhome / "ventri.yml").read_text()
    fake = FakeFeishu([{"error": "authorization_pending"}, "ok"])
    rc, out, err = setup(vhome, fake, qr_png=vhome / "qr.png")
    assert rc == 0, out + err
    # the secret: in the 0600 file, nowhere else
    sf = vhome / "secrets" / "feishu_app_secret"
    assert sf.read_text() == SECRET and stat.S_IMODE(sf.stat().st_mode) == 0o600
    cap = capsys.readouterr()
    backups = list(vhome.glob("ventri.yml.bak-*"))
    assert len(backups) == 1 and backups[0].read_text() == original
    for text in (out, err, cap.out, cap.err, caplog.text, (vhome / "ventri.yml").read_text(),
                 backups[0].read_text(), json.dumps([c[2] for c in fake.calls])):
        assert SECRET not in text
    assert "█" in out and str(sf) in out and "ou_scanner" in out and "va serve" in out and "200340" in out
    assert read_png(vhome / "qr.png")[0] > 0
    # the configuration loads and the channel is enabled for the scanner only
    doc = load_document(vhome / "ventri.yml")
    entry = doc.find("feishu")
    assert not entry.disabled and entry.validated.app_id == "cli_a1b2c3"
    assert entry.validated.allow_users == ["ou_scanner"] and entry.validated.app_secret.reveal() == SECRET
    assert wizard.secret_configured()


def test_setup_json_events(vhome):
    rc, out, err = setup(vhome, FakeFeishu(["ok"]), json=True)
    assert rc == 0
    events = [json.loads(line) for line in out.splitlines()]
    assert [e["event"] for e in events] == ["qr", "done"]
    assert events[0]["url"].startswith("https://accounts.feishu.cn/") and events[0]["expire_in"] == 600
    done = events[-1]
    assert done["app_id"] == "cli_a1b2c3" and done["open_id"] == "ou_scanner" and done["bot_name"] == "Jeff's agent"
    assert done["secret_store"] == "file" and done["edit"] == "uncommented"
    assert SECRET not in out and SECRET not in err and "█" in err


def test_setup_is_idempotent_and_force_replaces(vhome):
    assert setup(vhome, FakeFeishu(["ok"]))[0] == 0
    first = (vhome / "ventri.yml").read_text()
    fake = FakeFeishu(["ok"])
    rc, out, _ = setup(vhome, fake)
    assert rc == 0 and "already set up" in out and fake.calls == []      # no network, nothing changed
    assert (vhome / "ventri.yml").read_text() == first
    fake2 = FakeFeishu(["ok"], app_id="cli_second", open_id="ou_second", secret=SECRET2)
    rc, out, err = setup(vhome, fake2, force=True)
    assert rc == 0
    [item] = feishu_entries((vhome / "ventri.yml").read_text())
    assert item["config"]["app_id"] == "cli_second" and item["config"]["allow_users"] == ["ou_second"]
    assert (vhome / "secrets" / "feishu_app_secret").read_text() == SECRET2
    assert SECRET2 not in out + err
    assert len(list(vhome.glob("ventri.yml.bak-*"))) == 2


@pytest.mark.parametrize("error", ["access_denied", "expired_token"])
def test_failed_registration_changes_nothing(vhome, error):
    original = (vhome / "ventri.yml").read_text()
    rc, out, _ = setup(vhome, FakeFeishu([{"error": error}]))
    assert rc == 1 and "again" in out
    assert (vhome / "ventri.yml").read_text() == original
    assert not (vhome / "secrets").exists() and not list(vhome.glob("ventri.yml.bak-*"))


def test_setup_needs_a_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path))
    fake = FakeFeishu(["ok"])
    rc, out, _ = setup(tmp_path, fake)
    assert rc == 2 and "va init" in out and fake.calls == []


def test_env_override_is_flagged(vhome, monkeypatch):
    monkeypatch.setenv("VENTRI_SECRET_FEISHU_APP_SECRET", "stale-value")
    rc, out, _ = setup(vhome, FakeFeishu(["ok"]))
    assert rc == 0 and "takes precedence" in out and "stale-value" not in out


# ------------------------------------------------------------- keychain
def test_keychain_store_keeps_the_secret_off_argv(monkeypatch, tmp_path):
    monkeypatch.undo()   # the real _keychain_store, with a fake `security`
    seen: dict = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["input"] = argv, kw.get("input")
        return type("R", (), {"returncode": 0})()
    security = tmp_path / "security"
    security.touch()
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(wizard, "_SECURITY", str(security))
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(cfgmod.KeychainSecrets, "get", lambda self, name: SECRET if name == wizard.SECRET_NAME else None)
    stored = wizard.store_secret(SECRET, "auto")
    assert stored.kind == "keychain" and SECRET not in stored.where
    assert seen["argv"] == [str(security), "-i"] and SECRET not in " ".join(seen["argv"])
    assert seen["input"] == f'add-generic-password -U -s ventri -a feishu_app_secret -w "{SECRET}"\n'


def test_keychain_failure_falls_back_to_file_or_errors(vhome, monkeypatch):
    stored = wizard.store_secret(SECRET, "auto")       # no keychain here (fixture) -> file
    assert stored.kind == "file" and Path(stored.where) == vhome / "secrets" / "feishu_app_secret"
    with pytest.raises(wizard.SetupError, match="keychain"):
        wizard.store_secret(SECRET, "keychain")
    with pytest.raises(wizard.SetupError, match="unexpected format"):
        wizard.store_secret('bad"; rm -rf /', "file")


# ------------------------------------------------------------------ CLI
def test_cli_wiring(vhome, monkeypatch):
    got: list[wizard.Options] = []
    monkeypatch.setattr(wizard, "run_setup", lambda o: got.append(o) or 0)
    assert cli.main(["feishu", "setup", "--json", "--qr-png", "/tmp/x.png", "--domain", "lark", "--minimal",
                     "--secret-store", "file", "--timeout", "90"]) == 0
    o = got[-1]
    assert (o.config, o.json, o.qr_png, o.domain, o.minimal, o.secret_store, o.timeout, o.force) == (
        vhome / "ventri.yml", True, Path("/tmp/x.png"), "lark", True, "file", 90.0, False)
    assert cli.main(["init", "--no-key", "--feishu"]) == 0
    assert got[-1].config == vhome / "ventri.yml" and not got[-1].json


def test_init_mentions_feishu_setup(vhome, capsys):
    (vhome / "ventri.yml").unlink()
    assert cli.main(["init", "--no-key"]) == 0
    assert "va feishu setup" in capsys.readouterr().out


def test_without_qrcode_the_url_is_shown(vhome, monkeypatch):
    monkeypatch.setattr(wizard, "qr_matrix", lambda url: None)
    fake = FakeFeishu(["ok"])
    rc, out, _ = setup(vhome, fake, qr_png=vhome / "qr.png")
    assert rc == 2 and "qrcode" in out and fake.calls == []      # asked for a PNG it cannot draw
    rc, out, _ = setup(vhome, FakeFeishu(["ok"]))
    assert rc == 0 and "https://accounts.feishu.cn/page/cli?user_code=ABCD" in out and "█" not in out
