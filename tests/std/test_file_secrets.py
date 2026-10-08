"""FileSecrets: the 0600 file backend for hosts without a keychain."""
import os
import stat

import pytest

from ventri_std import config as cfgmod
from ventri_std.config import ChainSecrets, ConfigError, DictSecrets, FileSecrets, load_document


def test_round_trip_modes_and_missing(tmp_path):
    fs = FileSecrets(tmp_path / "secrets")
    assert fs.get("x") is None  # no directory yet
    p = fs.set("feishu_app_secret", "s3cr3t-value")
    assert p == tmp_path / "secrets" / "feishu_app_secret"
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700
    assert fs.get("feishu_app_secret") == "s3cr3t-value"
    fs.set("feishu_app_secret", "rotated\n")  # replace in place; trailing newline is not part of the secret
    assert fs.get("feishu_app_secret") == "rotated"
    assert fs.get("other") is None
    assert [q.name for q in p.parent.iterdir()] == ["feishu_app_secret"]  # no temp files left


def test_default_directory_follows_ventri_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path / "h"))
    fs = FileSecrets()
    assert fs.directory == tmp_path / "h" / "secrets" == cfgmod.secrets_dir()
    fs.set("deepseek", "k")
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path / "other"))
    assert fs.get("deepseek") is None  # resolved lazily


def test_group_or_world_readable_file_fails_closed(tmp_path):
    fs = FileSecrets(tmp_path / "s")
    p = fs.set("a", "v")
    os.chmod(p, 0o644)
    with pytest.raises(ConfigError, match="chmod 600"):
        fs.get("a")
    os.chmod(p, 0o600)
    os.chmod(p.parent, 0o777)
    with pytest.raises(ConfigError, match="chmod 700"):
        fs.get("a")


def test_symlinks_and_bad_names_are_refused(tmp_path):
    fs = FileSecrets(tmp_path / "s")
    fs.set("real", "v")
    (tmp_path / "s" / "link").symlink_to(tmp_path / "s" / "real")
    with pytest.raises(ConfigError):
        fs.get("link")
    for bad in ("../x", "a/b", "", ".hidden", "x" * 200):
        assert fs.get(bad) is None  # not a file-store name: other stores in the chain may know it
        with pytest.raises(ConfigError, match="invalid secret name"):
            fs.set(bad, "v")


def test_default_chain_order(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path))
    monkeypatch.setattr(cfgmod.KeychainSecrets, "get", lambda self, name: None)
    FileSecrets().set("tok", "from-file")
    chain = cfgmod.default_secrets()
    assert isinstance(chain, ChainSecrets)
    assert chain.get("tok") == "from-file"
    monkeypatch.setenv("VENTRI_SECRET_TOK", "from-env")
    assert chain.get("tok") == "from-env"  # env overrides the file


def test_config_resolves_file_secret_without_printing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path))
    monkeypatch.delenv("VENTRI_SECRET_FEISHU_APP_SECRET", raising=False)
    monkeypatch.setattr(cfgmod.KeychainSecrets, "get", lambda self, name: None)
    FileSecrets().set("feishu_app_secret", "fs-secret-xyz")
    cfg = tmp_path / "v.yml"
    cfg.write_text('version: 1\nplugins:\n  - use: tests.std.cfg_plugins.LLM\n'
                   '    config: {api_key: "${secret:feishu_app_secret}"}\n')
    entry = load_document(cfg).entries[0]
    assert entry.config["api_key"].reveal() == "fs-secret-xyz"
    assert "fs-secret-xyz" not in repr(entry.config) and "fs-secret-xyz" not in repr(entry.validated)
    assert "fs-secret-xyz" not in capsys.readouterr().out


def test_missing_secret_error_names_every_store(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path))
    with pytest.raises(ConfigError) as ei:
        cfgmod.interpolate("${secret:nope}", ChainSecrets(DictSecrets({})), {}, "p")
    msg = str(ei.value)
    assert "VENTRI_SECRET_NOPE" in msg and str(tmp_path / "secrets" / "nope") in msg
