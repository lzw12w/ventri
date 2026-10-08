"""Secret[T] and redaction in snapshots / trees (DESIGN.md 4.2, 4.7)."""
from dataclasses import dataclass

import pydantic
import pytest

from ventri import Kernel, Secret, State, redact

pytestmark = pytest.mark.anyio


class DeepSeekConfig(pydantic.BaseModel):
    api_key: Secret[str]
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"


def test_secret_value_semantics():
    s = Secret("sk-123")
    assert s.reveal() == "sk-123" == s.get_secret_value()
    assert repr(s) == "Secret('***')" and str(s) == "***" and f"{s}" == "***"
    assert s == Secret("sk-123") and s != Secret("other") and s != "sk-123"
    assert hash(s) == hash(Secret("sk-123"))
    assert "sk-123" not in repr(RuntimeError(s)) and "sk-123" not in repr([s])


def test_pydantic_integration():
    cfg = DeepSeekConfig(api_key="sk-abc")
    assert isinstance(cfg.api_key, Secret) and cfg.api_key.reveal() == "sk-abc"
    assert DeepSeekConfig(api_key=Secret("sk-abc")) == cfg
    assert "sk-abc" not in repr(cfg)
    assert "sk-abc" not in cfg.model_dump_json() and '"api_key":"***"' in cfg.model_dump_json()

    class IntSecret(pydantic.BaseModel):
        pin: Secret[int]
    assert IntSecret(pin="42").pin.reveal() == 42
    with pytest.raises(pydantic.ValidationError):
        IntSecret(pin="not a number")


def test_redact_recurses():
    @dataclass
    class Cfg:
        token: str
        nested: dict

    value = {
        "api_key": "plain-but-sensitive-name",
        "routes": {"default": {"model": "m", "auth": Secret("s1")}},
        "list": [Secret("s2"), {"password": "p"}],
        "dc": Cfg("t", {"x": Secret("s3")}),
        "model": DeepSeekConfig(api_key="s4"),
        "endpoint": "https://user:hunter2@host",  # documented non-guarantee
    }
    out = redact(value)
    assert out["api_key"] == "***"
    assert out["routes"]["default"] == {"model": "m", "auth": "***"}
    assert out["list"] == ["***", {"password": "***"}]
    assert out["dc"] == {"token": "***", "nested": {"x": "***"}}
    assert out["model"]["api_key"] == "***" and out["model"]["model"] == "deepseek-flash"
    assert out["endpoint"] == "https://user:hunter2@host"
    for s in ("s1", "s2", "s3", "s4", "plain-but-sensitive-name"):
        assert s not in repr(out)


async def test_secrets_never_reach_snapshot_or_tree():
    got = {}

    class DeepSeek:
        Config = DeepSeekConfig

        def __init__(self, ctx, cfg: DeepSeekConfig) -> None:
            got["key"] = cfg.api_key.reveal()

    async with Kernel() as app:
        f = await app.plugin(DeepSeek, {"api_key": "sk-live", "extra": {"token": "t-live"}})
        await app.plugin(lambda ctx, config: None, {"auth": Secret("sk-other")})
        assert f.state is State.ACTIVE and got["key"] == "sk-live"
        dump = repr(app.snapshot()) + app.tree() + repr(list(app.trace_log))
        for leaked in ("sk-live", "t-live", "sk-other"):
            assert leaked not in dump
