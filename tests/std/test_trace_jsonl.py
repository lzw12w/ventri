"""ventri_std.trace.jsonl: schema-v1 JSONL sink with rotation (DESIGN.md 4.10)."""
import anyio
import pytest

from ventri import Kernel, Secret, State
from ventri.trace import read_jsonl
from ventri_std.trace import jsonl

pytestmark = pytest.mark.anyio


async def _noise(app: Kernel, n: int) -> None:
    for i in range(n):
        f = await app.plugin(lambda ctx, config: ctx.provide(f"svc{config}", config), i)
        await f.dispose()


async def test_writes_every_record_once_in_order(tmp_path):
    async with Kernel() as app:
        sink = await app.plugin(jsonl, {"path": str(tmp_path), "flush_interval": 0.01})
        await _noise(app, 5)
        await anyio.sleep(0.05)
        assert (tmp_path / "trace.jsonl").stat().st_size > 0  # periodic flush happened
        await _noise(app, 3)
        last_seq = app.trace_log[-1].seq
        await sink.dispose()
    seqs = [r["seq"] for r in read_jsonl(tmp_path / "trace.jsonl")]
    assert seqs[0] == 1  # backfilled kernel.start
    assert seqs == list(range(1, len(seqs) + 1))  # contiguous, no duplicates
    assert seqs[-1] >= last_seq


async def test_no_backfill_and_file_path(tmp_path):
    target = tmp_path / "sub" / "k.jsonl"
    async with Kernel() as app:
        await _noise(app, 2)
        start = app.trace_log[-1].seq
        sink = await app.plugin(jsonl, {"path": str(target), "backfill": False})
        await _noise(app, 1)
        await sink.dispose()
    recs = list(read_jsonl(target))
    assert recs and all(r["seq"] > start for r in recs)
    assert any(r["kind"] == "service.bind" for r in recs)


async def test_rotation_keeps_newest(tmp_path):
    async with Kernel() as app:
        sink = await app.plugin(jsonl, {"path": str(tmp_path), "rotate_mb": 0.004, "keep": 2,
                                        "flush_interval": 0.005})
        for _ in range(6):
            await _noise(app, 10)
            await anyio.sleep(0.01)
        await sink.dispose()
    rotated = sorted(tmp_path.glob("trace-*.jsonl"))
    assert len(rotated) == 2
    for p in [*rotated, tmp_path / "trace.jsonl"]:
        assert p.stat().st_size <= 0.004 * 1024 * 1024 + 2048
    files = [*rotated, tmp_path / "trace.jsonl"]  # names sort chronologically
    seqs = [r["seq"] for p in files for r in read_jsonl(p)]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))  # retained tail is gap-free
    assert [int(p.stem.rsplit("-", 1)[1]) for p in rotated] == [
        next(read_jsonl(p))["seq"] for p in rotated]  # name carries the first seq


async def test_redaction(tmp_path):
    async with Kernel() as app:
        sink = await app.plugin(jsonl, {"path": str(tmp_path)})
        app.trace("agent.turn", api_key="sk-1", payload={"x": Secret("sk-2")})
        await app.plugin(lambda ctx, config: ctx.provide("llm", Secret("sk-3")))
        await sink.dispose()
    text = (tmp_path / "trace.jsonl").read_text()
    assert "agent.turn" in text and "sk-1" not in text and "sk-2" not in text and "sk-3" not in text


async def test_write_error_is_reported_once(tmp_path):
    (tmp_path / "trace.jsonl").mkdir()  # cannot open a directory for append
    async with Kernel() as app:
        sink = await app.plugin(jsonl, {"path": str(tmp_path), "flush_interval": 0.005})
        await _noise(app, 2)
        await anyio.sleep(0.03)
        await _noise(app, 2)
        await anyio.sleep(0.03)
        assert sink.state is State.ACTIVE
        errors = [e for e in app.trace_log if e.kind == "trace.sink_error"]
        assert len(errors) == 1
