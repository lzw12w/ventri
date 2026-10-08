"""Kernel micro-benchmarks for the DESIGN.md 4.13 targets.

    uv run python benchmarks/bench_kernel.py [--json] [--repeat N]

Each metric is measured ``repeat`` times and the median is reported (``ctx.get``
reports the median of per-call batch averages). Results are machine dependent;
docs/benchmarks.md records a run.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import anyio

from ventri import Kernel, plugin


class Service:
    pass


def empty(ctx: Any, config: Any) -> None:
    pass


def _ms(t0: int) -> float:
    return (time.perf_counter_ns() - t0) / 1e6


async def bench_get(repeat: int) -> float:
    """p50 of ``ctx.get(Type)`` from a fiber three scopes deep (key provided at root)."""
    async with Kernel() as app:
        app.provide(Service, Service())
        s1 = await app.scope("s1", isolate=["x"])
        s2 = await s1.ctx.scope("s2", isolate=["y"])
        s3 = await s2.ctx.scope("s3", isolate=["z"])
        leaf = await s3.ctx.plugin(empty)
        get = leaf.ctx.get
        batches = []
        n = 200
        for _ in range(max(200, repeat * 100)):
            t0 = time.perf_counter_ns()
            for _ in range(n):
                get(Service)
            batches.append((time.perf_counter_ns() - t0) / n / 1000)
        return statistics.median(batches)


async def bench_activate_500() -> float:
    async with Kernel() as app:
        t0 = time.perf_counter_ns()
        async with app.transaction() as tx:
            for i in range(500):
                await tx.plugin(plugin(empty, name=f"p{i}"))
        return _ms(t0)


async def bench_replace_in_500() -> float:
    """Replace the provider of one key in a 500-plugin tree (10 dependents)."""

    def provider(ctx: Any, config: Any) -> None:
        ctx.provide("k", config)

    def dependent(ctx: Any, config: Any, k: Any = None) -> None:
        pass

    async with Kernel() as app:
        async with app.transaction() as tx:
            prov = await tx.plugin(provider, 1)
            for i in range(10):
                await tx.plugin(plugin(empty, name=f"d{i}", inject=["k"]))
            for i in range(489):
                await tx.plugin(plugin(empty, name=f"p{i}"))
        t0 = time.perf_counter_ns()
        async with app.transaction() as tx:
            await tx.replace(prov, config=2)
        return _ms(t0)


async def bench_session_scope() -> float:
    async with Kernel() as app:
        app.provide(Service, Service())
        times = []
        for i in range(50):
            t0 = time.perf_counter_ns()
            s = await app.scope(f"session:{i}", isolate=["llm"])
            for j in range(3):
                await s.ctx.plugin(plugin(empty, name=f"e{j}"))
            await s.dispose()
            times.append(_ms(t0))
        return statistics.median(times[5:])


async def bench_emit_100() -> float:
    async with Kernel() as app:
        for _ in range(100):
            app.on("tick", lambda: None)
        times = []
        for _ in range(200):
            t0 = time.perf_counter_ns()
            await app.emit("tick")
            times.append(_ms(t0))
        return statistics.median(times[10:])


async def bench_fiber_memory() -> float:
    """KB per empty fiber (tracemalloc delta over 1000 loads)."""
    async with Kernel() as app:
        await app.plugin(plugin(empty, name="warmup"))
        gc.collect()
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for i in range(1000):
            await app.plugin(plugin(empty, name=f"m{i}"))
        gc.collect()
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        delta = sum(s.size_diff for s in after.compare_to(before, "filename"))
        return delta / 1000 / 1024


TARGETS: dict[str, tuple[float, str]] = {
    "ctx.get(Type), 3-level scope, p50": (2.0, "us"),
    "activate 500 empty plugins (one tx)": (150.0, "ms"),
    "replace one key in a 500-plugin tree": (20.0, "ms"),
    "create + reclaim a session scope (3 plugins)": (5.0, "ms"),
    "emit to 100 sync listeners": (1.0, "ms"),
    "memory per empty fiber": (8.0, "KB"),
}


async def run(repeat: int) -> dict[str, float]:
    runs: dict[str, Callable[[], Any]] = {
        "ctx.get(Type), 3-level scope, p50": lambda: bench_get(repeat),
        "activate 500 empty plugins (one tx)": bench_activate_500,
        "replace one key in a 500-plugin tree": bench_replace_in_500,
        "create + reclaim a session scope (3 plugins)": bench_session_scope,
        "emit to 100 sync listeners": bench_emit_100,
        "memory per empty fiber": bench_fiber_memory,
    }
    out = {}
    for name, fn in runs.items():
        vals = [await fn() for _ in range(1 if name.startswith(("ctx.get", "memory")) else repeat)]
        out[name] = statistics.median(vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    res = anyio.run(run, args.repeat, backend="asyncio")
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    ok_all = True
    for name, value in res.items():
        target, unit = TARGETS[name]
        ok = value <= target
        ok_all &= ok
        print(f"  {'PASS' if ok else 'MISS'}  {name:<46} {value:9.3f} {unit}  (target <= {target:g} {unit})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
