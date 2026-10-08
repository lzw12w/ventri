# Kernel benchmarks (DESIGN.md 4.13)

`uv run python benchmarks/bench_kernel.py [--repeat N] [--json]` -- medians of N runs (`ctx.get`: median of
200-call batch averages; memory: tracemalloc delta over 1000 empty fibers).

## M1 run (2026-10-08)

Machine: shared Linux box (Linux 6.12, x86_64 Intel Xeon, 8 vCPU), CPython 3.12.15, `--repeat 7`.
**Caveat**: 1.0 targets macOS; these numbers come from a Linux VM and only show the order of magnitude.
The CI job runs the same script on macOS (informational).

| metric | target | measured | |
|---|---|---|---|
| `ctx.get(Type)` from a fiber 3 scopes deep (key at root), p50 | ≤ 2 µs | 0.44 µs | pass |
| activate 500 empty plugins in one transaction | ≤ 150 ms | 41.8 ms | pass |
| replace one key in a 500-plugin tree (10 dependents restart) | ≤ 20 ms | 3.9 ms | pass |
| create + reclaim a session scope with 3 empty plugins | ≤ 5 ms | 0.41 ms | pass |
| `emit` to 100 sync listeners | ≤ 1 ms | 0.05 ms | pass |
| memory per empty fiber | ≤ 8 KB | 4.3 KB | pass |
| agent overhead before first token | ≤ 20 ms | – | M2 (no agent runtime yet) |

Python 3.13.5 on the same box: 0.29 µs / 41.4 ms / 3.7 ms / 0.44 ms / 0.05 ms / 4.6 KB.

### Notes

- The memory target was initially missed (≈ 10–11 KB/fiber): each fiber ran a task-group host task
  (task + task group + events ≈ 6 KB). Tasks were already owned by their fiber through a cancel-and-join
  effect, so the host task was removed and tasks now start in the kernel's task group
  (commit "kernel: fiber tasks run in the kernel task group"). No target needed adjusting, so no ADR.
- The memory figure includes the fiber's `Context`, its trace records in the ring buffer and the
  reconciler's bookkeeping; `Kernel(trace_limit=0)` saves ≈ 0.9 KB more.
