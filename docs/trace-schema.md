# Trace schema v1

Frozen in 0.2 (M1). Source of truth: `ventri/trace.py`; JSON Schema: [`trace-schema-v1.json`](trace-schema-v1.json).
The JSONL sink (`ventri_std.trace`) writes one record per line.

```json
{"v":1,"seq":17,"ts":1791427200.123,"kind":"fiber.state","fiber":"root/tools/github",
 "scope":null,"tx":3,"attrs":{"label":"github#7","old":"pending","new":"loading"}}
```

| field | type | meaning |
|---|---|---|
| `v` | int | schema version, `1` |
| `seq` | int | per-kernel sequence, strictly increasing from 1 |
| `ts` | float | wall clock, Unix epoch seconds (UTC) |
| `kind` | str | event kind (dotted) |
| `fiber` | str \| null | fiber path: `root/` + one segment per fiber, the config id (`meta["id"]`) if set, else the plugin name. `null` for kernel-level events |
| `scope` | str \| null | nearest enclosing scope name; `null` at root level |
| `tx` | int \| null | transaction id for staged fibers and `tx.*` events |
| `attrs` | object | kind-specific attributes, redacted with `ventri.redact`; fiber events carry `label` (`name#id`, unique per process) |

## Kernel kinds

| kind | guaranteed attrs |
|---|---|
| `kernel.start`, `kernel.stop` | – |
| `fiber.state` | `old`, `new` (+ `error` on `failed`) |
| `fiber.retry` | `attempt` (+ `gave_up`) |
| `service.bind`, `service.unbind` | `key`, `staged` (+ `realm` when not the root realm) |
| `task.spawn` | `task` |
| `task.error`, `effect.error`, `event.error` | `task` / `effect` / `event`, `error` |
| `tx.begin` | `tx`, `origin`, `reason`, `dry_run`, `scope` |
| `tx.commit` | `tx`, `services`, `added`, `removed` |
| `tx.rollback` | `tx`, `error` (+ `degraded` after a stop-first rollback) |
| `dep.cycle` | `cycle` (list of fiber labels) |

Upper layers emit their own kinds with `ctx.trace("agent.turn", ...)`; a custom kind must be dotted and
must not be a kernel kind.

## Compatibility

Consumers of v1 must ignore unknown kinds and unknown attrs. Adding kinds or attrs is not a schema change;
renaming or removing a top-level field, or changing its type, bumps `v`.

Non-guarantees: redaction is by `Secret` type and by key name (`key`, `token`, `secret`, `password`); a secret
inside an ordinary string (e.g. credentials in a URL) is not detected. The in-memory ring buffer
(`Kernel(trace_limit=...)`) drops the oldest records; only a sink sees every record.
