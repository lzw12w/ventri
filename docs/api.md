# Ventri 0.2 API reference

Frozen for the 0.2 series (M1, `ventri 0.2.0a1`). Everything listed here is public and follows
semver from 0.2 on (breaking changes only in 0.3). Names starting with `_`, and module internals not
listed here, are private. Source docstrings are authoritative on details.

All APIs are asyncio-only (anyio is an internal dependency). Python 3.12+.

## `ventri` (L0 kernel)

### `Kernel(*, trace_limit=10_000, load_timeout=30.0, retry=None)`

The root `Context`. Use as `async with Kernel() as app:`; leaving the block disposes every fiber
(children first, LIFO effects) and cancels every task.

| member | |
|---|---|
| `load_timeout` | default per-plugin load timeout in seconds (`None`: none); overridden by plugin metadata / `plugin(..., timeout=)` |
| `retry` | default `Retry` policy for FAILED fibers (`None`: never retry) |
| `await settle()` | run the reconciler to a fixpoint (only needed after a `provide` made outside any kernel operation) |
| `snapshot() -> dict` | comparable state: `{"fibers": tree, "services": {key: owner}, "realms": {scope: {...}}}`; config redacted |
| `tree() -> str` | human-readable tree (state, config id, staged, deps, provides, tasks, errors, pending reasons) |
| `on_trace(cb) -> unsubscribe` | call `cb(TraceEvent)` for every trace record (exceptions in `cb` are swallowed) |
| `trace_log` | in-memory ring buffer of `TraceEvent` (`trace_limit` records) |

### `Context` (`ctx`)

| method | semantics |
|---|---|
| `await ctx.plugin(plugin, config=None, *, meta=None, timeout=MISSING, retry=MISSING) -> Fiber` | load a child plugin now (returns once ACTIVE / PENDING / FAILED) |
| `await ctx.scope(name, isolate=(), *, meta=None) -> Fiber` | create a child scope; keys in `isolate` get a scope-local realm; `scope.ctx` loads into it |
| `ctx.provide(key, value, *, name=None) -> value` | bind a service for this fiber's lifetime (`ServiceConflict` if the realm already has `key`) |
| `ctx.get(key[, default])` / `ctx.has(key)` / `ctx.<name>` | resolve a service (`ServiceNotFound`); injected keys return the instance bound at activation |
| `ctx.on(event, handler, *, priority=0) -> off` | listener; higher priority first, then registration order |
| `ctx.intercept(event, fn, *, priority=0) -> off` | interceptor for `ctx.check` |
| `await ctx.emit(event, *args)` | sequential, errors isolated (traced as `event.error`) |
| `await ctx.parallel(event, *args)` | concurrent, errors isolated |
| `await ctx.serial(event, *args)` | sequential, first non-`None` result returned, errors propagate |
| `ctx.bail(event, *args)` | synchronous `serial` |
| `await ctx.check(event, value) -> value \| Deny` | run interceptors: `Deny` stops, `Rewrite(v)` replaces and continues, `None` passes; errors propagate |
| `ctx.effect(setup) -> async disposer` | `setup()` now; its returned callable runs at teardown (LIFO) |
| `ctx.on_dispose(cb)` | run `cb` at teardown |
| `await ctx.enter(cm)` | enter a (async) context manager now, exit at teardown |
| `ctx.spawn(fn, *args, name=None) -> TaskHandle` | background task owned by the fiber: cancelled and joined at teardown; a crash FAILs the fiber |
| `ctx.trace(kind, **attrs)` | custom trace record (dotted kind, not a kernel kind) |
| `ctx.transaction(*, wait=True, strict=False, origin=None, reason=None, timeout=None, dry_run=False, probe=None) -> Transaction` | see below |
| `await ctx.replace(fiber, plugin_or_config, *, plugin=, config=, strategy=None) -> Fiber` | one-operation transaction |
| `ctx.parent`, `ctx.fiber`, `ctx.kernel` | navigation |

Events are strings or `Event[T]("name")` (same channel as the string). Scope filter: an event emitted
in scope S reaches listeners in S's subtree and on S's ancestor chain, never sibling scopes.

### Plugins

A plugin is a function `fn(ctx, config, *deps)` (sync or async) or a class `Cls(ctx, config, *deps)`
(optional `start()` / `stop()`). Metadata, as attributes or through `@plugin(...)`:

`@plugin(fn=None, *, name=None, inject=(), config=None, provides=None, timeout=MISSING, retry=None, exclusive=False)`

- `config` / `Config`: if the config is a dict (or `None`), the plugin receives `Config(**config)` (dataclass, pydantic model...).
- Signature injection: parameters after `(ctx, config)`: `X` required; `X | None = None` optional
  (appearing/disappearing restarts the fiber); `Annotated[T, "key"]` for string keys; other defaulted
  parameters are not injected; an unannotated parameter without default is a `TypeError`. `inject=[...]` adds keys.
- `provides`: `{"name": Key}` or `[Key, ...]` -- used by diagnostics, `ventri stubgen` and to name
  `ctx.provide(Key, v)` for `ctx.<name>`.
- `timeout`: load timeout (seconds / `None`). `retry`: `Retry` or dict. `exclusive`: replace stop-first by default.

### `Fiber`

`state` (`State.PENDING|LOADING|ACTIVE|FAILED|UNLOADING|DISPOSED`), `name`, `id`, `label` (`name#id`),
`path` (trace path), `parent`, `children`, `ctx`, `config`, `raw_config`, `instance`, `error`,
`pending_reason`, `meta`, `inject`, `optional`, `deps`, `load_timeout`, `retry_policy`, `scope`,
`is_scope`, `isolate`, `tx`; `await dispose()`, `await restart()`, `spawn(...)`.

### `Transaction`

`async with ctx.transaction(...) as tx:` -- `await tx.plugin(plugin, config, *, parent=None, meta=None, timeout=, retry=)`,
`await tx.scope(name, isolate, *, parent=None, meta=None)`, `await tx.dispose(fiber)`,
`await tx.replace(fiber, plugin=, config=, *, timeout=, retry=, strategy=None|"blue-green"|"stop-first")`,
`await tx.reconfigure(fiber, config)`, `tx.get(key[, default])` (staged view), `tx.id`, `tx.report`.

Guarantees G1–G5 and non-guarantees: see README / DESIGN.md 4.5. `wait=False` raises `TransactionBusy`;
`timeout` raises `TransactionTimeout` after rolling back; `strict=True` fails on staged fibers that
stay PENDING (`DependencyCycle` when the reason is a cycle). A `dry_run` always rolls back after
settling, validating and running `probe(tx)` / `{name: probe}`.

### `TxReport`

`tx, origin, reason, dry_run, outcome ("committed"|"rolled_back"|"dry_run"), degraded, error, added,
removed, replaced, restarted, activated, services {added, removed, replaced}, failures, pending,
skipped, probes`; `ok`, `to_dict()`, `to_json()`, `str()`.

### Other public names

- `Retry(max=3, backoff="exp"|"fixed", base=0.5, cap=30.0, reset_after=60.0)`; `delay(n)`, `Retry.coerce(x)`.
- `Event[T](name)`, `Deny(reason)`, `Rewrite[T](value)`.
- `Secret[T](value)`: masked `repr`/`str`, `reveal()` / `get_secret_value()`, value equality, pydantic field support.
  `redact(value)` masks secrets (type and key-name based) recursively.
- `TraceEvent` (`seq, time/ts, kind, fiber (label), data/attrs, path, scope, tx`, `to_dict()` -> schema v1),
  `SCHEMA_VERSION`, `KERNEL_KINDS`; `ventri.trace.validate`, `read_jsonl`, `dumps`.
- `Binding`, `Realm`, `State`, `TaskHandle` (`cancel()`, `done`, `await wait()`).
- Errors: `KernelError` > `ServiceNotFound` (also `LookupError`), `ServiceConflict`, `PluginError` > `LoadTimeout`,
  `TransactionError` > `TransactionBusy`, `TransactionConflict`, `TransactionTimeout`, `DependencyCycle`.

## `ventri_std` (L1)

### `ventri_std.config`

- `load_document(path=None, *, text=None, profiles=None, secrets=None, environ=None, resolver=resolve_use) -> Document`
  (raises `ConfigError(errors)`; touches nothing).
- `plan(parent_fiber, doc) -> Plan` (`changes: [Change(op, id, use, reason)]`, `unchanged`, `empty`).
- `await apply_document(parent_fiber, doc, *, dry_run=False, strict=False, reason=None, timeout=None, on_staged=None) -> ApplyResult`
  (`plan`, `report: TxReport | None`, `document`, `ok`). Failures are reported, not raised.
- `Loader(parent_fiber, path, *, profiles=None, secrets=None, environ=None, resolver=..., strict=False)`:
  `load()`, `await plan()`, `await apply(dry_run=False, strict=None, reason=None)`,
  `await watch(poll=0.1, debounce=0.3, on_result=None)`, `document`, `last`.
- Plugin `loader` (`use: ventri_std.config.loader`, config `path, profiles, watch, poll, debounce, strict`),
  provides `"config.loader"`.
- Secrets: `SecretStore` protocol (`get(name) -> str | None`), `KeychainSecrets`, `EnvSecrets`, `DictSecrets`,
  `ChainSecrets`, `default_secrets()`; `interpolate(value, secrets, environ=None)`; `resolve_use(use)`.

### `ventri_std.trace`

Plugin `jsonl` (`use: ventri_std.trace.jsonl`; config `path, rotate_mb=64, keep=10, flush_interval=0.5, backfill=True`);
`JsonlWriter`.

### `ventri_std.stubgen`

`generate(out_dir, modules=(), *, use_entry_points=True) -> StubResult`; `discover`, `collect`.

### CLI

`ventri run | apply (--dry-run | --validate-only) [--json] | tree | doctor | stubgen`; each config
command takes `[config] [--profile P]... [--strict]`. Exit codes: 0 ok, 1 problems / failed apply,
2 configuration or usage error.
