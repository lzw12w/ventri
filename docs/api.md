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

## `ventri_agent` (M2, `ventri-agent 0.2.0a1`)

Alpha: public but may still change within 0.2.x. Design: DESIGN.md section 5 and the "M2 实施说明".

### Plugins (`use:` strings)

| use | provides | config |
|---|---|---|
| `ventri_agent.providers.deepseek` | `ModelProvider` (`ctx.llm`) | `api_key` (`Secret`, default `$DEEPSEEK_API_KEY`), `base_url`, `beta_url`, `routes`, `strict_tools=false`, `soft_context=256000`, `timeout=900`, `max_retries=4`, `probe=true`, `user_id`, `prices`, `holidays` |
| `ventri_agent.providers.openai_compat` | `ModelProvider` | `base_url`, `api_key`, `routes`, `context`, `max_output`, `concurrency`, `timeout`, `max_retries` |
| `ventri_agent.providers.fake` | `ModelProvider` | `script` (steps), `script_file` (JSON/YAML), `chunk_delay` |
| `ventri_agent.tools.registry` | `ToolRegistry` | -- |
| `ventri_agent.permission` | `Policy`, `ApprovalBroker`, `AuditLog` | `rules: [{tool, action, risk?, when?, origin?, agent?, session?}]`, `approval_timeout=120`, `audit`, `unattended: {ask: deny\|allow, irreversible: deny\|allow}` (both default `deny`; only consulted for headless sessions) |
| `ventri_agent.memory` | `LongTermMemory` | `path` (`~/.ventri/memory.db`, or `:memory:`) |
| `ventri_agent.sessions` | `SessionManager` | `dir`, `idle_timeout=1800`, `retention_days=7`, `budget`, `agents`, `extract_memory=true`, `sweep_interval` |
| `ventri_agent.tools.core` / `.fs` / `.shell` / `.web` / `.notes` / `.memory` / `.inspect` | tools | fs: `roots`, `write` (file behaviour adapted from Hermes Agent, see THIRD_PARTY_NOTICES.md); shell (tools `shell.run` with `background`, `shell.output(job, wait, pattern, since)`, `shell.jobs`, `shell.kill(job, signal)`): `cwd`, `policy`, `timeout`, `max_capture_bytes=4000000`, `tail_bytes=256000`, `preview_tokens=6000`, `persist=false`, `max_jobs=16`, `job_log_bytes=8000000`, `job_keep_bytes=2000000`, `jobs_on_dispose=kill\|keep`, `hide_runtime=true`, `hide_paths=[]`, `base_env_file` (`max_output` ignored); web: `allow_domains`, `timeout`, `max_bytes`, `max_redirects=5`, `allow_private_urls=false` (SSRF guard adapted from Hermes Agent); notes: `vault`, `write`; memory tools: `memory.search`, `memory.remember`, `memory.update`, `memory.forget` |
| `ventri_agent.channels.cli` | `CliChannel` | `session`, `agent`, `resume_last`, `show_thinking` (exclusive; reads an optional `"cli.terminal"` service) |

Session-scope plugins loaded by `SessionManager.open` (not used directly in `ventri.yml`):
`permission.gate` (provides `Grants`, intercepts `ToolCheck`), `context.context_builder` (`ContextBuilder`),
`loop.agent_loop` (`AgentLoop`; replaceable per agent preset with `agents: {name: {loop: "<use>"}}`).
Top-level `agents: {name: {persona, tools: [globs], route, loop, memory_k, system_prompt, time_notes, mode,
inline_tokens, compact_at, compact_keep_turns, compact_keep_steps}}` in `ventri.yml` defines presets. `system_prompt`
(file or inline) replaces persona + rules; `time_notes` (default on, off when headless); `mode: headless` presets open
only with `headless=True`; `inline_tokens` (8000) is the artifact spill threshold; compaction triggers at `compact_at`
(0.6, range 0.1-0.95) x the model's soft context, keeps the last `compact_keep_turns` (4) user turns and, inside a long
turn, the last `compact_keep_steps` (6) model steps.

### Types

- `messages`: `Message` (`system/user/assistant/tool` builders, `to_api()`, `to_json()/from_json()`), `ToolCall`,
  `Usage` (`prompt_tokens, completion_tokens, cache_hit, cache_miss, reasoning_tokens`, `hit_rate`, `+`), `Money`,
  `ChatRequest`, events `ReasoningDelta`, `ContentDelta`, `ToolCallStart`, `Done(message, usage, finish_reason, model)`.
- `providers.base`: `ModelProvider` protocol (`name, caps, routes, caps_for(model), route(name), stream(req), price(usage, at, model)`),
  `ModelCaps`, `Route` (`request(messages, **kw)`), `ProviderError(status, retryable, body)` > `RequestInvalid` > `ReasoningContentMissing`,
  `collect(stream)`, `complete_json(provider, req, Model)`.
- `providers.pricing`: `PriceTable` (`price`, `lookup`, `from_config`), `ModelPrice`, `PeakSchedule` (`is_peak`, `next_off_peak`).
- `tools.registry`: `Tool(name, description, handler, params, risk, idempotent, parallel_safe, default_action, default_allow,
  grantable, subject, untrusted, timeout)`, `Risk` (`READ < WRITE_LOCAL < EXTERNAL < IRREVERSIBLE < SPEND`), `ToolError`,
  `ToolContext` (`session_id, ctx, workdir, origin, extras, call_id`), `ToolRegistry` (`register(ctx, tool)` -- unregistered with the fiber, `get`, `select(globs)`, `version`, `watch`),
  `tool_schema(model, strict=True)`.
- `permission`: `ToolRequest`, `ToolCheck` (event), `Policy.decide`, `Rule`, `Grants`, `ApprovalBroker` (`bind(session, channel, ask)`,
  `request`, `verify`), `ApprovalRequest`, `ApprovalDecision`, `ApprovalRequested` (event), `AuditLog`.
- `loop`: `AgentLoop` (`turn(text, sink, plan=False) -> TurnResult`, `retry`,
  `compact(sink, keep_turns=, keep_steps=, goal=, force=False) -> bool`, `extract_memories`), `TurnEvent`
  (`turn.start | reasoning | content | tool.call | tool.start | tool.end | notice | error | turn.end`), `TurnResult`,
  events `MessageIn` / `AgentOutput`.
- `sessions`: `SessionManager` (`open(id=None, agent=, channel=, origin=, headless=False)`, `get`, `suspend`, `end`, `list`, `last_id`,
  `sweep_idle`, `sweep_retention`), `Session` (`turn`, `retry`, `suspend`, `end`, `alive`, `loop`, `ctx`), `SessionError`.
- `session`: `SessionLog` (JSONL; `replay(path) -> Replay`), `SessionInfo` (`… headless`), `AgentPreset`, `Budget` / `BudgetLimits`.
- `context`: `ContextBuilder` (`build`, `system_text`, `estimate_tokens`, `trigger_tokens(route)`, `hard_limit(route)`,
  `needs_compaction`, `compaction_plan(goal, keep_turns=, keep_steps=, force=) -> CompactionPlan | None`, `apply_compaction`),
  `CompactionPlan` (`drop, span, kept_steps, trims, before, after`), `transcript(msgs)`, `RULES`, `HEADLESS_SYSTEM`.
- `session.apply_compact(history, record)`: applies one `compact` record (live and on replay).
- `permission` (headless): `Unattended(ask, irreversible)`, `Policy.decide_unattended(req)`; audit `decided_by: unattended-policy`.
- `tools.shell`: `ShellJobs` (session service: jobs + persisted cwd/env; `close()` on dispose), `clean_env(base=None, hide_runtime=True, hide_paths=())`, `read_env_file`, `make_tools(cfg)`.
- `memory`: `LongTermMemory` (`remember -> Remembered(item, action=created|duplicate|superseded, previous)`, `add -> (item, created)`,
  `update(id, text) -> MemoryItem | None` (restores a superseded item), `confirm` (applies a pending supersede), `forget`, `get`,
  `list(status=active|pending|superseded|None)`, `search`, `top`, `export_markdown`, `import_markdown`), `WorkingMemory`,
  `MemoryItem` (`… status, supersedes, superseded_by`).
- `tokens`: `estimate_tokens(text)` (CJK 0.6 / other 0.3 token per char, DeepSeek's documented ratio), `prefix_within`, `suffix_within`.
- `threat_patterns`: `scan_for_threats(text, scope)`, `first_threat_message(text, scope="strict")` (adapted from Hermes Agent).
- `tools.output`: `strip_ansi`, `head_tail`, `preview(tc, text, kind, budget=6000)`, `write_artifact(tc, text, kind)`.

### Session log records (`~/.ventri/sessions/<id>.jsonl`)

`meta` (`headless` when set), `prefix` (epoch: system, memory, tool names, tools, strict, hash), `msg`, `compact` (drop, summary;
intra-turn: `span` [a, b) of the remaining history replaced by `progress`, `steps`, `trim` [{seq, content}]), `usage`
(model, route, usage, cost_usd, peak, finish), `turn`, `work`, `state` (`resumed | suspended | ended | memory`).
Every record has `t` and `ts`. Append-only; a torn last line is skipped on replay. `prune` records (written only by
0.2.0a1 development builds, feature removed) are ignored. Trace events include `context.compact` (parts, dropped, steps,
kept_steps, trimmed, before, after, forced).

### CLI

`va init [--force] [--no-key]`, `va chat [--config] [--profile]... [--session ID | --continue] [--agent] [--fake SCRIPT]
[--show-thinking] [--no-watch]`, `va run [TASK | --task-file F | stdin] [--config] [--profile]... [--agent] [--session]
[--headless] [--json] [-q] [--remember]` (one non-interactive turn; exit 0 when the turn ended `ok`), `va sessions [--json]`, `va cost [--days N] [--json]`,
`va memory list|pending|search Q|confirm ID|forget ID|export [-o F]|import F`, `va tree [--config]`, `va doctor [--config]`.
`serve / propose / history / rollback / reload` exit 2 (later milestones). `VENTRI_HOME` overrides `~/.ventri`.
