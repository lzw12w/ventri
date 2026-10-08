# Ventri —— 受 cordis 启发的 Python 插件内核

**Ventri** 取自拉丁语 *ventriculus*（心室）。cordis 意为“心脏”，而心室负责把血液泵向全身——
Ventri 就是那个“泵”：驱动插件的加载、运行与更替。

Ventri 是一个插件内核，核心创新有两点：

- **(A) 结构化并发**：基于 `anyio`（只跑在 asyncio 上），每个插件（Fiber）拥有自己的任务、监听器、服务与清理 effect，
  卸载时按确定顺序取消并等待结束，不泄漏；
- **(B) 事务化插件变更**：`async with ctx.transaction()` 批量加载/卸载/替换/重配置插件，失败整体回滚，其他插件看不到“半应用”状态。

在此之上，**M1（内核 Alpha，0.2.0a1）** 加入了：作用域（scope / realm）与会话隔离、声明式 YAML 配置（diff 后单事务热应用）、
stop-first 替换、dry-run 事务与 `TxReport`、加载超时与重试、依赖环/缺失提供者诊断、签名注入与 `ventri stubgen`、
`Secret[T]`、事件优先级/类型化事件/拦截器、trace schema v1 与 JSONL sink。

设计为原创实现（只借鉴 cordis 的语义），作为 DeepSeek 个人 Agent 的底座。

**M2（Agent MVP，`ventri-agent 0.2.0a1`）** 在内核之上实现了 **Ventri Agent**：DeepSeek 优先的个人助理，命令 `va`。
Agent 的每一部分都是普通插件：模型适配器、工具、权限引擎、记忆、会话（每个会话是一个隔离的内核 scope）、CLI 渠道，
因此热改配置、会话隔离、`/tree` 诊断都直接来自内核（见下文 [Ventri Agent](#ventri-agentm2va)）。

> **项目路线与完整设计见 [`docs/DESIGN.md`](docs/DESIGN.md)；API 参考见 [`docs/api.md`](docs/api.md)。**
> 许可证：MIT（见 [`LICENSE`](LICENSE)）。要求 Python 3.12+；1.0 只面向 macOS，但内核代码与平台无关（Linux 上同样测试）。

## 快速开始

```bash
uv sync                                   # 创建 .venv，安装 ventri + ventri-std（workspace）与开发依赖
uv run pytest -q                          # 只在 asyncio 上运行
uv run ruff check .
uv run python examples/demo.py
uv run python benchmarks/bench_kernel.py  # DESIGN 4.13 性能目标
```

```python
from typing import Annotated
from ventri import Kernel, plugin

class LLM: ...

@plugin(provides={"llm": LLM}, timeout=10)
async def deepseek(ctx, config):
    ctx.provide(LLM, LLM())                  # provides 里声明了名字 -> 也可写 ctx.llm

def tool(ctx, config, llm: LLM, cache: Annotated[object, "cache"] | None = None):
    # 签名即依赖：llm 必需；cache 可选（出现/消失会重启 tool）
    ctx.on("tool:call", lambda q: ..., priority=10)   # 监听器：高优先级先执行；fiber 卸载时自动移除
    ctx.spawn(background_job)                         # 后台任务：fiber 卸载时取消并等待结束

async with Kernel() as app:
    t = await app.plugin(tool)               # 缺少 LLM -> PENDING，pending_reason="missing: LLM (...)"
    await app.plugin(deepseek)               # LLM 出现 -> tool 自动 ACTIVE
    async with app.transaction(reason="upgrade") as tx:   # 原子变更
        await tx.replace(llm_fiber, config={...})
        await tx.plugin(other)
    print(tx.report)                         # TxReport：增删改、重启、服务变化、失败原因
    session = await app.scope("session:a1", isolate=["memory"])   # 会话作用域
    await session.ctx.plugin(memory_plugin)  # 只在本会话可见；session.dispose() 全部回收
    print(app.tree())
```

## Ventri Agent（M2，`va`）

```bash
uv sync
uv run va init                      # ~/.ventri：ventri.yml、personas/、workspace/；macOS 上把 API key 存进钥匙串
export DEEPSEEK_API_KEY=...         # 其他平台（或 ventri.yml 里 api_key: ${secret:deepseek}）
uv run va chat                      # 流式输出、思考折叠、内联审批；/help 查看斜杠命令
uv run va chat --fake script.json   # 离线：脚本化假模型（演示 / 测试，不需要 key）
uv run va chat --continue           # 恢复最近的会话（进程重启后同样可恢复）
uv run va sessions | va cost | va memory list|pending|search|confirm|forget|export|import | va tree | va doctor
```

```
── session 20261008-145145-aad9 (new) · agent default · deepseek-flash thinking high · 19 tools ──
> 帮我在工作区写个 hello.md
  ▸ thought (41 chars)
  ⚙ fs.write path=~/.ventri/workspace/hello.md
  [approval] fs.write path=~/.ventri/workspace/hello.md  (write-local)
    args: {"path": "hello.md", "content": "# hello\n", "mode": "create"}
    [y] allow once  [s] allow for this session  [n] deny
    approve? y
    ✓ wrote 8 chars to ~/.ventri/workspace/hello.md (create)
已写入 hello.md。
  [turn 2 · 2 steps · 1 tools · in 5.6k (cache 99%) · out 33 · $0.0001 / ¥0.001]
```

| 组件 | 模块 | 要点 |
|---|---|---|
| ModelProvider | `providers/` | DeepSeek 适配器（`deepseek-flash` / `deepseek-v4-pro`；思考模式与 effort；带工具时 `reasoning_content` 回传的本地校验；strict 工具走 `/beta`；JSON 输出 + 一次修复重试；SSE 流式；`usage` 缓存命中；峰谷计价；按模型并发上限；429/5xx 指数退避并遵守 `Retry-After`，只在首个事件之前重试；`/models` 探测）；通用 OpenAI 兼容适配器；离线 `fake`（模拟 DeepSeek 前缀缓存） |
| 路由 | `routes:` | `default`（flash，思考 high）、`plan`（Pro，max；`/think max` 作为独立子调用）、`cheap`（flash，非思考：压缩、记忆抽取） |
| ContextBuilder | `context.py` | ① system ② 工具（按名排序、规范 JSON）③ 记忆快照 ④ 只追加的历史 ⑤ 尾部块（时间、通知、计划）追加进历史、永不改写；epoch 冻结 ①–③ 并写入会话日志；工具集变化在下一 epoch 生效；60% 软上限时压缩一次；>8K token 的工具结果存为工件 |
| AgentLoop | `loop.py` | 预算（步数、工具调用、token、¥、墙钟）；工具崩溃/超时 → `ERROR` 结果；模型错误不丢会话（`/retry`）；只读且 `parallel_safe` 的调用并发；不可信输出加围栏 |
| 工具 | `tools/` | `fs.*`（限定 roots、真实路径）、`shell.run`（每次审批）、`web.fetch`（默认询问，`allow_domains` 可放行；无 web 搜索）、`notes.*`、`time.now`、`artifact.read`、`work.*`、`memory.*`、`inspect.*`；工具名在线上为 `fs__read` |
| 权限 | `permission.py` | 风险 read < write-local < external < irreversible < spend；规则 → 工具默认 → read 放行/其余询问；irreversible+ 永远询问且不可“本会话允许”；批准只能来自渠道（HMAC 令牌）；无渠道/超时 = 拒绝；无门禁 = 拒绝；`audit.jsonl` |
| 记忆 | `memory.py` | SQLite FTS5（trigram，中文可用）；去重合并；敏感项待确认；Markdown 导出/导入；会话结束时用 cheap 路由抽取 |
| 会话 | `sessions.py` | 每个会话 = `scope session:<id>`（隔离 SessionInfo/Budget/Grants/ContextBuilder/AgentLoop…）；JSONL 日志；空闲 30 分钟挂起、7 天保留后结束；崩溃 = 挂起，可从日志恢复 |
| CLI | `channels/cli.py`、`cli.py` | `/think /cost /tree /memory /epoch /compact /retry /sessions /end /suspend /exit` |

实测（2026-10-08，真实 API，`pytest -m live`）：6 轮 Agent 会话输入缓存命中率 **74.8%**，成本 $0.0011；
30 个个人任务评测（`uv run python -m evals.agent.run`）通过 **29/30（96.7%）**，总成本 $0.041。
费用数据以 `usage` 为准；`va cost` 输出每日报表（命中率、峰/谷调用数、USD/CNY）。

## 仓库结构（uv workspace）

| 路径 | 内容 |
|---|---|
| `packages/ventri/src/ventri/` | **L0 内核**（只依赖 `anyio`） |
| &nbsp;&nbsp;`kernel.py` | `Kernel`（根 Context）、realm 注册表、reconcile、事件分发、trace |
| &nbsp;&nbsp;`fiber.py` | `Fiber` 状态机、per-fiber 锁、effect 栈、`spawn`、加载超时与重试 |
| &nbsp;&nbsp;`context.py` | 插件 API：`plugin/scope/provide/get/on/intercept/check/effect/enter/spawn/emit/.../transaction/replace/trace` |
| &nbsp;&nbsp;`transaction.py` / `report.py` | 事务：作用域锁、暂存、stop-first、dry-run、probe、校验、原子交换、回滚；`TxReport` |
| &nbsp;&nbsp;`plugin.py` | 插件描述：签名注入、`provides`、`Config`、`timeout`、`Retry`、`exclusive`；`@plugin` |
| &nbsp;&nbsp;`diagnose.py` | 依赖诊断（Tarjan SCC 找环、缺失/失败提供者） |
| &nbsp;&nbsp;`events.py` / `secret.py` / `trace.py` / `observe.py` | `Event[T]`/`Deny`/`Rewrite`；`Secret[T]` 与打码；trace schema v1；`snapshot()`/`tree()` |
| `packages/ventri-std/src/ventri_std/` | **L1 标准插件**：`config.py`（声明式配置加载器）、`trace.py`（JSONL sink）、`stubgen.py`、`cli.py`（`ventri` 命令） |
| `packages/ventri-agent/src/ventri_agent/` | **Ventri Agent**：`providers/`、`tools/`、`permission.py`、`memory.py`、`session.py`/`sessions.py`、`context.py`、`loop.py`、`channels/cli.py`、`cli.py`（`va`） |
| `tests/` | 内核测试；`tests/std/` 为 ventri-std 测试；`tests/agent/` 为 Agent 测试（假模型 + httpx MockTransport；`-m live` 为真实 API 契约测试）；含 hypothesis 属性测试与混沌测试 |
| `evals/agent/` | 30 个个人任务评测集与运行器（M2 退出标准） |
| `benchmarks/bench_kernel.py` | 4.13 性能目标基准（结果见 [`docs/benchmarks.md`](docs/benchmarks.md)） |
| `docs/` | `DESIGN.md`、`api.md`、`trace-schema.md`（+ JSON Schema）、`benchmarks.md` |

## 架构

- **Context**：每个 Fiber 一个 Context；`Kernel` 本身就是根 Context。
- **插件**：函数 `fn(ctx, config, *deps)`（可 async）或类 `Cls(ctx, config, *deps)`（可选 `start()` / `stop()`）。
  元数据：`name`、`provides`（`{"llm": ModelProvider}` 或 key 列表，用于诊断与 stubgen）、`Config`、`timeout`、`retry`、`exclusive`，
  或旧式 `inject=[...]`。
- **签名注入**：`(ctx, config)` 之后的参数即依赖：`llm: LLM` 必需；`x: X | None = None` 可选（出现/消失会重启插件）；
  `Annotated[T, "key"]` 用字符串 key；其他带默认值的参数不注入；无注解无默认值报 `TypeError`。
- **服务**：`ctx.provide(key, value)`，key 是类或字符串；同一 realm 内同一 key 只能有一个提供者（`ServiceConflict`）。
  `ctx.get(Type) -> Type` 类型化访问；字符串 key、`name=` 或 `provides` 里声明了名字的服务支持 `ctx.<name>`。
  服务的生命周期 = 提供它的 Fiber 的生命周期。
- **Fiber 状态机**：`PENDING → LOADING → ACTIVE → UNLOADING → PENDING | DISPOSED`，加载失败/超时为 `FAILED`。
  依赖不全停在 `PENDING`（并给出 `pending_reason`）；依赖到齐自动加载；依赖消失自动 `ACTIVE → PENDING`。
  **依赖方先拆、服务后删**；**per-fiber 一致视图**（`ctx.get` 对已注入的 key 返回激活时绑定的实例）。
- **Reconciler**：最外层公开操作结束时，只对“脏”区域（被改动的 realm 所覆盖的子树）跑不动点循环。

### 作用域与隔离（M1）

`await ctx.scope("session:a1", isolate=["memory", WorkingMemory])` 创建一个作用域 fiber：

- **realm**：scope 内对 `isolate` 列出的 key 的 `provide/get` 使用 scope 自己的 realm；其余 key 解析到最近的、隔离了它的祖先 realm，
  否则根 realm（**不会向外回退**：在 scope 中隔离的 key 若 scope 内没有提供者，即视为缺失）。兄弟 scope 互不可见。
- **事件过滤**：在 scope S 内发出的事件只投递给 S 子树与 S 的祖先链上的监听器，不投递给兄弟 scope；根上发出的事件投递给所有人。
- **回收**：`await scope.dispose()` 确定性地回收其中所有 fiber、任务、监听器、服务；回收后 `snapshot()` 回到基线（测试断言）。
- **事务锁细化**：每个 scope 一把 FIFO 读写锁；事务对自己的 scope 取写锁、对祖先取读锁（从根开始），
  因此不同会话的事务可以并发，根事务与会话事务互斥。`wait=False` 时立即 `TransactionBusy`。

### 事件（M1）

- 四种分发：`emit`（顺序、错误隔离）、`parallel`（并发、错误隔离）、`serial`（顺序、返回首个非 None、错误上抛）、`bail`（同步版 serial）。
- `priority`：高者先，同优先级按注册序；所有分发模式一致。
- **类型化事件**：`MessageIn = Event[ChannelMessage]("message.in")`，`ctx.on(MessageIn, handler)` 的 handler 参数有类型；字符串事件名保留。
- **拦截器**：`ctx.intercept(ToolCall, fn)`，`await ctx.check(ToolCall, call)` 按优先级执行：返回 `Deny(reason)` 立即拒绝，
  `Rewrite(new)` 替换值并继续，`None` 放行；拦截器抛错向上传播（fail closed）。

## (A) 结构化并发

1. `Kernel` 持有根 task group；`ctx.spawn` 的任务在其中运行，但**归属**于所属 fiber：每个任务有独立 `CancelScope`，
   并作为 effect 压入该 fiber 的 LIFO 栈：卸载时**取消并等待其结束**，与其他 effect 严格按逆序交错执行。关闭 Kernel 会取消一切。
   （M0 中每个 fiber 还有一个 task group 宿主任务作兜底；M1 去掉了它——任务本来就由 effect 拥有——空 fiber 内存从 ~10 KB 降到 ~3.4 KB。）
2. **确定的拆除顺序**：子 fiber（新→旧，递归）→ 本 fiber 的 effect（LIFO：监听器、服务、任务、用户 disposer）。
   拆除过程被 shield，不会被外部取消打断；清理异常只记录到 trace，不中断后续清理。
3. **任务异常的归属**：spawn 任务抛异常 → 所属 fiber 被拆除并置为 `FAILED`（supervisor 语义），兄弟插件和 Kernel 不受影响。
4. **重入安全**（每个 fiber 一把锁，记录持锁任务）：加载中被 dispose、在 apply 里 dispose 自己、在自己的任务里 dispose 自己都不会死锁。
5. **取消即不泄漏**：调用 `await ctx.plugin(...)` 的任务被取消时，正在加载的 fiber 被完整拆除并 `DISPOSED`。
6. **加载超时**（M1）：插件元数据 `timeout`（默认 30 s，`Kernel(load_timeout=)`/配置可覆盖，`None` 关闭）；超时取消 apply →
   清理 → `FAILED(LoadTimeout)`。只覆盖加载，不覆盖运行期任务。
7. **重试策略**（M1）：`Retry(max, backoff="exp"|"fixed", base, cap, reset_after)`；默认不重试；staged fiber 不重试；
   dispose 取消已排期的重试；`restart()` 重置计数。

## (B) 事务化插件变更

采用 **蓝绿暂存（blue/green staging）**：

- `tx.plugin` / `tx.replace` 创建 **staged fiber**，真实运行，但其服务写入事务私有的 **overlay**（按 `(realm, key)`），
  只有同一事务内的 staged fiber 可见；它们的监听器只接收事务内部发出的事件。
- `tx.dispose` / `tx.replace` **不会停止旧 fiber**，只在 overlay 中打“墓碑”。
- **提交**：结算 staged fiber → 校验（`FAILED`、`strict` 下未 ACTIVE、事务作用域或 staged fiber 在事务期间被外部回收、外部抢占同一 key）
  → **在一个同步步骤内把 overlay 应用到 live 注册表** → 重启绑定已变化的 live 依赖方 → 拆除被移除/替换的旧 fiber → reconcile。
- **回滚**：逆序 dispose 所有 staged fiber，丢弃 overlay；旧 fiber 从未停止，保留原 config、原实例和内存状态。
- **stop-first 替换**（M1）：`exclusive=True` 的插件（如独占端口）默认用 `strategy="stop-first"`：新 fiber 停放到提交时，
  先停旧实例再启动新实例；新实例失败则以原 config 重启旧实例——**降级回滚**（配置与服务拓扑恢复，旧实例内存状态不保留，
  依赖方会观察到短暂空窗；`TxReport.degraded=True`）。
- **dry-run**（M1）：`transaction(dry_run=True, probe=...)` 执行一切到提交前（加载、结算、校验、probe），然后总是回滚；
  `tx.get()` 返回事务视角的服务，可用于 probe。stop-first 替换在 dry-run 中不启动（记入 `skipped`）。
- **元数据与超时**（M1）：`transaction(origin=, reason=, timeout=)`，写入 trace；超时回滚并抛 `TransactionTimeout`。
- **诊断**（M1）：每轮 reconcile 后对 PENDING fiber 做 Tarjan SCC；环上的 fiber 得到 `pending_reason="cycle: a#1 → b#2 → a#1"` 并发出
  `dep.cycle`；`strict=True` 的事务因环失败时抛 `DependencyCycle`。

### 精确的保证

| # | 保证 |
|---|---|
| G1 原子性 | live 注册表从“事务前”到“事务后”只在一个同步步骤内切换；任何观察者只能看到全部或全无。 |
| G2 隔离性 | 提交前：staged 服务对 live fiber 不可见；staged 监听器听不到外部事件；被移除/替换的 fiber 照常运行。 |
| G3 回滚 | apply 抛错、块内代码抛错、事务被取消/超时、提交前校验失败——都会回滚，回滚后 `app.snapshot()` 与事务前**完全相等**（属性测试断言）。 |
| G4 串行化 | 同一 scope 上的事务串行（FIFO）；不同会话 scope 的事务可并发；祖先 scope 的事务与子 scope 的事务互斥；嵌套事务报错。 |
| G5 冲突检测 | 事务期间允许非事务操作，若与 overlay 冲突（或回收了事务的 scope / staged fiber），提交时检测出并回滚。 |

**不保证的内容**：staged 插件对外部世界的副作用不会被撤销；不可回头点之后的失败不回滚（表现为 `FAILED` + trace）；
蓝绿模式要求新旧实例能短暂共存（否则用 stop-first，其回滚是降级的）；live 依赖方在交换后会短暂重启。

## 声明式配置（ventri-std）

```yaml
# ~/.ventri/ventri.yml
version: 1
profiles: [home]                 # 叠加 profiles/home.yml 的有序补丁
plugins:
  - use: ventri_std.trace.jsonl
    config: { path: ~/.ventri/trace/, rotate_mb: 64 }
  - use: mypkg.providers.deepseek
    id: ds
    config: { api_key: "${secret:deepseek}", model: deepseek-flash }   # 钥匙串，或 $VENTRI_SECRET_DEEPSEEK
  - group: tools
    plugins:
      - use: mypkg.tools.fs
        config: { roots: [~/notes] }
```

- 流程：读 YAML → 合并 profiles（`{id, config}` 深合并 / `disabled` / `add`（可 `under` 某 group）/ `remove`）→ 解析 secrets
  （`${secret:x}` → `Secret`；`${env:X}`、`${env:X:-默认}`）→ 解析 `use`（entry point `ventri.plugins`、`mod:attr`、点路径）→
  按插件 `Config` 校验（**收集全部错误**，任何事务开始前抛 `ConfigError`）→ 按稳定 id diff → **一个事务**（`origin="config"`）。
- **稳定 id**：显式 `id`，否则为 `use`（同一父节点下第 n 个相同 `use` 记为 `use@n`）；group 用其名字。
- **diff**：缺失或 `disabled` 的条目被卸载；`use`、校验后的 config、`timeout`、`retry` 变化或处于 `FAILED` 的条目被替换；新条目被加载。
  config 比较基于**校验后的模型**，格式变化或写出默认值不会触发重启。
- 失败 → 回滚，**旧配置继续服务**，`ApplyResult` 指出失败的 fiber 与原因；文件监视（轮询 + 300 ms 去抖：一次保存 = 一个事务）。
- secrets 永不进入快照/trace/报告：`Secret[T]` 类型 + 键名模式（`*key*`、`*token*`、`*secret*`、`*password*`）双重打码。

## `ventri` 命令

```bash
ventri apply --dry-run [ventri.yml] [--profile work] [--strict] [--json]   # 打印 plan 与 TxReport，不生效
ventri apply --validate-only                # 只解析/校验，不实例化任何插件
ventri tree                                 # 配置产生的插件树（dry-run 中的 staged 树）
ventri doctor                               # 环境、配置、secrets、插件解析、会停在 PENDING/失败的插件
ventri run                                  # 前台托管配置，编辑文件即热应用
ventri stubgen [-m mypkg.plugins] [--out .] # 生成 ventri_stubs.pyi：class Ctx(Context): llm: ModelProvider
```

M1 没有守护进程控制通道，所以 `ventri apply` 不带 `--dry-run` 会报错退出；`apply/tree/doctor` 在一次性 Kernel 中以 dry-run
事务执行（插件会被实例化并拆除；`--validate-only` 不会）。

## 可观测性

- `app.tree()` / `app.snapshot()`：状态、配置 id、staged 标记、config（打码）、依赖、provides、任务数、错误、`pending_reason`、scope 与 realm；
- trace：`app.on_trace(cb)` / `app.trace_log`（环形缓冲）；`event.to_dict()` 为 **schema v1**
  `{v, seq, ts, kind, fiber, scope, tx, attrs}`（见 [`docs/trace-schema.md`](docs/trace-schema.md)）；上层用 `ctx.trace("agent.turn", ...)` 发自定义事件；
- `use: ventri_std.trace.jsonl`：缓冲写入、按大小轮转、只保留最新 N 个文件、回填加载前的环形缓冲。

## 测试

`pytest` + anyio 插件，只在 **asyncio** 上运行。内核与 ventri-std 共 150+ 个测试：M0 的 24 个 + 作用域/锁/注入/超时重试/诊断/dry-run/stop-first/
事件/Secret/trace/JSONL/配置/CLI/stubgen（含 pyright `--strict` 检查 `ctx.llm` 的类型）测试、hypothesis 属性测试
（随机事务程序回滚后快照不变、作用域隔离与回收、事件顺序、配置 diff 收敛）、以及覆盖作用域与事务的混沌测试
（`VENTRI_CHAOS_SEEDS=1000 uv run pytest tests/test_chaos.py`，1000 个种子已验证）。在 Python 3.12 / 3.13 / 3.14 上通过。

Agent 测试（`tests/agent/`，116 个离线 + 8 个 live）默认全部离线：DeepSeek 适配器针对 httpx `MockTransport` 的 SSE 流测试，Agent 循环用脚本化
假模型（同时模拟 DeepSeek 的前缀单元缓存，用于验证 ≥ 70% 命中率）；红队用例让假模型“完全服从”注入内容，验证零未批准的
write-local 及以上动作。真实 API 契约测试只在设置 `DEEPSEEK_API_KEY` 时运行（`uv run pytest -m live -s`，每次 < $0.01），
并由 `.github/workflows/deepseek-contract.yml` 每晚（北京时间 01:30，谷时）运行。

## 已知限制

- 插件**源码**修改不会被检测（开发模式代码重载不在 M1）；OpenTelemetry 导出与配置目录 git 账本在 M3；沙箱不在 M1；
- `ventri apply` 无守护进程通道（见上）；
- 只支持 asyncio；1.0 只支持 macOS（CI 以 macOS 为主，Linux 做回归）。
- Agent（M2）：飞书渠道与本地 Web UI 在 M4；MCP、沙箱、Skills、例行任务与自我演化在 M3（`va propose/history/rollback/serve`
  已保留并以退出码 2 提示）；`shell.run` 不是沙箱（只限工作目录、超时、剥离敏感环境变量）；没有待审批队列（无渠道 = 拒绝）。
