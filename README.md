# Ventri —— 受 cordis 启发的 Python 插件内核原型

**Ventri** 取自拉丁语 *ventriculus*（心室）。cordis 意为“心脏”，而心室负责把血液泵向全身——
Ventri 就是那个“泵”：驱动插件的加载、运行与更替。

Ventri 是一个小而可运行的插件内核（库代码约 1000 行有效代码），用来验证两个创新点：

- **(A) 结构化并发**：基于 `anyio`，每个插件（Fiber）拥有自己的 task group，任务树与插件树一一对应；
- **(B) 事务化插件变更**：`async with ctx.transaction()` 批量加载/卸载/替换/重配置插件，失败整体回滚，其他插件看不到“半应用”状态。

设计为原创实现（只借鉴 cordis 的语义：Context / 服务 / inject / Fiber 生命周期 / effect），未参考任何现有 Python 移植。
后续计划作为 DeepSeek 个人 Agent 的底座。

> **项目路线与完整设计见 [`docs/DESIGN.md`](docs/DESIGN.md)（PDF：[`docs/DESIGN.pdf`](docs/DESIGN.pdf)）。**

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install anyio pytest trio
.venv/bin/python -m pytest -q          # asyncio + trio 两个后端各跑一遍
.venv/bin/python examples/demo.py
```

```python
from ventri import Kernel, plugin

class LLM: ...

@plugin(name="tool", inject=[LLM])
def tool(ctx, config):
    llm = ctx.get(LLM)                       # 类型化访问
    ctx.on("tool:call", lambda q: ...)       # 监听器：fiber 卸载时自动移除
    ctx.spawn(background_job)                # 后台任务：fiber 卸载时取消并等待结束

async with Kernel() as app:
    t = await app.plugin(tool)               # 缺少 LLM -> PENDING
    await app.plugin(llm_plugin, {...})      # LLM 出现 -> tool 自动 ACTIVE
    async with app.transaction() as tx:      # 原子变更
        await tx.replace(llm_fiber, config={...})
        await tx.plugin(other)
    print(app.tree())
```

## 文件结构

| 文件 | 内容 |
|---|---|
| `ventri/kernel.py` | `Kernel`（根 Context）、服务注册表、reconcile 循环、事件分发、trace |
| `ventri/fiber.py` | `Fiber` 生命周期状态机、per-fiber 锁、task group 宿主任务、effect 栈、`spawn` |
| `ventri/context.py` | 插件看到的 API：`plugin/provide/get/on/effect/enter/spawn/emit/.../transaction/replace` |
| `ventri/transaction.py` | 事务：暂存（staging）、提交校验、原子交换、回滚 |
| `ventri/observe.py` | `snapshot()` / `tree()` |
| `ventri/plugin.py` | 插件描述（函数或类、`inject`、`Config`）与 `@plugin` 装饰器 |
| `tests/` | 24 个测试 × 2 个后端（asyncio、trio）= 48 |
| `examples/demo.py` | 假 LLM 服务 + 工具插件 + 后台任务 + 失败事务回滚 |

## 架构

- **Context**：每个 Fiber 一个 Context（`ctx.parent` 可向上走）；`Kernel` 本身就是根 Context。
- **插件**：函数 `fn(ctx, config)`（可 async）或类 `Cls(ctx, config)`（可选 `start()` / `stop()`）。
  元数据取自属性：`name`、`inject`（类或字符串 key 列表）、`Config`（若存在且 config 为 dict，则 `Config(**config)`）。
- **服务**：`ctx.provide(key, value, name=None)`，key 是类或字符串；同一 key 只能有一个提供者（否则 `ServiceConflict`）。
  `ctx.get(Type) -> Type` 类型化访问；字符串 key 或带 `name=` 的服务支持 `ctx.<name>` 属性访问。
  服务的生命周期 = 提供它的 Fiber 的生命周期（作为 effect 注册）。
- **Fiber 状态机**：`PENDING → LOADING → ACTIVE → UNLOADING → PENDING | DISPOSED`，加载失败为 `FAILED`。
  - inject 依赖不全时停在 `PENDING`；依赖到齐自动加载；依赖消失自动 `ACTIVE → PENDING`，再出现自动恢复。
  - **依赖方先拆、服务后删**：移除服务前先把依赖它的插件卸载，所以依赖方在自己的清理代码里仍可使用该服务。
  - **per-fiber 一致视图**：`ctx.get` 对已 inject 的 key 返回该 fiber 激活时绑定的实例，直到它重启。
- **Reconciler**：每个公开操作（`plugin`/`dispose`/`restart`/事务操作）可以嵌套；最外层操作结束时跑一次
  不动点循环：先停掉依赖已失效的 ACTIVE fiber（深/新者优先），再按树序激活依赖已满足的 PENDING fiber。
  在任何操作之外直接调用 `provide`（如在后台任务里）会在后台调度一次 reconcile；需要确定性时 `await app.settle()`。
- **事件**：`emit`（顺序、错误隔离）、`parallel`（并发、错误隔离）、`serial`（顺序、返回首个非 None、错误上抛）、
  `bail`（同步版 serial）。监听器随 fiber 自动清理。

## (A) 结构化并发

1. `Kernel` 持有根 task group。Fiber 激活时，在**父 fiber 的 task group** 中启动一个宿主任务，宿主任务内开自己的 task group；
   `ctx.spawn` 的任务都在这里。于是任务树 = 插件树，关闭 Kernel 会取消一切。
2. 每个 spawn 的任务有独立 `CancelScope`，并作为 effect 压入 LIFO 栈：卸载时**取消并等待其结束**，与其他 effect 严格按逆序交错执行。
3. **确定的拆除顺序**：子 fiber（新→旧，递归）→ 本 fiber 的 effect（LIFO：监听器、服务、任务、用户 disposer）→ 关闭 task group 兜底。
   拆除过程被 shield，不会被外部取消打断；清理异常只记录到 trace，不中断后续清理。
4. **任务异常的归属**：spawn 任务抛异常 → 所属 fiber 被拆除并置为 `FAILED`（类似 supervisor），兄弟插件和 Kernel 不受影响。
5. **重入安全**（每个 fiber 一把锁，记录持锁任务）：
   - 加载中被另一任务 `dispose`：取消加载用的 CancelScope → 等锁 → 加载方展开时清理已注册的 effect → `DISPOSED`；
   - 插件在自己的 apply 里 dispose 自己（或子插件 dispose 正在加载的父插件）：检测到持锁者是当前任务，只做标记+取消，由持锁者收尾，不会死锁；
   - 在自己 spawn 的任务里 dispose 自己：不等待自身所在的 task group 退出，避免自等待死锁。
6. **取消即不泄漏**：调用 `await ctx.plugin(...)` 的任务若被取消，正在加载的 fiber 会被完整拆除并 `DISPOSED`
   （服务、监听器、任务、子插件都不残留——见 `test_cancellation_mid_load_leaks_nothing`）。

## (B) 事务化插件变更

采用 **蓝绿暂存（blue/green staging）**，而不是“先改再补偿”：

- `tx.plugin` / `tx.replace` 创建 **staged fiber**，它们真实运行（apply、spawn、监听），
  但其服务写入事务私有的 **overlay**，只有同一事务内的 staged fiber 可见；它们的监听器只接收事务内部发出的事件。
- `tx.dispose` / `tx.replace` **不会停止旧 fiber**，只在 overlay 中为其服务打“墓碑”，让 staged fiber 看不到；外部世界继续使用旧 fiber。
- **提交**：
  1. 结算 staged fiber（可能此时才满足依赖并加载）；任一 `FAILED`（或 `strict=True` 时未 ACTIVE）→ 回滚；
  2. 对 live 注册表做乐观校验（事务期间有外部插件抢占了同一 key → `TransactionConflict` → 回滚）；
  3. **在一个同步步骤内（中间没有 await）把 overlay 应用到 live 注册表** —— 不可回头点；
  4. 重启绑定已变化的 live 依赖方（旧实例此时仍存活，依赖方清理时看到的是旧实例），然后拆除被移除/被替换的旧 fiber，再 reconcile。
- **回滚**：按逆序 dispose 所有 staged fiber（执行它们的清理），丢弃 overlay。被“移除/替换”的旧 fiber 从未停止，
  因此无需“恢复”——保留原 config、原实例和内存状态（比“重启旧插件”更强）。
- `ctx.replace(fiber, new_plugin_or_config)` = 单操作事务；位置参数可调用视为新插件，否则视为新 config。

### 精确的保证

| # | 保证 |
|---|---|
| G1 原子性 | live 注册表从“事务前”到“事务后”只在一个同步步骤内切换；任何观察者（其他插件、`ctx.get`、trace 回调）只能看到全部或全无。测试中用 trace 回调在每个事件上断言 `LLM` 服务始终存在。 |
| G2 隔离性 | 提交前：staged 服务对 live fiber 不可见；staged 监听器听不到外部事件；被移除/替换的 fiber 照常运行。 |
| G3 回滚 | apply 抛错、块内用户代码抛错、事务被取消、提交前校验失败——都会回滚，回滚后 `app.snapshot()` 与事务前**完全相等**（测试断言）。 |
| G4 串行化 | 每个 Kernel 同时只有一个事务：`wait=True` 排队（先到先得），`wait=False` 立即 `TransactionBusy`；嵌套事务直接报错。 |
| G5 冲突检测 | 事务期间允许非事务操作，但若它们与 overlay 冲突，提交时检测出并回滚。 |

**不保证的内容**：
- staged 插件对外部世界的副作用（网络请求、写文件、后台任务做的事）不会被撤销，只会运行其清理 effect；
- 不可回头点之后的失败（依赖方重启失败、旧 fiber 清理抛错）不会回滚，表现为 `FAILED` fiber + trace 事件；
- 蓝绿模式要求新旧实例能短暂共存（例如独占端口的插件需要 “stop-first” 策略，尚未实现）；
- live 依赖方在交换后会经历短暂重启（UNLOADING→PENDING→LOADING），期间不服务；但它们**不会观察到服务缺失**。

## 可观测性

- `app.tree()`：树形文本（状态、staged 标记、config（敏感字段打码）、inject、provides、任务数、错误）+ 服务表；
- `app.snapshot()`：同样信息的 dict，可直接比较；
- `app.on_trace(cb)` / `app.trace_log`：`fiber.state`、`service.bind/unbind`、`task.spawn/error`、`effect.error`、`event.error`、`tx.begin/commit/rollback` 等事件。

## 测试

`pytest` + anyio 插件，**asyncio 与 trio 双后端**各跑一遍。覆盖：inject 挂起→激活、服务移除自动卸载与恢复、级联拆除顺序、
spawn 任务随卸载取消、Kernel 关闭取消所有任务、任务崩溃只影响所属 fiber、加载中 dispose、自我 dispose、取消加载不泄漏、
类型化访问/冲突、事件四种模式、快照/trace、事务提交与隔离、apply 抛错回滚、用户代码抛错回滚、提交期失败/strict 回滚、
replace 成功与回滚（含替换失败的 FAILED fiber）、并发事务排队/拒绝/嵌套报错、提交冲突、事务卸载、事务被取消回滚，
以及一个随机并发加载/卸载的混沌测试（40 个随机种子验证过）。

## 已知限制 / 下一步

- 服务是 Kernel 全局的：没有 cordis 的 isolate / 作用域服务、属性代理（mixin）、可选依赖；
- `emit` 是 async（cordis 是同步）；监听器无优先级/prepend；
- `FAILED` fiber 不会自动重试（用 `fiber.restart()` 或 `ctx.replace` 修复）；依赖环只会静默停在 PENDING，没有诊断；
- 事务锁是 Kernel 级的粗粒度锁，可细化为按 key/子树加锁；reconcile 每轮 O(N)，适合数百插件规模；
- apply 没有超时（可在插件内用 `anyio.fail_after`，或后续在内核层加）；
- 下一步（面向 DeepSeek Agent）：`stop-first` 替换策略；从配置文件计算期望状态并以单个事务 diff 应用（声明式热重载）；
  DeepSeek 客户端服务插件（`ctx.enter(httpx.AsyncClient())`）、工具注册表服务、MCP 桥接；trace 导出到 OpenTelemetry/JSONL。
