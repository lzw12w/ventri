# Terminal-Bench 初步成果：Ventri Agent + deepseek-flash

> 日期：2026-10-08 · 状态：**试点（12 题，非正式分数）**

## 结论

- **11/12 通过（91.7%）**，95% 置信区间约 65%–98%（Wilson）。样本太小，只能说明 Ventri 能稳定完成真实终端任务，不能和榜单分数直接比较。
- **很省钱**：12 题总花费 **$0.31**（按 DeepSeek 峰时价；谷时约一半），平均每题约 $0.026。
- **缓存命中率 97.9%**：输入 525 万 token 中 514 万命中缓存，这是低成本的主要原因。
- 唯一失败的一题主要是题面没写清判分要求，不是能力问题（见下文）。

## 设置

| 项 | 值 |
|---|---|
| Agent | Ventri Agent（M2，提交 81dfd0d 之后的版本），通过 Harbor 适配器运行 |
| 模型 | `deepseek-flash`（thinking 开启） |
| 基准 | Terminal-Bench 2.0（10 题）+ Terminal-Bench 2.1（2 题），每题 1 次 trial |
| 运行环境 | Harbor 框架，本地 Docker，默认超时和资源 |
| 选题 | 2.1 的 2 题取自按难度分层随机抽样的清单（seed 20261008）；2.0 的 10 题为试点选题 |

难度标注取自 2.1 的 `task.toml`。

## 逐题结果

| # | 版本 | 任务 | 难度 | 类别 | 结果 | Agent 用时 | 输入 token（缓存命中） | 输出 token | 成本（峰时） |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.0 | fix-git | easy | 软件工程 | ✅ | 22 s | 70K（94%） | 2.4K | $0.005 |
| 2 | 2.0 | prove-plus-comm | easy | 软件工程 | ✅ | 39 s | 19K（80%） | 1.2K | $0.003 |
| 3 | 2.1 | overfull-hbox | easy | 调试 | ✅ | 500 s | 528K（98%） | 20.8K | $0.032 |
| 4 | 2.0 | git-leak-recovery | medium | 软件工程 | ✅ | 27 s | 57K（91%） | 4.6K | $0.007 |
| 5 | 2.0 | log-summary-date-ranges | medium | 数据处理 | ✅ | 14 s | 28K（89%） | 1.9K | $0.003 |
| 6 | 2.0 | openssl-selfsigned-cert | medium | 安全 | ✅ | 20 s | 41K（94%） | 3.0K | $0.005 |
| 7 | 2.0 | regex-log | medium | 数据处理 | ✅ | 93 s | 181K（98%） | 16.5K | $0.022 |
| 8 | 2.0 | sqlite-db-truncate | medium | 调试 | ✅ | 326 s | 839K（99%） | 66.4K | $0.087 |
| 9 | 2.1 | build-cython-ext | medium | 调试 | ✅ | 369 s | 2,352K（98%） | 28.4K | $0.061 |
| 10 | 2.0 | cancel-async-tasks | hard | 软件工程 | ✅ | 144 s | 158K（99%） | 15.8K | $0.021 |
| 11 | 2.0 | fix-code-vulnerability | hard | 安全 | ✅ | 28 s | 123K（91%） | 3.3K | $0.008 |
| 12 | 2.0 | configure-git-webserver | hard | 系统管理 | ❌ | 258 s | 853K（99%） | 39.5K | $0.056 |

**按难度**：easy 3/3，medium 6/6，hard 2/3。

**合计**：输入 5.25M token（缓存命中 97.9%），输出 0.20M token，总成本 $0.31；agent 用时中位数 66 s，平均 154 s，最长 500 s。

## 失败分析：configure-git-webserver

- 判分脚本用 `git@localhost` 和密码 `password` 登录推送，题面没有写这一要求。
- Ventri 搭好了可用的 git 服务和 web 服务，但收尾时把测试账户和 web 根目录清空了，判分时拿到 404。
- 同题 Terminus 2 通过，主要是因为它测试时推上去的 hello.html 留在了目录里。
- 结论：主要是题目规范不足、判分宽松；Ventri 侧可以改进的是“交付时保留可演示成品，不要清场”。

## 跑分中发现的 Ventri 问题

1. **后台进程阻塞 `shell.run`**：`nohup ... &` 会卡住 20–120 s，限时紧的题容易超时。需要真正的后台执行或持久 shell。
2. **用了运行时自带的 Python**：agent 调用了适配器带进容器的 `/opt/ventri/py/bin/python3`，正式提交时可能被审计质疑。应隐藏运行时或在提示中禁止。
3. **上下文膨胀**：build-cython-ext 用了 61 步、235 万输入 token，这次靠缓存才便宜。需要长输出摘要和旧步骤裁剪。
4. **缺少无人值守模式**：系统规则、时间注入和不可撤销操作的强制审批都关不掉。

## 参考

- Terminal-Bench 2.0 榜已冻结；唯一的 DeepSeek 条目是 Terminus 2 + DeepSeek-V3.2 = 39.6%。
- Terminal-Bench 2.1 榜（22 条）没有 DeepSeek 条目，目前只收维护者运行的结果；同一模型换 harness 的差距约 0–8 分。
- 榜单：[Terminal-Bench leaderboard](https://www.tbench.ai/leaderboard)

## 正式跑分前的准备

- 适配器总时长写死在 870 s，需改为跟随每题限时。
- 输出 ATIF 格式轨迹（榜单提交必需）。
- 先修上面第 1、4 条问题。
- 全量估算（89 题 × 5 次）：谷时约 $16，峰时约 $32，本机并发 4 约 19 小时。
