# 飞书 / Lark 机器人接入（`va serve`）

Ventri Agent 的飞书渠道（`ventri_agent.channels.feishu`）用官方 SDK `lark-oapi` 的**长连接**收事件：
不需要公网地址、回调 URL 或内网穿透，`va serve` 主动连到飞书。设计说明见 `docs/DESIGN.md` M2 实施说明第 21 条。

两种接法：

- **推荐：扫码创建**——一条命令 `va feishu setup`，用飞书 App 扫码确认，自动创建机器人应用并写好配置（下一节）。
- **备选：手动创建**——在开发者后台自己建企业自建应用、开权限、订阅事件、发版本（下文“备选：手动创建”）。
  扫码不可用（比如企业管理员禁止成员创建应用）时用它。

## 推荐：扫码创建（`va feishu setup`）

```bash
uv sync --extra feishu            # 或 pip install 'ventri-agent[feishu]'：lark-oapi + qrcode
uv run va init                    # 还没有 ~/.ventri/ventri.yml 时（init 结束时也会问要不要扫码创建）
uv run va feishu setup            # 终端里显示二维码；加 --qr-png ~/feishu-qr.png 另存一张 PNG
uv run va serve -v                # 机器人上线；在飞书里给它发消息
```

`va feishu setup` 做的事（移植自 Hermes Agent 的扫码创建，协议细节与官方 SDK `lark-oapi` 的
`register_app` 一致，见 `THIRD_PARTY_NOTICES.md`）：

1. 向 `accounts.feishu.cn/oauth/v1/app/registration` 发起飞书的“一键创建智能体应用”注册（OAuth 设备码流程，
   `archetype=PersonalAgent`），拿到一个验证链接，显示成二维码（终端半块字符；`--qr-png PATH` 另存 PNG，
   权限 0600；没装 `qrcode` 时只显示链接，可在手机上直接打开）。
2. 你用飞书（或 Lark）App 扫码、确认创建。命令轮询直到确认、拒绝或过期（约 10 分钟；网络错误自动重试）。
   Lark 国际版账号会被自动识别并切到 `larksuite.com`（也可 `--domain lark`）。
3. 拿到新应用的 **App ID / App Secret** 和**扫码人的 open_id**，用新凭据调 `/open-apis/bot/v3/info` 确认机器人可用。
4. 保存 App Secret（**从不打印、不写进 ventri.yml、不进日志**）：macOS 存钥匙串（服务 `ventri`，账户
   `feishu_app_secret`）；其他平台或钥匙串不可用时存 `~/.ventri/secrets/feishu_app_secret`（目录 0700、文件 0600）。
   `--secret-store keychain|file` 可强制选择。
5. 改 `~/.ventri/ventri.yml`：先备份成 `ventri.yml.bak-<时间>`，然后启用飞书渠道块（`va init` 模板里注释掉的示例
   会被原地替换；已有的块会被更新、保留你的其他设置；其余内容和注释不动），写入 `app_id`、
   `app_secret: "${secret:feishu_app_secret}"`、`domain` 和 `allow_users: [<扫码人的 open_id>]`。
   白名单仍是**失败关闭**：除了扫码的你，谁都不能用。

再次运行是幂等的：已配置且密钥可读时直接说明“已配置”并退出，不联网；`--force` 才会再建一个新应用
（新应用 = 新 App ID，**open_id 按应用区分**，所以白名单会换成新应用下你的 open_id）。`--json` 在 stdout
输出 JSON 行事件（`qr` / `status` / `done` / `error`，从不含密钥），供脚本或其他 Agent 驱动；人类可读的输出与二维码改走 stderr。
退出码：0 成功，1 注册失败（拒绝 / 过期 / 网络），2 没有配置文件等用法错误。

> **安全提示**：二维码是一次性的，约 10 分钟过期。**谁先扫码，谁就创建这个应用并成为唯一的授权用户**——
> 只用你自己的手机扫，不要把 PNG 发给别人。确认后可在 [开发者后台](https://open.feishu.cn/app) 的应用列表里看到它。

### 扫码之后可能还需要在后台做的事

飞书官方文档（[一键创建飞书智能体应用](https://open.feishu.cn/document/mcp_open_tools/integrating-agents-with-feishu/overview)）
说明这个模板会预置常用权限、事件和回调，“扫码确认后即可投入使用”。区分已核实和未核实的部分：

| 项目 | 状态 |
|---|---|
| 机器人能力；权限 `im:message.p2p_msg:readonly`、`im:message.group_at_msg:readonly`、`im:message:send_as_bot`、`im:message:update`、卡片读写等 | 官方文档列为模板默认项（**文档已核实**，未用真实应用验证） |
| 事件 `im.message.receive_v1`（以及进出群、表情回复等）、回调 `card.action.trigger`，默认用**长连接**订阅 | 官方文档列为默认项（**文档已核实**，未用真实应用验证） |
| 是否还要「创建版本并发布」、企业是否需要管理员审核 | **未核实**：官方说明说可直接使用；Hermes 的通用文档仍要求为卡片回调发布版本。若私聊无响应或点审批按钮报 **200340**，见下 |
| 审批按钮（卡片回调）不做后台操作就能用 | **未核实**（Hermes 的扫码路径也没有额外后台步骤，但它的排查文档把 200340 归因于回调未配置/未发布） |
| `--minimal`（`addons.preset=false`，只申请 Ventri 用到的权限、事件和回调，不要文档/云空间等权限） | **未核实**：最小模板下事件订阅方式是否默认为长连接未见文档说明；默认不加 |

如果私聊没反应，或点审批按钮报 200340，在 [开发者后台](https://open.feishu.cn/app) 打开这个应用：

1. 「事件与回调」→「事件配置」：订阅方式为**使用长连接接收事件**，已添加 **接收消息 v2.0**（`im.message.receive_v1`）。
2. 「事件与回调」→「回调配置」（另一个页签）：订阅方式为**使用长连接接收回调**，已添加 **卡片回传交互**（`card.action.trigger`）。
   保存长连接方式前要先运行 `va serve`（后台会检测连接）。
3. 「版本管理与发布」→「创建版本」→ 可用范围至少包含你自己 →「申请发布」（企业应用需管理员审核）。

群聊：把机器人拉进群并 @ 它，它会回复群的 `chat_id`（`oc_...`），填进 `allow_chats` 保存即可（`va serve` 热应用）。

### 密钥存在哪里（`${secret:...}` 的查找顺序）

`${secret:name}` 依次查：macOS 钥匙串（服务 `ventri`、账户 `name`）→ 环境变量 `VENTRI_SECRET_<NAME>` →
文件 `$VENTRI_HOME/secrets/<name>`（默认 `~/.ventri/secrets/`）。文件后端为没有钥匙串的机器（Linux 服务器）准备：
一个密钥一个文件，末尾换行忽略；文件必须是当前用户拥有的普通文件且权限 0600（或 0400）、目录不能被他人写，
否则直接报错（失败关闭，不会悄悄跳过）；不跟随符号链接。注意环境变量优先于文件：设置了
`VENTRI_SECRET_FEISHU_APP_SECRET` 时文件里的值不会生效（`va feishu setup` 会提醒）。

## 备选：手动创建

需要你提供：一个飞书**企业自建应用**的 **App ID** 和 **App Secret**，以及你自己的 **open_id**（按 M7 获取）。

### M1. 创建应用

1. 打开 [飞书开放平台开发者后台](https://open.feishu.cn/app)（Lark 国际版：<https://open.larksuite.com/app>），
   「创建企业自建应用」，填名称、描述、图标。
2. 「凭证与基础信息」页记下 **App ID**（`cli_` 开头）和 **App Secret**。
3. 「添加应用能力」→ 添加 **机器人**。

### M2. 开通权限（「权限管理」→「开通权限」）

| 权限 | scope | 用途 |
|---|---|---|
| 读取用户发给机器人的单聊消息 | `im:message.p2p_msg:readonly` | 私聊 |
| 接收群聊中 @机器人消息事件 | `im:message.group_at_msg:readonly` | 群聊里 @机器人 的消息 |
| 以应用的身份发消息 | `im:message:send_as_bot` | 回复、发卡片、更新卡片（PATCH） |

可选：

- `im:message.group_msg`（获取群组中所有消息，**敏感权限**，需管理员审批）——只有把 `require_mention: false`
  （群里不 @ 也处理）时才需要；默认不要开。
- 不需要额外权限的：卡片回调 `card.action.trigger`、获取机器人自身信息 `/open-apis/bot/v3/info`。

### M3. 订阅事件（「事件与回调」→「事件配置」）

1. 订阅方式选 **使用长连接接收事件**。
2. 「添加事件」：**接收消息 v2.0**（`im.message.receive_v1`）。

> 保存长连接方式时，后台会检查是否已有客户端连上。如果提示未检测到连接，先完成 M6、运行 `va serve`
> （凭证正确即可建立连接，尚未发布版本也可以），再回来保存。

### M4. 配置卡片回调（「事件与回调」→「回调配置」，与「事件配置」是两个页签）

1. 订阅方式同样选 **使用长连接接收回调**。
2. 「添加回调」：**卡片回传交互**（`card.action.trigger`）。

审批卡片的按钮靠它工作。没配或没发布版本时，点按钮会报错（常见错误码 200340）。

### M5. 发布版本

权限、事件、回调的改动都要**发布版本后才生效**：「版本管理与发布」→「创建版本」→ 填可用范围（至少包含你自己）→
「保存」→「申请发布」。企业管理员在管理后台审核通过后生效（你是管理员就自己批）。之后每次改权限或事件都要再发一个版本。

### M6. 配置 Ventri

安装依赖（`lark-oapi` 是可选 extra）：

```bash
uv sync --extra feishu          # 或 pip install 'ventri-agent[feishu]'
```

保存 App Secret（不要写进 ventri.yml 明文）：

```bash
# macOS：存进钥匙串（服务 ventri，账户 feishu_app_secret），-w 不带值会提示输入，不留在 shell 历史里
security add-generic-password -U -s ventri -a feishu_app_secret -w
# 其他平台（Linux 等）：0600 权限的文件（见下文“密钥存在哪里”），或环境变量
mkdir -p -m 700 ~/.ventri/secrets && (umask 077; read -rs s && printf %s "$s" > ~/.ventri/secrets/feishu_app_secret)
export VENTRI_SECRET_FEISHU_APP_SECRET='...'   # 二选一；环境变量优先于文件
```

在 `~/.ventri/ventri.yml` 的 `plugins:` 里加上（`va init` 生成的模板里有注释掉的同样内容）：

```yaml
  - use: ventri_agent.channels.feishu
    id: feishu
    config:
      app_id: cli_xxxxxxxxxxxxxxxx
      app_secret: "${secret:feishu_app_secret}"
      domain: feishu              # Lark 国际版写 lark
      allow_users: []             # 先留空，下一步拿到 open_id 再填
      allow_chats: []             # 允许的群 chat_id（oc_...）
      require_mention: true       # 群里只处理 @机器人 的消息
```

其余可选项：`group_session: chat|thread`（群共享一个会话 / 每个话题一个会话）、`agent`（使用的 Agent 预设）、
`reply_unknown`（是否告诉未授权者其 open_id，默认 true）、`progress_interval`（进度卡片更新间隔，默认 1.5 秒）、
`max_message_age`（超过多少秒的旧消息丢弃，默认 600）、`show_thinking`、`state_dir`（默认 `~/.ventri/feishu`）。

### M7. 运行并把自己加入白名单

```bash
uv run va serve -v              # -v 在终端打印渠道日志；Ctrl-C 退出
```

1. 在飞书里搜索机器人，给它发一条私聊消息。白名单是空的，所以它会回复：
   「你还没有被授权使用这个机器人。你的 open_id 是：ou_xxx」（终端日志里也有）。
2. 把这个 `ou_...` 填进 `allow_users`，保存 ventri.yml——`va serve` 会热应用配置，不用重启。
3. 群聊：把机器人拉进群并 @ 它，它会回复这个群的 `chat_id`（`oc_...`）；填进 `allow_chats`。
   群里的发言者也必须在 `allow_users` 里。
4. 授权后可随时发 `/id` 查看自己的 open_id、当前 chat_id 和会话。

白名单是**失败关闭**的：`allow_users` 为空时谁都不能用。

## 使用

- 每条消息是会话的一轮。机器人先回一张「处理中」卡片并原地更新（工具调用、思考中、正文），结束后变成最终回答和用量；
  超长回答拆成多张卡片。
- 需要审批时会发一张审批卡片：**允许一次 / 本会话允许 / 拒绝**（不可撤销的操作没有“本会话允许”）。
  只有发起这一轮的人能点；超时（`permission.approval_timeout`，默认 120 秒）视为拒绝；卡片随后显示结果。
  在聊天里打字（“允许”“y”）不会批准任何东西。
- 斜杠命令：`/help`、`/new`（结束会话，下一条消息开新会话；`/end`、`/exit` 同义）、`/suspend`、`/retry`、
  `/think`、`/cost`、`/compact`、`/epoch`、`/memory`、`/sessions`、`/tree`、`/id`。
- 私聊一个会话；群默认一个群一个会话（`group_session: thread` 时按话题）。重启 `va serve` 后从会话日志恢复。

## 注意事项

- **同一个应用只运行一个 `va serve`**：飞书长连接是集群模式，同一应用的多个连接每个事件只随机投给其中一个。
  Ventri 用 `state_dir` 下的锁文件阻止同一台机器上的第二个进程；`va chat` 加载同一份配置时飞书渠道保持空闲（`run: serve`）。
  不要同时在别的机器或别的程序里连同一个应用。
- 暂不支持图片、文件、语音等消息（会回复提示）。
- 卡片更新有频率限制（单条消息 5 次/秒），进度更新已节流；遇到限流（230020）会自动放慢。
- 数据：飞书消息内容会作为对话发给模型提供方（DeepSeek），与 `va chat` 相同。
- SDK 自 1.7 起带一个高层的 `lark_oapi.channel.FeishuChannel`；Ventri 没有用它（它自带线程与事件循环、卡片回调固定返回空响应），
  而是直接用 `lark_oapi.ws.Client`，以便把卡片回调接到 Ventri 的审批流程上。

## 排查

| 现象 | 检查 |
|---|---|
| 私聊没反应 | `va serve -v` 是否显示 link up；版本是否已发布；`im.message.receive_v1` 是否订阅；`im:message.p2p_msg:readonly` 是否开通 |
| 群里没反应 | 是否 @ 了机器人；机器人是否在群里；`allow_chats` / `allow_users`；`im:message.group_at_msg:readonly`；日志里是否有 `bot-identity-unknown`（机器人能力未启用） |
| 点审批按钮报错 | 「回调配置」是否为长连接并添加了 `card.action.trigger`；改完是否发布了版本 |
| 提示“该审批已失效” | 已处理、已超时，或 `va serve` 重启过（重启时未决的审批按拒绝处理，旧卡片作废） |
| 启动报 `already served by another process` | 另一个 `va serve` 在用同一个 App ID |
| 启动报 `app_id and app_secret are required` | 钥匙串 / 环境变量 / `~/.ventri/secrets/` 里没有 `feishu_app_secret` |
| 启动报 `secret store ... is accessible by other users` | `chmod 600 ~/.ventri/secrets/feishu_app_secret`（目录 `chmod 700`） |
| `va feishu setup` 报 `unsupported_auth_method` / 一直无法完成 | 飞书侧未开放扫码创建（或企业禁止成员创建应用）：改用手动创建 |
| 扫码后提示被拒绝或过期 | 重新运行 `va feishu setup` 获取新二维码 |
