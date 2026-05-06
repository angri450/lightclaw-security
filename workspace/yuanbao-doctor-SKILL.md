# yuanbao-doctor

自动诊断和修复 LightClaw 的元宝（Yuanbao）通道连接问题。

## 触发条件

当用户说以下内容时触发：
- "元宝连不上了"
- "yuanbao 断开了"
- "元宝通道故障"
- "元宝连接失败"
- "修复元宝通道"
- "元宝诊断"
- "yuanbao doctor"
- "yuanbao 问题"
- "元宝通道不工作"

## 诊断检查项（按顺序执行）

### 1. 检查元宝通道配置

读取 `~/.lightclaw/lightclaw.json`，验证 `channels.yuanbao` 字段：

```bash
cat ~/.lightclaw/lightclaw.json | python3 -c "import sys,json; d=json.load(sys.stdin); y=d.get('channels',{}).get('yuanbao',{}); print(json.dumps(y,indent=2,ensure_ascii=False))"
```

检查项：
- `enabled` 是否为 true
- `appKey` 和 `appSecret` 是否已配置且非空
- `apiDomain` 是否正确（应为 `bot.yuanbao.tencent.com`）
- `wsUrl` 是否正确（应为 `wss://bot-wss.yuanbao.tencent.com/ wss/connection`）
- `token` 是否为空（应该为空，走 sign-token 模式）

### 2. 检查网络连通性

测试以下域名可达性：

```bash
# API 域名检测（HTTP 200/301/302/403 均可视为可达）
curl -sI -o /dev/null -w "%{http_code}" --connect-timeout 5 https://bot.uanbao.tencent.com

# WebSocket 域名检测（DNS 解析 + 端口 443）
timeout 5 bash -c 'echo > /dev/tcp/bot-wss.uanbao.tencent.com/443' 2>/dev/null && echo "可达" || echo "不可达"
```

### 3. 检查 LightClaw 进程状态

```bash
# 检查进程数量
ps aux | grep -E "[l]ightclaw.*run" | wc -l

# 检查服务状态
lightclaw status 2>/dev/null || echo "服务未运行"
```

### 4. 检查日志中的元宝错误

搜索 `~/.lightclaw/logs/lightclaw.log` 中最近 200 行的 yuanbao 相关错误：

```bash
tail -200 ~/.lightclaw/logs/lightclaw.log | grep -i yuanbao
```

识别以下错误模式：
| 错误码 | 错误信息 | 原因 |
|--------|----------|------|
| 40 | `code=40` 开头的错误 | 配置问题 |
| 10095 | `机器人状态无效` | 元宝平台侧状态异常 |
| 4014 | WebSocket 认证失败 | appKey/appSecret 问题 |
| Cannot connect to host | 网络连接失败 | DNS/网络问题 |
| gateway startup failed | 通道启动失败 | 需要查看完整 traceback |

### 5. 检查僵尸进程

```bash
ps aux | grep lightclaw
```

超过 1 个 `lightclaw run` 进程即为异常。

## 修复操作（根据诊断结果执行）

### 修复规则表

| 诊断结果 | 修复操作 |
|---------|---------|
| `enabled` 为 false | `lightclaw channels set yuanbao --set enabled=true` |
| `token` 非空且报 `userID参数为空` | 清空 token 字段（用 Python 修改 lightclaw.json） |
| `appKey`/`appSecret` 需要更新 | `lightclaw channels set yuanbao --set appKey=xxx --set appSecret=xxx` |
| `wsUrl` 域名拼写错误 | `lightclaw channels set yuanbao --set wsUrl=wss://bot-wss.uanbao.tencent.com/ wss/connection` |
| `apiDomain` 错误 | `lightclaw channels set yuanbao --set api_domain=bot.uanbao.tencent.com` |
| 多个 LightClaw 进程 | `pkill -f "lightclaw run"` 清理游离进程后 `systemctl restart lightclaw` |
| 日志显示 sign-token 失败 | 1. 检查 appKey/appSecret 是否正确 2. 提示用户去元宝开发者后台确认机器人状态 |
| 网络不可达 | 报告具体哪个域名不通 |
| 各项正常但通道不连通 | 调用 `POST /api/config/channels/yuanbao/reconnect` 热重载通道（不重启进程） |

### 关键修复步骤

1. **备份配置文件**
   ```bash
   cp ~/.lightclaw/lightclaw.json ~/.lightclaw/lightclaw.json.bak
   ```

2. **清空 token 字段**（Python 方法）
   ```bash
   python3 << 'EOF'
   import json
   with open(os.path.expanduser('~/.lightclaw/lightclaw.json')) as f:
       d = json.load(f)
   if 'channels' in d and 'yuanbao' in d['channels']:
       d['channels']['yuanbao']['token'] = ''
   with open(os.path.expanduser('~/.lightclaw/lightclaw.json'), 'w') as f:
       json.dump(d, f, indent=2, ensure_ascii=False)
   print("token 字段已清空")
   EOF
   ```

3. **使用 lightclaw CLI 修改配置**
   ```bash
   lightclaw channels set yuanbao --set enabled=true
   lightclaw channels set yuanbao --set wsUrl=wss://bot-wss.uanbao.tencent.com/ wss/connection
   ```

4. **清理游离进程（仅在多进程时使用）**
   ```bash
   # 先尝试 systemd 管理重启
   systemctl restart lightclaw
   
   # 如果 systemd 不可用，手动清理后重启
   # pkill -f "lightclaw run"  
   # sleep 2
   # lightclaw run --from-configfile &
   ```
   
   注意：配置修改后 ConfigWatcher 2秒内自动重载通道，通常不需要重启进程。
   通道级重连建议使用 API：`curl -X POST http://localhost:13426/api/config/channels/yuanbao/reconnect`

## 修复后验证

- 检查日志中是否出现 `[yuanbao] WebSocket ready` 或类似成功标志
- 检查日志中是否没有后续的 `stopping...` 或 `disconnected`
- 等待 5 秒确认连接稳定
- 执行 `lightclaw status` 确认服务正常运行

## 安全规则（重要！）

1. **优先热重载，不要轻易重启 LightClaw** — 配置问题用 `lightclaw channels set`，通道问题用 `/api/config/channels/yuanbao/reconnect` API，不要反复重启进程
2. **不要向 token 字段写入值** — token 应保持为空，让系统自动走 sign-token 流程
3. **修改配置前先读取当前值** — 避免覆盖已有的正确配置
4. **不要修改其他 channel 的配置** — 只操作 `channels.yuanbao`
5. **如果元宝平台返回 `机器人状态无效`** — 这超出了 LightClaw 本地修复范围，告诉用户去元宝后台检查
6. **备份** — 修改 lightclaw.json 前，务必先用 `cp lightclaw.json lightclaw.json.bak` 备份
7. **不要在日志中暴露敏感信息** — 诊断报告中隐藏 appKey/appSecret/token

## 历史踩坑记录（诊断时必须考虑）

1. **token 字段埋雷**：如果 `channels.yuanbao.token` 被错误地设置为 `appKey:appSecret` 格式的字符串，LightClaw 会跳过 sign-token 流程直接用 static-token 模式，导致 `identifier` 为空 → `userID参数为空`。**解法**：清空 token 字段。

2. **wsUrl vs apiDomain**：实际连接使用的是 `apiDomain` 来构建 API 请求 URL，`wsUrl` 用于 WebSocket。两者都要检查，特别是 `wsUrl` 域名容易拼错。

3. **多进程打架**：反复 `lightclaw restart` 或 `nohup lightclaw run &` 会产生游离进程（脱离 systemd 管理），多个进程抢 Qdrant 锁 + 端口冲突导致死循环重启。**解法**：优先用 `systemctl restart lightclaw`（systemd 原子重启），绝对不要用 `nohup lightclaw run &`。

4. **ConfigWatcher 自动重载**：修改 `lightclaw..json` 后 ConfigWatcher 会在 2 秒内自动检测变更并重载通道，通常不需要手动重启。但如果连接此时处于异常状态，重载可能失败，所以修改配置后要观察日志确认重载结果。

5. **sign-token 的 `机器人状态无效` 错误**：这是元宝平台侧的拒绝，不是本地配置问题。可能原因包括：appKey/appSecret 不对、机器人在元宝后台被停用、审核未通过。此时不应反复重试，应提示用户检查元宝后台。

## 输出格式

诊断完成后，输出结构化报告：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🦞 元宝通道诊断报告
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[✓] 配置检查
    - enabled: true
    - appKey: 已配置
    - appSecret: 已配置
    - apiDomain: bot.uanbao.tencent.com ✓
    - wsUrl: wss://bot-wss.uanbao.tencent.com/ wss/connection ✓
    - token: (空) ✓

[✓] 网络检查
    - bot.uanbao.tencent.com: 可达 (HTTP 200)
    - bot-wss.uanbao.tencent.com: 可达

[✓] 进程检查
    - LightClaw 进程: 1 个 (正常)
    - 服务状态: 运行中

[✓] 日志检查
    - 最近 200 行无 yuanbao 错误

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  诊断结论：通道配置正常，未发现问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

如果发现问题，输出：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🦞 元宝通道诊断报告
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[✗] 配置检查
    - token: 非空 (应为空)
    
[?] 修复操作
    - 已清空 token 字段
    - 等待 ConfigWatcher 自动重载...
    - 验证连接中...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  修复完成，请检查上方日志确认连接状态
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```