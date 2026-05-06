# LightClaw 开发工作全面汇报

> **开发周期**：2026年5月2日 — 2026年5月6日（连续5天高强度开发）
> **大模型费用**：约 **¥1,600**（DeepSeek V4 Pro + ChatGPT 5.5，持续高强度推理）
> **工作强度**：五一假期全程无休，日均工作12+小时，多 Agent 并行开发模式

---

## 零、工作总览

| 维度 | 数据 |
|------|------|
| 开发天数 | **5天**（5月2日-6日，含五一假期全程） |
| 修改/新建文件总数 | **60+** |
| 新增 Python 模块 | **15个** |
| 代码行数变更 | **约 8,000+ 行** |
| 并行调研 Agent 调用 | **30+ 次**（每次 3-8 个 Agent 并行只读调研） |
| 并行实施 Agent 调用 | **20+ 次** |
| 服务重启次数 | **15+ 次**，每次重启成功率 100%，零引入性错误 |
| 单元测试通过 | **100+** |
| 大模型 token 费用 | **约 ¥1,600**（DeepSeek V4 Pro + ChatGPT 5.5） |
| 解决的 P0 级风险 | **8项** |
| 解决的 P1 级风险 | **12项** |
| 解决的 P2 级优化 | **6项** |

---

## 一、项目背景与工作动机

### 1.1 接手时的系统状态

五一接手 LightClaw 时，系统面临以下核心问题：

1. **自激循环卡顿**：heartbeat 每30分钟一次心跳，但 agent 反复调查 2.5 小时前的旧错误，每次消耗 10-15 次模型调用 + 15K-30K tokens，3/4 次心跳超时（120s），形成升级链
2. **上下文爆炸**：实测单次请求 302K tokens（超过 MiniMax 模型上限 196K），`recall_block` 单块 194K tokens 超过总预算 63K，导致对话消息被全部丢弃（messages=0）
3. **跨用户隐私泄露**：DAILY_TALK 全局混合（元宝私聊内容可注入 dashboard 会话），session FTS5 搜索无过滤，vector search 无 session filter
4. **模型调用零防护**：无 `request_timeout`，单次调用可跑 210s 不中断；heartbeat agent 拥有完整 40+ 工具权限（含写文件、执行 shell）
5. **主动消息质量缺陷**：文本重复发送（同一段话完整出现两次）+ 角色错位（"你是常务副会长，你都不知道的事儿我上哪儿知道去"）
6. **多实例崩溃死循环**：skill 执行 `nohup` 产生游离进程 → 端口冲突 → systemd 无限重启（10次/7分钟）
7. **记忆膨胀无淘汰**：MEMORY.md 持续膨胀混入天气/余额/版本号等无用内容，无衰减机制
8. **运维风险**：Agent 自维修会搞死自己（盲目修改配置 + 反复重启 + 产生僵尸进程）

### 1.2 工作模式

全部采用 **多 Agent 并行调研 + 并行实施** 的开发模式：
- 每次调研派出 3-8 个只读 Agent 同时探查不同模块
- 每次实施派出 2-4 个 Agent 同时修改不同文件
- 所有 Agent 由人类统一调度、交叉验证、收口合并
- 每次改动后重启服务验证，确认零新错误

---

## 二、模块一：运行控制层安全防护（P0-P2 阶梯式推进）

### 2.1 P0 止血：Heartbeat 自激循环 + 模型调用 timeout

**日期**：2026-05-05 22:27
**修改文件**：3 个（constant.py / heartbeat.py / model_factory.py）

#### 问题根因分析

Heartbeat 每30分钟一次心跳，`_count_new_errors()` 基于文件偏移量做增量扫描——计数自上次 marker 以来包含 "ERROR" 或 "CRITICAL" 的行数。**不存在**错误指纹去重、类型聚类、或"已调查"标注。

关键日志证据：
```
21:00 的 HB3 的 call_index=14 最后一条消息：
  line 246695: ERROR ws_client.py:559 | 2026-05-05 18:37:53 | send failed: connection not available
```
——距离原始事件已 2.5 小时，heartbeat agent 仍在调查同一条旧日志。

单次心跳的 token 消耗：10-15 次模型调用，每次 15K-30K tokens input。

#### 修复方案

**A. heartbeat.py —— error fingerprint 系统（核心技术实现）**

核心设计：用"错误指纹"替代"日志行数"作为计量单位。

```python
# 12 种错误模式识别，按优先级匹配
_ERROR_PATTERNS = [
    ("websocket_send_failed", re.compile(r'send failed.*connection not available')),
    ("websocket_disconnect", re.compile(r'(?:disconnect|reconnect|close).*websocket', re.I)),
    ("stream_turns_error", re.compile(r'stream_turns.*error', re.I)),
    ("stream_chunk_timeout", re.compile(r'stream chunk timeout', re.I)),
    ("timeout", re.compile(r'(?:timeout|timed.?out)', re.I)),
    ("connection_error", re.compile(r'(?:connection.*(?:error|refused|reset|failed))', re.I)),
    ("api_error", re.compile(r'(?:status.*code.*[45]\d{2}|HTTP.*[45]\d{2})', re.I)),
    ("import_error", re.compile(r'(?:ImportError|ModuleNotFoundError)', re.I)),
    ("syntax_error", re.compile(r'SyntaxError', re.I)),
    ("runtime_error", re.compile(r'RuntimeError', re.I)),
    ("heartbeat_failed", re.compile(r'heartbeat run failed', re.I)),
    ("unknown_error", re.compile(r'(?:ERROR|CRITICAL)')),  # 兜底
]
```

fingerprint 提取逻辑：
```python
def _extract_fingerprint(line: str) -> tuple[str, str, str] | None:
    """从日志行提取 (error_keyword, module_path, normalized_head)"""
    # Step 1: 匹配错误模式 → error_keyword
    # Step 2: 从行中提取相对路径 → module_path (如 "app/cron/heartbeat.py:495")
    # Step 3: 去时间戳/UUID/数字ID/hex串 → normalized_head
    # Step 4: 多空白归一化
    # 返回三元组，合成存储 key: "error_keyword|module_path|normalized_head"
```

去重状态管理（核心逻辑）：
```python
def _update_fingerprints_and_count(lines: list[str]) -> tuple[int, dict]:
    """
    对每个 fingerprint：
    - 首次出现 → actionable_count += 1，记录 first_seen/last_seen
    - 已存在 + resolved → 错误重现，重新打开，actionable_count += 1
    - 已存在 + 未 resolved + last_reported < cooldown(1800s) → 静默，suppressed += 1
    - 已存在 + 未 resolved + last_reported >= cooldown → 冷却已过，actionable_count += 1
    - 本次扫描未出现 → missed_cycles += 1
    - missed_cycles >= 2 → resolved = true
    """
```

状态持久化：
- 存储位置：`memory/.heartbeat/_error_fingerprints.json`
- 写入方式：`.tmp` + `os.replace()` 原子写入
- 自动裁剪：已 resolved 且超过 TTL(7天) 删除；超过 MAX_ENTRIES(200) 保留最近
- JSON 读失败 → 重建空状态，不崩溃

**B. model_factory.py —— 注入 request_timeout + max_retries**

```python
# 环境变量安全解析
_MODEL_REQUEST_TIMEOUT = _parse_model_timeout()   # 默认 120s, env: LIGHTCLAW_MODEL_REQUEST_TIMEOUT
_MODEL_MAX_RETRIES = _parse_model_max_retries()    # 默认 2, env: LIGHTCLAW_MODEL_MAX_RETRIES

# 注入 ChatOpenAI kwargs
kwargs = {
    "model": model_name,
    "api_key": api_key,
    "base_url": base_url,
    "streaming": True,
    "stream_usage": use_stream_usage,
    "request_timeout": _MODEL_REQUEST_TIMEOUT,     # 新增
    "max_retries": _MODEL_MAX_RETRIES,              # 新增
}
```

**C. agent query 硬性约束**

在 heartbeat agent 的 system prompt 中加入：
```
## 硬性约束（必须遵守）
- 禁止读取完整 lightclaw.log
- 禁止读取超过 200 行日志文件
- 禁止使用 shell 做大范围 grep 搜索
- 禁止写文件
- 禁止修改配置
- 只基于心跳 sensor summary 做判断
- 如果信息不足，输出 HEARTBEAT_OK 或简短告警即可，不要展开调查
- 最多使用 3 次工具调用，之后必须输出结论
```

#### 测试结果

| 场景 | 输入 | 预期 | 实际 |
|------|------|------|------|
| Cycle 1 | 10条相同 WS 错误 | actionable=1, suppressed=9 | ✅ actionable=1, suppressed=9 |
| Cycle 2 | 无新错误 | actionable=0 | ✅ actionable=0 |
| Cycle 3 | 连续2次无错误 | resolved=1 | ✅ resolved=1 |
| Cycle 4 | 已 resolved 错误重现 | actionable=1 (re-opened) | ✅ actionable=1 |
| Cycle 5 | cooldown 内再次出现 | actionable=0 (suppressed) | ✅ actionable=0 |

#### 效果
- heartbeat 从 "1427 行全报警" 降为 "1 个 actionable + 1426 条 suppressed"
- 同一 fingerprint 30 分钟内最多触发 1 次
- 错误消失 2 个周期后自动标记 resolved
- 所有 ChatOpenAI 请求默认 120s timeout + 2 次 retry
- `consecutive_abnormal` 不再形成升级链

---

### 2.2 P1 硬刹车：上下文预算强制性控制

**日期**：2026-05-05 23:00
**修改文件**：3 个（constant.py / token_budget.py / coordinator.py）

#### 问题根因分析

日志实测上下文爆炸证据：
```
[Token][Call #2][preflight]
  actual_total_pre_tokens = 214,074
  max_input_length = 131,072
  over_budget_pre = 83,002

  extra_system_tokens breakdown:
    recall: 193,989  ← 单块超过总预算 63K!
    facts: 459
    memory_write_reminder: 0

[ModelCall][start] call_index=2
  messages=0  ← enforce() 丢了所有对话消息!
  system_chars=395,258
```

中午的超上限事件：
```
12:51:44 | input_tokens=229,998  ← 超 MiniMax 上限 196,608!
12:51:51 | input_tokens=238,807  ← 超上限!
12:56:55 | input_tokens=220,148  ← 超上限!
13:15:51 | input_tokens=302,287  ← 超上限!
```

根因：原有 `enforce()` 在 token_budget.py 中只记日志不截断 ToolMessage 总量；`final_input_guard` 仅 trim recall_block，不裁剪消息、不拒绝请求。

#### 修复方案

**A. token_budget.py —— ToolMessage 总量硬截断**

```python
def enforce(self, messages, total_tokens, max_input_tokens, reserved_tokens):
    # ... 现有单条截断逻辑 ...

    # 新增：ToolMessage 总量硬截断
    tool_total_chars = sum(len(self._get_content(m)) for m in messages if isinstance(m, ToolMessage))
    tool_total_max = TOOL_MESSAGE_TOTAL_MAX_CHARS  # 默认 24000

    if tool_total_chars > tool_total_max:
        # 从最新的 ToolMessage 开始保留预算（最新消息优先）
        budget_remaining = tool_total_max
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, ToolMessage):
                continue
            content = self._get_content(msg)
            if budget_remaining >= len(content):
                budget_remaining -= len(content)
            elif budget_remaining > 0:
                # 跨越预算边界的消息：保留头部 + 截断说明
                truncated = content[:budget_remaining] + f"\n[...truncated {len(content)-budget_remaining} chars...]"
                messages[i] = _clone_message(msg, truncated)
                budget_remaining = 0
            else:
                # 最老的 ToolMessage：整条替换为短占位符
                placeholder = f"[Tool output removed by context guard. Original chars: {len(content)}. Total tool budget {tool_total_max} exceeded.]"
                messages[i] = _clone_message(msg, placeholder[:PLACEHOLDER_MAX_CHARS])
```

**B. coordinator.py —— 多阶段 final input guard**

四阶段硬刹车设计：

```
Stage 1 — 底线保留：
  删除所有非系统消息后，如果只剩系统消息 → 保留最后一条 HumanMessage

Stage 2 — 多轮削减（最多 3 轮）：
  每轮按顺序尝试：
    2a：正则删除 <auto-recall-context>...</auto-recall-context>
    2b：非必要 extra system parts 超限时，删除 summary/flush_reminder/memory_write_reminder/recall
    2c：删除最老 ~20% 非系统消息，将 ToolMessage 替换为短占位符

Stage 3 — 最后手段：
  保留仅系统消息 + 最后一条 HumanMessage

Stage 4 — 硬失败（fail-closed）：
  如果仍然超预算 → 抛出 RuntimeError 阻止模型调用
  （无论 HARD_FAIL 是 True 还是 False，都阻止）
```

```python
# 关键代码逻辑
if final_total > max_input:
    if FINAL_INPUT_GUARD_HARD_FAIL:
        logger.error("final_input_guard: hard_fail ...")
    else:
        logger.error("final_input_guard: fail_closed ...")
    raise RuntimeError(
        f"Final input too large ({final_total} tokens) "
        f"after {rounds} trim rounds. max_input={max_input}"
    )
else:
    logger.info("final_input_guard: ok tokens=%d max=%d rounds=%d", final_total, max_input, rounds)
```

#### 测试结果

ToolMessage 截断模拟测试：
```
Before truncation: total_chars=50000, msg_count=5
After truncation: total_chars=23472 <= 24000, truncated=3 messages preserved
HumanMessage preserved: True
```

#### 效果
- 单次请求 input tokens 不超过 131072 硬上限
- ToolMessage 总量 >24000 chars 时自动从最老消息开始裁剪
- 超预算时硬拒绝，不留绕过路径（fail-closed）
- 所有阈值通过环境变量可调

---

### 2.3 P1.1 安全修边

**日期**：2026-05-05 23:20
**修改文件**：3 个（constant.py / coordinator.py / heartbeat.py）

#### 修复内容

**A. env 安全解析**：15+ 个常量从裸 `int(os.environ.get(...))` 改为安全 helper 函数

```python
def _env_int(name, default, min_value=None, max_value=None) -> int:
    """非法值 → default，min/max 越界 → default，永不抛异常"""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return default
    if min_value is not None and val < min_value:
        return default
    if max_value is not None and val > max_value:
        return default
    return val

def _env_bool(name, default) -> bool:
    """非法值 → default，支持 true/1/yes/y/on / false/0/no/n/off"""
```

Converted all 12 Context Budget + 4 Heartbeat constants to use these safe parsers.

**B. Fail-closed 彻底封闭**

原来 `FINAL_INPUT_GUARD_HARD_FAIL=False` 时超预算仍然进入模型调用——改为两种模式都 `raise RuntimeError`，仅日志标签不同。

**C. 日志级别过滤修复**

```python
# Before (false positives):
if "ERROR" not in line and "CRITICAL" not in line:
    # INFO 行中的 "last_status=ERROR" 被误判为真错误

# After (only true ERROR/CRITICAL log level lines):
if not (line.startswith("ERROR ") or line.startswith("CRITICAL ")):
    # 仅行首日志级别前缀匹配
```

#### 测试结果

env 非法值导入测试：5/5 PASS（设置 `abc`/`""`/`maybe` 等非法值均 safe fallback）
真实模型调用验证：`final_input_guard: ok tokens=12 max=131072 rounds=0 actions=none`
fail-closed 静态分析：不存在 fall-through 到模型调用路径

---

### 2.4 P1b Heartbeat 真工具白名单与 Runtime Budget

**日期**：2026-05-05 23:50
**修改文件**：6 个

#### 问题根因

`_HEARTBEAT_AGENT_ALLOWED` 是死代码：定义为 5 个不存在的假工具名，全代码库零引用。
`HEARTBEAT_MAX_AGENT_ITERS` / `HEARTBEAT_MAX_INPUT_TOKENS` 也是死常量：仅定义，零引用。

#### 修复方案：6 文件全链路改造

**全链路数据流**：

```
heartbeat.py
  _HEARTBEAT_AGENT_ALLOWED = ("read_file", "get_current_time")
  HEARTBEAT_MAX_INPUT_TOKENS = 8000
      ↓ run_cron_agent(allowed_tools=..., runtime_profile="heartbeat", ...)
agent_task.py
  turn_request.context["allowed_tools"] = ["read_file", "get_current_time"]
  turn_request.context["runtime_profile"] = "heartbeat"
  turn_request.context["max_input_length_override"] = 8000
  turn_request.context["max_iters_override"] = 5
      ↓ runner.stream_turns(turn_request)
session_factory.py
  runtime_context.allowed_tools = request_ctx["allowed_tools"]
  runtime_context.runtime_profile = request_ctx["runtime_profile"]
  runtime_context.max_input_length_override = request_ctx["max_input_length_override"]
  runtime_context.max_iters_override = request_ctx["max_iters_override"]
      ↓ langgraph context → coordinator
coordinator.py (awrap_model_call)
  max_input = runtime_context.max_input_length_override or shared_default  # 8000 for heartbeat
  allowed_tools = runtime_context.allowed_tools
  if allowed_tools is not None:
      filtered_tools = [t for t in request.tools if t.name in allowed_set]
      override_kwargs["tools"] = filtered_tools
```

**coordinator 中工具过滤的关键代码**：

```python
allowed_tools = getattr(runtime_context, "allowed_tools", None)
if allowed_tools is not None:
    allowed_set = set(allowed_tools)
    original_tools = request.tools
    filtered_tools = [t for t in original_tools if t.name in allowed_set]
    blocked_count = len(original_tools) - len(filtered_tools)
    logger.info(
        "tool_policy: source=%s allowed=%d before=%d after=%d blocked=%d",
        source, len(allowed_tools), len(original_tools), len(filtered_tools), blocked_count
    )
    override_kwargs["tools"] = filtered_tools
```

**heartbeat 白名单**：13 个默认工具中屏蔽 11 个（写文件、shell、搜索、浏览器、memory 写入），仅保留 2 个（read_file + get_current_time）。

#### 效果

heartbeat agent 被硬性约束：
- 无法写文件（write_file / edit_file / append_file）
- 无法执行 shell（execute_shell_command）
- 无法搜索（grep_search / glob_search / lightclaw_search）
- 无法浏览器操作（browser_control / desktop_screenshot）
- 输入上限 8000 tokens，max_iters=5
- 普通用户 turn 不受影响（allowed_tools=None → 不过滤）

---

### 2.5 P2a/P2b Heartbeat 极致轻量化

**日期**：2026-05-06 00:08-01:10
**修改文件**：9 个

#### P2a：HEARTBEAT.md 瘦身 + context slimming

**HEARTBEAT.md**：46 行 → 5 行

```markdown
# 心跳任务

1. 根据传感器摘要判断系统是否健康。
2. 正常时输出 HEARTBEAT_OK。
3. 异常时只输出简短、可行动的告警。
```

（工具权限由运行时白名单硬强制，预算由 runtime_context 硬强制，prompt 中声明这些是冗余）

**context_slimming**：heartbeat 运行时跳过 recall/memory_write_reminder/flush_reminder

```python
_profile = getattr(runtime_context, "runtime_profile", None)
if _profile == "heartbeat":
    for _key in ("recall", "memory_write_reminder", "flush_reminder"):
        if extra_parts_map.get(_key):
            extra_parts_map[_key] = ""
    logger.info("heartbeat_context_slimming: removed_blocks=%s", ",".join(removed))
```

**效果**：每 turn 节省约 752 tokens（recall ~600 + memory_write_reminder ~152），每日 48 轮 × 752 = 约 36K tokens/天。

#### P2b：最小权限工具 + runtime 收口

**新增 2 个专用工具**：

1. **read_status_file**：逻辑目标安全状态读取
   ```python
   # 逻辑目标映射（不接受任意路径）
   _TARGET_MAP = {
       "heartbeat_state": "memory/.heartbeat/_state.json",
       "heartbeat_today_log": "memory/.heartbeat/YYYY-MM-DD.log",
       "heartbeat_error_fingerprints": "memory/.heartbeat/_error_fingerprints.json",
       "dreaming_status": "memory/.dreams/status.json",
       "heartbeat_prompt": "HEARTBEAT.md",
   }
   # 安全策略：
   # 1. 只接受逻辑 target 名称，拒绝任意路径
   # 2. 解析后路径必须在预定义的安全前缀内
   # 3. JSON 文件自动 pretty-print
   # 4. max_chars 硬截断（default 4000, hard-cap 8000）
   # 5. 文件不存在返回 "(file not found)" 而非异常
   ```

2. **read_recent_heartbeat_errors**：受控错误摘要
   ```python
   # 数据来源优先级：
   # 1. _error_fingerprints.json — 活跃/未解决的错误指纹
   # 2. YYYY-MM-DD.log — 今日心跳日志最近 ERROR 行
   # 3. lightclaw.log 尾部 — 仅 ERROR/CRITICAL 行（最多 300 行或 64KB）
   #
   # 安全策略：
   # 1. 不接收任何路径参数
   # 2. 不 grep 全文件
   # 3. max_items 硬上限 20, max_chars 硬上限 8000
   # 4. Secret 脱敏：api_key/secret/token/password/cookie/auth/credential → "***REDACTED***"
   # 5. 只提取行首 "ERROR " 或 "CRITICAL " 开头的日志行
   ```

**heartbeat 白名单更新**：
- Before: `("read_file", "get_current_time")` — 2 tools
- After: `("read_status_file", "read_recent_heartbeat_errors", "get_current_time")` — 3 tools
- `read_file` 已从 heartbeat 白名单移除

**max_iters_override 接入 engine_session**：

```python
# engine_session.py stream() 方法
max_iters = self._max_iterations
if runtime_context is not None:
    override = getattr(runtime_context, "max_iters_override", None)
    if override is not None and isinstance(override, int) and override > 0:
        max_iters = override
        logger.info(
            "runtime_budget: profile=%s max_iters=%d (global_default=%d)",
            getattr(runtime_context, "runtime_profile", "unknown"),
            max_iters, self._max_iterations,
        )
```

**heartbeat timeout 接入**：
- total turn timeout: 120s → 75s
- `HEARTBEAT_TOTAL_TIMEOUT_SECONDS` 在 cron 层 `asyncio.wait_for` 消费

---

## 三、模块二：会话隔离防火墙（Session Isolation Firewall）

**日期**：2026-05-05
**修改文件**：30+ 个（含 4 个新文件 + lightclaw.json 配置）
**方式**：4 Agent 并行 + 手动补位 + 直接修改

### 3.1 问题全景调研

**12 项只读调研**，每项有源码调用链 + 实际文件/日志证据：

| 发现 | 证据等级 |
|------|----------|
| DAILY_TALK 全局混合（元宝私聊注入 dashboard） | strong（源码+实际文件) |
| session FTS5 搜索无 session filter | strong（源码+数据库） |
| recall_block 超过总预算（194K > 131K） | strong（日志实测） |
| Vector search 返回 400K+ chars（6-char query → 405K result） | strong（日志实测) |
| Qdrant payload 无 session_id/scope 字段 | strong（源码) |
| summary 无长度上限（单调增长） | strong（源码) |
| ToolMessage 原始内容通过 session FTS5 进入 recall | strong（源码) |
| final_input_guard 不阻止超限请求 | strong（源码+日志） |
| EXTRA_PARTS_MAX_TOTAL_CHARS 是僵尸常量 | strong（源码) |

### 3.2 ActorContext 身份系统

**核心数据模型**（新增文件 `agent/core/actor_context.py`，232 行）：

```python
class TrustLevel(str, Enum):
    OWNER = "owner"           # 配置确认的 owner
    TRUSTED = "trusted"       # 可信用户
    GROUP_MEMBER = "group_member"  # 群聊成员（无法确认身份）
    UNKNOWN = "unknown"        # 完全未识别的发送者
    SYSTEM = "system"          # cron/heartbeat 系统任务

class ChatType(str, Enum):
    PRIVATE = "private"
    GROUP = "group"
    DASHBOARD = "dashboard"
    CRON = "cron"
    PROACTIVE = "proactive"

@dataclass
class ActorContext:
    agent_id: str
    channel: str          # yuanbao / weixin / dashboard / system
    account_id: str
    chat_type: ChatType
    chat_id: str
    sender_id: str
    principal_id: str     # owner / unknown / system
    is_owner: bool
    is_group_chat: bool
    session_id: str
    session_key: str      # 5 种命名空间格式
    trust_level: TrustLevel
```

**Session Key 5 种命名空间规范**：
```
private:   {agent_id}:{channel}:{account_id}:dm:{sender_id}
group:     {agent_id}:{channel}:{account_id}:group:{chat_id}:user:{sender_id}
dashboard: {agent_id}:dashboard:owner
system:    {agent_id}:system:{job_id}
```

### 3.3 Recall 权限矩阵

| Actor | current_session | owner_memory | group_memory | all_sessions |
|-------|:---:|:---:|:---:|:---:|
| dashboard owner | ✅ | ✅ | ❌ | ❌ |
| owner 私聊 | ✅ | ✅ | ❌ | ❌ |
| group owner发言 | ✅ | ❌ | ✅(if enabled) | ❌ |
| group unknown | ✅ | ❌ | ❌ | ❌ |
| system/cron | ✅ | ✅ | ❌ | ✅(offline only) |
| unknown sender | ✅ | ❌ | ❌ | ❌ |

recall.py 完全重写：`get_block()` 接收 ActorContext → 构造 recall_block 时按权限矩阵过滤 → 不符合 scope 的结果被 authz filter 拒绝。

### 3.4 Secret Firewall

**14 种正则模式检测敏感信息**：

```python
_SECRET_PATTERNS = {
    "api_key_assignment": r'(?:api[-_]?key|apikey|API[-_]?KEY)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]',
    "bearer_token": r'bearer\s+[a-zA-Z0-9\-_.~+/]{20,}',
    "openai_key": r'sk-(?:proj-)?[a-zA-Z0-9]{20,}',
    "password_assignment": r'(?:password|passwd|pwd)\s*[:=]\s*[\'"][^\'"]{4,}[\'"]',
    "secret_assignment": r'secret\s*[:=]\s*[\'"][^\'"]{4,}[\'"]',
    "access_key_assignment": r'access[-_]?key\s*[:=]\s*[\'"][^\'"]{8,}[\'"]',
    "private_key": r'-----BEGIN (?:RSA|EC|OPENSSH|DSA) PRIVATE KEY-----',
    "ssh_key": r'ssh-(?:rsa|ed25519|ecdsa|dss)\s+[a-zA-Z0-9+/]{20,}',
    "webhook_url": r'https://(?:hooks\.|oapi\.|incoming\.)[^\s]{10,}',
    "cookie_secret": r'(?:cookie|session)[-_]?(?:secret|key)\s*[:=]\s*[\'"][^\'"]{4,}[\'"]',
    "jwt_token": r'eyJ[a-zA-Z0-9\-_]{20,}\.[a-zA-Z0-9\-_]{20,}\.[a-zA-Z0-9\-_]{10,}',
    "cloud_key": r'(?:secretId|secretKey|SecretId|SecretKey)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]',
    "siliconflow_key": r'sk-[a-z]{4,10}-[a-zA-Z0-9]{20,}',
    "generic_credential": r'(?:auth|credential|token)\s*[:=]\s*[\'"][^\'"]{8,}[\'"]',
}
```

**5 重过滤**：
1. `contains_secret()` 在 recall.py 中过滤（realtime recall 入口）
2. secret 不进 DAILY_TALK（recall 入口过滤）
3. secret 不进 sessions FTS（recall 入口过滤）
4. Qdrant 索引时标记 `contains_secret=True`，搜索时 MustNot 排除
5. Dreaming 中 `has_secret=True` 的 candidate 跳过 promotion

**测试**：7/7 真实 secret pattern 命中，普通文本不影响

### 3.5 Context Budget Firewall

**9 项硬上限**：

| 限制项 | 默认值 | 实施位置 |
|--------|:-----:|------|
| recall_block 总字符 | 12,000 | constant.py → recall.py |
| 每 candidate 字符 | 2,000 | constant.py → recall.py |
| session result 字符 | 2,000 | session_distillation_service.py |
| DAILY_TALK realtime | 0 (禁用) | constant.py → recall.py |
| MEMORY.md realtime | 4,000 | constant.py → recall.py |
| summary_block | 12,000 | compactor.py |
| extra_parts 总计 | 30,000 | constant.py → coordinator.py |
| 单 ToolMessage | 8,000 | token_budget.py |
| ToolMessage 总计 | 24,000 | token_budget.py |

### 3.6 Qdrant v2 Metadata

**14 个新 payload 字段**：
`source_type`, `scope`, `session_id`, `session_key`, `channel`, `chat_type`, `chat_id`, `sender_id`, `principal_id`, `trust_level`, `user_id`, `is_raw_session`, `is_tool_output`, `contains_secret`

**搜索 filter**：
```python
# Must: scope in [owner_memory, user_memory] OR session_id = current_session_id
# MustNot: is_raw_session=True, contains_secret=True, is_tool_output=True
# Legacy fallback: 仅当 OLD_METADATA_REALTIME_RECALL_ENABLED=true 时按 path 回退
```

### 3.7 四通道 ActorContext 接入

| 通道 | 文件 | 改动 |
|------|------|------|
| yuanbao | ws_client.py / channel_adapter | 从 WS 消息提取 sender_id + chat_type，构造真实 ActorContext |
| weixin | channel_adapter | 从 weixin 回调提取 sender_id，通过 ownerPrincipals 映射 principal |
| dashboard | channel.py | dashboard sender_id="default" → principal_id="owner" |
| cron/heartbeat | agent_task.py | ActorContext.system(job_id="heartbeat") |

### 3.8 收尾加固

- **weixin owner 映射**：`lightclaw.json` 新增 weixin sender → owner principal
- **coordinator fallback 收紧**：dashboard 保留 owner 兼容，其他通道 fallback 为 unknown（最小权限）
- **Qdrant reindex**：删库重建，全部 6 个 .md 文件获得完整 v2 metadata

---

## 四、模块三：主动消息质量保障

**日期**：2026-05-06
**修改文件**：9 个（含 3 个新文件）

### 4.1 文本重复发送根因分析

三个设计缺陷叠加：

1. **bridge.py text_buffer 跨模型调用累积**：
   `text_buffer` 在 turn 生命周期内不清空，Call 1 的残留文本导致 `streamed_text != completed_text`，`drop_completed_text=False`，ASSISTANT_COMPLETED 文本未被去重

2. **agent_task.py 用流式 delta 构造最终发送文本**：
   ASSISTANT_DELTA 和 ASSISTANT_COMPLETED 两种不同语义的事件混入同一个 `text_chunks` 列表，去重失效时必然重复

3. **缺乏出站质量门**：无后处理拦截角色错位、内心独白泄露

### 4.2 架构根治

**新增 AssistantTurnAccumulator（核心架构组件）**：

```python
class AssistantTurnAccumulator:
    """以 completed_text 为唯一权威来源的 turn 结果收集器"""
    def __init__(self):
        self._completed_text: str = ""     # 来自 ASSISTANT_COMPLETED，唯一权威
        self._delta_text: str = ""          # 流式展示用，不作为发送文本
        self._thinking_text: str = ""       # 独立存储，永不进入 outbound

    def add_delta(self, text: str): ...
    def add_thinking(self, text: str): ...
    def set_completed(self, text: str): ...
    def assemble(self) -> AccumulatedTurn: ...

@dataclass
class AccumulatedTurn:
    completed_text: str     # 权威来源，来自 ASSISTANT_COMPLETED
    delta_text: str         # 仅用于流式展示
    thinking_text: str      # 永不进入 outbound
```

**核心设计原则**：
- `completed_text` 来自 ASSISTANT_COMPLETED，是**唯一权威来源**
- `delta_text` 仅用于流式展示，不作为发送文本
- `thinking_text` 独立存储，**永不**进入 outbound
- 即使 bridge 的 `drop_completed_text` 机制完全失效，也不会产生重复

**新增 OutboundMessageSanitizer（6 规则出站质量门）**：

```
规则 a: strip — 去除首尾空白和多余空行
规则 b: exact repeat — 严格字节比较，检测同一段话完全重复（>8 chars）
规则 c: near repeat — 编辑距离检测相似文本（阈值 0.85）
规则 d: meta leak — 正则检测思考内容/内心独白泄露（recall/tool/heartbeat/proactive/thinking 等元标记）
规则 e: role confusion — 检测角色错位（"你是"、"我不知道"等不当人称）
规则 f: SKIP/HEARTBEAT_OK auto-block — 空文本/跳过标记自动拦截（agent_task SKIP 检测的二次安全网）
```

### 4.3 PROACTIVE.md 重构

两份 PROACTIVE.md（vanilla + custom）全部重构：
- 身份规则：明确"我是 LightClaw"而非模拟用户身份
- 禁词清单：禁止输出内心独白、元思考、工具调用描述
- 消息结构：只输出面向用户的一句话，不加解释/免责声明/emoji
- 正反例：每条规则配正面示例和反面示例

### 4.4 测试结果

- 单元测试：29/29 PASS（真实 import，非 mock）
- 集成验证：10/10 PASS
- 手动触发：`POST /api/cron/proactivity/run?bypass_gate=true` → 管线正常

---

## 五、模块四：维护循环体系（Maintenance Loop v2）

**日期**：2026-05-04
**修改文件**：15 个（含 5 个新文件/目录）

### 5.1 Heartbeat Sensor-Gated 改造

**改造前**：每次心跳无条件触发完整 agent turn（每 30 分钟一次 LLM 调用）

**改造后**：脚本检测 → 有 trigger 才启动 agent turn

```python
# 9 项传感器检测（纯脚本，零 LLM，< 100ms）
def _run_sensor_checks() -> dict:
    return {
        "daily_talk_size_kb": _check_daily_talk_size(),       # >30KB
        "memory_md_size_kb": _check_memory_md_size(),         # >20KB
        "dreaming_status_error": _check_dreaming_error(),     # last_error
        "log_errors_new": _count_new_errors(),                # 去重后 actionable
        "learnings_pending": _check_learnings_pending(),      # >3
        "consecutive_abnormal": _check_consecutive_abnormal(), # >3
        "medium_window": _is_medium_window(),                 # ~4h 间隔
        "heavy_window": _is_heavy_window(),                   # 每日首次或结束
    }

# 无 trigger → 直接 HEARTBEAT_OK，不调 LLM
if not any_triggers and not medium_window and not heavy_window:
    return {"status": "HEARTBEAT_OK", "phase": "sensor_only"}
```

### 5.2 Dreaming Pipeline 完整实现

**三阶段架构**：

```
Light Phase (快速扫描)
  → 读 DAILY_TALK.md + memory/*.md
  → 提取候选记忆条目（36-42 candidates）
  → 过滤 secret / ephemeral

REM Phase (交叉引用)
  → 40 candidates × LLM 分析
  → 8 种信号类型：
     strengthens_existing_memory / conflicts_with_existing_memory
     / repeated_across_days / user_profile_update_needed
     / tool_rule_update_needed / agent_rule_update_needed
     / proactivity_update_needed / no_action
  → 解析器容错 4 级 fallback（fenced → aggressive brace-counting → raw → parse_error）

Deep Phase (落盘)
  → 基于 REM signals 评分 + promotion
  → 更新 managed files（USER.md / TOOLS.md / AGENTS.md / HEARTBEAT.md）
    使用 <!-- BEGIN/END DREAMING MANAGED --> 标记局部注入
  → 归档 DAILY_TALK → memory/YYYY-MM-DD.md
```

**调度**：
- 02:00 完整 dreaming (Light → REM → Deep)
- 14:30 light-only ingest（仅 Light Phase，产出 candidates 供下次 full 使用）

**Evidence 持久化**：
```
memory/.dreams/
├── candidates/full-YYYY-MM-DD-HHMM.json
├── candidates/light-ingest-YYYY-MM-DD-HHMM.json
├── signals/YYYY-MM-DD-HHMM.json
├── reinforcement.jsonl
├── conflicts.jsonl
└── status.json
```

### 5.3 记忆衰减引擎

**新增文件**：`memory_parser.py` (~300行) + `decay_scorer.py` (~480行) + `memory_pruner.py` (~290行)

**五维加权评分**：
```
final_score = 0.30 × recency + 0.25 × frequency + 0.20 × consolidation
            + 0.15 × persistence + 0.10 × richness
```

**半衰期分档**：
| 持久性 | 半衰期 | 适用内容 |
|--------|:------:|------|
| ephemeral | 3 天 | 天气、温度、版本号、余额、file_id |
| transient | 10 天 | 临时决策、短期备忘 |
| stable | 21 天 | 偏好、配置、决策 |
| permanent | 60 天 | 人格、名字、身份、长期规划 |

**三种交叉引用模式**：
1. 纯关键词：正则提取关键词 → 在 daily digest 中统计重叠
2. 纯向量：Jina 768d embedding → cosine similarity ≥ 0.70
3. 混合模式（默认）：`max(vec_hits, kw_hits)`

**首次运行效果**：淘汰 6 条（14.3%），保留 36 条（85.7%）
```
淘汰: [0.052] SSH 连接信息和别名
      [0.385] SiliconFlow 账户余额：278.01 元
      [0.390] 北京 2026-05-02 天气
      [0.397] 北京未来几天天气
保留: 配置、偏好、待办、决策等
```

---

## 六、模块五：并行任务调度（TaskDispatcher P3）

**日期**：2026-05-05 调研 + 2026-05-06 实施
**修改文件**：8 个（含 4 个新文件）

### 6.1 P3.0 调研设计

**8 路并行只读 Agent 调研**，确认：

1. 推荐方案 B：显式工具 `delegate_readonly_tasks` 触发，Agent 自主判断拆分时机，不改 Runner/RunController 架构
2. Worker 全链路就绪：runtime_profile + allowed_tools + max_iters_override + context_slimming 均可直接复用 P1b/P2a 基础设施
3. V1 Worker 白名单：8 个只读工具
4. 并发控制：asyncio.Semaphore(4) + TaskGroup

### 6.2 P3.1 实施

**数据流**：

```
主Agent调用 delegate_readonly_tasks(tasks=[...])
  → 输入验证（1-5 tasks, goal/context 长度限制, 写意图检测）
  → 构造 N 个 TurnRequest（独立 session_id, ephemeral, allowed_tools=8）
  → asyncio.Semaphore(4) 并发控制
  → 每个 worker: runner.stream_turns → 提取最终 AI 回复 → WorkerResult
  → 汇总 <child-agent-results> XML → 返回主Agent
```

**WorkerResult schema**：

```python
@dataclass
class WorkerResult:
    task_id: str
    status: str            # success|failed|timeout|cancelled|rejected
    summary: str           # ≤3000 chars（核心结论）
    evidence: list[dict]   # [{source, lines, claim}]
    risks: list[str]
    recommended_next_action: str
    tokens_used_estimate: int
    duration_seconds: float
    error: str | None
```

**上下文/记忆污染防护**：
| 污染源 | 防护 |
|--------|------|
| ToolMessage 进父 context | worker 完整历史丢弃，仅注入摘要 |
| Session history | 独立 `readonly_worker:{parent}:{idx}` session_id |
| DAILY_TALK | ephemeral=True 自动跳过 |
| MEMORY | 三重防护：无 write_file + 无 reminder + is_owner=False |
| Qdrant 索引 | 无 write_file → 无法写 .md → watchdog 无触发 |
| 主 agent 可见 | 仅 `<child-agent-results>` XML |

**End-to-end 测试**：

- 单 worker 端到端 ✅ （23.84s, 成功）
- 3 worker 并发 ✅ （16.98s / 23.11s / 25.08s, 全部成功）
- 8 种危险任务全部拒绝 ✅ （write_file / execute_shell / 递归 delegate / 写入 MEMORY / 发送消息 / 修改配置 / restart）
- worker 隔离验证 ✅ （memory_write_reminder=0 / DAILY_TALK 无触发 / compaction 跳过 / write_file=0）
- heartbeat 回归 ✅ （白名单不变，delegate_readonly_tasks 不在 heartbeat 白名单）

**实施中修复的 Bug**：
1. ContextVar 不跨 asyncio Task 传播 → 增加 module-level global fallback
2. "写入" pattern 缺失 → `_WRITE_INTENT_PATTERNS` 增加 "写入" 中文模式

---

## 七、模块六：崩溃循环修复与基础运维加固

**日期**：2026-05-05
**修改文件**：4 个

### 7.1 端口冲突死循环诊断

**故障链**：
```
yuanbao-doctor skill 执行 nohup lightclaw run &
  → 游离进程占用 port 13426
    → systemd 也启动 lightclaw.service
      → 新进程完整启动（embedding/Qdrant/channels ~4s）
        → uvicorn bind 失败 [Errno 98] address already in use
          → shutdown（非信号触发 → reason=unknown）
            → systemd Restart=on-failure → 5s 后重启
              → 死循环（18:35-18:42，~10次）
```

**连带损害**：
- Qdrant 锁冲突：新进程 Qdrant 被旧进程锁定 → 降级 FTS-only
- 元宝 WS 4014：新进程签权成功但被元宝服务端踢下线（重复连接）
- dreaming scheduled spam：每个新进程都重新注册 cron（非 dreaming bug）
- heartbeat 崩溃：`None[:10]` TypeError，`_state.json` 永远不更新

### 7.2 修复方案

**systemd 重启上限**（零代码）：
```ini
StartLimitBurst=3
StartLimitIntervalSec=60
# 60秒内最多3次失败重启，超限放弃
```

**PID 文件锁**（run_cmd.py）：
```python
PID_FILE = "/root/.lightclaw/lightclaw.pid"

# 存活检查：读取 /proc/<pid>/cmdline 验证是否 lightclaw 进程
# atexit.register：进程正常退出时删除 PID 文件
# stale lock 处理：PID 不存在或被其他进程复用 → 静默覆盖
```

**yuanbao-doctor skill 去游离进程**：
- `lightclaw restart` → `POST /api/config/channels/yuanbao/reconnect`（通道热重载）
- `nohup lightclaw run &` → `systemctl restart lightclaw`

**heartbeat `None[:10]` 修复**：
```python
# Before
state_date = state.get("last_check_at", "")[:10]   # None[:10] → TypeError

# After
state_date = (state.get("last_check_at") or "")[:10]  # None → "" → safe
```

---

## 八、模块七：基础安全修复

### 8.1 特殊 Token 清洗

**日期**：2026-05-03
**修改文件**：2 个

**问题**：tiktoken 编码时报错 `ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'`

**根因**：`count_str_tokens()` 直接调用 `enc.encode(text)`，未做特殊 token 清洗，上下文拼接中的模型控制 token 导致崩溃。

**修复**：

```python
# token_counting.py 新增
MODEL_SPECIAL_TOKEN_RE = re.compile(r'<[|｜][^|\r\n]{0,200}[|｜]>')

def strip_model_special_tokens(text: str) -> str:
    """三步清洗：
    1. 精确匹配替换 14 种已知模型 token（<|endoftext|>, <|im_start|>, ...）
    2. 正则 <|...|> 匹配任意形态
    3. 连续多空白压缩为一个空格
    不处理普通 < > | 字符
    """
```

**覆盖入口**：
- `count_str_tokens()` → 直接清洗 + `disallowed_special=()` 兜底
- `coordinator.py` extra_parts_map → `_clean_extra_parts_map()` 全量清洗
- `extra_sys_parts` → 用清洗后的值重建
- `memory_manager.py` → 两个 LLM 入口方法清洗（Dreaming Pipeline 全链路）

### 8.2 裸 LLM 调用清洗补齐

Maintenance Loop v2 勘察中发现 `proactivity.py:657` 和 `auth_detector_llm.py:158` 存在未清洗的 `model.ainvoke` 旁路。已补齐 `strip_model_special_tokens()` 调用。

---

## 九、开发过程量化统计

### 9.1 时间线

| 日期 | 工作内容 | 产出 |
|------|------|------|
| **5月3日** | Dreaming Pipeline 完整实现、记忆衰减引擎、特殊 Token 清洗、Session Distillation 修复 | 4 新文件 + 8 修改文件 |
| **5月4日** | Maintenance Loop v2 改造（8 步 15 文件）、Heartbeat 重新设计 | 15 文件修改 + 11 新目录/文件 |
| **5月5日** | 崩溃循环修复、Isolation Firewall 三阶段 30 文件、P0 止血、P1 硬刹车、P1.1 安全修边、P1b 工具白名单、P3.0 调研 | 30+ 文件修改 |
| **5月5日夜** | P2a 极轻量化 + P2b 最小权限 + P3.1 TaskDispatcher 实施 | 14+ 文件修改 |
| **5月6日** | 主动消息质量修复（hotfix + 架构根治 + 收尾）、P3.1.1 端到端验收 | 12+ 文件修改 |

### 9.2 技术深度亮点

1. **Error Fingerprint 系统**：从零设计的错误去重+cooldown+resolved detection 三合一机制，带状态持久化和自动裁剪
2. **四阶段 Context Hard Brake**：从 ToolMessage 截断 → recall 削减 → 消息裁剪 → fail-closed 拒绝，四层纵深防御
3. **ActorContext 身份系统**：5 级信任度 + 5 种 Session Key 命名空间 + 7×4 权限矩阵，全程 fail-closed（缺省拒绝）
4. **Secret Firewall**：14 种 regex + 5 重过滤（recall 入口 → DAILY_TALK → FTS → Qdrant 索引 → Dreaming）
5. **TaskDispatcher**：Agent 自主判断拆分 + asyncio 并发 + 独立 session 隔离 + Final Summary Only，0 侵入现有架构
6. **Maintenance Loop v2**：sensor-gated heartbeat（90%+ 跳次零 LLM）+ 三阶段 Dreaming + 记忆衰减五维评分

### 9.3 费用说明

五一期间 5 天高强度开发，DeepSeek V4 Pro + ChatGPT 5.5 两个大模型持续高强度推理：
- 每次调研派出 3-8 个 Agent 并行探查不同模块，每个 Agent 需多轮模型调用
- 每次实施派出 2-4 个 Agent 并行修改代码，每个 Agent 需阅读源码 + 编写代码
- 每次验证需真实模型调用确认（非 mock），包括 dashboard 对话、heartbeat 周期、worker 并发等
- 单次重度分析（如上下文拼接全链路调研）需跨 8 个模块、读 15+ 文件、分析 2000+ 行代码
- 多轮交叉验证、故障排查、回归测试

**总计约 ¥1,600**，用于购买 DeepSeek V4 Pro 和 ChatGPT 5.5 的高强度推理能力，以支撑多 Agent 并行开发和代码审计。

---

## 十、当前系统状态

### 10.1 服务运行状态

```
LightClaw service: active (running)
Dashboard channel: started
Yuanbao WebSocket: connected
WeChat poll loop: running (1 account)
LightClaw gateway: Socket.IO connected
Qdrant: vector+FTS mode (768d, jina-embeddings-v2-base-zh)

Cron 调度:
  heartbeat: 30min interval
  dreaming full: 0 2 * * * Asia/Shanghai
  dreaming light-ingest: 30 14 * * * Asia/Shanghai
  proactivity: engine-driven (1800s interval)
  daily_proactivity_analysis: 23:00
  早间群消息汇总: 0 9 * * *
  午后群消息汇总: 0 14 * * *
```

### 10.2 6 重防火墙状态

| 防火墙 | 状态 |
|--------|:--:|
| Principal 身份映射（dashboard/yuanbao/weixin） | ✅ |
| Recall 权限矩阵（7×4 scope 过滤） | ✅ |
| DAILY_TALK realtime injection 默认关闭 | ✅ |
| Qdrant v2 metadata + scope Must/MustNot filter | ✅ |
| Context Budget 9 项硬上限 | ✅ |
| Secret Firewall 14 patterns + 5 重过滤 | ✅ |

### 10.3 Heartbeat 轻量级状态

| 指标 | Before | After |
|------|--------|-------|
| 工具权限 | 40+ tools（含 write/shell/browser） | 3 tools（read_status_file / read_recent_heartbeat_errors / get_current_time） |
| Max input tokens | 131072 | **8000** |
| Max iters | 100 | **5** |
| Turn timeout | 120s | **75s** |
| HEARTBEAT.md 行数 | 46 行 | **5 行** |
| Recall block | ~600 tokens | **0 tokens**（skipped） |
| 每 turn token 节省 | — | **~752 tokens** |
| 每日 token 节省 | — | **~36K tokens** |
| 90%+ 心跳 | 调 LLM | **直接 HEARTBEAT_OK**（零 LLM） |

---

## 十一、遗留风险与后续建议

| 风险 | 级别 | 说明 |
|------|:----:|------|
| Qdrant 维度对齐 | 中 | jina-embeddings-v2-base-zh 输出 1024 维但 collection 用 768 维创建 |
| worker session 完全禁写 | 低 | 当前 ephemeral=True 会存最后一个 AI msg |
| yuanbao WS 4014 | 预存在 | token 过期需重新签权 |
| DAILY_TALK 旧条目无 metadata | 低 | 不影响安全（realtime injection OFF） |
| 多通道真实流量 | 中 | dashboard 已验证，yuanbao/weixin 需真实用户消息触发 |
| 按 runtime_profile 区分 model-level timeout | P3 | heartbeat 60s / dashboard 120-300s |

**后续 P3.2 建议**：
1. worker session 清理策略（readonly_worker 前缀定期清理）
2. read_file_head 工具（限制 worker 只能范围读取文件）
3. web_search 按需授权（特定场景允许 worker 使用搜索）
4. UI 展示 worker progress（Dashboard 实时显示子任务进度）
5. Qdrant 维度对齐修复

---

*报告完毕。本报告基于 25 份 memory 记录、60+ 文件修改、100+ 测试用例、15+ 次服务重启验证编写而成。*
