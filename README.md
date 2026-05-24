# LightClaw 稳定性与安全加固 — 五一攻坚代码发布

> **开发周期**: 2026年5月2日-6日（连续5天，含五一假期全程）
> **大模型费用**: 约 ¥1,600（DeepSeek V4 Pro + ChatGPT 5.5 高强度推理）
> **代码量**: 58个文件, 22,862行 (Python: 21,837行, 配置/文档: 1,043行)

---

## 项目简介 / Project Summary

中文：本次发布围绕 LightClaw AI Agent 的研发过程展开，从需求拆解、风险识别到模块重构、联调验证，按“运行控制—会话隔离—主动消息质量—记忆系统—任务调度”的路径逐步推进。过程中先补齐 heartbeat 止血、上下文硬刹车与敏感信息过滤，再修复崩溃循环、完善主动消息与回归测试，保证每一步改动都可验证、可回滚、可持续迭代。

English: This release is built on Python, asyncio, FastAPI-style service APIs, Qdrant, and a modular agent architecture. It combines CLI tooling, cron workers, prompt files, and pytest-based verification to deliver a secure, observable, and maintainable AI agent stack.

---

## 一、项目概述

本次攻坚对 LightClaw AI Agent 框架进行了系统性稳定性加固和安全性改造，涵盖 **7大模块**：

| 模块 | 核心价值 |
|------|---------|
| **运行控制层 (P0-P2)** | heartbeat 自激循环止血 + 上下文硬刹车 + 工具权限最小化 + 极致轻量化 |
| **会话隔离防火墙** | 6重跨用户隐私隔离 + 14种 secret 模式检测 + 9项上下文预算硬上限 |
| **主动消息质量** | 文本重复发送根除 + 6规则出站质量门 |
| **维护循环 v2** | sensor-gated heartbeat (90%+无LLM) + Dreaming三阶段 + 记忆衰减引擎 |
| **TaskDispatcher (P3)** | Agent 自主并行任务拆分 + 8工具只读白名单 + 独立上下文隔离 |
| **崩溃循环修复** | 多实例死循环根除 + PID锁 + heartbeat持久化修复 |
| **基础安全** | 模型特殊 token 三层清洗 + 裸LLM调用补齐 |

---

## 二、目录结构

```
lightclaw-release/
├── lightclaw/                    # 核心源代码 (保留原目录结构)
│   ├── constant.py               # 全局常量 (env安全解析 + 全部阈值)
│   ├── agent/
│   │   ├── core/                 # 核心引擎层
│   │   │   ├── actor_context.py  # [新] Actor身份系统 (5级信任度)
│   │   │   ├── coordinator.py    # 中间件主协调器 (四阶段硬刹车)
│   │   │   ├── token_budget.py   # 上下文预算强制执行
│   │   │   ├── recall.py         # 自动召回 (权限矩阵过滤)
│   │   │   ├── compactor.py      # 记忆压缩器 (summary上限)
│   │   │   ├── runtime_context.py # [改] per-turn运行时上下文
│   │   │   ├── model_factory.py  # [改] timeout/max_retries注入
│   │   │   ├── engine_session.py # [改] max_iters_override接入
│   │   │   ├── bridge.py         # [改] text_buffer.clear()
│   │   │   ├── builtins.py       # [改] 工具注册 (新增4工具)
│   │   │   └── message_sanitizer.py # 消息清洗
│   │   ├── memory/               # 记忆系统层
│   │   │   ├── dreaming_pipeline.py  # [新] Dreaming三阶段编排
│   │   │   ├── dreaming_types.py     # [新] Dreaming数据模型
│   │   │   ├── memory_parser.py      # [新] 结构化记忆解析
│   │   │   ├── decay_scorer.py       # [新] 五维衰减评分
│   │   │   ├── memory_pruner.py      # [新] 记忆淘汰编排
│   │   │   ├── recall_candidate.py   # [新] Recall候选数据模型
│   │   │   ├── qdrant_schema.py      # [改] 14个新隔离payload字段
│   │   │   ├── qdrant_searcher.py    # [改] scope filter
│   │   │   ├── qdrant_indexer.py     # [改] v2 metadata索引
│   │   │   ├── session_distillation_service.py # [改] char截断
│   │   │   ├── memory_query_service.py   # [改] session filter
│   │   │   └── init_memory_system.py     # [改] embedding时序
│   │   ├── tools/                # 工具系统层
│   │   │   ├── delegate_readonly_tasks.py        # [新] 并行任务分发
│   │   │   ├── delegate_readonly_tasks_worker.py # [新] worker runner
│   │   │   ├── read_status_file.py               # [新] 安全状态读取
│   │   │   ├── read_recent_heartbeat_errors.py    # [新] 受控错误摘要
│   │   │   ├── __init__.py                        # [改] 工具注册
│   │   │   └── browser/auth_detector_llm.py       # [改] token清洗
│   │   └── utils/
│   │       └── token_counting.py  # [改] 三层特殊token清洗
│   ├── app/                      # 应用层
│   │   ├── _app.py               # [改] runner_context注入
│   │   ├── runner_context.py     # [新] ContextVar全局访问
│   │   ├── cron/
│   │   │   ├── heartbeat.py      # [改] 完全重写: fingerprint+whitelist
│   │   │   ├── dreaming.py       # [新] dreaming cron入口
│   │   │   ├── proactivity.py    # [改] token清洗+记忆注入重构
│   │   │   ├── agent_task.py     # [改] AssistantTurnAccumulator
│   │   │   ├── manager.py        # [改] 新增dreaming cron
│   │   │   ├── api.py            # [改] 新增proactivity/dreaming API
│   │   │   ├── outbound_sanitizer.py # [新] 6规则出站质量门
│   │   │   └── turn_accumulator.py   # [新] AssistantTurnAccumulator
│   │   └── runner/
│   │       ├── core/run_controller.py    # [改] ActorContext传递
│   │       └── session/
│   │           ├── session_factory.py    # [改] allowed_tools/budget注入
│   │           └── daily_talk_store.py   # [改] channel/session metadata
│   ├── config/
│   │   └── config.py             # [改] SecurityConfig + DreamingConfig
│   └── cli/
│       ├── run_cmd.py            # [改] PID文件锁
│       └── dreaming_cmd.py       # [新] dreaming CLI
├── configs/
│   ├── lightclaw.json            # 运行时配置 (security/dreaming段)
│   └── lightclaw.service         # systemd unit (StartLimitBurst)
├── workspace/
│   ├── HEARTBEAT.md              # [新] 极简5行版
│   ├── HEARTBEAT.en.md           # 英文prompt版
│   ├── HEARTBEAT.zh.md           # 中文prompt版
│   ├── PROACTIVE.md              # [新] 主动消息指令重构
│   ├── PROACTIVE.en.md           # 英文prompt版
│   ├── PROACTIVE.zh.md           # 中文prompt版
│   └── yuanbao-doctor-SKILL.md   # [改] 去游离进程+热重载
└── tests/
    ├── test_outbound_message_quality.py   # 29单元测试
    └── verify_proactive_pipeline.py       # 10集成验证
```

---

## 三、核心技术亮点

### 1. Error Fingerprint 去重系统
从零设计的错误去重+cooldown+resolved detection三合一机制，12种错误模式识别，JSON持久化+自动裁剪。

### 2. 四阶段上下文硬刹车
ToolMessage总量截断 → recall削减 → 消息裁剪 → fail-closed拒绝，四层纵深防御，消灭 302K tokens 超限请求。

### 3. ActorContext 身份系统
5级信任度(owner/trusted/group_member/unknown/system) + 5种Session Key命名空间 + 7×4权限矩阵，全程fail-closed。

### 4. Secret Firewall
14种regex模式 + 5重过滤链(recall→DAILY_TALK→FTS→Qdrant索引→Dreaming)，敏感信息无法进入LLM上下文。

### 5. TaskDispatcher 并行调度
Agent自主判断拆分时机 + asyncio并发 + 独立session隔离 + Final Summary Only，0侵入现有架构。

### 6. Heartbeat 极致轻量化
从40+工具权限→3工具, 131K输入→8K, 100 iters→5, 每turn节省~752 tokens, 90%+心跳零LLM。

---

## 四、变更统计

| 分类 | 数量 |
|------|:---:|
| 新增 Python 模块 | 15个 |
| 修改 Python 文件 | 33个 |
| 配置文件 | 2个 |
| Prompt 文件 | 5个 |
| 测试文件 | 2个 |
| **总计** | **58文件** |

---

## 五、测试验证

- 语法检查: 全部 py_compile 通过
- 单元测试: 29/29 PASS (真实import, 非mock)
- 集成验证: 10/10 PASS
- fingerprint 场景: 5/5 PASS
- env非法值: 5/5 PASS
- secret检测: 7/7 PASS
- worker并发: 3 worker全部成功
- 危险任务拒绝: 8/8 pattern全部拒绝
- heartbeat回归: 白名单不变
- 服务重启: 15+次, 0引入性错误

---

## 六、详细信息

完整的开发工作汇报见 [LightClaw-开发工作全面汇报.md](../LightClaw-开发工作全面汇报.md)（1123行，包含所有代码逻辑详解、测试数据、效果对比）。
