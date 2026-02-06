# 可观测性指南：事件总线与日志配置

## 概述

Comate Agent SDK 提供了完整的可观测性支持，帮助开发者追踪、调试和监控 Agent 的运行状态。主要包含两个核心机制：

- **事件总线 (EventBus)**: 追踪所有上下文变更事件，实现审计和监控
- **日志系统 (Logging)**: 基于 Python logging 模块的分层日志配置

## 事件总线 (EventBus)

### 核心组件

- **ContextEventBus** (`comate_agent_sdk/context/observer.py`): 事件总线实现
- **ContextEvent** (`comate_agent_sdk/context/observer.py`): 事件数据结构
- **EventType** (`comate_agent_sdk/context/observer.py`): 事件类型枚举

### 设计模式：观察者模式

事件总线是典型的**观察者模式(Observer Pattern)**实现：

```
ContextIR (发布者)
    │
    ├─ emit(event) ──► EventBus
    │                     │
    │                     ├─ 记录到日志
    │                     ├─ 存入事件历史
    │                     └─ 通知所有订阅者
    │
订阅者 1, 2, 3...
```

**核心优势**：
1. **解耦设计**: ContextIR 不需要知道谁在监听，订阅者也不需要修改 ContextIR 代码
2. **可追溯性**: 保留最近 1000 条事件历史，方便排查问题
3. **扩展性**: 通过订阅机制灵活添加自定义监控逻辑

### 事件类型

| 事件类型 | 说明 | 触发场景 |
|---------|------|---------|
| `ITEM_ADDED` | 添加上下文条目 | 添加消息、设置 system prompt、注入工具等 |
| `ITEM_REMOVED` | 移除上下文条目 | 移除 skill strategy、MCP tools 等 |
| `ITEM_DESTROYED` | 销毁临时内容 | Ephemeral 工具输出被销毁 |
| `COMPACTION_PERFORMED` | 执行上下文压缩 | Token 超限触发自动压缩 |
| `REMINDER_REGISTERED` | 注册系统提醒 | 注册 system reminder |
| `REMINDER_REMOVED` | 移除系统提醒 | 清理过期提醒 |
| `CONTEXT_CLEARED` | 清空所有上下文 | 调用 `context.clear()` |
| `CONVERSATION_REPLACED` | 替换对话段 | 从持久化恢复会话 |
| `BUDGET_EXCEEDED` | 预算超限 | Token 超出配置限额 |
| `TODO_STATE_UPDATED` | TODO 状态变更 | 更新 TODO 列表 |

### 使用示例

#### 1. 基本事件发送（SDK 内部）

在 SDK 内部，每当上下文发生变化时都会发送事件：

```python
# 在 context/ir.py 中
event_bus.emit(ContextEvent(
    event_type=EventType.ITEM_ADDED,
    item_type=ItemType.MCP_TOOL,
    item_id=item.id,
    detail="mcp_tools set"
))
```

#### 2. 订阅事件（用户代码）

开发者可以订阅事件来实现自定义监控：

```python
from comate_agent_sdk import Agent
from comate_agent_sdk.agent import ComateAgentOptions

# 自定义事件处理函数
def my_event_handler(event):
    """记录所有上下文变更到自定义日志"""
    if event.event_type.value == "compaction_performed":
        print(f"⚠️  触发压缩: {event.detail}")
    elif event.event_type.value == "item_added":
        print(f"✅ 添加条目: {event.item_type} - {event.detail}")

# 创建 Agent
agent = Agent(options=ComateAgentOptions())

# 订阅事件
agent._context.event_bus.subscribe(my_event_handler)

# 后续所有上下文变更都会触发 my_event_handler
```

#### 3. 查询事件历史

```python
# 获取最近的事件历史（最多 1000 条）
event_log = agent._context.event_bus.event_log

# 分析事件
for event in event_log:
    print(f"{event.timestamp}: {event.event_type.value} - {event.detail}")
```

#### 4. 实战：性能监控

```python
from collections import defaultdict

def performance_monitor(event):
    """统计各类条目的添加频率"""
    if event.event_type.value == "item_added":
        performance_monitor.stats[event.item_type.value] += 1

performance_monitor.stats = defaultdict(int)

agent._context.event_bus.subscribe(performance_monitor)

# 运行一段时间后查看统计
print("上下文条目统计:")
for item_type, count in performance_monitor.stats.items():
    print(f"  {item_type}: {count} 次")
```

## 日志系统 (Logging)

### 架构设计

SDK 采用 Python 标准 `logging` 模块的**分层架构**：

```
根 logger
    │
    └─ comate_agent_sdk
          │
          ├─ comate_agent_sdk.context
          │     ├─ comate_agent_sdk.context.observer  ← event_bus 的 debug 日志
          │     ├─ comate_agent_sdk.context.ir
          │     └─ comate_agent_sdk.context.compaction
          │
          ├─ comate_agent_sdk.agent
          │     ├─ comate_agent_sdk.agent.core
          │     └─ comate_agent_sdk.agent.runner
          │
          └─ comate_agent_sdk.llm
                ├─ comate_agent_sdk.llm.anthropic
                └─ comate_agent_sdk.llm.openai
```

**设计原则**：
- SDK 核心代码只创建 logger，不配置输出
- 日志配置由使用者决定（应用代码或示例代码）
- 支持细粒度控制（可以只调整某个模块的日志级别）

### 默认行为

如果不配置任何 handler，Python logging 的默认行为：
- 输出位置：**stderr（标准错误流）**，即控制台
- 日志级别：**WARNING** 及以上
- 结果：**DEBUG 和 INFO 级别的日志不会显示**

### 配置方法

#### 1. 基础配置（应用级）

在你的应用入口配置日志：

```python
import logging

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

#### 2. 使用环境变量（推荐）

支持通过环境变量动态调整日志级别：

```python
import logging
import os

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),  # 从环境变量读取，默认 INFO
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

使用方式：
```bash
# 查看 DEBUG 级别日志（包括 event_bus 的事件日志）
export LOG_LEVEL=DEBUG
uv run python your_script.py

# 只显示 WARNING 及以上
export LOG_LEVEL=WARNING
uv run python your_script.py
```

#### 3. 细粒度控制

只调整特定模块的日志级别：

```python
import logging

# 全局设置为 INFO
logging.basicConfig(level=logging.INFO)

# 只让 event_bus 输出 DEBUG 日志
logging.getLogger("comate_agent_sdk.context.observer").setLevel(logging.DEBUG)

# 关闭某个模块的日志
logging.getLogger("comate_agent_sdk.llm").setLevel(logging.ERROR)
```

#### 4. 写入文件

同时输出到控制台和文件：

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("agent.log"),      # 写入文件
        logging.StreamHandler(),               # 输出到控制台
    ]
)
```

#### 5. 结构化日志（高级）

使用 JSON 格式输出日志，方便机器解析：

```python
import logging
import json
import sys

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_data, ensure_ascii=False)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())

root_logger = logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(logging.DEBUG)
```

### Event Bus 的日志输出

Event Bus 在 `emit()` 时会自动输出 DEBUG 级别的日志：

```python
# 在 observer.py:86-91
logger.debug(
    f"ContextEvent: {event.event_type.value} "
    f"item_type={event.item_type.value if event.item_type else 'N/A'} "
    f"item_id={event.item_id or 'N/A'} "
    f"detail={event.detail}"
)
```

要看到这些日志，需要：
1. 设置日志级别为 DEBUG
2. 确保 `comate_agent_sdk.context.observer` 这个 logger 的级别至少是 DEBUG

示例输出：
```
2025-02-06 10:30:15 DEBUG comate_agent_sdk.context.observer: ContextEvent: item_added item_type=mcp_tool item_id=a1b2c3d4 detail=mcp_tools set
2025-02-06 10:30:20 DEBUG comate_agent_sdk.context.observer: ContextEvent: compaction_performed item_type=N/A item_id=N/A detail=auto_compact: 8500 → 6200 tokens
```

## 实战示例

### 完整监控系统

结合事件总线和日志系统，构建完整的监控方案：

```python
import logging
import os
from datetime import datetime
from comate_agent_sdk import Agent
from comate_agent_sdk.agent import ComateAgentOptions

# 1. 配置日志
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"agent_{datetime.now():%Y%m%d}.log"),
        logging.StreamHandler(),
    ]
)

# 2. 创建自定义监控器
class AgentMonitor:
    def __init__(self):
        self.compaction_count = 0
        self.token_saved = 0

    def handle_event(self, event):
        if event.event_type.value == "compaction_performed":
            self.compaction_count += 1
            # 解析 token 变化
            if "→" in event.detail:
                parts = event.detail.split("→")
                if len(parts) == 2:
                    before = int(parts[0].split(":")[1].strip().split()[0])
                    after = int(parts[1].strip().split()[0])
                    saved = before - after
                    self.token_saved += saved
                    logging.info(f"压缩节省 {saved} tokens (总计节省: {self.token_saved})")

    def report(self):
        print("\n=== 监控报告 ===")
        print(f"压缩次数: {self.compaction_count}")
        print(f"节省 tokens: {self.token_saved}")

# 3. 创建 Agent 并注册监控器
monitor = AgentMonitor()
agent = Agent(options=ComateAgentOptions())
agent._context.event_bus.subscribe(monitor.handle_event)

# 4. 使用 Agent...
# session = agent.chat()
# ...

# 5. 查看监控报告
# monitor.report()
```

### 调试模式快速开关

在开发环境快速启用详细日志：

```python
import os

DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

if DEBUG_MODE:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    print("🔍 Debug mode enabled")
```

使用方式：
```bash
# 正常模式
uv run python my_agent.py

# 调试模式
DEBUG=true uv run python my_agent.py
```

## 最佳实践

### 1. 日志级别使用建议

| 级别 | 使用场景 | 示例 |
|-----|---------|------|
| DEBUG | 详细的诊断信息 | Event bus 事件、上下文变更细节 |
| INFO | 关键流程节点 | Agent 初始化、Subagent 发现、会话恢复 |
| WARNING | 异常但可恢复的情况 | 配置缺失使用默认值、Hook 失败 |
| ERROR | 错误但不致命 | 工具执行失败、LLM 调用超时 |
| CRITICAL | 致命错误 | 无法初始化 Agent、核心依赖缺失 |

### 2. 事件订阅的生命周期管理

```python
def temporary_monitor(event):
    print(f"Event: {event.event_type.value}")

# 订阅
agent._context.event_bus.subscribe(temporary_monitor)

# ... 使用一段时间 ...

# 取消订阅（避免内存泄漏）
agent._context.event_bus.unsubscribe(temporary_monitor)
```

### 3. 生产环境配置建议

```python
import logging
from logging.handlers import RotatingFileHandler

# 生产环境：文件日志 + 日志轮转
handler = RotatingFileHandler(
    "agent.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,           # 保留 5 个备份
)
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))

logging.getLogger("comate_agent_sdk").addHandler(handler)
logging.getLogger("comate_agent_sdk").setLevel(logging.INFO)
```

### 4. 避免日志泄露敏感信息

```python
# ❌ 错误：直接记录用户输入
logger.info(f"User query: {user_input}")

# ✅ 正确：脱敏或使用摘要
logger.info(f"User query length: {len(user_input)}")
logger.debug(f"User query preview: {user_input[:50]}...")
```

## 常见问题

### Q: 为什么设置了 DEBUG 但看不到 event_bus 日志？

A: 确认两点：
1. 日志级别是否为 DEBUG：`logging.basicConfig(level=logging.DEBUG)`
2. 是否有自定义配置覆盖了 logger 级别

### Q: 事件历史最多保留多少条？

A: 默认保留最近 1000 条事件（`MAX_EVENT_LOG_SIZE = 1000`），超出后自动丢弃最早的事件。

### Q: 可以禁用事件总线吗？

A: 事件总线是核心机制的一部分，无法禁用。但如果不订阅事件，性能开销非常小（只有内存中的事件历史）。

### Q: 如何在 Jupyter Notebook 中配置日志？

A: Jupyter 需要特殊配置避免重复日志：

```python
import logging

# 清除现有 handlers
logger = logging.getLogger("comate_agent_sdk")
logger.handlers = []

# 重新配置
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True  # 强制重新配置
)
```

## 参考资料

- Event Bus 实现：`comate_agent_sdk/context/observer.py`
- Context IR 事件发送：`comate_agent_sdk/context/ir.py`
- 示例配置：`comate_agent_sdk/examples/chat_session_repl.py`
- Python logging 官方文档：https://docs.python.org/3/library/logging.html
