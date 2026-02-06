# ChatSession 使用指南

## 概述

`ChatSession` 是 Comate Agent SDK 提供的会话管理层，在 `Agent` 基础上增加了**持久化**、**恢复**、**分叉**和**流式输入**能力。它将对话历史、上下文状态自动保存到文件系统，支持长时间运行、多分支对话等复杂场景。

### 核心能力

| 能力 | 说明 | 适用场景 |
|------|------|---------|
| **持久化** | 自动保存对话历史到 JSONL 文件 | 聊天机器人、需要重启后恢复的应用 |
| **恢复（Resume）** | 从磁盘恢复已保存的会话 | 服务重启、跨进程对话 |
| **分叉（Fork）** | 从当前会话创建副本，探索不同对话分支 | A/B 测试、回退到历史状态 |
| **流式输入** | 支持消息迭代器，批量处理或实时接收 | 批量任务、消息队列、异步输入 |

### Agent vs ChatSession

```python
# Agent：适合单次查询或简单多轮对话
agent = Agent(llm=ChatOpenAI(model="gpt-4o"), options=ComateAgentOptions(tools=[...]))
result = await agent.query("帮我查询天气")

# ChatSession：适合需要持久化、恢复、分叉的场景
session = ChatSession(agent, session_id="user-123")
async for event in session.query_stream("帮我查询天气"):
    ...  # 对话历史自动保存
```

**选择建议**：
- 如果只需要简单的工具调用，使用 `Agent`
- 如果需要会话管理、历史恢复、多分支对话，使用 `ChatSession`

---

## 快速开始

### 基础示例：单轮对话

```python
from comate_agent_sdk import Agent
from comate_agent_sdk.agent import ComateAgentOptions, ChatSession
from comate_agent_sdk.llm import ChatOpenAI
from comate_agent_sdk.tools import tool

@tool("获取当前时间")
async def get_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 创建 Agent
agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    options=ComateAgentOptions(tools=[get_time]),
)

# 创建 ChatSession
session = ChatSession(agent, session_id="demo-session")

# 流式对话
async for event in session.query_stream("现在几点了？"):
    if isinstance(event, TextEvent):
        print(event.content)
    elif isinstance(event, StopEvent):
        print(f"完成：{event.reason}")
```

**存储位置**：默认保存在 `~/.agent/sessions/demo-session/`

---

### 多轮对话示例

```python
from comate_agent_sdk.agent import ChatSession

session = ChatSession(agent, session_id="user-alice")

# 第一轮
async for event in session.query_stream("我叫 Alice"):
    ...

# 第二轮（Agent 会记住 "Alice"）
async for event in session.query_stream("我叫什么名字？"):
    if isinstance(event, TextEvent):
        print(event.content)  # 输出：你叫 Alice

# 历史自动保存到 ~/.agent/sessions/user-alice/context.jsonl
```

---

## 核心功能

### 3.1 会话持久化与恢复

`ChatSession` 使用**增量持久化**策略，每次对话后自动保存变更，重启后可完整恢复。

#### 持久化格式

```
~/.agent/sessions/{session_id}/
  ├── context.jsonl       # 对话历史（增量事件流）
  ├── todos.json          # TODO 状态
  └── offload/            # 大型工具输出卸载目录
      ├── index.json
      └── {item_id}.txt
```

#### 恢复会话

```python
from comate_agent_sdk.agent import ChatSession

# 首次创建
session = ChatSession(agent, session_id="persistent-chat")
async for event in session.query_stream("记住我喜欢蓝色"):
    ...

# === 程序重启后 ===

# 恢复会话（历史自动加载）
session = ChatSession.resume(
    agent,
    session_id="persistent-chat"  # 使用相同的 session_id
)

async for event in session.query_stream("我喜欢什么颜色？"):
    ...  # Agent 记得之前的对话
```

#### 自定义存储路径

```python
from pathlib import Path

session = ChatSession(
    agent,
    session_id="custom-session",
    storage_root=Path("/data/sessions/custom-session")  # 自定义路径
)

# 恢复时也需要指定相同路径
session = ChatSession.resume(
    agent,
    session_id="custom-session",
    storage_root=Path("/data/sessions/custom-session")
)
```

---

### 3.2 会话分叉（Fork）

从当前会话创建副本，探索不同的对话分支，互不影响。

#### 基础用法

```python
# 原会话
session = ChatSession(agent, session_id="main")
async for event in session.query_stream("帮我写一个排序函数"):
    ...

# 分叉出两个方向
fork_bubble = session.fork_session()  # 探索冒泡排序
fork_quick = session.fork_session()   # 探索快速排序

# 分叉后的会话有独立的 session_id
async for event in fork_bubble.query_stream("用冒泡排序实现"):
    ...

async for event in fork_quick.query_stream("用快速排序实现"):
    ...

# 原会话不受影响
async for event in session.query_stream("用归并排序实现"):
    ...
```

#### 对话树场景

```python
from pathlib import Path

# 主对话
main_session = ChatSession(agent, session_id="story-main")
async for event in main_session.query_stream("写一个科幻故事的开头"):
    ...

# 探索不同结局
ending_a = main_session.fork_session(
    storage_root=Path("./sessions/story-ending-a")
)
async for event in ending_a.query_stream("主角选择留在地球"):
    ...

ending_b = main_session.fork_session(
    storage_root=Path("./sessions/story-ending-b")
)
async for event in ending_b.query_stream("主角选择离开地球"):
    ...
```

**注意**：
- `fork_session()` 会完整复制 `context.jsonl` 和 `offload/` 目录
- 分叉后的会话获得新的 `session_id`（自动生成 UUID）
- 分叉时原会话不能已关闭（`ChatSessionClosedError`）

---

### 3.3 流式输入（MessageSource）

`ChatSession` 支持通过 `message_source` 参数传入消息迭代器，实现批量处理或异步消息队列。

#### 类型定义

```python
ChatMessage = str | list[ContentPartTextParam | ContentPartImageParam]
MessageSource = AsyncIterator[ChatMessage] | Iterator[ChatMessage] | Iterable[ChatMessage]
```

#### 示例 1：批量处理消息列表

```python
from comate_agent_sdk.agent import ChatSession

# 准备批量消息
messages = [
    "总结这篇文章",
    "翻译成英文",
    "生成标题"
]

# 创建 ChatSession，传入消息列表
session = ChatSession(
    agent,
    session_id="batch-task",
    message_source=messages  # 可迭代对象
)

# 自动依次处理每条消息
async for event in session.events():
    if isinstance(event, TextEvent):
        print(event.content)
    elif isinstance(event, StopEvent):
        print("--- 一条消息处理完成 ---")
```

**工作流程**：
1. `events()` 从 `message_source` 取第一条消息
2. 调用 `agent.query_stream(message)` 处理
3. 产出 `StopEvent` 后，取下一条消息
4. 重复直到消息耗尽

#### 示例 2：异步消息队列

```python
import asyncio
from comate_agent_sdk.agent import ChatSession

async def message_generator():
    """模拟实时消息流（如 WebSocket、消息队列）"""
    messages = [
        "查询订单 #001",
        "创建新工单",
        "发送通知给用户"
    ]
    for msg in messages:
        await asyncio.sleep(1)  # 模拟延迟
        yield msg

session = ChatSession(
    agent,
    session_id="realtime-queue",
    message_source=message_generator()  # 异步生成器
)

async for event in session.events():
    ...  # 处理事件
```

#### 示例 3：手动发送消息（send/close）

```python
from comate_agent_sdk.agent import ChatSession

# 不提供 message_source
session = ChatSession(agent, session_id="interactive")

# 启动事件监听（后台任务）
async def event_loop():
    async for event in session.events():
        if isinstance(event, TextEvent):
            print(event.content)

asyncio.create_task(event_loop())

# 主线程手动发送消息
await session.send("第一条消息")
await asyncio.sleep(2)
await session.send("第二条消息")

# 关闭会话
await session.close()  # 触发 events() 结束
```

**注意**：
- 如果提供了 `message_source`，则不能调用 `send()`（抛出 `ChatSessionError`）
- `send()` 和 `close()` 配合使用时，适合 WebSocket、REPL 等交互场景

---

### 3.4 清空历史

清空对话历史和 token 统计，保持会话持久化。

```python
session = ChatSession.resume(agent, session_id="test")

# 已有对话历史
async for event in session.query_stream("我叫 Bob"):
    ...

# 清空历史
session.clear_history()

# 再次查询时，Agent 不记得之前的内容
async for event in session.query_stream("我叫什么？"):
    ...  # Agent 回答：我不知道

# context.jsonl 中会写入 conversation_reset 事件
```

**注意**：
- `clear_history()` **不可逆**，清空后无法恢复
- 如需保留历史，先使用 `fork_session()` 创建备份
- 不要在 `query_stream()` 执行过程中调用

---

## 高级用法

### 4.1 异步上下文管理

使用 `async with` 自动管理资源：

```python
from comate_agent_sdk.agent import ChatSession

async with ChatSession(agent, session_id="auto-close") as session:
    async for event in session.query_stream("帮我查询天气"):
        ...
# 自动调用 session.close()
```

---

### 4.2 Token 使用统计

```python
from comate_agent_sdk.agent import ChatSession

session = ChatSession(agent, session_id="stats")

# 执行对话
async for event in session.query_stream("写一个斐波那契函数"):
    ...

# 获取 token 统计
usage = await session.get_usage()
print(f"总 tokens: {usage.total_tokens}")
print(f"总成本: ${usage.total_cost:.4f}")
print(f"按模型统计: {usage.by_model}")
```

**返回字段**：
- `total_tokens`: 总 token 数（包括 prompt + completion + cache）
- `total_cost`: 总成本（USD）
- `by_model`: 按模型分组的统计
- `by_level`: 按档位（LOW/MID/HIGH）分组的统计

---

### 4.3 上下文信息查询

```python
session = ChatSession(agent, session_id="context-info")

# 查询上下文使用情况
info = await session.get_context_info()

print(f"当前 token 数: {info.current_tokens}")
print(f"预算上限: {info.budget_tokens}")
print(f"使用率: {info.current_tokens / info.budget_tokens * 100:.1f}%")
print(f"对话轮次: {len(info.conversation_items)}")
```

**注意**：即使会话已关闭（`close()`），仍可调用 `get_usage()` 和 `get_context_info()`。

---

## 完整示例

### 示例 1：聊天机器人（持久化 + 恢复）

```python
import asyncio
from comate_agent_sdk import Agent
from comate_agent_sdk.agent import ComateAgentOptions, ChatSession
from comate_agent_sdk.llm import ChatOpenAI
from comate_agent_sdk.tools import tool

@tool("查询用户信息")
async def get_user_info(user_id: str) -> str:
    # 模拟数据库查询
    return f"用户 {user_id} 的信息：VIP 会员，积分 1200"

agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    options=ComateAgentOptions(tools=[get_user_info]),
)

async def chatbot(user_id: str, message: str):
    """聊天机器人入口，支持断点续聊"""
    # 尝试恢复会话
    try:
        session = ChatSession.resume(agent, session_id=f"user-{user_id}")
        print(f"[恢复会话] user-{user_id}")
    except Exception:
        session = ChatSession(agent, session_id=f"user-{user_id}")
        print(f"[新建会话] user-{user_id}")

    # 处理消息
    async for event in session.query_stream(message):
        if isinstance(event, TextEvent):
            print(f"[Bot]: {event.content}")
        elif isinstance(event, StopEvent):
            print(f"[完成]: {event.reason}")

# 模拟用户对话
await chatbot("alice", "帮我查询我的信息")
# === 程序重启 ===
await chatbot("alice", "我的积分是多少？")  # 能记住之前的对话
```

---

### 示例 2：批量任务处理（MessageSource）

```python
from comate_agent_sdk.agent import ChatSession

async def process_batch_tasks():
    """批量处理文件分析任务"""
    tasks = [
        "分析 report_q1.pdf 的关键数据",
        "总结 meeting_notes.txt 的要点",
        "翻译 contract_cn.docx 到英文",
    ]

    session = ChatSession(
        agent,
        session_id="batch-processor",
        message_source=tasks  # 传入任务列表
    )

    # 自动依次处理每个任务
    task_results = []
    current_result = []

    async for event in session.events():
        if isinstance(event, TextEvent):
            current_result.append(event.content)
        elif isinstance(event, StopEvent):
            task_results.append("".join(current_result))
            current_result = []

    # 输出所有任务结果
    for i, result in enumerate(task_results):
        print(f"任务 {i+1} 结果: {result}")

await process_batch_tasks()
```

---

### 示例 3：对话分支探索（Fork）

```python
from comate_agent_sdk.agent import ChatSession

async def explore_story_branches():
    """生成故事的不同结局"""
    # 主故事线
    main = ChatSession(agent, session_id="story-main")
    async for event in main.query_stream("写一个侦探故事的开头：一个雨夜，侦探收到神秘委托"):
        if isinstance(event, TextEvent):
            print(f"[开头]: {event.content}")

    # 分叉：结局 A - 凶手是管家
    fork_a = main.fork_session()
    async for event in fork_a.query_stream("揭示真相：凶手是管家"):
        if isinstance(event, TextEvent):
            print(f"[结局 A]: {event.content}")

    # 分叉：结局 B - 凶手是委托人
    fork_b = main.fork_session()
    async for event in fork_b.query_stream("揭示真相：委托人才是凶手"):
        if isinstance(event, TextEvent):
            print(f"[结局 B]: {event.content}")

    # 原会话继续：开放式结局
    async for event in main.query_stream("留下悬念，真相未明"):
        if isinstance(event, TextEvent):
            print(f"[开放结局]: {event.content}")

await explore_story_branches()
```

---

## 最佳实践

### 存储路径管理

```python
from pathlib import Path

# ✅ 推荐：为不同用户/场景使用独立的 session_id
session = ChatSession(agent, session_id=f"user-{user_id}-chat")

# ✅ 推荐：生产环境使用自定义存储路径
session = ChatSession(
    agent,
    session_id=session_id,
    storage_root=Path(f"/data/sessions/{session_id}")
)

# ❌ 避免：多个会话共享相同的 session_id（会导致数据混乱）
```

---

### 错误处理

```python
from comate_agent_sdk.agent import ChatSession, ChatSessionClosedError, ChatSessionError

try:
    session = ChatSession.resume(agent, session_id="unknown")
except Exception as e:
    # 会话不存在，创建新会话
    session = ChatSession(agent, session_id="unknown")

try:
    async for event in session.query_stream("你好"):
        ...
except ChatSessionClosedError:
    print("会话已关闭，无法继续")
except ChatSessionError as e:
    print(f"会话错误: {e}")
```

---

### 资源清理

```python
# ✅ 推荐：使用 async with 自动清理
async with ChatSession(agent, session_id="temp") as session:
    async for event in session.query_stream("临时查询"):
        ...

# ✅ 推荐：手动关闭长期运行的会话
session = ChatSession(agent, session_id="long-running")
try:
    async for event in session.events():
        ...
finally:
    await session.close()
```

---

### 会话 ID 命名规范

```python
# ✅ 推荐：语义化命名
session_id = f"user-{user_id}-chat"           # 用户聊天
session_id = f"task-{task_id}-{timestamp}"    # 任务处理
session_id = f"debug-{issue_id}"              # 调试会话

# ❌ 避免：使用随机 UUID（难以管理和查找）
session_id = str(uuid.uuid4())  # 不推荐
```

---

## API 参考

### ChatSession 构造函数

```python
ChatSession(
    agent: Agent,
    *,
    session_id: str | None = None,           # 会话 ID（默认自动生成 UUID）
    storage_root: Path | None = None,        # 存储路径（默认 ~/.agent/sessions/{session_id}）
    message_source: MessageSource | None = None  # 消息迭代器（可选）
)
```

---

### 核心方法

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `query_stream(message)` | 发送单条消息并流式返回事件 | `AsyncIterator[AgentEvent]` |
| `events()` | 消费 `message_source` 的所有消息 | `AsyncIterator[AgentEvent]` |
| `send(message)` | 手动发送消息到内部队列 | `None` |
| `close()` | 关闭会话 | `None` |
| `clear_history()` | 清空对话历史 | `None` |
| `get_usage()` | 获取 token 使用统计 | `UsageSummary` |
| `get_context_info()` | 获取上下文使用情况 | `ContextInfo` |

---

### 类方法

| 方法 | 说明 |
|------|------|
| `ChatSession.resume(agent, *, session_id, storage_root=None)` | 从磁盘恢复已保存的会话 |

---

### 实例方法

| 方法 | 说明 |
|------|------|
| `fork_session(*, storage_root=None, message_source=None)` | 从当前会话创建副本 |

---

## 常见问题

**Q1：`query_stream()` 和 `events()` 有什么区别？**

- `query_stream(message)`：发送**单条**消息，返回事件流
- `events()`：消费 `message_source` 的**所有**消息，依次调用 `query_stream()`

```python
# 单条消息
async for event in session.query_stream("你好"):
    ...

# 批量消息
session = ChatSession(agent, message_source=["消息1", "消息2"])
async for event in session.events():  # 自动处理所有消息
    ...
```

---

**Q2：会话数据存储在哪里？**

默认路径：`~/.agent/sessions/{session_id}/`

可通过 `storage_root` 参数自定义。

---

**Q3：如何删除历史会话？**

```python
import shutil
from pathlib import Path

session_id = "old-session"
storage_root = Path.home() / ".agent" / "sessions" / session_id
shutil.rmtree(storage_root)  # 删除整个会话目录
```

---

**Q4：`fork_session()` 后原会话和分叉会话是否互相影响？**

不影响。`fork_session()` 会完整复制存储目录，两个会话完全独立。

---

**Q5：能否在 `query_stream()` 执行过程中调用 `clear_history()`？**

不推荐。可能导致上下文状态不一致，应该在对话完成后（收到 `StopEvent`）再调用。

---

## 相关文档

- [Agent 快速开始](../comate_agent_sdk/agent/README.md)
- [可观测性指南](./observability-guide.md)
- [上下文文件系统](./context_filesystem.md)
- [上下文压缩流程](./compaction-flow.md)
