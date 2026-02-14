# Comate Agent SDK 开发者接入文档

## 概述

Comate Agent SDK 是一个用于构建基于 LLM 的智能体应用的 Python 框架。它提供了完整的工具调用、会话管理、上下文压缩、子代理和可观测性支持。

## 核心概念

### Agent 三层架构

SDK 采用三层架构设计，实现配置与状态的分离：

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentTemplate                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Config     │  │    Tools     │  │  LLM Levels      │  │
│  │  (frozen)    │  │  (tuple)     │  │  (dict)          │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                      不可变模板层                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ create_runtime()
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    AgentRuntime                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  ContextIR   │  │  Tool Map    │  │  Token Cost      │  │
│  │  (mutable)   │  │  (dict)      │  │  (tracking)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                      可变运行时层                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ chat()
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    ChatSession                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Persistence  │  │   History    │  │  Fork/Resume     │  │
│  │  (jsonl)     │  │  (messages)  │  │  (state mgmt)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                      会话管理层                              │
└─────────────────────────────────────────────────────────────┘
```

#### 1. AgentTemplate - 不可变模板层

- **职责**：定义 Agent 的静态配置（工具、系统提示词、LLM 档位等）
- **特性**：使用 `frozen=True` 的 dataclass，配置不可变，线程安全
- **缓存**：解析后的配置（resolved_agents、resolved_skills 等）会被缓存

```python
from comate_agent_sdk import Agent, AgentConfig

# 创建 Agent 模板
agent = Agent(config=AgentConfig(
    system_prompt="你是一个 helpful assistant",
    max_iterations=100,
))

# 模板可以重复使用创建多个运行时
runtime1 = agent.create_runtime()
runtime2 = agent.create_runtime(session_id="session-2")
```

#### 2. AgentRuntime - 可变运行时层

- **职责**：处理单次查询的执行，维护运行时的可变状态
- **核心组件**：
  - `ContextIR`：上下文中间表示，结构化管理对话历史
  - `TokenCost`：Token 使用和成本追踪
  - `CompactionService`：上下文压缩服务
  - `ToolMap`：工具名称到工具实例的映射

```python
# 创建运行时（单次查询）
runtime = agent.create_runtime()
result = await runtime.query("Hello!")

# 运行时包含完整的执行状态
print(runtime.max_iterations)  # 访问配置
print(runtime.tools)           # 访问工具列表
```

#### 3. ChatSession - 会话管理层

- **职责**：维护跨多次交互的状态，支持持久化、恢复、Fork
- **存储位置**：`~/.agent/sessions/{session_id}/`
- **持久化格式**：JSONL 文件，包含完整的对话历史和元数据

```python
# 创建新会话
session = agent.chat()

# 恢复已有会话
session = agent.chat(session_id="previous-session-id")

# Fork 会话（创建分支）
new_session = await session.fork()
```

### ContextIR - 上下文中间表示

ContextIR 是 SDK 的核心创新，将传统的扁平消息列表提升为结构化的中间表示：

```
┌─────────────────────────────────────────────────────────────┐
│                      ContextIR                              │
├─────────────────────────────────────────────────────────────┤
│  Header Segment                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │SystemPrompt │ │ AgentLoop   │ │ Tool/Subagent/Skill │   │
│  │  (priority  │ │  (priority  │ │    Strategies       │   │
│  │    100)     │ │    100)     │ │   (priority 90-95)  │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Conversation Segment                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │UserMessage  │ │AssistantMsg │ │    ToolResult       │   │
│  │ (priority   │ │  (priority  │ │   (priority 10)     │   │
│  │    50)      │ │    30)      │ │  最先被压缩         │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Memory (独立字段)                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  从文件加载的静态背景知识，作为 meta message 注入    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### ItemType - 语义类型系统

```python
from comate_agent_sdk.context import ItemType

# Header 类型（永不压缩，priority=100）
ItemType.SYSTEM_PROMPT      # 系统提示词
ItemType.AGENT_LOOP         # Agent 循环控制指令
ItemType.MEMORY             # Memory 静态背景知识
ItemType.TOOL_STRATEGY      # 工具策略
ItemType.SUBAGENT_STRATEGY  # 子代理策略
ItemType.SKILL_STRATEGY     # Skill 策略
ItemType.SYSTEM_ENV         # 系统环境信息
ItemType.GIT_ENV            # Git 环境信息

# Conversation 类型（按优先级压缩）
ItemType.USER_MESSAGE       # 用户消息 (priority=50)
ItemType.ASSISTANT_MESSAGE  # 助手消息 (priority=30)
ItemType.TOOL_RESULT        # 工具结果 (priority=10, 最先压缩)

# 特殊类型
ItemType.COMPACTION_SUMMARY # 压缩摘要（永不压缩）
ItemType.OFFLOAD_PLACEHOLDER # 卸载占位符
```

#### 压缩优先级

数值越低越先被压缩，数值越高越重要：

| 优先级 | 类型 | 策略 |
|--------|------|------|
| 10 | TOOL_RESULT | TRUNCATE (保留最近5个) |
| 20 | SKILL_PROMPT | DROP |
| 30 | ASSISTANT_MESSAGE | TRUNCATE (保留最近5个) |
| 50 | USER_MESSAGE | TRUNCATE (保留最近5个) |
| 80 | COMPACTION_SUMMARY | NONE (永不压缩) |
| 90-95 | STRATEGY 类型 | NONE (永不压缩) |
| 100 | SYSTEM/MEMORY/ENV | NONE (永不压缩) |"

## 快速开始

### 1. 基础 Agent 创建

```python
import asyncio
import os
from comate_agent_sdk import Agent, AgentConfig
from comate_agent_sdk.llm import ChatOpenAI

async def main():
    # 创建 LLM 实例
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # 创建 Agent
    agent = Agent(
        llm=llm,
        config=AgentConfig(
            system_prompt="你是一个 helpful assistant",
            max_iterations=100,
        ),
    )
    
    # 简单查询
    result = await agent.query("Hello, world!")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 流式响应

```python
from comate_agent_sdk.agent.events import TextEvent, ToolCallEvent, ToolResultEvent, StopEvent

async def stream_example():
    agent = Agent(config=AgentConfig())
    
    async for event in agent.query_stream("Tell me a joke"):
        match event:
            case TextEvent(content=text):
                print(text, end="", flush=True)
            case ToolCallEvent(tool=name, args=args):
                print(f"\n[Calling tool: {name}]")
            case ToolResultEvent(tool=name, result=result):
                print(f"\n[Tool result: {result[:100]}...]")
            case StopEvent(reason=reason):
                print(f"\n[Stopped: {reason}]")
```

## 工具执行机制

### 执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                   Tool Execution Flow                       │
├─────────────────────────────────────────────────────────────┤
│  1. LLM 返回 tool_calls                                     │
│                    ↓                                        │
│  2. 参数类型修复 (_coerce_tool_arguments)                   │
│     - string → integer/number/boolean/array/object          │
│                    ↓                                        │
│  3. 依赖注入解析 (Depends)                                  │
│                    ↓                                        │
│  4. 执行工具函数                                            │
│                    ↓                                        │
│  5. 结果格式化 (OutputFormatter)                            │
│     - 支持图片、文本、结构化数据                            │
│                    ↓                                        │
│  6. 生成 ToolMessage 返回给 LLM                             │
└─────────────────────────────────────────────────────────────┘
```

### 参数类型修复

SDK 自动修复 LLM 返回的字符串化参数：

```python
# LLM 可能返回字符串形式的参数
tool_args = {
    "count": "5",        # string → int
    "price": "19.99",    # string → float
    "enabled": "true",   # string → bool
    "tags": '["a","b"]', # string → list
    "config": '{"k":"v"}' # string → dict
}

# SDK 自动根据 schema 转换
# count: "5" → 5 (int)
# price: "19.99" → 19.99 (float)
# enabled: "true" → True (bool)
# tags: '["a","b"]' → ["a", "b"] (list)
# config: '{"k":"v"}' → {"k": "v"} (dict)
```

### 错误检测

工具执行错误自动检测：

```python
# 错误结果会被标记
{
    "ok": False,
    "error": "File not found",
    "data": None
}

# ContextItem 会标记 is_tool_error=True
# 压缩时会特殊处理错误项（保留更长时间）
```

## 工具系统

### 创建工具

使用 `@tool` 装饰器创建类型安全的工具：

```python
from comate_agent_sdk.tools import tool, Depends
from pydantic import BaseModel

# 简单工具
@tool("Add two numbers")
async def add(a: int, b: int) -> int:
    return a + b

# 带 Pydantic 模型的工具
class SearchParams(BaseModel):
    query: str
    max_results: int = 10

@tool("Search with complex params")
async def search(params: SearchParams) -> str:
    return f"Searching for {params.query}..."

# 依赖注入
async def get_database():
    return DatabaseConnection()

@tool("Query database")
async def query_db(sql: str, db: Depends(get_database)) -> str:
    return await db.execute(sql)
```

### 注册工具到 Agent

```python
from comate_agent_sdk import Agent, AgentConfig

agent = Agent(
    config=AgentConfig(
        tools=[add, search, query_db],
    ),
)
```

### 工具注册表

```python
from comate_agent_sdk.tools import get_default_registry

# 获取 SDK 内置工具注册表
registry = get_default_registry()

# 使用内置工具创建 Agent
agent = Agent(
    config=AgentConfig(
        tools=registry.all(),
    ),
)
```

## 会话管理

### 创建和管理会话

```python
from comate_agent_sdk import Agent
from comate_agent_sdk.agent import ChatSession

agent = Agent(config=AgentConfig())

# 创建新会话
session = agent.chat()

# 发送消息
async for event in session.send("Hello"):
    if isinstance(event, TextEvent):
        print(event.content)

# 获取会话信息
session_id = session.session_id
usage = await session.get_usage()
context_info = await session.get_context_info()
```

### 会话持久化

```python
# 会话自动持久化到 ~/.agent/sessions/{session_id}/
# 包含完整的对话历史和状态

# 恢复会话
from comate_agent_sdk.agent.chat_session import ChatSession

restored_session = await ChatSession.resume(
    session_id="previous-session-id",
    template=agent,
)

# Fork 会话
forked_session = await session.fork()
```

### 会话事件

```python
from comate_agent_sdk.agent.events import (
    SessionInitEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    StopEvent,
    UsageDeltaEvent,
)

async for event in session.send("Query"):
    match event:
        case SessionInitEvent(session_id=sid):
            print(f"Session started: {sid}")
        case TextEvent(content=text):
            print(text, end="")
        case ThinkingEvent(content=text):
            print(f"[Thinking: {text[:100]}...]")
        case ToolCallEvent(tool=name, args=args):
            print(f"[Tool: {name}({args})]")
        case ToolResultEvent(tool=name, result=result, is_error=err):
            status = "error" if err else "ok"
            print(f"[Result ({status}): {result[:100]}...]")
        case UsageDeltaEvent(usage=usage):
            print(f"[Tokens: {usage.total_tokens}]")
        case StopEvent(reason=reason):
            print(f"[Done: {reason}]")
```

## LLM 支持

### 支持的提供商

SDK 支持多种 LLM 提供商：

```python
from comate_agent_sdk.llm import (
    ChatOpenAI,      # OpenAI
    ChatAnthropic,   # Anthropic Claude
    ChatGoogle,      # Google Gemini
    ChatAzureOpenAI, # Azure OpenAI
    ChatDeepSeek,    # DeepSeek
    ChatGroq,        # Groq
    ChatOllama,      # Ollama (本地)
    # ... 更多
)

# OpenAI
llm = ChatOpenAI(model="gpt-4o", api_key="...")

# Anthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key="...")

# Google
llm = ChatGoogle(model="gemini-2.0-flash", api_key="...")
```

### 预配置模型实例

SDK 提供预配置的模型实例：

```python
from comate_agent_sdk.llm import (
    openai_gpt_4o,
    openai_gpt_4o_mini,
    google_gemini_2_0_flash,
    # ...
)

agent = Agent(llm=openai_gpt_4o)
```

### LLM 档位系统

支持根据任务复杂度切换不同档位：

```python
from comate_agent_sdk.agent.llm_levels import LLMLevel

agent = Agent(
    llm=openai_gpt_4o_mini,  # 默认低档
    config=AgentConfig(
        llm_levels={
            LLMLevel.LOW: openai_gpt_4o_mini,
            LLMLevel.MID: openai_gpt_4o,
            LLMLevel.HIGH: openai_gpt_4o,  # 或更强的模型
        },
    ),
)
```

## 上下文管理 (ContextIR)

ContextIR 是 SDK 的核心上下文管理系统，提供结构化的对话历史管理。

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      ContextIR                              │
├─────────────────────────────────────────────────────────────┤
│  Header Segment (System-level)                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │SystemPrompt │ │ AgentLoop   │ │ Tool/Subagent/Skill │   │
│  │  priority   │ │  priority   │ │    Strategies       │   │
│  │    100      │ │    100      │ │   priority 90-95    │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Conversation Segment (User-Assistant-Tools)                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │UserMessage  │ │AssistantMsg │ │    ToolResult       │   │
│  │  priority   │ │  priority   │ │   priority 10       │   │
│  │    50       │ │    30       │ │  最先被压缩         │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Memory (独立字段)                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  静态背景知识，lowering 时作为 meta message 注入     │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Supporting Components                                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │BudgetConfig │ │TokenCounter │ │   SystemReminder    │   │
│  │ EventBus    │ │   FS        │ │   Compaction        │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

#### ContextItem - 上下文条目

```python
from comate_agent_sdk.context import ContextItem, ItemType

item = ContextItem(
    item_type=ItemType.USER_MESSAGE,
    message=UserMessage(content="Hello"),
    content_text="Hello",
    token_count=2,
    priority=50,
    ephemeral=False,      # 是否为临时内容
    destroyed=False,      # 是否已被销毁
    cache_hint=False,     # 是否建议缓存
    created_turn=0,       # 创建时的轮次号
)
```

#### Segment - 段落管理

```python
from comate_agent_sdk.context import Segment, SegmentName

# Header 段 - 系统级配置
header = Segment(name=SegmentName.HEADER)

# Conversation 段 - 对话历史
conversation = Segment(name=SegmentName.CONVERSATION)

# 查找指定类型的条目
items = conversation.find_by_type(ItemType.TOOL_RESULT)
item = conversation.find_one_by_type(ItemType.USER_MESSAGE)

# 移除条目
removed = conversation.remove_by_type(ItemType.SKILL_PROMPT)
item = conversation.remove_by_id("item-id")

# 获取段落的总 token 数
total_tokens = conversation.total_tokens
```

### 上下文压缩 (Compaction)

SDK 提供智能的上下文压缩机制，按优先级逐步压缩：

```python
from comate_agent_sdk.agent.compaction import CompactionConfig
from comate_agent_sdk.context import CompactionStrategy, TypeCompactionRule

agent = Agent(
    config=AgentConfig(
        compaction=CompactionConfig(
            enabled=True,
            threshold_ratio=0.80,  # 80% 时触发压缩
            model="gpt-4o-mini",   # 用于生成摘要的模型
        ),
    ),
)
```

#### 压缩策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `NONE` | 不压缩 | System Prompt、Memory 等重要内容 |
| `DROP` | 直接丢弃 | Skill Prompt、Metadata 等可丢弃内容 |
| `TRUNCATE` | 保留最近 N 个 | Tool Result、Assistant Message |
| `SUMMARIZE` | LLM 摘要 | 长对话历史（暂未实现） |

#### 默认压缩规则

```python
from comate_agent_sdk.context.compaction import DEFAULT_COMPACTION_RULES

# 默认规则：
{
    ItemType.TOOL_RESULT:    TypeCompactionRule(TRUNCATE, keep_recent=5),
    ItemType.ASSISTANT_MESSAGE: TypeCompactionRule(TRUNCATE, keep_recent=5),
    ItemType.USER_MESSAGE:   TypeCompactionRule(TRUNCATE, keep_recent=5),
    ItemType.SKILL_PROMPT:   TypeCompactionRule(DROP),
    ItemType.SYSTEM_PROMPT:  TypeCompactionRule(NONE),  # 永不压缩
    ItemType.MEMORY:         TypeCompactionRule(NONE),  # 永不压缩
}
```

#### 压缩元事件

启用压缩元事件以获取详细的压缩信息：

```python
from comate_agent_sdk.agent.events import CompactionMetaEvent, PreCompactEvent

agent = Agent(
    config=AgentConfig(
        emit_compaction_meta_events=True,  # 启用压缩元事件
    ),
)

async for event in session.send("Query"):
    match event:
        case PreCompactEvent(current_tokens, threshold):
            print(f"即将压缩: {current_tokens} tokens >= {threshold}")
        case CompactionMetaEvent(
            phase=phase,
            tokens_before=before,
            tokens_after=after,
            tool_blocks_dropped=dropped,
        ):
            print(f"压缩阶段 {phase}: {before} -> {after} (-{dropped} blocks)")
```

### 上下文卸载 (Offload)

大内容自动卸载到文件系统，避免上下文窗口溢出：

```python
from comate_agent_sdk.context import OffloadPolicy

agent = Agent(
    config=AgentConfig(
        offload_enabled=True,
        offload_token_threshold=2000,  # 超过此阈值卸载
        offload_root_path="~/.agent/offload",  # 卸载目录
        offload_policy=OffloadPolicy(
            max_size=100_000,  # 最大卸载大小
            compress=True,     # 是否压缩
        ),
    ),
)
```

#### 卸载机制

```
┌─────────────────────────────────────────────────────────────┐
│                    Offload Flow                             │
├─────────────────────────────────────────────────────────────┤
│  1. 工具返回大内容 (> threshold)                            │
│                    ↓                                        │
│  2. 内容写入文件系统 ~/.agent/offload/{session_id}/         │
│                    ↓                                        │
│  3. ContextItem 标记为 offloaded=True                       │
│  4. 存储 offload_path（相对路径）                           │
│                    ↓                                        │
│  5. Lowering 时生成占位符消息                               │
│     "[Content offloaded to {path}, {size} bytes]"           │
└─────────────────────────────────────────────────────────────┘
```

### Memory 配置

注入静态背景知识：

```python
from comate_agent_sdk.context import MemoryConfig

agent = Agent(
    config=AgentConfig(
        memory=MemoryConfig(
            files=["docs/context.md", "docs/rules.txt"],
            max_tokens=2000,
            cache=True,  # 启用 prompt caching
        ),
    ),
)
```

## 子代理 (Subagent)

### 定义子代理

```python
from comate_agent_sdk.subagent import AgentDefinition

code_reviewer = AgentDefinition(
    name="code_reviewer",
    description="Review code for bugs and style issues",
    system_prompt="You are a code reviewer. Focus on...",
    tools=[some_tool],
)

doc_writer = AgentDefinition(
    name="doc_writer", 
    description="Write documentation",
    system_prompt="You are a technical writer...",
)

# 注册到主 Agent
agent = Agent(
    config=AgentConfig(
        agents=[code_reviewer, doc_writer],
    ),
)
```

### 子代理事件

```python
from comate_agent_sdk.agent.events import (
    SubagentStartEvent,
    SubagentStopEvent,
    SubagentProgressEvent,
)

async for event in session.send("Review this code"):
    match event:
        case SubagentStartEvent(agent_name=name, task=task):
            print(f"[Subagent {name} started: {task}]")
        case SubagentProgressEvent(agent_name=name, progress=pct):
            print(f"[Subagent {name} progress: {pct}%]")
        case SubagentStopEvent(agent_name=name, result=result):
            print(f"[Subagent {name} completed]")
```

## Skill 系统

### 发现和加载 Skills

Skills 是兼容 Claude Code SKILL.md 格式的可复用能力单元：

```python
from comate_agent_sdk.skill import discover_skills, SkillDefinition

# 自动发现目录中的 skills
skills = discover_skills("./skills")

# 或手动定义
skill = SkillDefinition(
    name="web_search",
    description="Search the web",
    prompt="When searching, always...",
)

# 使用 skills
agent = Agent(
    config=AgentConfig(
        skills=tuple(skills),
    ),
)
```

## Hook 系统

Hook 系统允许你在 Agent 执行过程中插入自定义逻辑，实现诸如权限控制、日志记录、内容增强等功能。

### 概述

Hook 系统支持两种类型的 handler：

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| `command` | 执行外部命令 | 跨语言集成、权限控制 |
| `python` | 执行 Python 回调函数 | 快速原型、性能敏感 |

### 配置来源

Hook 配置通过以下顺序加载（后者优先级更高）：

1. **user** - `~/.agent/settings.json`
2. **project** - `{project}/.agent/settings.json`
3. **local** - `{project}/.agent/local-settings.json`

### 事件类型

| 事件名 | 触发时机 | 支持 matcher |
|--------|----------|---------------|
| `SessionStart` | 会话开始时 | ❌ |
| `UserPromptSubmit` | 用户提交 prompt 时 | ❌ |
| `PreToolUse` | 工具执行前 | ✅ |
| `PostToolUse` | 工具执行成功后 | ✅ |
| `PostToolUseFailure` | 工具执行失败后 | ✅ |
| `Stop` | Agent 停止前 | ❌ |
| `SessionEnd` | 会话结束时 | ❌ |
| `SubagentStart` | 子代理启动时 | ❌ |
| `SubagentStop` | 子代理停止时 | ❌ |

### 配置格式

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read|Grep|Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python .agent/hooks/validate_tool.py",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

### Matcher 匹配规则

`matcher` 支持正则表达式，用于过滤特定工具：

| Matcher 示例 | 匹配的工具 |
|-------------|-----------|
| `"*"` | 所有工具 |
| `"^Read$"` | 仅 Read |
| `"Read\|Grep\|Bash"` | Read、Grep 或 Bash |
| `"^Bash.*"` | Bash 开头的工具 |
| `".*File.*"` | 包含 File 的工具 |

### Command Hook

#### 输入

Command hook 通过 **stdin** 接收 JSON 格式的输入：

```json
{
  "session_id": "xxx",
  "cwd": "/path/to/project",
  "permission_mode": "default",
  "hook_event_name": "PreToolUse",
  "tool_name": "Read",
  "tool_input": {"file_path": "main.py"},
  "tool_call_id": "call_xxx"
}
```

#### 输出

Command hook 通过 **stdout** 返回 JSON 结果：

```json
{
  "additionalContext": "返回给 LLM 的额外上下文",
  "permissionDecision": "allow",
  "updatedInput": {"file_path": "main.py"},
  "reason": "拒绝/询问的原因"
}
```

#### 阻止执行

**方式一：Exit Code 2 + stderr**

```python
#!/usr/bin/env python3
import sys

print("禁止执行此操作", file=sys.stderr)
sys.exit(2)
```

**方式二：JSON 返回 deny**

```json
{
  "permissionDecision": "deny",
  "reason": "禁止访问敏感文件"
}
```

**方式三：JSON 返回 ask + 用户拒绝**

```json
{
  "permissionDecision": "ask",
  "reason": "需要用户确认"
}
```

#### 完整示例：阻止访问 .env 文件

```python
#!/usr/bin/env python3
"""阻止访问 .env 文件的 Hook"""
import sys
import json

# 读取 hook 输入
hook_input = json.load(sys.stdin)
tool_name = hook_input.get("tool_name", "")
tool_input = hook_input.get("tool_input", {})

# 检查是否访问 .env
file_path = str(tool_input.get("file_path", ""))
if ".env" in file_path or ".env" in file_path:
    print(json.dumps({
        "additionalContext": "⚠️ 注意：此操作试图读取 .env 文件，可能包含敏感信息",
        "permissionDecision": "allow",
        "reason": "检测到 .env 文件访问"
    }))
else:
    print(json.dumps({
        "permissionDecision": "allow"
    }))
```

### Python Hook

#### 注册方式

```python
from comate_agent_sdk import Agent, AgentConfig
from comate_agent_sdk.agent.hooks.models import HookInput, HookResult

def my_pre_tool_hook(hook_input: HookInput) -> HookResult:
    print(f"Tool: {hook_input.tool_name}")
    return HookResult(permission_decision="allow")

agent = Agent(config=AgentConfig(...))
agent.register_python_hook(
    event_name="PreToolUse",
    callback=my_pre_tool_hook,
    matcher="Bash",  # 可选，默认 "*"
    order=0,          # 可选，执行顺序
    name="my_hook",   # 可选，hook 名称
)
```

#### 回调函数签名

```python
from comate_agent_sdk.agent.hooks.models import HookInput, HookResult

def hook_callback(hook_input: HookInput) -> HookResult | dict | None:
    # hook_input 包含事件相关数据
    # 返回值可以是：
    # - HookResult 对象
    # - dict (会自动转换为 HookResult)
    # - None (忽略此 hook)
    return HookResult(
        additional_context="额外上下文",
        permission_decision="allow",  # allow/ask/deny
        updated_input={"key": "value"},  # 修改后的参数
        reason="原因",
    )

# 支持异步
async def async_hook_callback(hook_input: HookInput) -> HookResult:
    # 异步逻辑
    return HookResult(permission_decision="allow")
```

### HookResult 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `additional_context` | `str \| None` | 返回给 LLM 的额外上下文 |
| `permission_decision` | `Literal["allow", "ask", "deny"]` | 权限决策 |
| `updated_input` | `dict \| None` | 修改后的工具参数 |
| `reason` | `str \| None` | 拒绝/询问的原因 |

### 权限模式

通过 `permission_mode` 配置不同的权限级别：

```python
from comate_agent_sdk import AgentConfig

agent = Agent(
    config=AgentConfig(
        permission_mode="default",  # 默认模式
    )
)
```

Hook 可以通过 `hook_input.permission_mode` 获取当前权限模式。

### 工具审批回调

当 Hook 返回 `permissionDecision: "ask"` 时，可以使用 `tool_approval_callback` 配置审批逻辑：

```python
from comate_agent_sdk import AgentConfig

async def approve_tool(session_id: str, tool_name: str, tool_input: dict, **kwargs):
    # 自定义审批逻辑
    if tool_name == "Bash" and "rm" in str(tool_input):
        return False
    return True

agent = Agent(
    config=AgentConfig(
        tool_approval_callback=approve_tool,
    )
)
```

### 完整配置示例

**~/.agent/settings.json (用户级)**

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/log_tool.py"
          }
        ]
      }
    ]
  }
}
```

**项目级 .claude/settings.json**

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read|Grep",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/check_file_access.py"
          }
        ]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/check_command.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/log_result.py"
          }
        ]
      }
    ]
  }
}
```

## MCP 集成

### 创建 MCP 服务器

```python
from comate_agent_sdk.mcp import create_sdk_mcp_server, mcp_tool

@mcp_tool("Search web")
async def web_search(query: str) -> str:
    return f"Results for {query}"

# 创建 MCP 服务器
server = create_sdk_mcp_server(
    name="my-mcp-server",
    tools=[web_search],
)
```

### 使用 MCP 工具

```python
from comate_agent_sdk.mcp import McpStdioServerConfig

agent = Agent(
    config=AgentConfig(
        mcp_enabled=True,
        mcp_servers=[
            McpStdioServerConfig(
                command="python",
                args=["mcp_server.py"],
            ),
        ],
    ),
)
```

## 可观测性

### 基础追踪

```python
from comate_agent_sdk.observability import observe, Laminar

@observe(name="my_function")
async def my_function():
    # 自动追踪
    result = await do_something()
    return result

# 手动 span
if Laminar is not None:
    with Laminar.start_as_current_span(
        name="action",
        input={"key": "value"},
        span_type="TOOL",
    ):
        result = await do_something()
        Laminar.set_span_output(result)
```

### 调试模式

```python
from comate_agent_sdk.observability import observe_debug

@observe_debug(name="debug_only")  # 仅在调试模式追踪
async def debug_function():
    pass
```

启用调试模式：
```bash
export LMNR_LOGGING_LEVEL=debug
# 或
export BU_DEBUG=1
```

## 配置参考

### AgentConfig 完整选项

```python
from comate_agent_sdk import AgentConfig

config = AgentConfig(
    # 核心配置
    tools=None,                    # 工具列表
    system_prompt=None,            # 系统提示词
    max_iterations=200,            # 最大迭代次数
    tool_choice="auto",            # 工具选择策略
    
    # 压缩配置
    compaction=None,               # CompactionConfig
    
    # 上下文配置
    offload_enabled=True,
    offload_token_threshold=2000,
    offload_root_path=None,
    offload_policy=None,
    
    # 成本追踪
    include_cost=False,
    
    # LLM 重试
    llm_max_retries=5,
    llm_retry_base_delay=1.0,
    llm_retry_max_delay=60.0,
    
    # 子代理
    agents=None,                   # AgentDefinition 列表
    task_parallel_enabled=True,
    task_parallel_max_concurrency=4,
    
    # Skills
    skills=None,                   # SkillDefinition 列表
    
    # Memory
    memory=None,                   # MemoryConfig
    
    # MCP
    mcp_enabled=True,
    mcp_servers=None,
    
    # LLM 档位
    llm_levels=None,               # Dict[LLMLevel, BaseChatModel]
    
    # 会话
    session_id=None,
    role=None,
)
```

## 完整示例

### REPL 聊天机器人

```python
"""Simple chat session REPL example."""
import asyncio
import os
import readline
from pathlib import Path

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.agent.events import (
    SessionInitEvent,
    StopEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from comate_agent_sdk.llm import ChatOpenAI


async def main():
    # 设置日志
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    
    # 创建 Agent
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = Agent(
        llm=llm,
        config=AgentConfig(
            system_prompt="你是一个 helpful assistant",
            include_cost=True,
        ),
    )
    
    # 创建会话
    session = agent.chat()
    print("=== Chat Session ===")
    print("Type /exit to quit, /help for commands")
    
    while True:
        try:
            line = await asyncio.to_thread(input, "> ")
        except EOFError:
            break
            
        line = line.strip()
        if not line:
            continue
            
        if line == "/exit":
            break
        elif line == "/help":
            print("Commands: /exit, /help, /session, /usage")
        elif line == "/session":
            print(f"Session ID: {session.session_id}")
        elif line == "/usage":
            usage = await session.get_usage()
            print(f"Total tokens: {usage.total_tokens}")
            print(f"Total cost: ${usage.total_cost:.4f}")
        else:
            # 发送消息
            async for event in session.send(line):
                match event:
                    case SessionInitEvent(session_id=sid):
                        print(f"[Session: {sid}]")
                    case TextEvent(content=text):
                        print(text, end="", flush=True)
                    case ToolCallEvent(tool=name):
                        print(f"\n[Using tool: {name}]")
                    case ToolResultEvent(result=result):
                        print(f"[Result: {result[:100]}...]")
                    case StopEvent():
                        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
```

## 最佳实践

### 1. 错误处理

```python
from comate_agent_sdk.agent.chat_session import ChatSessionClosedError

try:
    async for event in session.send("query"):
        # 处理事件
        pass
except ChatSessionClosedError:
    print("Session has been closed")
```

### 2. 资源清理

```python
# 会话使用完毕后关闭
await session.close()

# 或使用上下文管理器
async with agent.chat() as session:
    async for event in session.send("query"):
        pass
```

### 3. 并发控制

```python
# 限制子代理并发数
config = AgentConfig(
    task_parallel_enabled=True,
    task_parallel_max_concurrency=4,
)
```

### 4. 成本优化

```python
# 启用成本追踪
config = AgentConfig(include_cost=True)

# 使用低档模型处理简单任务
config = AgentConfig(
    llm=gpt_4o_mini,
    llm_levels={
        LLMLevel.LOW: gpt_4o_mini,
        LLMLevel.MID: gpt_4o,
        LLMLevel.HIGH: gpt_4o,
    },
)
```

### 5. 会话中断控制

```python
from comate_agent_sdk.agent.interrupt import SessionRunController
from comate_agent_sdk.agent.events import StopEvent

# 获取会话的运行控制器
controller = session.run_controller

# 中断会话（协作式）
controller.interrupt(reason="user_cancelled")

# 检查中断状态
if controller.is_interrupted:
    print(f"Session interrupted: {controller.reason}")

# 流式响应中处理中断
async for event in session.send("Long running task"):
    if isinstance(event, StopEvent) and event.reason == "interrupted":
        print("Task was interrupted")
```

### 6. Token 计数优化

```python
from comate_agent_sdk.context import TokenCounter

# 创建 TokenCounter（自动使用 tiktoken）
counter = TokenCounter()

# 估算文本 token 数
tokens = counter.count("Hello, world!")  # 使用 cl100k_base

# 按 provider/model 精确计数
tokens = counter.count_for_model(
    text="Hello",
    provider="openai",
    model="gpt-4o",
)

# 估算消息列表 token 数
messages = [UserMessage(content="Hello"), AssistantMessage(content="Hi")]
tokens = await counter.count_messages_for_model(
    messages=messages,
    llm=chat_openai_instance,
    timeout_ms=300,
)
```

### 7. 上下文预算控制

```python
from comate_agent_sdk.context import BudgetConfig

# 配置上下文预算
config = AgentConfig(
    budget=BudgetConfig(
        total_limit=100000,           # 全局 token 上限
        type_limits={
            "tool_result": 30000,     # Tool Result 类型上限
            "assistant_message": 20000,
        },
        compact_threshold_ratio=0.75,  # 75% 利用率时触发压缩
    ),
)
```

## API 参考

### 主要类

| 类名 | 描述 |
|------|------|
| `Agent` / `AgentTemplate` | Agent 模板，不可变配置 |
| `AgentRuntime` | 运行时实例，处理单次查询 |
| `ChatSession` | 会话管理，支持持久化 |
| `AgentConfig` | Agent 配置数据类 |
| `RuntimeAgentOptions` | 运行时配置选项 |
| `Tool` | 工具包装类 |
| `SessionRunController` | 会话运行控制器（中断支持） |

### Context 模块类

| 类名 | 描述 |
|------|------|
| `ContextIR` | 上下文中间表示 |
| `ContextItem` | 上下文条目 |
| `ItemType` | 条目语义类型枚举 |
| `Segment` | 上下文段落 |
| `SegmentName` | 段落名称枚举 |
| `BudgetConfig` | 上下文预算配置 |
| `TokenCounter` | Token 计数器 |
| `CompactionStrategy` | 压缩策略枚举 |
| `TypeCompactionRule` | 类型压缩规则 |
| `SelectiveCompactionPolicy` | 选择性压缩策略 |
| `OffloadPolicy` | 卸载策略 |
| `MemoryConfig` | Memory 配置 |
| `SystemReminder` | 系统提醒 |
| `ContextEventBus` | 上下文事件总线 |
| `ContextFileSystem` | 上下文文件系统 |

### Compaction 模块类

| 类名 | 描述 |
|------|------|
| `CompactionConfig` | 压缩配置 |
| `CompactionResult` | 压缩结果 |
| `CompactionService` | 压缩服务 |
| `CompactionMetaRecord` | 压缩元记录 |
| `PreCompactEvent` | 压缩前事件 |
| `CompactionMetaEvent` | 压缩元事件 |

### 主要事件

| 事件 | 描述 |
|------|------|
| `SessionInitEvent` | 会话初始化 |
| `TextEvent` | 文本输出 |
| `ThinkingEvent` | 思考/推理内容 |
| `ToolCallEvent` | 工具调用 |
| `ToolResultEvent` | 工具结果 |
| `StopEvent` | 停止信号 |
| `UsageDeltaEvent` | Token 使用更新 |
| `SubagentStartEvent` | 子代理启动 |
| `SubagentStopEvent` | 子代理停止 |

### 装饰器

| 装饰器 | 描述 |
|--------|------|
| `@tool` | 创建工具 |
| `@observe` | 追踪函数 |
| `@observe_debug` | 调试模式追踪 |

## 环境变量

| 变量 | 描述 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 |
| `GOOGLE_API_KEY` | Google API 密钥 |
| `LMNR_PROJECT_API_KEY` | Laminar 项目密钥 |
| `LMNR_LOGGING_LEVEL` | Laminar 日志级别 |
| `BU_DEBUG` | 调试模式开关 |
| `LOG_LEVEL` | 日志级别 |

## 架构流程图

### Agent 执行完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Agent Execution Flow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                           │
│  │ User Query   │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    AgentTemplate.create_runtime()                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
│  │  │ Resolve LLM │  │ Resolve     │  │ Resolve Tools/Agents/Skills │  │   │
│  │  │   Levels    │  │   Memory    │  │                             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         AgentRuntime                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                      ContextIR                                │   │   │
│  │  │  ┌─────────────┐  ┌─────────────────────────────────────┐  │   │   │
│  │  │  │   Header    │  │           Conversation              │  │   │   │
│  │  │  │  Segment    │  │  ┌─────────┐ ┌─────────┐ ┌────────┐ │  │   │   │
│  │  │  │             │  │  │  User   │ │Assistant│ │ Tool   │ │  │   │   │
│  │  │  │ SystemPrompt│  │  │ Message │ │ Message │ │ Result │ │  │   │   │
│  │  │  │ AgentLoop   │  │  └─────────┘ └─────────┘ └────────┘ │  │   │   │
│  │  │  │ Strategies  │  │                                     │  │   │   │
│  │  │  └─────────────┘  └─────────────────────────────────────┘  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                    LoweringPipeline                         │   │   │
│  │  │         ContextIR ──► list[BaseMessage]                     │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                      LLM.invoke()                           │   │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │   │
│  │  │  │   Retry     │  │   Token     │  │   Cost Tracking     │  │   │   │
│  │  │  │   Logic     │  │   Counter   │  │                     │  │   │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                    Response Handling                        │   │   │
│  │  │                    ┌──────────────┐                         │   │   │
│  │  │              ┌─────┤  Text Content├─────┐                  │   │   │
│  │  │              │     └──────────────┘     │                  │   │   │
│  │  │              │                          │                  │   │   │
│  │  │              ▼                          ▼                  │   │   │
│  │  │     ┌──────────────┐           ┌────────────────┐          │   │   │
│  │  │     │  Yield Text  │           │  Tool Calls    │          │   │   │
│  │  │     │   Events     │           │                │          │   │   │
│  │  │     └──────────────┘           └───────┬────────┘          │   │   │
│  │  │                                        │                   │   │   │
│  │  │                                        ▼                   │   │   │
│  │  │     ┌──────────────────────────────────────────────────┐   │   │   │
│  │  │     │              Tool Execution                      │   │   │   │
│  │  │     │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │   │   │   │
│  │  │     │  │  Coerce  │  │  Inject  │  │    Execute       │  │   │   │   │
│  │  │     │  │   Args   │  │  Depends │  │    Tool          │  │   │   │   │
│  │  │     │  └──────────┘ └──────────┘ └──────────────────┘  │   │   │   │
│  │  │     └──────────────────────────────────────────────────┘   │   │   │
│  │  │                              │                              │   │   │
│  │  │                              ▼                              │   │   │
│  │  │     ┌──────────────────────────────────────────────────┐   │   │   │
│  │  │     │           Check Compaction Need                  │   │   │   │
│  │  │     │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │   │   │   │
│  │  │     │  │  Token   │  │  Budget  │  │   Selective      │  │   │   │   │
│  │  │     │  │  Usage   │  │  Check   │  │   Compaction     │  │   │   │   │
│  │  │     │  └──────────┘ └──────────┘ └──────────────────┘  │   │   │   │
│  │  │     └──────────────────────────────────────────────────┘   │   │   │
│  │  │                              │                              │   │   │
│  │  └──────────────────────────────┼──────────────────────────────┘   │   │
│  │                                 │                                  │   │
│  │                                 ▼                                  │   │
│  │                    ┌────────────────────────┐                      │   │
│  │                    │   Next Iteration or    │                      │   │
│  │                    │        Stop            │                      │   │
│  │                    └────────────────────────┘                      │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 会话持久化流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Session Persistence Flow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ChatSession                                                               │
│        │                                                                    │
│        ├──► Create ──► ~/.agent/sessions/{session_id}/                      │
│        │                    │                                               │
│        │                    ├──► messages.jsonl (对话历史)                  │
│        │                    ├──► context/ (卸载的大内容)                    │
│        │                    └──► state.json (会话状态)                      │
│        │                                                                    │
│        ├──► Send Message ──► Update ContextIR ──► Persist Incrementally    │
│        │                                                                    │
│        ├──► Fork ──► Copy to new session_id ──► Independent history        │
│        │                                                                    │
│        └──► Resume ──► Load from disk ──► Restore ContextIR                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*文档版本: 1.0*
*SDK 版本: 最新*
