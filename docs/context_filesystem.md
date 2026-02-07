# Context FileSystem 上下文卸载功能

## 概述

Context FileSystem 实现了将大型上下文内容卸载到文件系统的功能，避免占用过多 token，同时保持上下文结构完整。模型可以通过 Read 工具访问卸载的内容。

## 架构

### 核心组件

- **ContextFileSystem** (`comate_agent_sdk/context/fs.py`): 负责文件系统操作
- **OffloadPolicy** (`comate_agent_sdk/context/offload.py`): 配置卸载策略
- **ContextItem扩展** (`comate_agent_sdk/context/items.py`): 新增 `offloaded`, `offload_path` 字段
- **ToolMessage扩展** (`comate_agent_sdk/llm/messages.py`): 新增 `offloaded`, `offload_path` 字段

### 协作关系

```
Ephemeral 机制 ──────► Context FileSystem
    │                       │
    ├─ 决定何时销毁          ├─ 负责持久化存储
    ├─ 标记 destroyed       ├─ 生成索引
    └─ 触发卸载             └─ 提供占位符
```

## 使用方法

### 1. Agent 配置

```python
from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.llm import ChatAnthropic

agent = Agent(
    llm=ChatAnthropic(),
    config=AgentConfig(
        tools=[...],

        # Context FileSystem 配置
        offload_enabled=True,              # 是否启用卸载（默认 True）
        offload_token_threshold=2000,      # 超过此 token 数才卸载（默认 2000）
        offload_root_path=None,            # 存储路径，None 使用默认 ~/.agent/context/{session_id}

        # Ephemeral 配置（与 offload 协作）
        ephemeral_keep_recent=None,        # 全局覆盖工具的 ephemeral 值
    ),
)
```

### 2. Ephemeral 工具定义

```python
from comate_agent_sdk.tools import tool

@tool("读取大文件", ephemeral=3)  # 保留最近 3 个输出
async def read_large_file(path: str) -> str:
    return open(path).read()
```

### 3. 文件系统结构

```
~/.agent/context/{session_id}/
├── index.json              # 索引文件
├── tool_result/
│   ├── Read/
│   │   └── call_abc123.json
│   └── Bash/
│       └── call_def456.json
├── assistant_message/
│   └── msg_001.json
└── user_message/
    └── msg_002.json
```

### 4. 卸载文件格式

```json
{
  "schema_version": "1.0",
  "item_id": "abc123",
  "item_type": "tool_result",
  "offloaded_at": 1705300100,
  "metadata": {
    "tool_name": "Read"
  },
  "content": "... 原始内容 ..."
}
```

### 5. 占位符格式

当内容被卸载后，serializer 会生成以下占位符：

```
[Content offloaded]
Path: /home/user/.agent/context/session_20240115_abc/tool_result/Read/call_abc123.json
Use Read tool to view details.
```

## 工作流程

### Ephemeral 触发流程

```python
# 每次 LLM 调用前执行
agent._destroy_ephemeral_messages()
    ↓
1. 获取每个工具的 keep_recent 配置
    ↓
2. 遍历工具，找出需要销毁的 items
    ↓
3. 如果启用了 offload，调用 ContextFS.offload()
    ↓
4. 标记 item.destroyed = True
    ↓
5. 同步更新 message.offloaded / offload_path
```

### Compaction 触发流程

```python
# Token 超阈值时执行
SelectiveCompactionPolicy.compact()
    ↓
1. 按优先级逐类型压缩
    ↓
2. 对于 TRUNCATE 策略：
    ├─ 跳过已 destroyed 的（避免重复）
    ├─ 检查是否应该卸载 (_should_offload)
    ├─ 调用 ContextFS.offload()
    └─ 从列表移除
    ↓
3. 如果仍超阈值，回退全量摘要
```

## 压缩策略配置

默认卸载策略（`OffloadPolicy.type_enabled`）：

| Type | 是否卸载 | Token 阈值 |
|------|----------|-----------|
| tool_result | ✓ | 2000 |
| user_message | ✓ | 2000 |
| skill_prompt | ✓ | 2000 |
| assistant_message | ✗ | - |
| compaction_summary | ✗ | - |
| system_prompt | ✗ | - |

## 注意事项

### 1. 与 Ephemeral 的协作

- **Ephemeral 机制**：负责决定哪些内容需要销毁（基于 keep_recent 和工具配置）
- **Context FileSystem**：负责将被销毁的内容持久化到文件系统
- **两者分工明确**：Ephemeral 决策 + Context FS 存储

### 2. 已压缩的 Summary 不卸载

`COMPACTION_SUMMARY` 类型的内容已经经过 LLM 压缩，不应再被卸载。

### 3. 占位符与模型交互

- 占位符显示文件绝对路径
- 模型可使用 Read 工具访问卸载文件
- 占位符保持消息结构完整（满足 API 要求）

### 4. Session ID 生成

格式：`session_{timestamp}_{random}`
例如：`session_20240115_123456_abc123`

- timestamp: YYYYMMDD_HHMMSS
- random: 6位随机字符

## 示例：完整工作流

```python
from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.llm import ChatAnthropic
from comate_agent_sdk.tools import tool

@tool("读取文件", ephemeral=2)
async def read_file(path: str) -> str:
    return open(path).read()

agent = Agent(
    llm=ChatAnthropic(),
    config=AgentConfig(
        tools=[read_file],
        offload_enabled=True,
        offload_token_threshold=1000,
    ),
)

# 第1次调用：输出保留
await agent.query("Read file1.txt")

# 第2次调用：输出保留
await agent.query("Read file2.txt")

# 第3次调用：file1.txt 的输出被卸载
await agent.query("Read file3.txt")

# 第4次调用：file2.txt 的输出被卸载
await agent.query("Read file4.txt")

# 查看卸载情况
for item in agent._context.conversation.items:
    if item.offloaded:
        print(f"Offloaded: {item.offload_path}")
```

## 测试

运行测试验证功能：

```bash
# 基本功能测试
uv run python -c "
from comate_agent_sdk.context import ContextFileSystem, OffloadPolicy
print('✓ Import successful')
"

# 集成测试（需要 API key）
# 见 examples/ 目录
```

## 后续扩展（暂未实现）

- 自动清理过期文件
- 压缩卸载文件（gzip）
- 摘要生成（卸载时生成内容摘要嵌入占位符）
- 卸载历史记录和恢复
