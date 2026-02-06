# 上下文压缩机制详解

## 概述

本文档详细说明 Agent SDK 中上下文压缩的完整链路和数据结构。

## 目录

- [压缩触发条件](#压缩触发条件)
- [完整调用链路](#完整调用链路)
- [数据结构详解](#数据结构详解)
- [压缩流程](#压缩流程)
- [关键代码位置](#关键代码位置)

---

## 压缩触发条件

```python
# comate_agent_sdk/agent/service.py:679-720
async def _check_and_compact(self, response: ChatInvokeCompletion) -> bool:
    # 1. 更新 token 使用量
    self._compaction_service.update_usage(response.usage)

    # 2. 检查是否需要压缩
    threshold = await self._compaction_service.get_threshold_for_model(self.llm.model)
    actual_tokens = TokenUsage.from_usage(response.usage).total_tokens

    # 3. 触发条件
    if actual_tokens >= threshold:
        # 执行压缩
        await self._context.auto_compact(policy=policy, current_total_tokens=actual_tokens)
```

**触发条件**：`total_tokens >= threshold`（默认阈值为模型上下文窗口的 80%）

---

## 完整调用链路

```
Agent.query()
  │
  ├─→ 添加用户消息
  │   self._context.add_message(UserMessage(content=message))
  │
  ├─→ LLM 推理循环
  │   │
  │   ├─→ 销毁临时消息
  │   │   self._destroy_ephemeral_messages()
  │   │
  │   ├─→ 调用 LLM
  │   │   response = await self.llm.ainvoke(messages=self._context.lower())
  │   │
  │   ├─→ 添加 assistant 消息
  │   │   self._context.add_message(assistant_msg)
  │   │
  │   └─→ 检查并压缩 ⭐
  │       await self._check_and_compact(response)
  │       │
  │       └─→ ContextIR.auto_compact()
  │           │
  │           └─→ SelectiveCompactionPolicy.compact(context)
  │               │
  │               ├─→ [路径1] 选择性压缩
  │               │   │
  │               │   ├─→ DROP 策略
  │               │   │   context.conversation.remove_by_type(item_type)
  │               │   │
  │               │   └─→ TRUNCATE 策略
  │               │       context.conversation.remove_by_id(item.id)
  │               │
  │               └─→ [路径2] 全量摘要回退
  │                   │
  │                   └─→ _fallback_full_summary(context)
  │                       │
  │                       ├─→ 提取 conversation 消息
  │                       │   conversation_messages = context.conversation_messages
  │                       │
  │                       ├─→ 调用 CompactionService
  │                       │   result = await service.compact(conversation_messages, llm)
  │                       │   │
  │                       │   └─→ LLM.ainvoke(messages + summary_prompt)
  │                       │
  │                       └─→ 替换 conversation
  │                           context.replace_conversation([summary_item])
```

---

## 数据结构详解

### 1. ContextIR 核心结构

```python
# comate_agent_sdk/context/ir.py
@dataclass
class ContextIR:
    """上下文中间表示"""

    # 两个独立的 Segment
    header: Segment       # ← system_prompt, subagent_strategy, skill_strategy
    conversation: Segment # ← 被压缩的部分

    # 辅助组件
    token_counter: TokenCounter
    event_bus: EventBus
```

**Segment 结构**：

```python
@dataclass
class Segment:
    """消息段"""
    name: SegmentName  # HEADER | CONVERSATION
    items: list[ContextItem]  # 条目列表
```

**ContextItem 结构**：

```python
@dataclass
class ContextItem:
    """上下文条目"""
    id: str
    item_type: ItemType  # TOOL_RESULT, USER_MESSAGE, etc.
    message: BaseMessage | None  # 底层消息对象
    content_text: str
    token_count: int
    priority: int
```

### 2. 发送给 LLM 的数据结构

```python
# comate_agent_sdk/agent/compaction/service.py:173-179
prepared_messages = [
    # 原 conversation 中的消息
    UserMessage("帮我实现一个功能"),
    AssistantMessage("好的，我来帮你..."),
    ToolMessage("..."),
    AssistantMessage(content="...", tool_calls=[...]),
    # ... (最后一条 assistant 消息的 tool_calls 会被移除)

    # 摘要 prompt
    UserMessage(content=DEFAULT_SUMMARY_PROMPT),
]
```

### 3. DEFAULT_SUMMARY_PROMPT

```python
# comate_agent_sdk/agent/compaction/models.py:16-45
DEFAULT_SUMMARY_PROMPT = """You have been working on the task described above but have not yet completed it. Write a continuation summary that will allow you (or another instance of yourself) to resume work efficiently in a future context window where the conversation history will be replaced with this summary. Your summary should be structured, concise, and actionable. Include:

1. Task Overview
The user's core request and success criteria
Any clarifications or constraints they specified

2. Current State
What has been completed so far
Files created, modified, or analyzed (with paths if relevant)
Key outputs or artifacts produced

3. Important Discoveries
Technical constraints or requirements uncovered
Decisions made and their rationale
Errors encountered and how they were resolved
What approaches were tried that didn't work (and why)

4. Next Steps
Specific actions needed to complete the task
Any blockers or open questions to resolve
Priority order if multiple steps remain

5. Context to Preserve
User preferences or style requirements
Domain-specific details that aren't obvious
Any promises made to the user

Be concise but complete - err on the side of including information that would prevent duplicate work or repeated mistakes. Write in a way that enables immediate resumption of the task.

Wrap your summary in <summary></summary> tags."""
```

---

## 压缩流程

### 阶段 1: 选择性压缩（优先）

按类型优先级从低到高逐个压缩：

```python
# comate_agent_sdk/context/compaction.py:131-167
for item_type, _priority in sorted_types:
    rule = self.rules.get(item_type.value)

    if rule.strategy == CompactionStrategy.DROP:
        # 直接移除所有该类型的条目
        removed = context.conversation.remove_by_type(item_type)

    elif rule.strategy == CompactionStrategy.TRUNCATE:
        # 保留最近 N 个，删除更早的
        items_to_remove = items[:-rule.keep_recent]
        for item in items_to_remove:
            context.conversation.remove_by_id(item.id)

    # 检查是否已降到阈值以下
    if context.total_tokens < threshold:
        return True
```

**默认压缩规则**：

| 类型 | 策略 | 参数 |
|------|------|------|
| `TOOL_RESULT` | TRUNCATE | keep_recent=3 |
| `SKILL_PROMPT` | DROP | - |
| `SKILL_METADATA` | DROP | - |
| `ASSISTANT_MESSAGE` | TRUNCATE | keep_recent=5 |
| `USER_MESSAGE` | TRUNCATE | keep_recent=5 |
| `SYSTEM_PROMPT` | NONE | 永不压缩 |
| `SUBAGENT_STRATEGY` | NONE | 永不压缩 |
| `COMPACTION_SUMMARY` | NONE | 永不压缩 |

### 阶段 2: 全量摘要回退

当选择性压缩不足时触发：

```python
# comate_agent_sdk/context/compaction.py:194-232
async def _fallback_full_summary(self, context: ContextIR) -> bool:
    # 1. 提取 conversation 中的底层消息
    conversation_messages = context.conversation_messages

    # 2. 调用 LLM 生成摘要
    service = CompactionService(llm=self.llm)
    result = await service.compact(conversation_messages, self.llm)

    # 3. 用摘要替换整个 conversation
    summary_item = ContextItem(
        item_type=ItemType.COMPACTION_SUMMARY,
        message=UserMessage(content=result.summary),
        ...
    )
    context.replace_conversation([summary_item])
```

### 阶段 3: 上下文更新

```python
# comate_agent_sdk/context/ir.py:503-513
def replace_conversation(self, items: list[ContextItem]) -> None:
    """替换整个 conversation 段"""
    self.conversation.items = list(items)
    self.event_bus.emit(ContextEvent(
        event_type=EventType.CONVERSATION_REPLACED,
        detail=f"conversation replaced with {len(items)} items",
    ))
```

**重要**：是 `replace` 而不是 `append`，确保整个 conversation 被摘要替换。

### 阶段 4: 下次 LLM 调用自动生效

```python
# comate_agent_sdk/agent/service.py:543
messages = self._context.lower()  # 基于修改后的 IR 生成新 messages
```

`lower()` 会重新遍历修改后的 `header` 和 `conversation`，生成新的 `list[BaseMessage]`。

---

## 数据结构变化示意图

### 选择性压缩前

```
ContextIR
├── header: [system_prompt, subagent_strategy, skill_strategy]
└── conversation: [
    ContextItem(TOOL_RESULT, token_count=5000),     # ← 将被 DROP
    ContextItem(SKILL_PROMPT, token_count=3000),    # ← 将被 DROP
    ContextItem(SKILL_METADATA, token_count=1000),  # ← 将被 DROP
    ContextItem(USER_MESSAGE, token_count=200),     # ← 保留（最近 5 个）
    ContextItem(ASSISTANT_MESSAGE, token_count=150),
    ContextItem(TOOL_RESULT, token_count=8000),     # ← TRUNCATE（保留最近 3 个）
    ...
]
```

### 选择性压缩后

```
ContextIR
├── header: [system_prompt, subagent_strategy, skill_strategy]  # ← 不变
└── conversation: [
    ContextItem(USER_MESSAGE, token_count=200),      # 保留
    ContextItem(ASSISTANT_MESSAGE, token_count=150), # 保留
    ContextItem(TOOL_RESULT, token_count=8000),      # 保留（最近 3 个之一）
]
```

### 全量摘要回退后

```
ContextIR
├── header: [system_prompt, subagent_strategy, skill_strategy]  # ← 不变
└── conversation: [
    ContextItem(COMPACTION_SUMMARY, UserMessage("<summary>...</summary>"), token_count=500)
]
```

---

## 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 压缩检查入口 | `comate_agent_sdk/agent/service.py` | 679-720 |
| auto_compact 调用 | `comate_agent_sdk/context/ir.py` | 449-477 |
| 选择性压缩策略 | `comate_agent_sdk/context/compaction.py` | 103-192 |
| 全量摘要回退 | `comate_agent_sdk/context/compaction.py` | 194-232 |
| CompactionService | `comate_agent_sdk/agent/compaction/service.py` | 135-195 |
| 默认摘要 prompt | `comate_agent_sdk/agent/compaction/models.py` | 16-45 |
| 默认压缩规则 | `comate_agent_sdk/context/compaction.py` | 46-72 |
| replace_conversation | `comate_agent_sdk/context/ir.py` | 503-513 |

---

## 为什么 system prompt 不会被压缩？

1. **位置隔离**：system prompt 存在 `header` Segment，而压缩只处理 `conversation` Segment

2. **策略保护**：即使被遍历到，`CompactionStrategy.NONE` 也会跳过

```python
# compaction.py:131-135
for item_type, _priority in sorted_types:
    rule = self.rules.get(item_type.value)
    if rule is None or rule.strategy == CompactionStrategy.NONE:
        continue  # ← system_prompt 在这里被跳过
```

3. **语义必要**：系统提示是 Agent 行为的核心定义，压缩会改变意图

---

## 为什么是 replace 而不是 append？

```python
# 如果 append（错误做法）
conversation.items = [
    ...原消息 50000 tokens...,
    UserMessage("<summary>...</summary>")  # 500 tokens
]
# 总 token: 50500（反而增加了！）

# 实际 replace（正确做法）
conversation.items = [
    UserMessage("<summary>...</summary>")  # 500 tokens
]
# 总 token: 500（大幅减少）
```

**核心原因**：压缩的目的是减少 token，必须整体替换才能达到效果。
