# 上下文压缩机制深度研究

## 研究目标

本文档是对现有 `compaction-flow.md` 的深度补充，聚焦于：
1. 与其他上下文管理机制（prompt cache、thinking protection、offload）的交叉点
2. 当前实现的潜在问题与优化方向
3. 完整的数据流与模块边界

> **前置阅读**：建议先阅读 `compaction-flow.md` 了解基础流程，本文在此基础上做深度分析。

---

## 1. 术语定义

| 术语 | 定义 | 所在模块 |
|------|------|----------|
| **ContextIR** | 上下文中间表示，将扁平的 `list[BaseMessage]` 提升为结构化 IR | `context/ir.py` |
| **SelectiveCompactionPolicy** | 选择性压缩策略：按类型优先级逐步压缩 + LLM 摘要回退 | `context/compaction.py` |
| **CompactionService** | 独立的压缩服务，负责 LLM 摘要生成（与 Policy 不同职责） | `agent/compaction/service.py` |
| **ToolInteractionBlock** | 工具交互块：AssistantMessage(tool_calls) + 对应 ToolMessages | `context/compaction.py` |
| **TruncationRecord** | 三层共享的截断记录（Formatter → Offload → Compaction） | `context/truncation.py` |
| **Offload** | 上下文落盘：将大体积内容写入文件系统，用占位符替换 | `context/offload.py` |

---

## 2. 架构图

### 2.1 压缩相关的四大模块

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Agent Runtime (agent/core.py)                      │
│                                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐                     │
│  │   CompactionService   │    │ SelectiveCompaction  │                     │
│  │  (agent/compaction/   │    │    Policy             │                     │
│  │      service.py)      │    │ (context/compaction) │                     │
│  │                       │    │                      │                     │
│  │  - token 计数         │    │  - 选择性压缩规则     │                     │
│  │  - 阈值计算           │    │  - 工具块截断        │                     │
│  │  - LLM 摘要生成       │    │  - 摘要回退          │                     │
│  │  - 冷却逻辑           │    │  - 回滚机制          │                     │
│  └───────────┬──────────┘    └──────────┬───────────┘                     │
│              │                          │                                  │
│              │    ┌──────────────────────┴───────────┐                      │
│              │    │        ContextIR (context/ir.py) │                      │
│              │    │                                   │                      │
│              │    │  ┌─────────────┐ ┌─────────────┐  │                      │
│              │    │  │   header    │ │ conversation│  │                      │
│              │    │  │  (Segment)   │ │  (Segment)  │  │                      │
│              │    │  └─────────────┘ └─────────────┘  │                      │
│              │    │         │              │          │                      │
│              │    │         └──────────────┬──────────┘                      │
│              │    │                        │                                  │
│              │    │  ┌─────────────────────────────────────┐                  │
│              │    │  │  LoweringPipeline (context/lower)  │                  │
│              │    │  │   IR → list[BaseMessage]            │                  │
│              │    │  └─────────────────────────────────────┘                  │
│              │    └──────────────────────────────────────────────────────────┘
│              │                               │
│              └───────────────────────────────┘
│                                      │
│                                      ▼
│  ┌─────────────────────────────────────────────────────────────────────────┐
│  │                      LLM API (llm/)                                     │
│  │   - openai/chat.py  - anthropic/chat.py  - deepseek/chat.py          │
│  └─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 压缩触发时机

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Agent Loop                                       │
│                                                                               │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                 │
│   │ Tool Result │      │   LLM       │      │   Check &   │                 │
│   │  Added      │ ──▶  │  Invoke     │ ──▶  │  Compact    │                 │
│   └─────────────┘      └─────────────┘      └──────┬──────┘                 │
│                                                    │                         │
│                       ┌────────────────────────────┴───────────┐            │
│                       │                                        │            │
│                       ▼                                        ▼            │
│         ┌─────────────────────────┐            ┌─────────────────────────┐   │
│         │  precheck_and_compact   │            │  check_and_compact     │   │
│         │  (tool result 后)       │            │  (LLM response 后)      │   │
│         │                         │            │                         │   │
│         │  - 估算下一步 token     │            │  - 使用实际 usage       │   │
│         │  - 附加安全缓冲        │            │  - 判断是否超阈值       │   │
│         └─────────────────────────┘            └─────────────────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 关键代码路径

### 3.1 入口点

| 入口函数 | 文件:行号 | 触发时机 | 职责 |
|----------|-----------|----------|------|
| `check_and_compact()` | `agent/runner.py:138` | LLM response 返回后 | 基于实际 token 使用量判断是否压缩 |
| `precheck_and_compact()` | `agent/runner.py:217` | tool result 添加后 | 基于估算值预测是否需要压缩 |
| `ContextIR.auto_compact()` | `context/ir.py:1045` | 内部调用 | 执行实际的压缩逻辑 |

### 3.2 核心类与函数

| 类/函数 | 文件:行号 | 职责 | 关键属性/参数 |
|---------|-----------|------|---------------|
| `SelectiveCompactionPolicy` | `context/compaction.py:125` | 压缩策略执行器 | `threshold`, `rules`, `llm`, `fallback_to_full_summary` |
| `SelectiveCompactionPolicy.compact()` | `context/compaction.py:156` | 压缩主流程 | 返回 `bool` 表示是否执行了压缩 |
| `SelectiveCompactionPolicy._truncate_tool_blocks()` | `context/compaction.py:320` | 工具块截断 | 按块保留最近 N 个 |
| `SelectiveCompactionPolicy._fallback_full_summary_with_retry()` | `context/compaction.py:528` | LLM 摘要回退 | 带重试的摘要生成 |
| `CompactionService.compact()` | `agent/compaction/service.py:139` | LLM 摘要生成 | 独立于 Policy，负责调用 LLM |
| `CompactionService.should_compact()` | `agent/compaction/service.py:119` | 判断是否需要压缩 | 基于 token 使用量 |
| `ContextIR.replace_conversation()` | `context/ir.py:1104` | 替换 conversation | 压缩后用摘要替换 |
| `LoweringPipeline.lower()` | `context/lower.py:42` | IR → messages | 生成发送给 LLM 的消息列表 |

### 3.3 数据结构

| 数据结构 | 文件:行号 | 用途 |
|----------|-----------|------|
| `ContextItem` | `context/items.py:78` | 上下文条目（对应一条消息或结构化信息） |
| `Segment` | `context/items.py:128` | 段落（header / conversation） |
| `ItemType` | `context/items.py:20` | 条目类型枚举（TOOL_RESULT, USER_MESSAGE 等） |
| `CompactionMetaRecord` | `context/compaction.py:54` | 压缩过程元数据记录 |
| `TruncationRecord` | `context/truncation.py:10` | 截断记录（Formatter → Offload → Compaction 共享） |

---

## 4. 压缩策略细节

### 4.1 默认压缩规则

| ItemType | 策略 | 参数 | 压缩优先级 |
|----------|------|------|------------|
| `TOOL_RESULT` | TRUNCATE | keep_recent=5 | 10 (最先) |
| `SKILL_PROMPT` | DROP | - | 20 |
| `SKILL_METADATA` | DROP | - | 40 |
| `ASSISTANT_MESSAGE` | TRUNCATE | keep_recent=5 | 30 |
| `USER_MESSAGE` | TRUNCATE | keep_recent=5 | 50 |
| `SYSTEM_PROMPT` | NONE | 永不压缩 | 100 |
| `MEMORY` | NONE | 永不压缩 | 100 |
| `COMPACTION_SUMMARY` | NONE | 永不压缩 | 80 |

> **优先级规则**：数值越低越先被压缩，`SYSTEM_PROMPT` / `MEMORY` 等核心信息永不压缩。

### 4.2 工具块 (ToolInteractionBlock) 处理

工具调用是一个 **块结构**：

```
AssistantMessage(tool_calls=[...]) → ToolMessage(result) → ToolMessage(result) → ...
```

压缩时不是按单条消息处理，而是按 **块** 处理：

```python
# context/compaction.py:320-358
def _truncate_tool_blocks():
    # 1. 提取所有工具交互块
    blocks = self._extract_tool_blocks(context)
    
    # 2. 排序并保留最近 keep_recent 个块
    blocks_sorted = sorted(blocks, key=lambda b: b.start_idx)
    kept_blocks = blocks_sorted[-keep_recent:]
    dropped_blocks = blocks_sorted[:-keep_recent]
    
    # 3. 对保留块内的字段做阈值截断
    for block in kept_blocks:
        self._truncate_tool_block_fields(context, block)
    
    # 4. 删除被丢弃的块
    for block in dropped_blocks:
        del context.conversation.items[block.start_idx:block.end_idx + 1]
```

**块内字段截断阈值**（可通过 Policy 配置）：

| 字段 | 阈值 (tokens) | 截断后保留 |
|------|---------------|------------|
| `tool_call.arguments` | 500 | 200 (preview) |
| `tool_result.content` | 600 | 200 (preview) |

### 4.3 摘要回退流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SelectiveCompactionPolicy.compact()                     │
│                                                                              │
│  1. 选择性压缩（按优先级遍历类型）                                            │
│     ├── DROP: 直接移除整类条目                                               │
│     └── TRUNCATE: 保留最近 N 条                                              │
│                                                                              │
│  2. 检查 token 是否降至阈值以下                                              │
│     └── 如果仍超阈值 → 进入摘要回退                                           │
│                                                                              │
│  3. 摘要回退 (_fallback_full_summary_with_retry)                           │
│     │                                                                        │
│     ├─ 序列化 conversation messages → 纯文本                                 │
│     ├─ 构建摘要 prompt (DEFAULT_SUMMARY_PROMPT)                              │
│     ├─ 调用 LLM 生成摘要                                                    │
│     ├─ 提取 <summary>...</summary> 标签内容                                │
│     │                                                                        │
│     └─ 替换 conversation 为:                                                │
│         ContextItem(                                                         │
│             item_type=COMPACTION_SUMMARY,                                    │
│             message=UserMessage(content=摘要),                               │
│             ...                                                              │
│         )                                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**摘要 prompt 结构**（来自 `agent/compaction/models.py`）：

```python
DEFAULT_SUMMARY_PROMPT = """You have been working on the task described above...
# 要求 LLM 生成结构化摘要，包含:
# 1. Task Overview
# 2. Current State
# 3. Important Discoveries
# 4. Next Steps
# 5. Context to Preserve
...
Wrap your summary in <summary></summary> tags."""
```

---

## 5. 与其他系统的交叉点

### 5.1 Prompt Cache

**交叉点**：
- `ContextItem.cache_hint` 属性（`context/items.py:115`）
- `LoweringPipeline.lower()` 在生成 SystemMessage 时检查 `cache_hint`（`context/lower.py:58`）

**当前行为**：
- header 中的 items 设置 `cache_hint=True` 时，生成的 SystemMessage 会带缓存提示
- **但压缩会修改 conversation，不会影响 header 的 cache 行为**

**潜在问题**：
- 如果 prompt cache 的 key 包含 conversation 内容哈希，压缩后会导致 cache miss
- 需确认 cache key 的组成（当前未在 compaction 模块中处理）

### 5.2 Thinking Protection

**交叉点**：
- `ContextIR._thinking_protected_assistant_ids`（`context/ir.py:119`）
- `SelectiveCompactionPolicy._collect_recent_round_protected_ids()`（`context/compaction.py:553`）

**当前行为**：
- Tool loop 中含 thinking blocks 的 assistant message ID 会被保护，不被选择性压缩删除
- 压缩时会跳过 `thinking_protected_assistant_ids` 中的项

**保护机制**：
```python
# context/compaction.py:230-235
if item_type in (ItemType.USER_MESSAGE, ItemType.ASSISTANT_MESSAGE):
    candidates = [it for it in items if it.id not in protected_ids]
```

### 5.3 Offload (上下文落盘)

**交叉点**：
- `TruncationRecord` 是三层共享的截断记录（`context/truncation.py`）
- Formatter 截断 → Offload 检查 `is_formatter_truncated` → Compaction 跳过已截断项

**Offload 触发条件**：
- 工具输出超过 `offload_token_threshold`（默认 1000 tokens）
- `destroy_ephemeral_messages()` 中对ephemeral 工具结果处理

**与 Compaction 的协作**：
```python
# context/compaction.py:409-418
if (
    result_item.truncation_record is not None
    and result_item.truncation_record.is_formatter_truncated
):
    # 跳过已被 Formatter 截断的 tool_result
    # 避免重复截断
    continue
```

### 5.4 三层截断协作示意图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OutputFormatter (system_tools)                     │
│  - 大文件读取时截断输出                                                      │
│  - 生成 TruncationRecord(is_formatter_truncated=True)                      │
│  └── 写入 ContextItem.truncation_record                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Offload (context/offload)                            │
│  - 检查 is_formatter_truncated                                              │
│  - 对大体积内容落盘到文件系统                                                │
│  - 用占位符替换 ContextItem                                                 │
│  └── 写入 offload_path / offloaded 标志                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Compaction (context/compaction)                         │
│  - 检查 is_formatter_truncated，避免重复截断                                 │
│  - 按块压缩工具交互历史                                                      │
│  - 对保留块内的字段做二次截断                                               │
│  └── 最终生成摘要或替换 conversation                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 可观测性与调试

### 6.1 日志点

| 事件 | 日志级别 | 位置 | 内容 |
|------|----------|------|------|
| 开始选择性压缩 | INFO | `context/compaction.py:176` | `current=X, threshold=Y` |
| DROP 策略执行 | INFO | `context/compaction.py:210` | `DROP {item_type}: 移除 N 条` |
| TRUNCATE 策略执行 | INFO | `context/compaction.py:224` | `TRUNCATE tool_blocks: 移除 N 个块` |
| 压缩失败回滚 | WARNING | `context/compaction.py:317` | `压缩失败，已回滚: {exc}` |
| 摘要成功 | INFO | `context/compaction.py:291` | `选择性压缩完成 → 执行 LLM Summary` |
| 压缩冷却跳过 | WARNING | `agent/runner.py:175` | `压缩冷却中，跳过本轮压缩` |

### 6.2 事件系统

| 事件类型 | 发射位置 | 内容 |
|----------|----------|------|
| `COMPACTION_PERFORMED` | `context/ir.py:1068` | `auto_compact: X → Y tokens` |
| `CONVERSATION_REPLACED` | `context/ir.py:1111` | `conversation replaced with N items` |

### 6.3 调试标志

```python
# 开启详细的压缩元事件（默认关闭）
agent = Agent(emit_compaction_meta_events=True)

# 事件内容示例：
# phase=selective_start, tokens_before=50000, tokens_after=50000
# phase=selective_done, tokens_before=50000, tokens_after=35000
# phase=summary_start, tokens_before=35000, tokens_after=35000
# phase=summary_done, tokens_before=35000, tokens_after=800
```

---

## 7. 潜在问题与优化方向

> **以下是基于代码分析的问题猜想，需结合实际使用场景验证。**

### 7.1 信息丢失风险

| 问题 | 描述 | 严重程度 |
|------|------|----------|
| **摘要质量不可控** | LLM 生成的摘要可能丢失关键信息（如文件路径、错误细节） | 高 |
| **工具输出截断** | `_truncate_tool_block_fields` 只保留前 200 tokens，可能丢失重要输出 | 中 |
| **多轮摘要累积** | 多次压缩后，摘要再被摘要，信息逐层丢失 | 高 |

**优化方向**：
- 增加摘要的结构化输出（如 JSON）而非纯文本
- 对关键信息（文件路径、错误码）做标记保留
- 限制摘要压缩次数或合并多个摘要

### 7.2 性能问题

| 问题 | 描述 | 严重程度 |
|------|------|----------|
| **同步 token 计数阻塞** | `token_counter.count()` 在主线程调用，可能阻塞 | 中 |
| **LLM 摘要延迟** | 摘要需要额外一次 LLM 调用，增加延迟 | 高 |
| **截断阈值硬编码** | `tool_call_threshold=500`, `tool_result_threshold=600` 不可配置 | 低 |

**优化方向**：
- 异步化 token 计数
- 考虑流式压缩（边处理边释放 token）
- 将截断阈值暴露为 Policy 配置参数

### 7.3 边界条件

| 问题 | 描述 | 严重程度 |
|------|------|----------|
| **摘要为空** | LLM 返回空摘要时，压缩会失败并回滚 | 中 |
| **冷却期间堆积** | 压缩失败后进入冷却，但 token 未减少，可能导致连续失败 | 中 |
| **跨轮次状态** | `_thinking_protected_assistant_ids` 跨轮次累积，可能导致保护项越来越多 | 低 |

**优化方向**：
- 增强摘要失败的 fallback（如直接丢弃旧消息）
- 冷却期间强制执行 DROP 策略而非完全跳过
- 定期清理过期的 thinking protection 记录

### 7.4 与其他模块的集成问题

| 问题 | 描述 | 严重程度 |
|------|------|----------|
| **Offload 后再压缩** | 已 offload 的内容如果在压缩时被引用，可能需要特殊处理 | 低 |
| **Session resume** | 从持久化恢复 session 时，压缩状态如何同步 | 中 |
| **多 Agent 共享 Context** | Subagent 的压缩行为与主 Agent 的交互 | 低 |

---

## 8. 测试覆盖

### 8.1 现有测试文件

| 测试文件 | 覆盖场景 |
|----------|----------|
| `context/tests/test_compaction_round_guard_and_rollback.py` | 选择性压缩 + 摘要回退 + 回滚机制 |
| `context/tests/test_tool_block_compaction.py` | 工具块截断逻辑 |
| `context/tests/test_thinking_protection.py` | Thinking block 保护机制 |
| `context/tests/test_compaction_skip_formatter_truncated.py` | 跳过 Formatter 已截断的内容 |
| `context/tests/test_three_layer_integration.py` | Formatter → Offload → Compaction 三层集成 |
| `agent/tests/test_compaction_summary_cooldown.py` | 摘要压缩冷却逻辑 |
| `agent/tests/test_compaction_meta_events.py` | 压缩元事件 |

### 8.2 测试可复现性

```python
# 快速复现压缩触发
from comate_agent_sdk.context.compaction import SelectiveCompactionPolicy

policy = SelectiveCompactionPolicy(threshold=1)  # 强制触发
await policy.compact(context)
```

---

## 9. 优化讨论切入点

基于以上分析，以下是可供讨论的优化方向（按优先级排序）：

### 高优先级

1. **摘要质量改进**
   - 当前：纯文本摘要，信息丢失风险高
   - 方向：结构化摘要（JSON）、关键信息标记保留

2. **多轮摘要累积问题**
   - 当前：每次压缩生成新摘要，多次后信息逐层丢失
   - 方向：限制压缩次数、合并多个历史摘要

3. **摘要失败 fallback**
   - 当前：摘要为空时整个压缩失败回滚
   - 方向：增强失败处理（强制 DROP 策略）

### 中优先级

4. **截断阈值可配置化**
   - 当前：硬编码 `tool_call_threshold=500`, `tool_result_threshold=600`
   - 方向：暴露为 Policy 参数

5. **压缩冷却优化**
   - 当前：冷却期间完全跳过，可能导致 token 堆积
   - 方向：冷却期间执行保守压缩（DROP 策略）

6. **性能优化**
   - 当前：同步 token 计数 + 同步 LLM 摘要
   - 方向：异步化、边处理边释放

### 低优先级

7. **Session resume 状态同步**
   - 压缩状态如何持久化与恢复

8. **Offload 与 Compaction 的更紧密协作**
   - 已 offload 内容的引用处理

---

## 10. 文件清单

| 文件 | 职责 |
|------|------|
| `context/ir.py` | ContextIR 主类，auto_compact 入口 |
| `context/compaction.py` | 选择性压缩策略与执行 |
| `context/items.py` | ContextItem, Segment, ItemType 定义 |
| `context/lower.py` | IR → messages 转换 |
| `context/budget.py` | TokenCounter, BudgetConfig |
| `context/truncation.py` | TruncationRecord 定义 |
| `context/offload.py` | 上下文落盘 |
| `context/observer.py` | ContextEventBus, EventType |
| `agent/compaction/service.py` | CompactionService (LLM 摘要生成) |
| `agent/runner.py` | check_and_compact, precheck_and_compact |
| `agent/events.py` | PreCompactEvent, CompactionMetaEvent |

---

*文档版本: 2026-02-17*
*基于代码分析生成，可能与实际行为有偏差，欢迎验证与修正。*
