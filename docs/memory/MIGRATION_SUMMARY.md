# Memory 从 SystemMessage 迁移到 UserMessage - 实施总结

## 迁移目标

将 Memory（CLAUDE.md / AGENTS.md 等仓库文件）从 SystemMessage 中移除，作为第一条 UserMessage(is_meta=True) 注入，用 `<instructions>` 标签包裹。这确保用户可控的内容不会获得 system 级别的优先级，提升安全性。

## 核心设计

### 数据结构变更

Memory 不再属于 `header.items` 或 `conversation.items`，而是作为 ContextIR 的独立字段 `_memory_item` 存储。

```python
@dataclass
class ContextIR:
    header: Segment
    conversation: Segment
    _memory_item: ContextItem | None  # 新增独立字段
```

### Lowering 流程

```
SystemMessage (header)
    ↓
UserMessage(is_meta=True, content="<instructions>...</instructions>")  ← Memory
    ↓
Conversation items (UserMessage, AssistantMessage, ToolMessage, ...)
    ↓
System Reminders (UserMessage, is_meta=True)
```

## 修改文件清单

### 1. `comate_agent_sdk/context/ir.py` ✅

**修改点**：
- L49-59: 从 `_HEADER_ITEM_ORDER` 中移除 `ItemType.MEMORY: 2`
- L78-104: 更新 docstring 和添加 `_memory_item` 字段
- L186-198: `set_agent_loop()` - 从 elif 条件中移除 `ItemType.MEMORY`
- L207-249: **完全重写 `set_memory()`** - 使用 `<instructions>` 标签，创建 UserMessage(is_meta=True)，存入 `_memory_item`
- L266-273: `set_subagent_strategy()` - 从 position 查找中移除 `ItemType.MEMORY`
- L281-312: `set_tool_strategy()` - 从 position 查找和 docstring 中移除 `ItemType.MEMORY`
- L799-801: `total_tokens` - 添加 memory_item 的 token 统计
- L803-821: `get_budget_status()` - 添加 memory_item 的 token 统计
- L870-881: `clear()` - 添加 `self._memory_item = None`
- L905-910: 添加只读属性 `memory_item`

### 2. `comate_agent_sdk/context/lower.py` ✅

**修改点**：
- L52-60: `lower()` - 在 SystemMessage 之后注入 Memory UserMessage
- L77-95: `_build_header_text()` - 从 type_order 中移除 `ItemType.MEMORY`，更新 docstring

### 3. `comate_agent_sdk/context/info.py` ✅

**修改点**：
- L139-147: `_build_categories()` - 额外统计 `context.memory_item`

### 4. `comate_agent_sdk/context/compaction.py` ✅

**修改点**：
- L107-109: 为 `ItemType.MEMORY` 压缩规则添加注释说明（该规则现已无效）

### 5. `comate_agent_sdk/examples/test_tool_strategy.py` ✅

**修改点**：
- L40-44: 修正 TOOL_STRATEGY 格式断言（使用 `<tools>` 而非 `[SYSTEM_TOOLS_DEFINITION]`）
- L70-76: 修正自定义工具测试逻辑
- L74-110: **完全重写 `test_system_message_order()`** - 验证 Memory 在 UserMessage[1] 中

### 6. 新增测试文件 ✅

**文件**: `comate_agent_sdk/examples/test_memory_migration.py`

**测试覆盖**：
1. Memory 作为 UserMessage(is_meta=True) 注入
2. Memory 不在 header.items 中，而在独立字段
3. clear_history 后 Memory 能正确重建
4. Memory 的 token 统计正确
5. Memory 的幂等更新
6. Memory 的 cache hint

**测试结果**: ✅ 6/6 通过

## 不需要修改的文件

| 文件 | 原因 |
|------|------|
| `agent/history.py` | `clear_history()` 先调用 `clear()`（会清除 `_memory_item`），再调用 `_setup_memory()`（会重建） |
| `agent/setup.py` | `setup_memory()` 仍调用 `context.set_memory()`，接口不变 |
| `agent/init.py` | 调用链不变 |
| `context/memory.py` | 纯文件加载逻辑，与注入位置无关 |
| `context/items.py` | `ItemType.MEMORY` 枚举和优先级保留 |
| `context/accounting.py` | `estimate_next_step()` 用 lowering 获取完整消息列表，memory 会自动包含 |
| `agent/compaction/service.py` | `_serialize_messages_to_text()` 序列化 conversation.items，memory 不在其中（正确：静态内容不需要被摘要） |
| `agent/chat_session.py` | 通过 `agent.clear_history()` 间接操作 |

## 验证结果

### 1. 功能测试 ✅

```bash
$ uv run python comate_agent_sdk/examples/test_tool_strategy.py
测试结果: 3/3 通过
✓ 所有测试通过！
```

### 2. Memory 迁移测试 ✅

```bash
$ uv run python comate_agent_sdk/examples/test_memory_migration.py
测试结果: 6/6 通过
✓ 所有测试通过！Memory 迁移成功！
```

### 3. 消息结构验证 ✅

**正常对话**：
```
[0] SystemMessage(system_prompt + agent_loop + tool_strategy + ...)  # Header（不含 Memory）
[1] UserMessage("<instructions>...memory...</instructions>", is_meta=True)  # Memory
[2] UserMessage("用户输入")  # 第一条用户消息
[3] AssistantMessage(...)
...
```

**clear_history 后**：
```
[0] SystemMessage(...)  # Header（重建）
[1] UserMessage("<instructions>...memory...</instructions>", is_meta=True)  # Memory（重建）
```

## 安全性提升

### Before (有风险)
```
SystemMessage:
  - system_prompt: "You are an AI assistant..."
  - memory: "<memory>...CLAUDE.md 内容...</memory>"  ← 用户可控内容
  - tool_strategy: "..."
```

**风险**: 用户可通过 CLAUDE.md 注入指令，覆盖 system_prompt 的安全约束。

### After (安全)
```
[0] SystemMessage:
      - system_prompt: "You are an AI assistant..."
      - tool_strategy: "..."
[1] UserMessage(is_meta=True):
      "<instructions>...CLAUDE.md 内容...</instructions>"  ← 降级为 user 权限
```

**改进**:
1. Memory 作为 UserMessage，优先级低于 SystemMessage
2. 使用 `<instructions>` 标签而非 `<memory>`，语义更准确
3. 安全约束（system_prompt）不会被用户可控内容污染

## 向后兼容性

- ✅ `context.set_memory()` API 不变
- ✅ `context.memory_item` 提供只读访问
- ✅ `context.total_tokens` 自动包含 memory tokens
- ✅ `get_budget_status()` 自动统计 memory tokens
- ✅ `clear()` 正确清除 memory
- ✅ clear_history 流程不受影响

## 性能影响

- **Token 统计**: 正确，memory tokens 计入 `conversation_tokens`（因为它在 conversation 之前注入）
- **缓存**: 支持，Memory UserMessage 保留 `cache=True` 属性
- **压缩**: 正确，Memory 不在 conversation.items 中，不会被压缩（符合预期：静态内容不应被摘要）

## 总结

本次迁移成功将 Memory 从 SystemMessage 降级到 UserMessage，实现了以下目标：

1. ✅ **安全性提升**: 用户可控内容不再拥有 system 权限
2. ✅ **架构清晰**: Memory 作为独立字段，不污染 header 或 conversation
3. ✅ **语义准确**: 使用 `<instructions>` 而非 `<memory>` 标签
4. ✅ **向后兼容**: API 不变，现有代码无需修改
5. ✅ **测试覆盖**: 9 个测试全部通过
6. ✅ **文档完善**: 注释和 docstring 更新

**迁移状态**: 🎉 **完成并验证通过**
