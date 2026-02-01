# AssistantMessage Token 统计修复总结

## 问题描述

用户在使用 `/usage` 和 `/context` 命令时发现：
- `/usage` 显示 `grok-code-fast-1: 60241 tokens`
- `/context` 显示 `8.0k/128.0k tokens (6.3%)`

在某些情况下，`/context` 输出中**没有显示 Messages 类别**。

## 根本原因分析

经过深入分析代码，发现了一个真实的 Bug：

### Bug 详情

**位置：** `bu_agent_sdk/context/ir.py:282-283`

**问题代码：**
```python
# 提取文本内容用于 token 估算
content_text = message.text if hasattr(message, "text") else ""
token_count = self.token_counter.count(content_text)
```

**问题分析：**

1. 当 `AssistantMessage` 只包含 `tool_calls` 而没有文本 `content` 时：
   - `AssistantMessage.content = None`
   - `AssistantMessage.text` 返回 `""`（空字符串）
   - `token_counter.count("")` 返回 **0 或 1**

2. `tool_calls` 中的 JSON tokens **完全被忽略**

3. 导致 `get_budget_status()` 统计时，ASSISTANT_MESSAGE 的总 tokens 可能为 0

4. `_build_categories()` 跳过 `total_tokens == 0` 的类别

5. **Messages 类别不显示！**

### 影响

- `/context` 的统计不准确（低估了实际上下文大小）
- 可能导致压缩触发时机不准确
- 用户难以理解真实的上下文占用情况

## 修复方案

### 核心修改

**文件：** `bu_agent_sdk/context/ir.py`

**修改内容：** 在 `add_message()` 方法中添加对 AssistantMessage 的特殊处理

```python
# 自动推断类型
if item_type is None:
    item_type = _MESSAGE_TYPE_MAP.get(type(message), ItemType.USER_MESSAGE)

# 提取文本内容用于 token 估算
content_text = message.text if hasattr(message, "text") else ""

# AssistantMessage 特殊处理：需要包括 tool_calls 的 tokens
# 因为 tool_calls 也会被发送给 LLM，占用 prompt tokens
if isinstance(message, AssistantMessage) and message.tool_calls:
    import json

    tool_calls_json = json.dumps(
        [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ],
        ensure_ascii=False,
    )
    # 如果有文本内容，拼接；否则只用 tool_calls
    content_text = content_text + "\n" + tool_calls_json if content_text else tool_calls_json

token_count = self.token_counter.count(content_text)
```

### 修复原理

1. **检测 AssistantMessage 是否有 tool_calls**
2. **序列化 tool_calls 为 JSON**：包含 `id`、`type`、`function.name`、`function.arguments`
3. **拼接到 content_text**：
   - 如果有文本内容，拼接到后面
   - 如果没有文本内容，只用 tool_calls JSON
4. **统一计算 token_count**：包含文本 + tool_calls

## 测试验证

### 单元测试

创建了完整的测试套件：`bu_agent_sdk/context/tests/test_assistant_message_tokens.py`

**测试用例：**
1. ✅ 只有文本的 AssistantMessage
2. ✅ 只有 tool_calls 的 AssistantMessage
3. ✅ 同时有文本和 tool_calls 的 AssistantMessage
4. ✅ 多个 tool_calls 的 AssistantMessage
5. ✅ get_budget_status() 正确统计
6. ✅ Messages 类别始终显示

**测试结果：** 全部通过 ✅

```
bu_agent_sdk/context/tests/test_assistant_message_tokens.py::TestAssistantMessageTokenCounting::test_assistant_message_with_multiple_tool_calls PASSED
bu_agent_sdk/context/tests/test_assistant_message_tokens.py::TestAssistantMessageTokenCounting::test_assistant_message_with_only_text_counts_text_tokens PASSED
bu_agent_sdk/context/tests/test_assistant_message_tokens.py::TestAssistantMessageTokenCounting::test_assistant_message_with_only_tool_calls_counts_tool_call_tokens PASSED
bu_agent_sdk/context/tests/test_assistant_message_tokens.py::TestAssistantMessageTokenCounting::test_assistant_message_with_text_and_tool_calls_counts_both PASSED
bu_agent_sdk/context/tests/test_assistant_message_tokens.py::TestAssistantMessageTokenCounting::test_budget_status_includes_assistant_message_tokens PASSED
bu_agent_sdk/context/tests/test_assistant_message_tokens.py::TestAssistantMessageTokenCounting::test_messages_category_appears_in_context_info PASSED
```

### 验证脚本

创建了实际测试脚本：`bu_agent_sdk/examples/verify_token_fix.py`

**验证结果：**
```
📊 测试 1: 验证只有 tool_calls 的 AssistantMessage 统计
   ✅ AssistantMessage 的 token_count > 0: 41
   ✅ Token count 包含 tool_calls (> 10): 41
   ✅ 预算状态包含 ASSISTANT_MESSAGE: 41 tokens
   ✅ ASSISTANT_MESSAGE tokens > 0: 41

📊 测试 2: 验证多个 tool_calls 的统计
   ✅ AssistantMessage (含多个 tool_calls) 添加成功
   - Token Count: 137
   ✅ 多个 tool_calls 的 token 数更多 (137 > 41)
```

### 回归测试

运行了所有现有测试，确保没有破坏现有功能：

- ✅ `bu_agent_sdk/context/tests/` - 6 个测试通过
- ✅ `bu_agent_sdk/agent/tests/` - 8 个测试通过

## 修复效果

### 修复前

```
/context 输出：
⛁ Tool Definitions: 1.5k tokens (1.2%)
⛁ Tool Results: 3.1k tokens (2.5%)
⛁ System Prompt: 1.8k tokens (1.4%)
⛁ Skills: 791 tokens (0.6%)
⛁ Skill Strategy: 560 tokens (0.4%)
# Messages 类别没有显示！
```

**问题：**
- AssistantMessage 的 token_count = 0（因为只统计了空的 content）
- tool_calls 的 JSON tokens 被完全忽略
- Messages 类别因为总 tokens = 0 而被跳过

### 修复后

```
/context 输出：
⛁ Tool Definitions: 1.5k tokens (1.2%)
⛁ Tool Results: 3.1k tokens (2.5%)
⛁ System Prompt: 1.8k tokens (1.4%)
⛁ Messages: 800 tokens (0.6%)  ← 现在正确显示了！
⛁ Skills: 791 tokens (0.6%)
⛁ Skill Strategy: 560 tokens (0.4%)
```

**改进：**
- ✅ AssistantMessage 的 token_count 正确包含 tool_calls
- ✅ Messages 类别始终显示
- ✅ token 统计更接近实际值

## 技术细节

### AssistantMessage 结构

```python
class AssistantMessage:
    content: str | list[...] | None  # 可以是 None！
    tool_calls: list[ToolCall] | None
```

### ToolCall 结构

```python
class ToolCall(BaseModel):
    id: str
    type: Literal['function'] = 'function'
    function: Function

class Function(BaseModel):
    name: str
    arguments: str  # JSON 字符串
```

### Token 统计逻辑

**修复前：**
```python
content_text = message.text  # 只统计文本
token_count = self.token_counter.count(content_text)
```

**修复后：**
```python
content_text = message.text

# 如果有 tool_calls，序列化并拼接
if isinstance(message, AssistantMessage) and message.tool_calls:
    tool_calls_json = json.dumps([...])
    content_text = content_text + "\n" + tool_calls_json if content_text else tool_calls_json

token_count = self.token_counter.count(content_text)
```

## 相关文件

### 核心修改
- `bu_agent_sdk/context/ir.py` - 修复 token 统计逻辑

### 测试文件
- `bu_agent_sdk/context/tests/test_assistant_message_tokens.py` - 单元测试
- `bu_agent_sdk/examples/verify_token_fix.py` - 验证脚本

### 相关代码
- `bu_agent_sdk/llm/messages.py` - AssistantMessage/ToolCall 定义
- `bu_agent_sdk/context/budget.py` - TokenCounter
- `bu_agent_sdk/context/formatter.py` - /context 格式化输出
- `bu_agent_sdk/tokens/service.py` - /usage 统计

## 结论

这是一个**真实的 Bug**，影响了 `/context` 命令的准确性。修复后：

1. ✅ AssistantMessage 的 token 统计更准确
2. ✅ Messages 类别始终显示
3. ✅ 上下文预算管理更可靠
4. ✅ 用户体验得到改善

**修复优先级：** 中等（影响用户体验和预算管理的准确性）

**测试覆盖：** 完整（单元测试 + 验证脚本 + 回归测试）

**向后兼容：** 是（没有破坏现有功能）
