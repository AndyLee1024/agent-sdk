# Task 工具 Subagent 黑盒问题

## 问题描述

当前 Task 工具创建的 Subagent 使用 `query()` 模式运行，而非 `query_stream()` 模式。这导致 Subagent 的执行过程对外不透明，无法观察到内部细节。

## 代码位置

- **Task 工具实现**: `comate_agent_sdk/subagent/task_tool.py` 第 149-156 行

```python
# 当前实现
if agent_def.timeout:
    result = await asyncio.wait_for(
        subagent_runtime.query(prompt), timeout=agent_def.timeout
    )
else:
    result = await subagent_runtime.query(prompt)
```

## 影响

| 缺失的信息 | 说明 |
|-----------|------|
| 工具调用过程 | 无法看到 Subagent 正在调用什么工具 |
| LLM 中间输出 | 无法看到 LLM 的思考过程和推理 |
| 工具执行结果 | 无法看到每个工具的具体返回内容 |
| 执行进度 | 无法知道 Subagent 当前处于第几轮迭代 |

## `query()` vs `query_stream()` 对比

| 模式 | 事件输出 | 可观察性 | 复杂度 |
|------|---------|----------|--------|
| `query()` | ❌ 无 | 黑盒，只能看到最终结果 | 低 |
| `query_stream()` | ✅ 有 | 透明，可以看到每一步 | 高 |

## 可能的解决方案

### 方案 1：Task 工具改用 `query_stream()`

```python
# 改造思路
async for event in subagent_runtime.query_stream(prompt):
    # 转发事件给上层（需要定义新的 SubagentEvent 类型）
    yield SubagentEvent(subagent_name=agent_def.name, inner_event=event)
    
    if isinstance(event, FinalResponseEvent):
        result = event.content
```

**挑战**：
- Task 工具需要变成 async generator
- 主 Agent 需要支持处理嵌套事件
- 并行 Task 执行时事件流交错的处理

### 方案 2：添加日志/回调机制

保持 `query()` 模式，但通过日志或回调机制暴露内部状态。

### 方案 3：可选的透明模式

在 `AgentDefinition` 中添加 `transparent: bool` 配置，让用户选择是否需要观察 Subagent 内部。

## 当前设计的合理性

可能的设计考量：
1. **简化并发控制** - 多个 Task 并行执行时，事件流合并复杂
2. **性能优化** - 不处理事件流更轻量
3. **接口简洁** - 调用者只关心最终结果

## 相关文件

- `comate_agent_sdk/subagent/task_tool.py` - Task 工具实现
- `comate_agent_sdk/agent/runner.py` - `query()` 实现
- `comate_agent_sdk/agent/runner_stream.py` - `query_stream()` 实现
- `comate_agent_sdk/agent/events.py` - 事件定义
