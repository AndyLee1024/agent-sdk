# Runner Engine 问题清单

## 架构级问题

### 1. `query_sync` 与 `query_stream` 逻辑分叉

两个文件实现**同一个 ReAct Agent Loop**，但完全独立维护（171 行 vs 1263 行）。任何 bug fix 必须在两处同步修改，否则行为不一致。

**理想方案**：让 `query_sync` 成为 `query_stream` 的瘦包装（消费事件流），消除重复。

---

## `query_stream.py` 问题

### 2. 函数体量失控（~900 行）

`run_query_stream()` 内嵌 6 个闭包 + 1 个 250+ 行嵌套 async generator `_run_all_tool_calls()`，通过共享变量耦合，构成隐式状态机。

### 3. `_stop_signal` hack

用 `list[str | None]` 模拟可变引用，让嵌套 generator 通过副作用控制外层循环退出，控制流晦涩。

### 4. `txn_committed` 手动事务管理

消息提交必须恰好一次，但 `txn_committed` flag 在 5+ 个分支中检查/设置，遗漏任何路径会导致消息丢失或重复提交。

### 5. 取消 ToolMessage 构造重复 5+ 次（DRY）

```python
ToolMessage(
    tool_call_id=call.id,
    tool_name=str(call.function.name or "").strip() or "Tool",
    content=json.dumps({"status": "cancelled", "reason": "user_interrupt"}, ...),
    is_error=True,
)
```

几乎相同的代码出现在至少 5 处。应提取为 `_make_cancelled_tool_message(call)` 工厂函数。

### 6. Task 生命周期事件重复 3 套

SubagentStart → SubagentProgress → SubagentStop 事件发射出现在：串行 Task、并行 Task 组、AskUser repair 路径，三处几乎相同。

### 7. AskUser 策略内联 100+ 行

`query_sync.py` 已将其提取为 `enforce_ask_user_exclusive_policy()`，但 `query_stream.py` 仍然内联。

### 8. `_drain_usage_events()` + `_drain_hidden_events()` 散落 20+ 处

每个 yield 点前后都需要手动调用，仪式性 boilerplate。

### 9. 串行/并行/流式三条执行路径

`_run_all_tool_calls()` 内部 Task 工具存在三条不同路径，各自处理中断、事件发射、状态管理，嵌套深度达 8-9 层。

---

## `query_sync.py` 问题

### 10. 缺少中断支持

整个文件没有 interrupt 检查。用于交互式场景时，用户无法优雅停止。

### 11. 并行 Task `_run` 闭包定义在循环内

目前安全，但未来重构时有变量捕获陷阱风险。

---

## 改进优先级

| 优先级 | 改进项 | 对应问题 |
|--------|--------|---------|
| P0 | 提取 `_make_cancelled_tool_message()` | #5 |
| P0 | 将 `_run_all_tool_calls` 从闭包提升为独立类/函数 | #2 |
| P1 | 提取 `TaskLifecycleEmitter` 统一生命周期事件 | #6 |
| P1 | 用 context manager 替代 `txn_committed` | #4 |
| P1 | `query_stream` 复用 `enforce_ask_user_exclusive_policy` | #7 |
| P2 | 封装 drain 调用为中间件 | #8 |
| P2 | `query_sync` 改为消费 `query_stream` 事件流 | #1 |
| P2 | 用 `nonlocal` 或 dataclass 替代 `_stop_signal` | #3 |
