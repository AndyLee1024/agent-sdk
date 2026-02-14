# Hook 系统分析总结

> 生成日期: 2026-02-14

---

## 一、Hook 系统执行流程

### 1.1 配置加载阶段

```
settings.json 文件
    ↓
load_hook_config_from_sources()
    ├── ~/.agent/settings.json        (user)
    ├── .agent/settings.json          (project)
    └── .agent/settings.local.json    (local)
    ↓
HookEngine(config, project_root, session_id)
```

### 1.2 事件触发阶段

```
AgentRuntime.run_hook_event(event_name, **kwargs)
    ↓
HookEngine.run_event(event_name, hook_input)
    ↓
1. 获取配置中该事件的所有 HookMatcherGroup
2. 获取运行时注册的 Python Hook
3. 合并并按 order 排序
    ↓
遍历每个 HookMatcherGroup → _matches_group() 正则匹配
    ↓
执行匹配的 Handler (_execute_handler)
    ├── Python Handler → callback(hook_input)
    └── Command Handler → shell 命令 + stdin/stdout JSON
    ↓
AggregatedHookOutcome 聚合结果
```

### 1.3 结果应用阶段

```
AggregatedHookOutcome
    ├── additional_context → 注入为隐藏用户消息
    ├── permission_decision=deny → 阻止工具执行
    ├── permission_decision=ask → 调用 tool_approval_callback
    └── updated_input → 修改工具参数后执行
```

---

## 二、支持的特性

### 2.1 配置格式

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read|Grep|Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/validate.py",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

### 2.2 Matcher 匹配规则

| Matcher 示例 | 匹配的工具 |
|-------------|-----------|
| `"*"` | 所有工具 |
| `"^Read$"` | 仅 Read |
| `"Read\|Grep\|Bash"` | Read、Grep 或 Bash |
| `"^Bash.*"` | Bash 开头的工具 |

### 2.3 Command Hook

- **输入**: stdin JSON (`hook_input.to_dict()`)
- **输出**: stdout JSON
- **Exit Code**: 0=成功, 2=阻止执行

### 2.4 Python Hook

```python
def my_hook(hook_input: HookInput) -> HookResult | dict | None:
    return HookResult(
        additional_context="额外上下文",
        permission_decision="allow",  # allow/ask/deny
        updated_input={"key": "value"},
        reason="原因",
    )
```

### 2.5 Timeout 支持

| Handler 类型 | timeout 支持 |
|-------------|-------------|
| `command` | ✅ 默认 10 秒，可配置 |
| `python` | ❌ 不支持（需修复） |

### 2.6 HookResult 字段

| 字段 | 说明 |
|------|------|
| `additional_context` | 返回给 LLM 的额外上下文 |
| `permission_decision` | allow / ask / deny |
| `updated_input` | 修改后的工具参数 |
| `reason` | 拒绝/询问的原因 |

---

## 三、发现的问题

### 问题 1: additional_context 消息顺序问题（已修复）

**问题描述**: Hook 返回的 `additional_context` 可能插入到 tool_call 和 tool_result 之间，导致 LLM API 调用失败。

**修复方案**: Tool Barrier + Pending Hook Injections

```python
# Tool barrier 追踪
if isinstance(message, AssistantMessage) and message.tool_calls:
    _inflight_tool_call_ids.add(tool_call.id)  # 打开 barrier
elif isinstance(message, ToolMessage):
    _inflight_tool_call_ids.discard(message.tool_call_id)  # 关闭 barrier
    _flush_pending_hook_injections_if_unblocked()  # flush pending
```

```
有进行中的 tool_call → additional_context 存入 pending
tool_result 返回后 → pending flush 到 conversation
```

### 问题 2: Python Handler 不支持 Timeout

**问题描述**: `handler.type = "python"` 的 hook 不支持 timeout 配置。

**修复建议**:

```python
async def _execute_python_handler(self, handler, hook_input):
    timeout = handler.timeout or 10
    
    result = callback(hook_input)
    if inspect.isawaitable(result):
        result = await asyncio.wait_for(result, timeout=timeout)
```

---

## 四、待优化点

### 4.1 additional_context 在 deny 时多余

当 `permissionDecision = "deny"` 时：
- `reason` 字段已经说明了拒绝原因
- `additional_context` 是多余的
- 但当前实现仍会 pending 并 flush

**建议**: deny 时直接丢弃 additional_context，不需要 pending。

---

## 五、相关代码路径

| 功能 | 文件 |
|------|------|
| Hook 配置加载 | `agent/hooks/loader.py` |
| Hook 引擎执行 | `agent/hooks/engine.py` |
| Hook 模型定义 | `agent/hooks/models.py` |
| Tool Barrier | `context/ir.py` |
| Hook 入口 | `agent/core.py:637` |
| Hook 应用 | `agent/tool_exec.py:228` |
