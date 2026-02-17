# System Reminder 机制深度排查

## 1. 结论摘要

本仓库中你给出的空 todo reminder 文案只有一个来源：

- `comate_agent_sdk/context/nudge.py` 的 `render_reminders()` 空 todo 分支。

`system-reminder` 注入机制当前有两条通道：

1. **静态 Reminder 通道**：`ContextIR.reminders` + `LoweringPipeline._inject_reminders()`
2. **Nudge 通道（当前主用）**：`render_reminders(NudgeState)` 在 lowering 末尾拼接到最后一条 `UserMessage`

你日志中的这条文案来自第 2 条通道。

## 2. 全链路机制（从状态到 prompt）

### 2.1 状态源

- `ContextIR` 保存 `_nudge: NudgeState`
- turn 在真实用户输入 `add_message(UserMessage(..., is_meta=False))` 时递增
- `TodoWrite` 通过 `set_todo_state()` 更新：
  - `todo_active_count`
  - `todo_last_changed_turn`
  - `last_todo_write_turn`

关键路径：

- `comate_agent_sdk/context/ir.py`
- `comate_agent_sdk/agent/tool_exec.py`
- `comate_agent_sdk/system_tools/tools.py` (`TodoWrite`)

### 2.2 注入时机

在 `LoweringPipeline.lower()` 中：

1. 先构造 header + conversation messages
2. 再注入 `context.reminders`
3. 最后调用 `render_reminders(context._nudge)`，把返回文本追加到最后一条 `UserMessage`

关键文件：

- `comate_agent_sdk/context/lower.py`

### 2.3 空 todo 提醒触发条件（排查前）

原始逻辑（排查时）：

- `todo_active_count == 0`
- `gap_empty = turn - todo_last_changed_turn`
- `gap_empty % 8 == 0`
- `cooldown_todo >= 3`

这会导致“空状态持续时的周期性提醒”，典型轮次为 `1/9/17`（或从其他初始 turn 偏移后的等差序列）。

## 3. 为什么会出现“连续三次类似 reminder”

### 3.1 主因：周期策略本身允许重复

空 todo 分支是“按轮次周期触发”，不是“同一 empty 状态只触发一次”。
因此在某些对话节奏里，会多次看到同文案，三次是预期可出现的次数。

仓库里已有对应频率验证脚本：

- `comate_agent_sdk/context/tests/test_todo_empty_reminder_frequency.py`

其中明确展示了第 1、9、17 轮触发（即第三次触发）。

### 3.2 不是同一次 lower() 叠加三条

排查中做了最小复现：同一 `ContextIR` 连续调用 `lower()` 三次，不会在同一次结果里堆叠三条 identical empty reminder。
原因是 `render_reminders()` 触发后会更新 `last_nudge_todo_turn`，下一次调用会被节流。

也就是说，“三次”通常来自**多轮触发**，而不是“一次 prompt 中重复注入三条”。

## 4. 本次修复（方案 2：Balanced）

目标：把空 todo 提醒改为

- 仍保留 `gap % 8 == 0` 节点判断
- 但在**同一 empty 状态只提醒一次**
- 只有 todo 状态再次变化（再次进入 empty）才允许下一次 empty 提醒

### 4.1 代码变更

修改文件：

- `comate_agent_sdk/context/nudge.py`

核心变更：

- 空 todo 分支新增状态门：
  - `empty_state_not_nudged = last_nudge_todo_turn <= todo_last_changed_turn`
- 去掉 empty 分支对 cooldown 的依赖，避免同轮状态切换时被误挡。
- 增加 `turn > 0` 防止初始空状态（turn=0）误触发。

### 4.2 单测变更

修改文件：

- `comate_agent_sdk/context/tests/test_nudge.py`

新增/调整断言：

- 同一 empty 状态在第 17 轮（再次命中 8 的倍数）不再重复提醒
- 状态变化后（再次变空）可以重新提醒

### 4.3 验证结果

已执行：

- `uv run python -m pytest comate_agent_sdk/context/tests/test_nudge.py -q` -> 通过
- `uv run python -m pytest comate_agent_sdk/context/tests/test_todo_integration.py -q` -> 通过

## 5. 风险与后续建议

### 5.1 风险

- 行为语义从“周期提醒”变为“同一空状态一次提醒”，提醒频率会显著降低。
- 若业务希望“长期空 todo 也持续提醒”，需要再定义新策略（例如更长周期 + 上限次数）。

### 5.2 建议

1. 若要进一步稳定，可把 `render_reminders()` 副作用与文本渲染解耦（避免 token 估算路径提前消费 nudge 状态）。
2. 为“主 agent + subagent + resume”补一组端到端回归，覆盖 reminder 可见性与频率。

