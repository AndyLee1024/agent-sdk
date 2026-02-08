# 上下文压缩策略说明（当前实现）

本文档描述 `comate_agent_sdk` 当前生效的压缩行为。

## 1. 总体流程

压缩分两段触发：

1. `precheck`：工具结果写入后、下一次 LLM 调用前
2. `check`：每次 LLM 返回后，使用真实 usage 判定

两段都执行同一个 `SelectiveCompactionPolicy`，并默认启用：

- 先做选择性压缩
- 再始终执行 LLM summary
- 任一步失败回滚到压缩前快照

## 2. 工具块压缩（核心）

入口：`comate_agent_sdk/context/compaction.py` 的 `_truncate_tool_blocks()`

### 2.1 工具块定义

- 一个工具块 = `assistant(tool_calls)` + 对应 `tool_result` 集合

### 2.2 保留与删除规则

- 全局按时间顺序提取工具块
- 保留最近 `5` 块
- 更早块整块删除（assistant/tool 一起删除）
- 不再生成 `OFFLOAD_PLACEHOLDER`

### 2.3 保留块内字段截断

- `tool_call.arguments`：仅当 token `> 500` 才截断
- `tool_result.content`：仅当 token `> 600` 才截断
- 截断后保留前 `200` tokens，并追加：

```text
[TRUNCATED original~N tokens]
```

未超过阈值的字段保持原文。

## 3. user/assistant 压缩保底（按轮）

入口：`SelectiveCompactionPolicy._collect_recent_round_protected_ids()`

- 轮次定义：真实 `UserMessage(is_meta=False)` 到下一轮开始前
- `is_meta=True` 不计入轮次
- 最近 `12` 轮内的 `user/assistant` 不删除
- 仅超过 12 轮的部分才参与 `USER_MESSAGE` / `ASSISTANT_MESSAGE` 的类型压缩

## 4. 事务一致性

入口：`SelectiveCompactionPolicy.compact()`

压缩开始前会深拷贝 `conversation.items` 快照。

任一场景失败都会回滚：

- 选择性压缩异常
- summary 调用异常
- summary 返回空结果
- summary 执行失败（`compacted=False` 或无可用摘要）

仅在 summary 成功时提交结果。

### 4.1 summary 失败恢复

- summary 失败会自动重试（默认 2 次重试，总计最多 3 次尝试）
- 重试仍失败时才回滚
- 失败原因会写入 rollback reason（如 `content_none`、`empty_summary`、`max_tokens_no_content`、`thinking_only_no_content`）

## 5. 调试事件（默认关闭）

新增事件：`CompactionMetaEvent`（`comate_agent_sdk/agent/events.py`）

字段：

- `phase`: `selective_start | selective_done | summary_start | summary_done | rollback`
- `tokens_before` / `tokens_after`
- `tool_blocks_kept` / `tool_blocks_dropped`
- `tool_calls_truncated` / `tool_results_truncated`
- `reason`

配置开关：

- `AgentConfig.emit_compaction_meta_events = False`
- `RuntimeAgentOptions.emit_compaction_meta_events = False`

开启后：

- `query_stream()` 会额外产出 `CompactionMetaEvent`
- 非流式 `query()` 仅写 debug 日志，不改变用户可见返回

## 6. 关键默认值

- `tool_blocks_keep_recent = 5`
- `tool_call_threshold = 500`
- `tool_result_threshold = 600`
- `preview_tokens = 200`
- `dialogue_rounds_keep_min = 12`
- summary：始终执行

## 7. 排查建议

建议优先观察：

- `PreCompactEvent`（触发来源：`precheck/check`）
- `CompactionMetaEvent`（需开启开关）
- 日志关键字：`开始选择性压缩`、`执行 LLM Summary`、`压缩失败，已回滚`

当 summary 连续失败时，runner 会进入短冷却窗口（默认 8 秒），期间跳过重复压缩尝试并打印冷却日志，避免高频无效重试。
