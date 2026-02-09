from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from comate_agent_sdk.agent.history import destroy_ephemeral_messages
from comate_agent_sdk.agent.llm import invoke_llm
from comate_agent_sdk.agent.tool_exec import execute_tool_call
from comate_agent_sdk.context import SelectiveCompactionPolicy
from comate_agent_sdk.context.offload import OffloadPolicy
from comate_agent_sdk.llm.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from comate_agent_sdk.llm.views import ChatInvokeCompletion

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime

_SUMMARY_FAILURE_COOLDOWN_SECONDS = 8.0
_SUMMARY_FAILURE_STREAK_FOR_COOLDOWN = 2


def _build_compaction_meta_events(
    agent: "AgentRuntime",
    policy: SelectiveCompactionPolicy,
) -> list["CompactionMetaEvent"]:
    if not getattr(agent, "emit_compaction_meta_events", False):
        return []
    if not policy.meta_records:
        return []

    from comate_agent_sdk.agent.events import CompactionMetaEvent

    return [
        CompactionMetaEvent(
            phase=record.phase,
            tokens_before=record.tokens_before,
            tokens_after=record.tokens_after,
            tool_blocks_kept=record.tool_blocks_kept,
            tool_blocks_dropped=record.tool_blocks_dropped,
            tool_calls_truncated=record.tool_calls_truncated,
            tool_results_truncated=record.tool_results_truncated,
            reason=record.reason,
        )
        for record in policy.meta_records
    ]


def _log_compaction_meta_events(events: list["CompactionMetaEvent"]) -> None:
    for event in events:
        logger.debug(f"Compaction meta event: {event}")


def _extract_summary_failure_reason(policy: SelectiveCompactionPolicy) -> str | None:
    if not policy.meta_records:
        return None
    last = policy.meta_records[-1]
    if last.phase != "rollback":
        return None
    reason = (last.reason or "").strip()
    if not reason.startswith("summary_failed_or_empty"):
        return None
    if ":" in reason:
        return reason.split(":", 1)[1]
    return "summary_failed_or_empty"


def _record_compaction_outcome(agent: "AgentRuntime", policy: SelectiveCompactionPolicy) -> None:
    reason = _extract_summary_failure_reason(policy)
    if reason is None:
        setattr(agent, "_summary_compaction_failure_streak", 0)
        setattr(agent, "_summary_compaction_last_reason", "")
        return

    streak = int(getattr(agent, "_summary_compaction_failure_streak", 0)) + 1
    setattr(agent, "_summary_compaction_failure_streak", streak)
    setattr(agent, "_summary_compaction_last_reason", reason)

    if streak >= _SUMMARY_FAILURE_STREAK_FOR_COOLDOWN:
        until = time.monotonic() + _SUMMARY_FAILURE_COOLDOWN_SECONDS
        setattr(agent, "_summary_compaction_cooldown_until", until)
        logger.warning(
            "Compaction summary repeatedly failed; entering cooldown "
            f"for {_SUMMARY_FAILURE_COOLDOWN_SECONDS:.1f}s (reason={reason}, streak={streak})"
        )


def _is_summary_compaction_cooldown_active(agent: "AgentRuntime") -> bool:
    cooldown_until = float(getattr(agent, "_summary_compaction_cooldown_until", 0.0))
    if cooldown_until <= 0:
        return False
    return time.monotonic() < cooldown_until


def _cooldown_remaining_seconds(agent: "AgentRuntime") -> float:
    cooldown_until = float(getattr(agent, "_summary_compaction_cooldown_until", 0.0))
    return max(0.0, cooldown_until - time.monotonic())


async def generate_max_iterations_summary(agent: "AgentRuntime") -> str:
    """max_iterations 达到上限时，使用 LLM 生成简短总结。"""
    summary_prompt = """The task has reached the maximum number of steps allowed.
Please provide a concise summary of:
1. What was accomplished so far
2. What actions were taken
3. What remains incomplete (if anything)
4. Any partial results or findings

Keep the summary brief but informative."""

    # Add the summary request as a user message temporarily
    temp_item = agent._context.add_message(UserMessage(content=summary_prompt, is_meta=True))

    try:
        # Invoke LLM without tools to get a summary response
        response = await agent.llm.ainvoke(
            messages=agent._context.lower(),
            tools=None,
            tool_choice=None,
        )
        summary = response.content or "Unable to generate summary."
    except Exception as e:
        logger.warning(f"Failed to generate max iterations summary: {e}")
        summary = (
            f"Task stopped after {agent.max_iterations} iterations. "
            "Unable to generate summary due to error."
        )
    finally:
        # Remove the temporary summary prompt
        agent._context.conversation.remove_by_id(temp_item.id)

    return f"[Max iterations reached]\n\n{summary}"


async def check_and_compact(
    agent: "AgentRuntime",
    response: ChatInvokeCompletion,
) -> tuple[bool, "PreCompactEvent | None", list["CompactionMetaEvent"]]:
    """检查 token 使用并在需要时压缩。

    Returns:
        tuple[bool, PreCompactEvent | None, list[CompactionMetaEvent]]:
            (是否执行了压缩, 压缩前事件或None, 调试压缩事件列表)
    """
    if agent._compaction_service is None:
        return False, None, []

    # Update token usage tracking
    agent._compaction_service.update_usage(response.usage)

    # 检查是否需要压缩
    if not await agent._compaction_service.should_compact(agent.llm.model):
        return False, None, []

    # 获取压缩阈值
    threshold = await agent._compaction_service.get_threshold_for_model(agent.llm.model)

    # 使用 token usage 中的实际 total_tokens
    from comate_agent_sdk.agent.compaction.models import TokenUsage
    from comate_agent_sdk.agent.events import PreCompactEvent

    actual_tokens = TokenUsage.from_usage(response.usage).total_tokens

    # 创建 PreCompactEvent
    event = PreCompactEvent(
        current_tokens=actual_tokens,
        threshold=threshold,
        trigger='check',
    )

    if _is_summary_compaction_cooldown_active(agent):
        remaining = _cooldown_remaining_seconds(agent)
        reason = str(getattr(agent, "_summary_compaction_last_reason", "summary_failed_or_empty"))
        logger.warning(
            f"压缩冷却中，跳过本轮压缩: remaining={remaining:.1f}s, reason={reason}"
        )
        return False, event, []

    # 创建/复用 OffloadPolicy
    offload_policy = None
    if agent.offload_enabled and agent._context_fs:
        offload_policy = agent.offload_policy or OffloadPolicy(
            enabled=True,
            token_threshold=agent.offload_token_threshold,
        )

    # 创建选择性压缩策略
    compaction_llm = agent._compaction_service.llm or agent.llm
    is_subagent = bool(getattr(agent, "_is_subagent", False))
    agent_name = getattr(agent, "name", None)
    source_prefix = (
        f"subagent:{agent_name}"
        if is_subagent and agent_name
        else None
    )
    policy = SelectiveCompactionPolicy(
        threshold=threshold,
        llm=compaction_llm,
        fallback_to_full_summary=True,
        fs=agent._context_fs,
        offload_policy=offload_policy,
        token_cost=agent._token_cost,
        level=agent._effective_level,
        source_prefix=source_prefix,
    )

    compacted = await agent._context.auto_compact(
        policy=policy,
        current_total_tokens=actual_tokens,
    )
    _record_compaction_outcome(agent, policy)

    return compacted, event, _build_compaction_meta_events(agent, policy)


async def precheck_and_compact(
    agent: "AgentRuntime",
) -> tuple[bool, "PreCompactEvent | None", list["CompactionMetaEvent"]]:
    """基于 provider-aware 估算值预检查并压缩。

    在工具结果添加后调用,防止下一次 invoke_llm 超限。
    优先使用模型相关计数器估算当前上下文，并附加安全缓冲。

    Returns:
        tuple[bool, PreCompactEvent | None, list[CompactionMetaEvent]]:
            (是否执行了压缩, 压缩前事件或None, 调试压缩事件列表)
    """
    if agent._compaction_service is None:
        return False, None, []

    if not agent._compaction_service.config.enabled:
        return False, None, []

    # 优先使用 provider-aware 的 message 估算（失败时回退 IR 估算）
    lowered_messages = agent._context.lower()
    estimated_tokens = await agent._context.token_counter.count_messages_for_model(
        lowered_messages,
        llm=agent.llm,
        timeout_ms=agent.token_count_timeout_ms,
    )
    if estimated_tokens <= 0:
        estimated_tokens = agent._context.total_tokens

    buffer_ratio = max(0.0, float(agent.precheck_buffer_ratio))
    buffered_tokens = int(estimated_tokens * (1.0 + buffer_ratio))

    # 获取压缩阈值
    threshold = await agent._compaction_service.get_threshold_for_model(agent.llm.model)

    # 如果估算值（含缓冲）未超阈值,无需压缩
    if buffered_tokens < threshold:
        return False, None, []

    logger.info(
        f"预检查触发压缩: 估算 {estimated_tokens} tokens + {buffer_ratio:.1%} 缓冲"
        f" = {buffered_tokens} >= 阈值 {threshold}"
    )

    # 创建 PreCompactEvent
    from comate_agent_sdk.agent.events import PreCompactEvent

    event = PreCompactEvent(
        current_tokens=buffered_tokens,
        threshold=threshold,
        trigger='precheck',
    )

    if _is_summary_compaction_cooldown_active(agent):
        remaining = _cooldown_remaining_seconds(agent)
        reason = str(getattr(agent, "_summary_compaction_last_reason", "summary_failed_or_empty"))
        logger.warning(
            f"压缩冷却中，跳过本轮压缩: remaining={remaining:.1f}s, reason={reason}"
        )
        return False, event, []

    # 复用现有的压缩策略创建逻辑
    offload_policy = None
    if agent.offload_enabled and agent._context_fs:
        offload_policy = agent.offload_policy or OffloadPolicy(
            enabled=True,
            token_threshold=agent.offload_token_threshold,
        )

    compaction_llm = agent._compaction_service.llm or agent.llm
    is_subagent = bool(getattr(agent, "_is_subagent", False))
    agent_name = getattr(agent, "name", None)
    source_prefix = (
        f"subagent:{agent_name}"
        if is_subagent and agent_name
        else None
    )
    policy = SelectiveCompactionPolicy(
        threshold=threshold,
        llm=compaction_llm,
        fallback_to_full_summary=True,
        fs=agent._context_fs,
        offload_policy=offload_policy,
        token_cost=agent._token_cost,
        level=agent._effective_level,
        source_prefix=source_prefix,
    )

    compacted = await agent._context.auto_compact(
        policy=policy,
        current_total_tokens=buffered_tokens,
    )
    _record_compaction_outcome(agent, policy)

    return compacted, event, _build_compaction_meta_events(agent, policy)


async def query(agent: "AgentRuntime", message: str) -> str:
    """非流式执行：发送消息并返回最终文本。"""
    # Add the user message to context
    agent._context.add_message(UserMessage(content=message))

    # 注册初始 TODO 提醒（如果需要）
    agent._context.register_initial_todo_reminder_if_needed()

    iterations = 0

    while iterations < agent.max_iterations:
        iterations += 1

        # Destroy ephemeral messages from previous iteration before LLM sees them again
        destroy_ephemeral_messages(agent)

        # Invoke the LLM
        response = await invoke_llm(agent)

        # Add assistant message to history
        assistant_msg = AssistantMessage(
            content=response.content,
            tool_calls=response.tool_calls if response.tool_calls else None,
        )
        agent._context.add_message(assistant_msg)

        # If no tool calls, check if should finish
        if not response.has_tool_calls:
            compacted, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
            if pre_compact_event:
                logger.info(f"Pre-compact event: {pre_compact_event}")
            _log_compaction_meta_events(compaction_meta_events)
            return response.content or ""

        # Execute tool calls (Task calls can run in parallel when contiguous)
        tool_calls = response.tool_calls
        idx = 0
        while idx < len(tool_calls):
            tool_call = tool_calls[idx]
            tool_name = tool_call.function.name

            # Parallelize contiguous Task tool calls only (avoid reordering across other tools)
            if agent.task_parallel_enabled and tool_name == "Task":
                group: list[ToolCall] = []
                while idx < len(tool_calls) and tool_calls[idx].function.name == "Task":
                    group.append(tool_calls[idx])
                    idx += 1

                semaphore = asyncio.Semaphore(max(1, int(agent.task_parallel_max_concurrency)))

                async def _run(tc: ToolCall) -> tuple[str, ToolMessage]:
                    async with semaphore:
                        try:
                            return tc.id, await execute_tool_call(agent, tc)
                        except asyncio.CancelledError:
                            # 用户取消 - 传播以触发 TaskGroup 取消其他任务
                            raise
                        except Exception as e:
                            # 业务异常 - 返回错误消息，不影响其他 subagent
                            logger.warning(f"Subagent {tc.id} failed: {e}")
                            error_msg = ToolMessage(
                                tool_call_id=tc.id,
                                content=f"[Subagent Error] {type(e).__name__}: {e}",
                            )
                            return tc.id, error_msg

                tasks_map: dict[str, asyncio.Task[tuple[str, ToolMessage]]] = {}
                async with asyncio.TaskGroup() as tg:
                    for tc in group:
                        tasks_map[tc.id] = tg.create_task(_run(tc))

                results_by_id = {tc_id: task.result()[1] for tc_id, task in tasks_map.items()}

                # Write results to context in original order for reproducibility
                for tc in group:
                    tool_result = results_by_id[tc.id]
                    agent._context.add_message(tool_result)

                    # 检查是否有待注入的 Skill items（必须在 ToolMessage 之后注入）
                    if agent._context.has_pending_skill_items:
                        agent._context.flush_pending_skill_items()

                # 新增:并行任务完成后预检查
                compacted, pre_compact_event, compaction_meta_events = await precheck_and_compact(agent)
                if pre_compact_event:
                    logger.info(f"Pre-compact event: {pre_compact_event}")
                _log_compaction_meta_events(compaction_meta_events)
                continue

            # Default: serial execution
            tool_result = await execute_tool_call(agent, tool_call)
            agent._context.add_message(tool_result)

            # 新增:预检查压缩
            compacted, pre_compact_event, compaction_meta_events = await precheck_and_compact(agent)
            if pre_compact_event:
                logger.info(f"Pre-compact event: {pre_compact_event}")
            _log_compaction_meta_events(compaction_meta_events)

            # 检查是否有待注入的 Skill items（必须在 ToolMessage 之后注入）
            if agent._context.has_pending_skill_items:
                agent._context.flush_pending_skill_items()

            idx += 1

        # Check for compaction after tool execution
        compacted, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
        if pre_compact_event:
            logger.info(f"Pre-compact event: {pre_compact_event}")
        _log_compaction_meta_events(compaction_meta_events)

    # Max iterations reached - generate summary of what was accomplished
    return await generate_max_iterations_summary(agent)
