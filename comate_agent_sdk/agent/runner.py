from __future__ import annotations

import asyncio
import json
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

    from comate_agent_sdk.agent.events import PreCompactEvent

    actual_tokens = int(response.usage.total_tokens) if response.usage else 0

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

    estimate = None
    estimated_tokens = agent._context.total_tokens
    buffer_ratio = max(0.0, float(agent.precheck_buffer_ratio))
    buffered_tokens = int(estimated_tokens * (1.0 + buffer_ratio))

    token_accounting = getattr(agent, "_token_accounting", None)
    if token_accounting is not None:
        estimate = await token_accounting.estimate_next_step(
            context=agent._context,
            llm=agent.llm,
            tool_definitions=agent.tool_definitions,
            timeout_ms=agent.token_count_timeout_ms,
        )
        estimated_tokens = estimate.raw_total_tokens
        buffered_tokens = estimate.buffered_tokens
        buffer_ratio = estimate.safety_margin_ratio
    else:
        lowered_messages = agent._context.lower()
        estimated_tokens = await agent._context.token_counter.count_messages_for_model(
            lowered_messages,
            llm=agent.llm,
            timeout_ms=agent.token_count_timeout_ms,
        )
        if estimated_tokens <= 0:
            estimated_tokens = agent._context.total_tokens
        buffered_tokens = int(estimated_tokens * (1.0 + buffer_ratio))

    # 获取压缩阈值
    threshold = await agent._compaction_service.get_threshold_for_model(agent.llm.model)

    # 如果估算值（含缓冲）未超阈值,无需压缩
    if buffered_tokens < threshold:
        return False, None, []

    if estimate is not None:
        logger.info(
            f"预检查触发压缩: 估算 {estimated_tokens} tokens × ratio {estimate.calibration_ratio:.3f} "
            f"+ {buffer_ratio:.1%} 缓冲 = {buffered_tokens} >= 阈值 {threshold}"
        )
    else:
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


async def _fire_session_start_if_needed(agent: "AgentRuntime") -> None:
    if agent._hooks_session_started:
        return
    await agent.run_hook_event("SessionStart")
    agent._hooks_session_started = True


async def _fire_user_prompt_submit(agent: "AgentRuntime", message: str) -> None:
    await agent.run_hook_event("UserPromptSubmit", prompt=message)


def _resolve_subagent_hook_payload(tool_call: ToolCall) -> tuple[str, str]:
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        args = {}

    subagent_name = args.get("subagent_type", "Task")
    description = args.get("description", subagent_name)

    if not isinstance(subagent_name, str) or not subagent_name.strip():
        subagent_name = "Task"
    if not isinstance(description, str) or not description.strip():
        description = subagent_name
    return subagent_name.strip(), description.strip()


async def _fire_subagent_hook(
    agent: "AgentRuntime",
    *,
    event_name: str,
    tool_call: ToolCall,
    subagent_status: str | None = None,
) -> None:
    subagent_name, description = _resolve_subagent_hook_payload(tool_call)
    await agent.run_hook_event(
        event_name,
        tool_call_id=tool_call.id,
        subagent_name=subagent_name,
        subagent_description=description,
        subagent_status=subagent_status,
    )


async def _maybe_block_stop(agent: "AgentRuntime", stop_reason: str) -> bool:
    outcome = await agent.run_hook_event(
        "Stop",
        stop_reason=stop_reason,
        stop_hook_active=agent._stop_hook_block_count > 0,
    )
    if outcome is None or not outcome.should_block_stop:
        agent.reset_stop_hook_state()
        return False

    should_continue = agent.mark_stop_blocked_once()
    block_reason = outcome.reason or f"Stop blocked by hook: {stop_reason}"
    agent.add_hidden_user_message(block_reason)
    agent.drain_hidden_user_messages()
    if not should_continue:
        logger.warning(
            f"Stop hook blocked {agent._stop_hook_block_count} times, forcing stop (reason={stop_reason})"
        )
        agent.reset_stop_hook_state()
        return False
    return True


async def query(agent: "AgentRuntime", message: str) -> str:
    """非流式执行：发送消息并返回最终文本。"""
    await _fire_session_start_if_needed(agent)
    agent.reset_stop_hook_state()

    # Add the user message to context
    agent._context.add_message(UserMessage(content=message))

    await _fire_user_prompt_submit(agent, message)

    iterations = 0
    askuser_repair_attempted = False

    while True:
        if iterations >= agent.max_iterations:
            summary = await generate_max_iterations_summary(agent)
            if await _maybe_block_stop(agent, "max_iterations"):
                iterations = max(0, agent.max_iterations - 1)
                continue
            return summary

        iterations += 1

        # Destroy ephemeral messages from previous iteration before LLM sees them again
        destroy_ephemeral_messages(agent)

        # Invoke the LLM
        response = await invoke_llm(agent)

        # If no tool calls, check if should finish
        if not response.has_tool_calls:
            assistant_msg = AssistantMessage(
                content=response.content,
                tool_calls=None,
            )
            agent._context.add_message(assistant_msg)
            compacted, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
            if pre_compact_event:
                logger.info(f"Pre-compact event: {pre_compact_event}")
            _log_compaction_meta_events(compaction_meta_events)
            if await _maybe_block_stop(agent, "completed"):
                continue
            return response.content or ""

        tool_calls = list(response.tool_calls or [])

        def _is_ask_user_question(call: ToolCall) -> bool:
            return str(call.function.name or "").strip() == "AskUserQuestion"

        def _policy_violation_message(
            *,
            call: ToolCall,
            code: str,
            message: str,
            required_fix: str,
        ) -> ToolMessage:
            content = json.dumps(
                {
                    "error": {
                        "code": code,
                        "message": message,
                        "required_fix": required_fix,
                    }
                },
                ensure_ascii=False,
            )
            return ToolMessage(
                tool_call_id=call.id,
                tool_name=str(call.function.name or "").strip() or "Tool",
                content=content,
                is_error=True,
            )

        has_ask = any(_is_ask_user_question(call) for call in tool_calls)
        has_other = any(not _is_ask_user_question(call) for call in tool_calls)
        if has_ask and has_other:
            if not askuser_repair_attempted:
                askuser_repair_attempted = True

                assistant_msg = AssistantMessage(
                    content=response.content,
                    tool_calls=tool_calls,
                )
                expected_ids = [call.id for call in tool_calls if str(call.id).strip()]
                agent._context.begin_tool_barrier(expected_ids)

                results_by_id: dict[str, ToolMessage] = {}
                for call in tool_calls:
                    if _is_ask_user_question(call):
                        results_by_id[call.id] = _policy_violation_message(
                            call=call,
                            code="ASKUSER_EXCLUSIVE",
                            message="AskUserQuestion must be called alone in a single assistant tool_calls response.",
                            required_fix="Retry now with ONLY AskUserQuestion. Run other tools after the user answer in the next turn.",
                        )
                    else:
                        results_by_id[call.id] = _policy_violation_message(
                            call=call,
                            code="ASKUSER_BLOCKED_BY_ASK",
                            message="Blocked because AskUserQuestion was present in the same tool_calls response.",
                            required_fix="Retry without mixing tools with AskUserQuestion.",
                        )

                agent._context.add_messages_atomic(
                    [assistant_msg] + [results_by_id[call.id] for call in tool_calls]
                )
                agent.drain_hidden_user_messages()
                continue

            ask_call = next((call for call in tool_calls if _is_ask_user_question(call)), None)
            if ask_call is not None:
                tool_calls = [ask_call]

        assistant_msg = AssistantMessage(
            content=response.content,
            tool_calls=tool_calls,
        )

        expected_ids = [call.id for call in tool_calls if str(call.id).strip()]
        agent._context.begin_tool_barrier(expected_ids)

        results_by_id: dict[str, ToolMessage] = {}

        # Execute tool calls (Task calls can run in parallel when contiguous)
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

                for tc in group:
                    await _fire_subagent_hook(agent, event_name="SubagentStart", tool_call=tc)

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

                group_results_by_id = {tc_id: task.result()[1] for tc_id, task in tasks_map.items()}

                # Record results by id in original order for reproducibility
                for tc in group:
                    tool_result = group_results_by_id[tc.id]
                    await _fire_subagent_hook(
                        agent,
                        event_name="SubagentStop",
                        tool_call=tc,
                        subagent_status="error" if tool_result.is_error else "completed",
                    )
                    results_by_id[tc.id] = tool_result
                continue

            # Default: serial execution
            if tool_name == "Task":
                await _fire_subagent_hook(agent, event_name="SubagentStart", tool_call=tool_call)

            tool_result = await execute_tool_call(agent, tool_call)
            results_by_id[tool_call.id] = tool_result
            if tool_name == "Task":
                await _fire_subagent_hook(
                    agent,
                    event_name="SubagentStop",
                    tool_call=tool_call,
                    subagent_status="error" if tool_result.is_error else "completed",
                )

            idx += 1

        agent._context.add_messages_atomic(
            [assistant_msg] + [results_by_id[call.id] for call in tool_calls]
        )
        agent.drain_hidden_user_messages()

        if len(tool_calls) == 1 and _is_ask_user_question(tool_calls[0]):
            tool_result = results_by_id.get(tool_calls[0].id)
            if tool_result is not None and not tool_result.is_error:
                if await _maybe_block_stop(agent, "waiting_for_input"):
                    continue
                return tool_result.text

        # Check for compaction after tool execution
        compacted, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
        if pre_compact_event:
            logger.info(f"Pre-compact event: {pre_compact_event}")
        _log_compaction_meta_events(compaction_meta_events)
