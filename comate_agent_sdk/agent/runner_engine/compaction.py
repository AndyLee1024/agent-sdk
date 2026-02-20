from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from comate_agent_sdk.context import SelectiveCompactionPolicy
from comate_agent_sdk.context.offload import OffloadPolicy
from comate_agent_sdk.llm.messages import UserMessage
from comate_agent_sdk.llm.views import ChatInvokeCompletion

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime
    from comate_agent_sdk.agent.events import CompactionMetaEvent, PreCompactEvent

_SUMMARY_FAILURE_COOLDOWN_SECONDS = 8.0
_SUMMARY_FAILURE_STREAK_FOR_COOLDOWN = 2


def _build_compaction_policy(agent: "AgentRuntime", threshold: int) -> SelectiveCompactionPolicy:
    offload_policy = None
    if agent.options.offload_enabled and agent._context_fs:
        offload_policy = agent.options.offload_policy or OffloadPolicy(
            enabled=True,
            token_threshold=agent.options.offload_token_threshold,
        )

    compaction_llm = agent._compaction_service.llm or agent.llm
    is_subagent = bool(getattr(agent, "_is_subagent", False))
    agent_name = getattr(agent, "name", None)
    source_prefix = f"subagent:{agent_name}" if is_subagent and agent_name else None
    return SelectiveCompactionPolicy(
        threshold=threshold,
        llm=compaction_llm,
        fallback_to_full_summary=True,
        fs=agent._context_fs,
        offload_policy=offload_policy,
        token_cost=agent._token_cost,
        level=agent._effective_level,
        source_prefix=source_prefix,
    )


def _build_compaction_meta_events(
    agent: "AgentRuntime",
    policy: SelectiveCompactionPolicy,
) -> list["CompactionMetaEvent"]:
    if not bool(getattr(agent.options, "emit_compaction_meta_events", False)):
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


def log_compaction_meta_events(events: list["CompactionMetaEvent"]) -> None:
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
    return cooldown_until > 0 and time.monotonic() < cooldown_until


def _cooldown_remaining_seconds(agent: "AgentRuntime") -> float:
    cooldown_until = float(getattr(agent, "_summary_compaction_cooldown_until", 0.0))
    return max(0.0, cooldown_until - time.monotonic())


async def generate_max_iterations_summary(agent: "AgentRuntime") -> str:
    summary_prompt = """The task has reached the maximum number of steps allowed.
Please provide a concise summary of:
1. What was accomplished so far
2. What actions were taken
3. What remains incomplete (if anything)
4. Any partial results or findings

Keep the summary brief but informative."""

    temp_item = agent._context.add_message(UserMessage(content=summary_prompt, is_meta=True))

    try:
        response = await agent.llm.ainvoke(
            messages=agent._context.lower(),
            tools=None,
            tool_choice=None,
        )
        summary = response.content or "Unable to generate summary."
    except Exception as exc:
        logger.warning(f"Failed to generate max iterations summary: {exc}")
        summary = (
            f"Task stopped after {agent.options.max_iterations} iterations. "
            "Unable to generate summary due to error."
        )
    finally:
        agent._context.conversation.remove_by_id(temp_item.id)

    return f"[Max iterations reached]\n\n{summary}"


async def _get_threshold(agent: "AgentRuntime") -> int:
    try:
        return await agent._compaction_service.get_threshold_for_model(
            agent.llm.model,
            llm=agent.llm,
        )
    except TypeError:
        return await agent._compaction_service.get_threshold_for_model(agent.llm.model)


def _pre_purge_system_reminders(agent: "AgentRuntime", *, phase: str) -> None:
    purge_func = getattr(agent._context, "purge_system_reminders", None)
    if callable(purge_func):
        purged = int(purge_func(include_persistent=True))
        if purged > 0:
            logger.info(f"Compaction pre-purge removed {purged} system reminders ({phase})")


async def check_and_compact(
    agent: "AgentRuntime",
    response: ChatInvokeCompletion,
) -> tuple[bool, "PreCompactEvent | None", list["CompactionMetaEvent"]]:
    del response
    if agent._compaction_service is None:
        return False, None, []

    tracker = getattr(agent, "_context_usage_tracker", None)
    if tracker is None or not tracker.should_compact_post_response():
        return False, None, []

    from comate_agent_sdk.agent.events import PreCompactEvent

    threshold = await _get_threshold(agent)
    actual_tokens = tracker.context_usage
    event = PreCompactEvent(current_tokens=actual_tokens, threshold=threshold, trigger="check")

    if _is_summary_compaction_cooldown_active(agent):
        remaining = _cooldown_remaining_seconds(agent)
        reason = str(getattr(agent, "_summary_compaction_last_reason", "summary_failed_or_empty"))
        logger.warning(f"压缩冷却中，跳过本轮压缩: remaining={remaining:.1f}s, reason={reason}")
        return False, event, []

    _pre_purge_system_reminders(agent, phase="check")
    policy = _build_compaction_policy(agent, threshold)
    compacted = await agent._context.auto_compact(policy=policy, current_total_tokens=actual_tokens)
    _record_compaction_outcome(agent, policy)
    if compacted:
        tracker.reset_after_compaction()

    return compacted, event, _build_compaction_meta_events(agent, policy)


async def precheck_and_compact(
    agent: "AgentRuntime",
) -> tuple[bool, "PreCompactEvent | None", list["CompactionMetaEvent"]]:
    if agent._compaction_service is None or not agent._compaction_service.config.enabled:
        return False, None, []

    tracker = getattr(agent, "_context_usage_tracker", None)
    if tracker is None:
        return False, None, []

    ir_total = agent._context.total_tokens
    if not tracker.should_compact_precheck(ir_total):
        return False, None, []

    from comate_agent_sdk.agent.events import PreCompactEvent

    estimated_tokens = tracker.estimate_precheck(ir_total)
    threshold = await _get_threshold(agent)
    logger.info(f"预检查触发压缩: 估算 {estimated_tokens} tokens >= 阈值 {threshold}")
    event = PreCompactEvent(current_tokens=estimated_tokens, threshold=threshold, trigger="precheck")

    if _is_summary_compaction_cooldown_active(agent):
        remaining = _cooldown_remaining_seconds(agent)
        reason = str(getattr(agent, "_summary_compaction_last_reason", "summary_failed_or_empty"))
        logger.warning(f"压缩冷却中，跳过本轮压缩: remaining={remaining:.1f}s, reason={reason}")
        return False, event, []

    _pre_purge_system_reminders(agent, phase="precheck")
    policy = _build_compaction_policy(agent, threshold)
    compacted = await agent._context.auto_compact(policy=policy, current_total_tokens=estimated_tokens)
    _record_compaction_outcome(agent, policy)
    if compacted:
        tracker.reset_after_compaction()

    return compacted, event, _build_compaction_meta_events(agent, policy)
