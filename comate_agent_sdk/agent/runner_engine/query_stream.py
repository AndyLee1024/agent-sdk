from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from comate_agent_sdk.agent.events import (
    AgentEvent,
    StepCompleteEvent,
    StepStartEvent,
    StopEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    SubagentToolCallEvent,
    SubagentToolResultEvent,
    TextEvent,
    ThinkingEvent,
    TodoUpdatedEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageDeltaEvent,
    UserQuestionEvent,
)
from comate_agent_sdk.agent.history import destroy_ephemeral_messages
from comate_agent_sdk.agent.llm import invoke_llm
from comate_agent_sdk.agent.tool_exec import _coerce_tool_arguments, execute_tool_call, extract_screenshot
from comate_agent_sdk.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from .compaction import check_and_compact, generate_max_iterations_summary, precheck_and_compact
from .lifecycle import (
    drain_hidden_events as _drain_hidden_events,
    fire_session_start_if_needed as _fire_session_start_if_needed,
    fire_user_prompt_submit as _fire_user_prompt_submit,
    is_interrupt_requested as _is_interrupt_requested,
)
from .stop_policy import should_continue_after_stop_block
from .tool_policy import (
    is_ask_user_question,
    policy_violation_message,
)
from .tool_result_projection import extract_diff_metadata, extract_questions, extract_todos

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime
    from comate_agent_sdk.tokens.views import TokenUsageEntry


@dataclass
class _RunningTaskState:
    tool_call_id: str
    subagent_name: str
    description: str
    source_prefix: str
    started_at_monotonic: float
    tokens: int = 0
    usage_tracking_mode: Literal["queue", "stream"] = "queue"


@dataclass
class _TaskStreamOut:
    """Result holder for _iter_task_stream_events."""

    tool_message: ToolMessage | None = None


def _source_matches_prefix(*, source: str, source_prefix: str) -> bool:
    if not source or not source_prefix:
        return False
    return source == source_prefix or source.startswith(f"{source_prefix}:")


def _usage_belongs_to_task(event: UsageDeltaEvent, state: _RunningTaskState) -> bool:
    return _source_matches_prefix(source=event.source, source_prefix=state.source_prefix)


def _usage_delta_tokens_for_progress(event: UsageDeltaEvent) -> int:
    """Subagent progress uses effective token consumption.

    We intentionally exclude cached prompt tokens to avoid inflated progress
    numbers when prompt cache hits are high.
    """
    prompt_tokens = max(int(event.delta_prompt_tokens), 0)
    cached_prompt_tokens = max(int(event.delta_prompt_cached_tokens), 0)
    completion_tokens = max(int(event.delta_completion_tokens), 0)
    prompt_new_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    return prompt_new_tokens + completion_tokens


def _resolve_task_metadata(
    *,
    tool_call_id: str,
    args: dict,
) -> tuple[str, str, str]:
    subagent_name = ""
    description = ""

    raw_name = args.get("subagent_type")
    if isinstance(raw_name, str):
        subagent_name = raw_name.strip()

    raw_desc = args.get("description")
    if isinstance(raw_desc, str):
        description = raw_desc.strip()

    if not subagent_name:
        subagent_name = "Task"
    if not description:
        description = subagent_name

    source_prefix = f"subagent:{subagent_name}:{tool_call_id}"
    return subagent_name, description, source_prefix


def _normalize_tool_call_event_args(
    agent: "AgentRuntime",
    *,
    tool_name: str,
    raw_args: object,
) -> dict:
    if not isinstance(raw_args, dict):
        return {"_raw": str(raw_args)}

    tool = agent._tool_map.get(tool_name)
    if tool is None:
        return raw_args

    schema = tool.definition.parameters
    if not isinstance(schema, dict):
        return raw_args

    try:
        coerced = _coerce_tool_arguments(raw_args, schema)
    except Exception as exc:
        logger.debug(f"Failed to normalize ToolCallEvent args for {tool_name}: {exc}")
        return raw_args

    if isinstance(coerced, dict):
        return coerced
    return raw_args


async def _run_stop_hook(agent: "AgentRuntime", stop_reason: str) -> bool:
    return await should_continue_after_stop_block(
        agent,
        stop_reason,
        drain_hidden_immediately=False,
    )


async def _iter_task_stream_events(
    agent: "AgentRuntime",
    tool_call: ToolCall,
    subagent_name: str,
    description: str,
    current_task_state: _RunningTaskState | None,
    out: _TaskStreamOut,
    *,
    check_interrupt: Callable[[], bool] | None = None,
) -> AsyncIterator[AgentEvent]:
    task_tool = agent._tool_map.get("Task")
    create_runtime_func = getattr(task_tool, "_create_subagent_runtime", None)
    if create_runtime_func is None:
        out.tool_message = ToolMessage(
            tool_call_id=tool_call.id,
            tool_name="Task",
            content="Error: Task tool is not properly configured for streaming",
            is_error=True,
        )
        return

    try:
        parsed_args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        parsed_args = {"_raw": tool_call.function.arguments}
    args = parsed_args if isinstance(parsed_args, dict) else {"_raw": tool_call.function.arguments}

    raw_subagent_type = args.get("subagent_type", "")
    subagent_type = raw_subagent_type if isinstance(raw_subagent_type, str) else str(raw_subagent_type)
    raw_prompt = args.get("prompt", "")
    prompt = raw_prompt if isinstance(raw_prompt, str) else str(raw_prompt)

    try:
        subagent_runtime, agent_def = await create_runtime_func(subagent_type, tool_call.id)
        if current_task_state is not None:
            current_task_state.usage_tracking_mode = "stream"
            runtime_source_prefix = getattr(subagent_runtime, "_subagent_source_prefix", None)
            if isinstance(runtime_source_prefix, str) and runtime_source_prefix:
                current_task_state.source_prefix = runtime_source_prefix

        result_text = ""
        tool_call_times: dict[str, float] = {}
        timeout = agent_def.timeout if agent_def and agent_def.timeout else None
        timeout_ctx = asyncio.timeout(timeout) if timeout else nullcontext()

        async with timeout_ctx:
            async for event in subagent_runtime.query_stream(prompt):
                if check_interrupt and check_interrupt():
                    out.tool_message = ToolMessage(
                        tool_call_id=tool_call.id,
                        tool_name="Task",
                        content=f"Error: Subagent '{subagent_name}' cancelled due to user interrupt",
                        is_error=True,
                    )
                    return

                if isinstance(event, UsageDeltaEvent):
                    if current_task_state is not None and _usage_belongs_to_task(event, current_task_state):
                        delta_tokens = _usage_delta_tokens_for_progress(event)
                        current_task_state.tokens += delta_tokens
                        elapsed_ms = (time.monotonic() - current_task_state.started_at_monotonic) * 1000
                        yield SubagentProgressEvent(
                            tool_call_id=tool_call.id,
                            subagent_name=subagent_name,
                            description=description,
                            status="running",
                            elapsed_ms=elapsed_ms,
                            tokens=current_task_state.tokens,
                        )
                    continue

                if isinstance(event, ToolCallEvent):
                    tool_call_times[event.tool_call_id] = time.time()
                    yield SubagentToolCallEvent(
                        parent_tool_call_id=tool_call.id,
                        subagent_name=subagent_name,
                        tool=event.tool,
                        args=event.args,
                        tool_call_id=event.tool_call_id,
                    )
                    continue

                if isinstance(event, ToolResultEvent):
                    start_time = tool_call_times.get(event.tool_call_id, time.time())
                    duration_ms = (time.time() - start_time) * 1000
                    yield SubagentToolResultEvent(
                        parent_tool_call_id=tool_call.id,
                        subagent_name=subagent_name,
                        tool=event.tool,
                        tool_call_id=event.tool_call_id,
                        is_error=event.is_error,
                        duration_ms=duration_ms,
                    )
                    continue

                if isinstance(event, TextEvent):
                    result_text += event.content

        out.tool_message = ToolMessage(
            tool_call_id=tool_call.id,
            tool_name="Task",
            content=result_text or "(no output)",
            is_error=False,
        )
    except asyncio.TimeoutError:
        out.tool_message = ToolMessage(
            tool_call_id=tool_call.id,
            tool_name="Task",
            content=f"Error: Subagent '{subagent_name}' timeout after {timeout}s",
            is_error=True,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error_msg = f"Error in subagent '{subagent_name}': {exc}"
        logger.error(error_msg, exc_info=True)
        out.tool_message = ToolMessage(
            tool_call_id=tool_call.id,
            tool_name="Task",
            content=error_msg,
            is_error=True,
        )


async def run_query_stream(
    agent: "AgentRuntime", message: str | list[ContentPartTextParam | ContentPartImageParam]
) -> AsyncIterator[AgentEvent]:
    """流式执行：发送消息并逐步产出事件。"""
    usage_queue: asyncio.Queue[UsageDeltaEvent] = asyncio.Queue()
    subagent_event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    running_tasks: dict[str, _RunningTaskState] = {}
    usage_poll_interval_seconds = 0.08

    def _on_usage_entry(entry: "TokenUsageEntry") -> None:
        source = entry.source or "unknown"
        usage = entry.usage
        usage_queue.put_nowait(
            UsageDeltaEvent(
                source=source,
                model=entry.model,
                level=entry.level,
                delta_prompt_tokens=int(usage.prompt_tokens),
                delta_prompt_cached_tokens=int(usage.prompt_cached_tokens or 0),
                delta_completion_tokens=int(usage.completion_tokens),
                delta_total_tokens=int(usage.total_tokens),
            )
        )

    async def _drain_usage_events() -> AsyncIterator[AgentEvent]:
        while True:
            try:
                event = usage_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            yield event
            delta_tokens = _usage_delta_tokens_for_progress(event)
            if delta_tokens <= 0:
                continue
            for state in running_tasks.values():
                # 流式 Task 的 token 由子流直接累计，避免在全局 usage 队列中重复累计。
                if state.usage_tracking_mode == "stream":
                    continue
                if not _usage_belongs_to_task(event, state):
                    continue
                state.tokens += delta_tokens
                elapsed_ms = (time.monotonic() - state.started_at_monotonic) * 1000
                yield SubagentProgressEvent(
                    tool_call_id=state.tool_call_id,
                    subagent_name=state.subagent_name,
                    description=state.description,
                    status="running",
                    elapsed_ms=elapsed_ms,
                    tokens=state.tokens,
                )
                break  # 找到匹配的 task 后立即停止，避免重复累加

    async def _drain_subagent_events() -> AsyncIterator[AgentEvent]:
        """排空 subagent 事件队列（用于并行执行时的流式事件）"""
        while True:
            try:
                event = subagent_event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            yield event

    async def _execute_task_streaming(
        agent: AgentRuntime,
        tool_call: ToolCall,
        meta: tuple[str, str],
    ) -> ToolMessage:
        """流式执行 Task 工具（用于并行执行）"""
        subagent_name, description = meta
        current_task_state = running_tasks.get(tool_call.id)
        out = _TaskStreamOut()
        async for event in _iter_task_stream_events(
            agent,
            tool_call,
            subagent_name,
            description,
            current_task_state,
            out,
        ):
            subagent_event_queue.put_nowait(event)
        return out.tool_message or ToolMessage(
            tool_call_id=tool_call.id,
            tool_name="Task",
            content="Error: streaming task produced no result",
            is_error=True,
        )

    async def _yield_interrupted_stop() -> AsyncIterator[AgentEvent]:
        async for usage_event in _drain_usage_events():
            yield usage_event
        async for hidden_event in _drain_hidden_events(agent):
            yield hidden_event
        yield StopEvent(reason="interrupted")

    async def _emit_cancelled_task(tool_call_id: str) -> AsyncIterator[AgentEvent]:
        state = running_tasks.pop(tool_call_id, None)
        elapsed_ms = 0.0
        total_tokens = 0
        subagent_name = "Task"
        description = "Task"
        if state is not None:
            elapsed_ms = (time.monotonic() - state.started_at_monotonic) * 1000
            total_tokens = state.tokens
            subagent_name = state.subagent_name
            description = state.description
            yield SubagentProgressEvent(
                tool_call_id=tool_call_id,
                subagent_name=subagent_name,
                description=description,
                status="cancelled",
                elapsed_ms=elapsed_ms,
                tokens=total_tokens,
            )
            yield SubagentStopEvent(
                tool_call_id=tool_call_id,
                subagent_name=subagent_name,
                status="cancelled",
                duration_ms=elapsed_ms,
                error="Interrupted by user",
            )
            await agent.run_hook_event(
                "SubagentStop",
                tool_call_id=tool_call_id,
                subagent_name=subagent_name,
                subagent_description=description,
                subagent_status="cancelled",
            )
            async for hidden_event in _drain_hidden_events(agent):
                yield hidden_event
        yield StepCompleteEvent(
            step_id=tool_call_id,
            status="cancelled",
            duration_ms=elapsed_ms,
        )

    usage_observer_id = agent._token_cost.subscribe_usage(_on_usage_entry)

    try:
        await _fire_session_start_if_needed(agent)
        agent.reset_stop_hook_state()
        async for hidden_event in _drain_hidden_events(agent):
            yield hidden_event

        # Add the user message to context (supports both string and multi-modal content)
        agent._context.add_message(UserMessage(content=message))
        await _fire_user_prompt_submit(agent, message)
        async for hidden_event in _drain_hidden_events(agent):
            yield hidden_event
        iterations = 0
        askuser_repair_attempted = False
        while True:
            if iterations >= agent.options.max_iterations:
                summary = await generate_max_iterations_summary(agent)
                async for usage_event in _drain_usage_events():
                    yield usage_event
                yield TextEvent(content=summary)
                if await _run_stop_hook(agent, "max_iterations"):
                    async for hidden_event in _drain_hidden_events(agent):
                        yield hidden_event
                    iterations = max(0, agent.options.max_iterations - 1)
                    continue
                async for hidden_event in _drain_hidden_events(agent):
                    yield hidden_event
                yield StopEvent(reason="max_iterations")
                return
            if _is_interrupt_requested(agent):
                async for event in _yield_interrupted_stop():
                    yield event
                return
            iterations += 1
            # Destroy ephemeral messages from previous iteration before LLM sees them again
            destroy_ephemeral_messages(agent)
            # Pre-check compaction before invoking LLM to reduce context overflow risk.
            _, precheck_event, precheck_meta_events = await precheck_and_compact(agent)
            async for usage_event in _drain_usage_events():
                yield usage_event
            if precheck_event:
                yield precheck_event
            for compaction_meta_event in precheck_meta_events:
                yield compaction_meta_event
            # Invoke the LLM (polling to support cooperative interruption)
            llm_task = asyncio.create_task(invoke_llm(agent))
            while True:
                async for usage_event in _drain_usage_events():
                    yield usage_event
                if _is_interrupt_requested(agent):
                    llm_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await llm_task
                    async for event in _yield_interrupted_stop():
                        yield event
                    return
                done, _ = await asyncio.wait(
                    {llm_task},
                    timeout=usage_poll_interval_seconds,
                    return_when=asyncio.ALL_COMPLETED,
                )
                async for usage_event in _drain_usage_events():
                    yield usage_event
                if done:
                    break
            response = await llm_task
            async for usage_event in _drain_usage_events():
                yield usage_event

            # Check for thinking content and yield it
            if response.thinking:
                yield ThinkingEvent(content=response.thinking)

            if _is_interrupt_requested(agent):
                async for event in _yield_interrupted_stop():
                    yield event
                return

            # If no tool calls, finish
            if not response.has_tool_calls:
                from comate_agent_sdk.llm.messages import AssistantMessage

                assistant_msg = AssistantMessage(
                    content=response.raw_content_blocks or response.content,
                    tool_calls=None,
                )
                agent._context.add_message(assistant_msg)
                compacted, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
                async for usage_event in _drain_usage_events():
                    yield usage_event
                if pre_compact_event:
                    yield pre_compact_event
                for compaction_meta_event in compaction_meta_events:
                    yield compaction_meta_event
                if response.content:
                    yield TextEvent(content=response.content)
                if await _run_stop_hook(agent, "completed"):
                    async for hidden_event in _drain_hidden_events(agent):
                        yield hidden_event
                    continue
                async for hidden_event in _drain_hidden_events(agent):
                    yield hidden_event
                yield StopEvent(reason="completed")
                return

            # Yield text content if present alongside tool calls
            if response.content:
                yield TextEvent(content=response.content)

            from comate_agent_sdk.llm.messages import AssistantMessage

            tool_calls = list(response.tool_calls or [])

            has_ask = any(is_ask_user_question(call) for call in tool_calls)
            has_other = any(not is_ask_user_question(call) for call in tool_calls)

            if has_ask and has_other:
                if not askuser_repair_attempted:
                    askuser_repair_attempted = True

                    assistant_msg = AssistantMessage(
                        content=response.raw_content_blocks or response.content,
                        tool_calls=tool_calls,
                    )
                    expected_ids = [call.id for call in tool_calls if str(call.id).strip()]
                    agent._context.begin_tool_barrier(expected_ids)

                    results_by_id: dict[str, ToolMessage] = {}
                    for call in tool_calls:
                        if is_ask_user_question(call):
                            results_by_id[call.id] = policy_violation_message(
                                call=call,
                                code="ASKUSER_EXCLUSIVE",
                                message="AskUserQuestion must be called alone in a single assistant tool_calls response.",
                                required_fix="Retry now with ONLY AskUserQuestion. Run other tools after the user answer in the next turn.",
                            )
                        else:
                            results_by_id[call.id] = policy_violation_message(
                                call=call,
                                code="ASKUSER_BLOCKED_BY_ASK",
                                message="Blocked because AskUserQuestion was present in the same tool_calls response.",
                                required_fix="Retry without mixing tools with AskUserQuestion.",
                            )

                    step_number = 0
                    for call in tool_calls:
                        step_number += 1
                        tool_name = str(call.function.name or "").strip()

                        try:
                            args = json.loads(call.function.arguments)
                        except json.JSONDecodeError:
                            args = {"_raw": call.function.arguments}
                        args_dict = _normalize_tool_call_event_args(
                            agent,
                            tool_name=tool_name,
                            raw_args=args,
                        )

                        is_task = tool_name == "Task"
                        task_subagent_name = "Task"
                        task_description = tool_name
                        if is_task:
                            task_subagent_name, task_description, _ = _resolve_task_metadata(
                                tool_call_id=call.id,
                                args=args_dict,
                            )

                        yield StepStartEvent(
                            step_id=call.id,
                            title=task_description if is_task else tool_name,
                            step_number=step_number,
                        )
                        yield ToolCallEvent(
                            tool=tool_name,
                            args=args_dict,
                            tool_call_id=call.id,
                            display_name=task_description if is_task else tool_name,
                        )

                        if is_task:
                            await agent.run_hook_event(
                                "SubagentStart",
                                tool_call_id=call.id,
                                subagent_name=task_subagent_name,
                                subagent_description=task_description,
                                subagent_status="running",
                            )
                            yield SubagentStartEvent(
                                tool_call_id=call.id,
                                subagent_name=task_subagent_name,
                                description=task_description,
                            )
                            yield SubagentProgressEvent(
                                tool_call_id=call.id,
                                subagent_name=task_subagent_name,
                                description=task_description,
                                status="running",
                                elapsed_ms=0.0,
                                tokens=0,
                            )

                        tool_result = results_by_id[call.id]
                        if is_task:
                            await agent.run_hook_event(
                                "SubagentStop",
                                tool_call_id=call.id,
                                subagent_name=task_subagent_name,
                                subagent_description=task_description,
                                subagent_status="error",
                            )
                            yield SubagentProgressEvent(
                                tool_call_id=call.id,
                                subagent_name=task_subagent_name,
                                description=task_description,
                                status="error",
                                elapsed_ms=0.0,
                                tokens=0,
                            )
                            yield SubagentStopEvent(
                                tool_call_id=call.id,
                                subagent_name=task_subagent_name,
                                status="error",
                                duration_ms=0.0,
                                error=tool_result.text,
                            )

                        yield ToolResultEvent(
                            tool=tool_name,
                            result=tool_result.text,
                            tool_call_id=call.id,
                            is_error=True,
                            screenshot_base64=None,
                        )
                        yield StepCompleteEvent(
                            step_id=call.id,
                            status="error",
                            duration_ms=0.0,
                        )
                        async for hidden_event in _drain_hidden_events(agent):
                            yield hidden_event

                    agent._context.add_messages_atomic(
                        [assistant_msg] + [results_by_id[call.id] for call in tool_calls]
                    )
                    async for hidden_event in _drain_hidden_events(agent):
                        yield hidden_event

                    continue

                ask_call = next((call for call in tool_calls if is_ask_user_question(call)), None)
                if ask_call is not None:
                    tool_calls = [ask_call]

            assistant_msg = AssistantMessage(
                content=response.raw_content_blocks or response.content,
                tool_calls=tool_calls,
            )

            expected_ids = [call.id for call in tool_calls if str(call.id).strip()]
            agent._context.begin_tool_barrier(expected_ids)

            _stop_signal: list[str | None] = [None]

            async def _run_all_tool_calls() -> AsyncIterator[AgentEvent]:
                results_by_id: dict[str, ToolMessage] = {}
                txn_committed = False

                # Execute tool calls, yielding events for each.
                # Contiguous Task tool calls can run in parallel (streamed by completion order),
                # while ToolMessage writes remain in original order for reproducibility.
                step_number = 0
                idx = 0
                while idx < len(tool_calls):
                    if _is_interrupt_requested(agent):
                        for call in tool_calls[idx:]:
                            results_by_id[call.id] = ToolMessage(
                                tool_call_id=call.id,
                                tool_name=str(call.function.name or "").strip() or "Tool",
                                content=json.dumps(
                                    {
                                        "status": "cancelled",
                                        "reason": "user_interrupt",
                                    },
                                    ensure_ascii=False,
                                ),
                                is_error=True,
                            )
                        if not txn_committed:
                            agent._context.add_messages_atomic(
                                [assistant_msg] + [results_by_id[call.id] for call in tool_calls]
                            )
                            txn_committed = True
                        _stop_signal[0] = "interrupted"
                        async for event in _yield_interrupted_stop():
                            yield event
                        return

                    tool_call = tool_calls[idx]
                    tool_name = tool_call.function.name

                    # Parallel Task group (contiguous only)
                    if agent.options.task_parallel_enabled and tool_name == "Task":
                        group: list[ToolCall] = []
                        while idx < len(tool_calls) and tool_calls[idx].function.name == "Task":
                            group.append(tool_calls[idx])
                            idx += 1

                        args_by_id: dict[str, dict] = {}
                        meta_by_id: dict[str, tuple[str, str]] = {}

                        # Emit step start + tool call + subagent start for each call (in call order)
                        for tc in group:
                            step_number += 1

                            try:
                                args = json.loads(tc.function.arguments)
                            except json.JSONDecodeError:
                                args = {"_raw": tc.function.arguments}
                            args_by_id[tc.id] = _normalize_tool_call_event_args(
                                agent,
                                tool_name="Task",
                                raw_args=args,
                            )
                            task_args = args_by_id[tc.id]

                            subagent_name, description, source_prefix = _resolve_task_metadata(
                                tool_call_id=tc.id,
                                args=task_args,
                            )
                            meta_by_id[tc.id] = (subagent_name, description)
                            running_tasks[tc.id] = _RunningTaskState(
                                tool_call_id=tc.id,
                                subagent_name=subagent_name,
                                description=description,
                                source_prefix=source_prefix,
                                started_at_monotonic=time.monotonic(),
                            )
                            await agent.run_hook_event(
                                "SubagentStart",
                                tool_call_id=tc.id,
                                subagent_name=subagent_name,
                                subagent_description=description,
                                subagent_status="running",
                            )

                            yield StepStartEvent(
                                step_id=tc.id,
                                title=description,
                                step_number=step_number,
                            )

                            yield ToolCallEvent(
                                tool="Task",
                                args=args_by_id[tc.id],
                                tool_call_id=tc.id,
                                display_name=description,
                            )

                            yield SubagentStartEvent(
                                tool_call_id=tc.id,
                                subagent_name=subagent_name,
                                description=description,
                            )
                            yield SubagentProgressEvent(
                                tool_call_id=tc.id,
                                subagent_name=subagent_name,
                                description=description,
                                status="running",
                                elapsed_ms=0.0,
                                tokens=0,
                            )
                            async for hidden_event in _drain_hidden_events(agent):
                                yield hidden_event

                        semaphore = asyncio.Semaphore(
                            max(1, int(agent.options.task_parallel_max_concurrency))
                        )

                        # 检查是否启用流式 Task
                        task_tool = agent._tool_map.get("Task")
                        is_streaming = getattr(task_tool, "_is_streaming_task", False)

                        async def _run(tc: ToolCall) -> tuple[ToolCall, ToolMessage, float]:
                            start = time.time()
                            async with semaphore:
                                # 流式执行 Task
                                if is_streaming:
                                    tool_result = await _execute_task_streaming(agent, tc, meta_by_id[tc.id])
                                else:
                                    tool_result = await execute_tool_call(agent, tc)
                            duration_ms = (time.time() - start) * 1000
                            return tc, tool_result, duration_ms

                        futures = [asyncio.create_task(_run(tc)) for tc in group]
                        pending = set(futures)
                        group_results_by_id: dict[str, ToolMessage] = {}

                        while pending:
                            async for usage_event in _drain_usage_events():
                                yield usage_event
                            async for subagent_event in _drain_subagent_events():
                                yield subagent_event

                            if _is_interrupt_requested(agent):
                                for fut in pending:
                                    fut.cancel()
                                await asyncio.gather(*pending, return_exceptions=True)

                                # Mark unfinished task calls as cancelled.
                                completed_ids = set(group_results_by_id.keys())
                                for tc in group:
                                    if tc.id in completed_ids:
                                        continue
                                    cancelled = ToolMessage(
                                        tool_call_id=tc.id,
                                        tool_name="Task",
                                        content=json.dumps(
                                            {"status": "cancelled", "reason": "user_interrupt"},
                                            ensure_ascii=False,
                                        ),
                                        is_error=True,
                                    )
                                    group_results_by_id[tc.id] = cancelled
                                    async for event in _emit_cancelled_task(tc.id):
                                        yield event

                                for tc in group:
                                    results_by_id[tc.id] = group_results_by_id[tc.id]
                                for call in tool_calls[idx:]:
                                    results_by_id[call.id] = ToolMessage(
                                        tool_call_id=call.id,
                                        tool_name=str(call.function.name or "").strip() or "Tool",
                                        content=json.dumps(
                                            {"status": "cancelled", "reason": "user_interrupt"},
                                            ensure_ascii=False,
                                        ),
                                        is_error=True,
                                    )
                                if not txn_committed:
                                    agent._context.add_messages_atomic(
                                        [assistant_msg] + [results_by_id[call.id] for call in tool_calls]
                                    )
                                    txn_committed = True
                                _stop_signal[0] = "interrupted"
                                async for event in _yield_interrupted_stop():
                                    yield event
                                return

                            done, pending = await asyncio.wait(
                                pending,
                                timeout=usage_poll_interval_seconds,
                                return_when=asyncio.FIRST_COMPLETED,
                            )

                            async for usage_event in _drain_usage_events():
                                yield usage_event
                            async for subagent_event in _drain_subagent_events():
                                yield subagent_event

                            if not done:
                                continue

                            for fut in done:
                                tc, tool_result, duration_ms = await fut
                                subagent_name, description = meta_by_id.get(tc.id, ("Task", "Task"))

                                state = running_tasks.pop(tc.id, None)
                                total_tokens = state.tokens if state is not None else 0

                                stop_status: Literal["completed", "error", "timeout", "cancelled"] = "completed"
                                error_msg: str | None = None
                                if tool_result.is_error:
                                    error_msg = tool_result.text
                                    if "timeout" in tool_result.text.lower():
                                        stop_status = "timeout"
                                    elif "cancelled" in tool_result.text.lower():
                                        stop_status = "cancelled"
                                    else:
                                        stop_status = "error"
                                await agent.run_hook_event(
                                    "SubagentStop",
                                    tool_call_id=tc.id,
                                    subagent_name=subagent_name,
                                    subagent_description=description,
                                    subagent_status=stop_status,
                                )

                                yield SubagentProgressEvent(
                                    tool_call_id=tc.id,
                                    subagent_name=subagent_name,
                                    description=description,
                                    status=stop_status,
                                    elapsed_ms=duration_ms,
                                    tokens=total_tokens,
                                )

                                yield SubagentStopEvent(
                                    tool_call_id=tc.id,
                                    subagent_name=subagent_name,
                                    status=stop_status,
                                    duration_ms=duration_ms,
                                    error=error_msg,
                                )

                                screenshot_base64 = extract_screenshot(tool_result)
                                yield ToolResultEvent(
                                    tool="Task",
                                    result=tool_result.text,
                                    tool_call_id=tc.id,
                                    is_error=tool_result.is_error,
                                    screenshot_base64=screenshot_base64,
                                )

                                yield StepCompleteEvent(
                                    step_id=tc.id,
                                    status="error" if tool_result.is_error else "completed",
                                    duration_ms=duration_ms,
                                )
                                async for hidden_event in _drain_hidden_events(agent):
                                    yield hidden_event

                                group_results_by_id[tc.id] = tool_result

                        for tc in group:
                            results_by_id[tc.id] = group_results_by_id[tc.id]

                        continue

                    # Default: serial execution
                    step_number += 1

                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {"_raw": tool_call.function.arguments}
                    args_dict = _normalize_tool_call_event_args(
                        agent,
                        tool_name=tool_name,
                        raw_args=args,
                    )

                    is_task = tool_name == "Task"
                    # 检查 Task 工具是否启用了流式模式
                    is_task_stream = False
                    if is_task:
                        task_tool = agent._tool_map.get("Task")
                        is_task_stream = getattr(task_tool, "_is_streaming_task", False)

                    task_subagent_name = "Task"
                    task_description = tool_name
                    if is_task:
                        task_subagent_name, task_description, task_source_prefix = _resolve_task_metadata(
                            tool_call_id=tool_call.id,
                            args=args_dict,
                        )
                    else:
                        task_source_prefix = ""

                    yield StepStartEvent(
                        step_id=tool_call.id,
                        title=task_description if is_task else tool_name,
                        step_number=step_number,
                    )

                    yield ToolCallEvent(
                        tool=tool_name,
                        args=args_dict,
                        tool_call_id=tool_call.id,
                        display_name=task_description if is_task else tool_name,
                    )

                    if is_task:
                        running_tasks[tool_call.id] = _RunningTaskState(
                            tool_call_id=tool_call.id,
                            subagent_name=task_subagent_name,
                            description=task_description,
                            source_prefix=task_source_prefix,
                            started_at_monotonic=time.monotonic(),
                        )
                        await agent.run_hook_event(
                            "SubagentStart",
                            tool_call_id=tool_call.id,
                            subagent_name=task_subagent_name,
                            subagent_description=task_description,
                            subagent_status="running",
                        )
                        yield SubagentStartEvent(
                            tool_call_id=tool_call.id,
                            subagent_name=task_subagent_name,
                            description=task_description,
                        )
                        yield SubagentProgressEvent(
                            tool_call_id=tool_call.id,
                            subagent_name=task_subagent_name,
                            description=task_description,
                            status="running",
                            elapsed_ms=0.0,
                            tokens=0,
                        )
                        async for hidden_event in _drain_hidden_events(agent):
                            yield hidden_event

                    step_start_time = time.time()

                    # === Task 流式处理：流式执行 ===
                    if is_task_stream:
                        current_task_state = running_tasks.get(tool_call.id)
                        out = _TaskStreamOut()
                        async for event in _iter_task_stream_events(
                            agent,
                            tool_call,
                            task_subagent_name,
                            task_description,
                            current_task_state,
                            out,
                            check_interrupt=lambda: _is_interrupt_requested(agent),
                        ):
                            yield event
                        tool_result = out.tool_message or ToolMessage(
                            tool_call_id=tool_call.id,
                            tool_name="Task",
                            content="Error: streaming task produced no result",
                            is_error=True,
                        )

                    # === Task 或普通工具：调用 execute_tool_call ===
                    else:
                        tool_result_task = asyncio.create_task(execute_tool_call(agent, tool_call))
                        while True:
                            async for usage_event in _drain_usage_events():
                                yield usage_event

                            if _is_interrupt_requested(agent):
                                tool_result_task.cancel()
                                with suppress(asyncio.CancelledError):
                                    await tool_result_task

                                step_duration_ms = (time.time() - step_start_time) * 1000
                                if is_task:
                                    cancelled_result = ToolMessage(
                                        tool_call_id=tool_call.id,
                                        tool_name="Task",
                                        content=json.dumps(
                                            {"status": "cancelled", "reason": "user_interrupt"},
                                            ensure_ascii=False,
                                        ),
                                        is_error=True,
                                    )
                                    results_by_id[tool_call.id] = cancelled_result
                                    async for event in _emit_cancelled_task(tool_call.id):
                                        yield event
                                else:
                                    yield StepCompleteEvent(
                                        step_id=tool_call.id,
                                        status="cancelled",
                                        duration_ms=step_duration_ms,
                                    )
                                    results_by_id[tool_call.id] = ToolMessage(
                                        tool_call_id=tool_call.id,
                                        tool_name=str(tool_name or "").strip() or "Tool",
                                        content=json.dumps(
                                            {"status": "cancelled", "reason": "user_interrupt"},
                                            ensure_ascii=False,
                                        ),
                                        is_error=True,
                                    )

                                for call in tool_calls[idx + 1 :]:
                                    results_by_id[call.id] = ToolMessage(
                                        tool_call_id=call.id,
                                        tool_name=str(call.function.name or "").strip() or "Tool",
                                        content=json.dumps(
                                            {"status": "cancelled", "reason": "user_interrupt"},
                                            ensure_ascii=False,
                                        ),
                                        is_error=True,
                                    )

                                if not txn_committed:
                                    agent._context.add_messages_atomic(
                                        [assistant_msg] + [results_by_id[call.id] for call in tool_calls]
                                    )
                                    txn_committed = True

                                _stop_signal[0] = "interrupted"
                                async for event in _yield_interrupted_stop():
                                    yield event
                                return

                            done, _ = await asyncio.wait(
                                {tool_result_task},
                                timeout=usage_poll_interval_seconds,
                                return_when=asyncio.ALL_COMPLETED,
                            )
                            async for usage_event in _drain_usage_events():
                                yield usage_event
                            if done:
                                break

                        tool_result = await tool_result_task
                    step_duration_ms = (time.time() - step_start_time) * 1000
                    results_by_id[tool_call.id] = tool_result
                    async for hidden_event in _drain_hidden_events(agent):
                        yield hidden_event

                    if is_task:
                        state = running_tasks.pop(tool_call.id, None)
                        total_tokens = state.tokens if state is not None else 0
                        stop_status: Literal["completed", "error", "timeout", "cancelled"] = "completed"
                        error_msg: str | None = None
                        if tool_result.is_error:
                            error_msg = tool_result.text
                            if "timeout" in tool_result.text.lower():
                                stop_status = "timeout"
                            elif "cancelled" in tool_result.text.lower():
                                stop_status = "cancelled"
                            else:
                                stop_status = "error"
                        await agent.run_hook_event(
                            "SubagentStop",
                            tool_call_id=tool_call.id,
                            subagent_name=task_subagent_name,
                            subagent_description=task_description,
                            subagent_status=stop_status,
                        )
                        yield SubagentProgressEvent(
                            tool_call_id=tool_call.id,
                            subagent_name=task_subagent_name,
                            description=task_description,
                            status=stop_status,
                            elapsed_ms=step_duration_ms,
                            tokens=total_tokens,
                        )
                        yield SubagentStopEvent(
                            tool_call_id=tool_call.id,
                            subagent_name=task_subagent_name,
                            status=stop_status,
                            duration_ms=step_duration_ms,
                            error=error_msg,
                        )

                    screenshot_base64 = extract_screenshot(tool_result)
                    diff_metadata = extract_diff_metadata(str(tool_name or ""), tool_result)
                    yield ToolResultEvent(
                        tool=tool_name,
                        result=tool_result.text,
                        tool_call_id=tool_call.id,
                        is_error=tool_result.is_error,
                        screenshot_base64=screenshot_base64,
                        metadata=diff_metadata,
                    )

                    yield StepCompleteEvent(
                        step_id=tool_call.id,
                        status="error" if tool_result.is_error else "completed",
                        duration_ms=step_duration_ms,
                    )
                    async for hidden_event in _drain_hidden_events(agent):
                        yield hidden_event

                    # Check if this was TodoWrite - if so, yield TodoUpdatedEvent
                    if tool_name == "TodoWrite" and not tool_result.is_error:
                        todos = extract_todos(tool_result)
                        if todos:
                            yield TodoUpdatedEvent(todos=todos)

                    # Check if this was AskUserQuestion - if so, yield UserQuestionEvent and stop
                    if tool_name == "AskUserQuestion" and not tool_result.is_error:
                        questions = extract_questions(tool_result)
                        if questions:
                            yield UserQuestionEvent(
                                questions=questions,
                                tool_call_id=tool_call.id,
                            )
                            if not txn_committed:
                                agent._context.add_messages_atomic(
                                    [assistant_msg] + [results_by_id[call.id] for call in tool_calls]
                                )
                                txn_committed = True
                                async for hidden_event in _drain_hidden_events(agent):
                                    yield hidden_event
                            async for usage_event in _drain_usage_events():
                                yield usage_event
                            if await _run_stop_hook(agent, "waiting_for_input"):
                                async for hidden_event in _drain_hidden_events(agent):
                                    yield hidden_event
                                idx += 1
                                continue
                            async for hidden_event in _drain_hidden_events(agent):
                                yield hidden_event
                            yield StopEvent(reason="waiting_for_input")
                            _stop_signal[0] = "waiting_for_input"
                            return

                    if _is_interrupt_requested(agent):
                        for call in tool_calls[idx + 1 :]:
                            results_by_id[call.id] = ToolMessage(
                                tool_call_id=call.id,
                                tool_name=str(call.function.name or "").strip() or "Tool",
                                content=json.dumps(
                                    {"status": "cancelled", "reason": "user_interrupt"},
                                    ensure_ascii=False,
                                ),
                                is_error=True,
                            )
                        if not txn_committed:
                            agent._context.add_messages_atomic(
                                [assistant_msg] + [results_by_id[call.id] for call in tool_calls]
                            )
                            txn_committed = True
                        _stop_signal[0] = "interrupted"
                        async for event in _yield_interrupted_stop():
                            yield event
                        return

                    idx += 1

                if not txn_committed:
                    agent._context.add_messages_atomic(
                        [assistant_msg] + [results_by_id[call.id] for call in tool_calls]
                    )
                    async for hidden_event in _drain_hidden_events(agent):
                        yield hidden_event

            async for event in _run_all_tool_calls():
                yield event
            if _stop_signal[0]:
                return

            # Check for compaction after tool execution
            compacted, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
            async for usage_event in _drain_usage_events():
                yield usage_event
            if pre_compact_event:
                yield pre_compact_event
            for compaction_meta_event in compaction_meta_events:
                yield compaction_meta_event

    finally:
        agent._token_cost.unsubscribe_usage(usage_observer_id)
