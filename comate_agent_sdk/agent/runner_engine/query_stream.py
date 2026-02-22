from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import nullcontext, suppress
from typing import TYPE_CHECKING

from comate_agent_sdk.agent.events import (
    AgentEvent,
    StepCompleteEvent,
    StepStartEvent,
    StopEvent,
    SubagentProgressEvent,
    SubagentToolCallEvent,
    SubagentToolResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageDeltaEvent,
)
from comate_agent_sdk.agent.history import destroy_ephemeral_messages
from comate_agent_sdk.agent.llm import invoke_llm
from comate_agent_sdk.agent.tool_exec import _coerce_tool_arguments
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from .compaction import check_and_compact, generate_max_iterations_summary, precheck_and_compact
from .execution.state import RunningTaskState, TaskStreamOut, TurnEngineState
from .execution.stream_drain import EventDrainHub
from .execution.task_lifecycle import TaskLifecycleEmitter
from .execution.tool_execution import execute_tool_calls
from .lifecycle import (
    fire_session_start_if_needed as _fire_session_start_if_needed,
    fire_user_prompt_submit as _fire_user_prompt_submit,
    is_interrupt_requested as _is_interrupt_requested,
)
from .policy.stop_policy import should_continue_after_stop_block
from .policy.tool_policy import (
    enforce_ask_user_exclusive_policy,
    is_ask_user_question,
    policy_violation_message,
)

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime
    from comate_agent_sdk.tokens.views import TokenUsageEntry


def _source_matches_prefix(*, source: str, source_prefix: str) -> bool:
    if not source or not source_prefix:
        return False
    return source == source_prefix or source.startswith(f"{source_prefix}:")


def _usage_belongs_to_task(event: UsageDeltaEvent, state: RunningTaskState) -> bool:
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
    current_task_state: RunningTaskState | None,
    out: TaskStreamOut,
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


async def _emit_policy_repair_events(
    agent: "AgentRuntime",
    *,
    tool_calls: list[ToolCall],
    task_lifecycle: TaskLifecycleEmitter,
    drain_hub: EventDrainHub,
) -> AsyncIterator[AgentEvent]:
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

        tool_result = results_by_id[call.id]
        if is_task:
            async for event in task_lifecycle.emit_start(
                tool_call_id=call.id,
                subagent_name=task_subagent_name,
                description=task_description,
            ):
                yield event
            async for event in task_lifecycle.emit_stop(
                tool_call_id=call.id,
                subagent_name=task_subagent_name,
                description=task_description,
                duration_ms=0.0,
                tokens=0,
                tool_result=tool_result,
            ):
                yield event

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
        async for hidden_event in drain_hub.drain_hidden_events():
            yield hidden_event


async def run_query_stream(
    agent: "AgentRuntime", message: str | list[ContentPartTextParam | ContentPartImageParam]
) -> AsyncIterator[AgentEvent]:
    """流式执行：发送消息并逐步产出事件。"""

    usage_poll_interval_seconds = 0.08
    engine_state = TurnEngineState()
    task_lifecycle = TaskLifecycleEmitter(agent)

    drain_hub = EventDrainHub(
        agent,
        usage_queue=engine_state.usage_queue,
        subagent_event_queue=engine_state.subagent_event_queue,
        running_tasks=engine_state.running_tasks,
        usage_tokens_for_progress=_usage_delta_tokens_for_progress,
        usage_belongs_to_task=_usage_belongs_to_task,
    )

    def _on_usage_entry(entry: "TokenUsageEntry") -> None:
        source = entry.source or "unknown"
        usage = entry.usage
        engine_state.usage_queue.put_nowait(
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

    async def _execute_task_streaming(
        tool_call: ToolCall,
        meta: tuple[str, str],
    ) -> ToolMessage:
        subagent_name, description = meta
        current_task_state = engine_state.running_tasks.get(tool_call.id)
        out = TaskStreamOut()
        async for event in _iter_task_stream_events(
            agent,
            tool_call,
            subagent_name,
            description,
            current_task_state,
            out,
            check_interrupt=lambda: _is_interrupt_requested(agent),
        ):
            engine_state.subagent_event_queue.put_nowait(event)
        return out.tool_message or ToolMessage(
            tool_call_id=tool_call.id,
            tool_name="Task",
            content="Error: streaming task produced no result",
            is_error=True,
        )

    async def _yield_interrupted_stop() -> AsyncIterator[AgentEvent]:
        async for usage_event in drain_hub.drain_usage_events():
            yield usage_event
        async for hidden_event in drain_hub.drain_hidden_events():
            yield hidden_event
        yield StopEvent(reason="interrupted")

    async def _emit_cancelled_task(tool_call_id: str) -> AsyncIterator[AgentEvent]:
        state = engine_state.running_tasks.pop(tool_call_id, None)
        elapsed_ms = 0.0
        total_tokens = 0
        subagent_name = "Task"
        description = "Task"
        if state is not None:
            elapsed_ms = (time.monotonic() - state.started_at_monotonic) * 1000
            total_tokens = state.tokens
            subagent_name = state.subagent_name
            description = state.description
            async for event in task_lifecycle.emit_cancelled(
                tool_call_id=tool_call_id,
                subagent_name=subagent_name,
                description=description,
                elapsed_ms=elapsed_ms,
                tokens=total_tokens,
            ):
                yield event
            async for hidden_event in drain_hub.drain_hidden_events():
                yield hidden_event

        yield StepCompleteEvent(
            step_id=tool_call_id,
            status="cancelled",
            duration_ms=elapsed_ms,
        )

    usage_observer_id = agent._token_cost.subscribe_usage(_on_usage_entry)

    try:
        mode_snapshot = agent.get_mode()
        agent._active_mode_snapshot = mode_snapshot
        agent._context.set_plan_mode(mode_snapshot == "plan")
        if mode_snapshot == "plan":
            plan_prompt = str(agent.options.plan_mode_prompt_template or "").strip()
            if plan_prompt:
                agent.add_hidden_user_message(plan_prompt)

        await _fire_session_start_if_needed(agent)
        agent.reset_stop_hook_state()
        async for hidden_event in drain_hub.drain_hidden_events():
            yield hidden_event

        agent._context.add_message(UserMessage(content=message))
        await _fire_user_prompt_submit(agent, message)
        async for hidden_event in drain_hub.drain_hidden_events():
            yield hidden_event

        iterations = 0
        askuser_repair_attempted = False

        while True:
            if iterations >= agent.options.max_iterations:
                summary = await generate_max_iterations_summary(agent)
                async for usage_event in drain_hub.drain_usage_events():
                    yield usage_event
                yield TextEvent(content=summary)
                if await _run_stop_hook(agent, "max_iterations"):
                    async for hidden_event in drain_hub.drain_hidden_events():
                        yield hidden_event
                    iterations = max(0, agent.options.max_iterations - 1)
                    continue
                async for hidden_event in drain_hub.drain_hidden_events():
                    yield hidden_event
                yield StopEvent(reason="max_iterations")
                return

            if _is_interrupt_requested(agent):
                async for event in _yield_interrupted_stop():
                    yield event
                return

            iterations += 1
            destroy_ephemeral_messages(agent)

            _, precheck_event, precheck_meta_events = await precheck_and_compact(agent)
            async for usage_event in drain_hub.drain_usage_events():
                yield usage_event
            if precheck_event:
                yield precheck_event
            for compaction_meta_event in precheck_meta_events:
                yield compaction_meta_event

            llm_task = asyncio.create_task(invoke_llm(agent))
            while True:
                async for usage_event in drain_hub.drain_usage_events():
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
                async for usage_event in drain_hub.drain_usage_events():
                    yield usage_event
                if done:
                    break

            response = await llm_task
            async for usage_event in drain_hub.drain_usage_events():
                yield usage_event

            if response.thinking:
                yield ThinkingEvent(content=response.thinking)

            if _is_interrupt_requested(agent):
                async for event in _yield_interrupted_stop():
                    yield event
                return

            if not response.has_tool_calls:
                assistant_msg = AssistantMessage(
                    content=response.raw_content_blocks or response.content,
                    tool_calls=None,
                )
                agent._context.add_message(assistant_msg)
                _, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
                async for usage_event in drain_hub.drain_usage_events():
                    yield usage_event
                if pre_compact_event:
                    yield pre_compact_event
                for compaction_meta_event in compaction_meta_events:
                    yield compaction_meta_event
                if response.content:
                    yield TextEvent(content=response.content)
                if await _run_stop_hook(agent, "completed"):
                    async for hidden_event in drain_hub.drain_hidden_events():
                        yield hidden_event
                    continue
                async for hidden_event in drain_hub.drain_hidden_events():
                    yield hidden_event
                yield StopEvent(reason="completed")
                return

            if response.content:
                yield TextEvent(content=response.content)

            tool_calls = list(response.tool_calls or [])
            assistant_msg = AssistantMessage(
                content=response.raw_content_blocks or response.content,
                tool_calls=tool_calls,
            )

            ask_policy = await enforce_ask_user_exclusive_policy(
                agent,
                assistant_message=assistant_msg,
                tool_calls=tool_calls,
                askuser_repair_attempted=askuser_repair_attempted,
            )
            askuser_repair_attempted = ask_policy.askuser_repair_attempted

            if ask_policy.repaired:
                async for event in _emit_policy_repair_events(
                    agent,
                    tool_calls=tool_calls,
                    task_lifecycle=task_lifecycle,
                    drain_hub=drain_hub,
                ):
                    yield event
                async for hidden_event in drain_hub.drain_hidden_events():
                    yield hidden_event
                continue

            tool_calls = ask_policy.tool_calls
            assistant_msg = AssistantMessage(
                content=response.raw_content_blocks or response.content,
                tool_calls=tool_calls,
            )

            expected_ids = [call.id for call in tool_calls if str(call.id).strip()]
            agent._context.begin_tool_barrier(expected_ids)

            engine_state.stop_reason = None
            async for event in execute_tool_calls(
                agent,
                assistant_msg=assistant_msg,
                tool_calls=tool_calls,
                engine_state=engine_state,
                usage_poll_interval_seconds=usage_poll_interval_seconds,
                task_lifecycle=task_lifecycle,
                is_interrupt_requested=lambda: _is_interrupt_requested(agent),
                run_stop_hook=lambda reason: _run_stop_hook(agent, reason),
                normalize_tool_call_event_args=lambda tool_name, raw_args: _normalize_tool_call_event_args(
                    agent,
                    tool_name=tool_name,
                    raw_args=raw_args,
                ),
                resolve_task_metadata=lambda tool_call_id, args: _resolve_task_metadata(
                    tool_call_id=tool_call_id,
                    args=args,
                ),
                execute_task_streaming=_execute_task_streaming,
                drain_usage_events=drain_hub.drain_usage_events,
                drain_hidden_events=drain_hub.drain_hidden_events,
                drain_subagent_events=drain_hub.drain_subagent_events,
                emit_cancelled_task=_emit_cancelled_task,
                yield_interrupted_stop=_yield_interrupted_stop,
            ):
                yield event

            if engine_state.stop_reason:
                return

            _, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
            async for usage_event in drain_hub.drain_usage_events():
                yield usage_event
            if pre_compact_event:
                yield pre_compact_event
            for compaction_meta_event in compaction_meta_events:
                yield compaction_meta_event

    finally:
        agent._active_mode_snapshot = None
        agent._context.set_plan_mode(agent.get_mode() == "plan")
        agent._token_cost.unsubscribe_usage(usage_observer_id)
