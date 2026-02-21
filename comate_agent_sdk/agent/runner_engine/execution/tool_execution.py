from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import TYPE_CHECKING

from comate_agent_sdk.agent.events import (
    AgentEvent,
    StepCompleteEvent,
    StepStartEvent,
    StopEvent,
    TodoUpdatedEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserQuestionEvent,
)
from comate_agent_sdk.agent.tool_exec import execute_tool_call, extract_screenshot
from comate_agent_sdk.llm.messages import ToolCall, ToolMessage

from .state import RunningTaskState, TurnEngineState
from .task_lifecycle import TaskLifecycleEmitter
from .tool_messages import make_cancelled_tool_message
from .txn_commit import ToolTxnCommitter
from ..projection.tool_result_projection import extract_diff_metadata, extract_questions, extract_todos

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


async def execute_tool_calls(
    agent: "AgentRuntime",
    *,
    assistant_msg,
    tool_calls: list[ToolCall],
    engine_state: TurnEngineState,
    usage_poll_interval_seconds: float,
    task_lifecycle: TaskLifecycleEmitter,
    is_interrupt_requested: Callable[[], bool],
    run_stop_hook: Callable[[str], Awaitable[bool]],
    normalize_tool_call_event_args: Callable[[str, object], dict],
    resolve_task_metadata: Callable[[str, dict], tuple[str, str, str]],
    execute_task_streaming: Callable[[ToolCall, tuple[str, str]], Awaitable[ToolMessage]],
    drain_usage_events: Callable[[], AsyncIterator[AgentEvent]],
    drain_hidden_events: Callable[[], AsyncIterator[AgentEvent]],
    drain_subagent_events: Callable[[], AsyncIterator[AgentEvent]],
    emit_cancelled_task: Callable[[str], AsyncIterator[AgentEvent]],
    yield_interrupted_stop: Callable[[], AsyncIterator[AgentEvent]],
) -> AsyncIterator[AgentEvent]:
    results_by_id: dict[str, ToolMessage] = {}
    committer = ToolTxnCommitter(agent, assistant_message=assistant_msg, tool_calls=tool_calls)

    step_number = 0
    idx = 0
    while idx < len(tool_calls):
        if is_interrupt_requested():
            for call in tool_calls[idx:]:
                results_by_id[call.id] = make_cancelled_tool_message(call)
            if await committer.commit_if_needed(results_by_id):
                async for hidden_event in drain_hidden_events():
                    yield hidden_event
            engine_state.stop_reason = "interrupted"
            async for event in yield_interrupted_stop():
                yield event
            return

        tool_call = tool_calls[idx]
        tool_name = tool_call.function.name

        if agent.options.task_parallel_enabled and tool_name == "Task":
            group: list[ToolCall] = []
            while idx < len(tool_calls) and tool_calls[idx].function.name == "Task":
                group.append(tool_calls[idx])
                idx += 1

            args_by_id: dict[str, dict] = {}
            meta_by_id: dict[str, tuple[str, str]] = {}

            for task_call in group:
                step_number += 1
                try:
                    args = json.loads(task_call.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": task_call.function.arguments}
                task_args = normalize_tool_call_event_args("Task", args)
                args_by_id[task_call.id] = task_args

                subagent_name, description, source_prefix = resolve_task_metadata(
                    task_call.id,
                    task_args,
                )
                meta_by_id[task_call.id] = (subagent_name, description)
                engine_state.running_tasks[task_call.id] = RunningTaskState(
                    tool_call_id=task_call.id,
                    subagent_name=subagent_name,
                    description=description,
                    source_prefix=source_prefix,
                    started_at_monotonic=time.monotonic(),
                )

                yield StepStartEvent(
                    step_id=task_call.id,
                    title=description,
                    step_number=step_number,
                )
                yield ToolCallEvent(
                    tool="Task",
                    args=task_args,
                    tool_call_id=task_call.id,
                    display_name=description,
                )
                async for event in task_lifecycle.emit_start(
                    tool_call_id=task_call.id,
                    subagent_name=subagent_name,
                    description=description,
                ):
                    yield event
                async for hidden_event in drain_hidden_events():
                    yield hidden_event

            semaphore = asyncio.Semaphore(max(1, int(agent.options.task_parallel_max_concurrency)))
            task_tool = agent._tool_map.get("Task")
            is_streaming = getattr(task_tool, "_is_streaming_task", False)

            async def _run(task_call: ToolCall) -> tuple[ToolCall, ToolMessage, float]:
                start = time.time()
                async with semaphore:
                    if is_streaming:
                        tool_result = await execute_task_streaming(task_call, meta_by_id[task_call.id])
                    else:
                        tool_result = await execute_tool_call(agent, task_call)
                duration_ms = (time.time() - start) * 1000
                return task_call, tool_result, duration_ms

            futures = [asyncio.create_task(_run(task_call)) for task_call in group]
            pending = set(futures)
            group_results_by_id: dict[str, ToolMessage] = {}

            while pending:
                async for usage_event in drain_usage_events():
                    yield usage_event
                async for subagent_event in drain_subagent_events():
                    yield subagent_event

                if is_interrupt_requested():
                    for future in pending:
                        future.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

                    completed_ids = set(group_results_by_id.keys())
                    for task_call in group:
                        if task_call.id in completed_ids:
                            continue
                        cancelled = make_cancelled_tool_message(task_call)
                        group_results_by_id[task_call.id] = cancelled
                        async for event in emit_cancelled_task(task_call.id):
                            yield event

                    for task_call in group:
                        results_by_id[task_call.id] = group_results_by_id[task_call.id]
                    for call in tool_calls[idx:]:
                        results_by_id[call.id] = make_cancelled_tool_message(call)

                    if await committer.commit_if_needed(results_by_id):
                        async for hidden_event in drain_hidden_events():
                            yield hidden_event
                    engine_state.stop_reason = "interrupted"
                    async for event in yield_interrupted_stop():
                        yield event
                    return

                done, pending = await asyncio.wait(
                    pending,
                    timeout=usage_poll_interval_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                async for usage_event in drain_usage_events():
                    yield usage_event
                async for subagent_event in drain_subagent_events():
                    yield subagent_event

                if not done:
                    continue

                for future in done:
                    task_call, tool_result, duration_ms = await future
                    subagent_name, description = meta_by_id.get(task_call.id, ("Task", "Task"))

                    state = engine_state.running_tasks.pop(task_call.id, None)
                    total_tokens = state.tokens if state is not None else 0

                    async for event in task_lifecycle.emit_stop(
                        tool_call_id=task_call.id,
                        subagent_name=subagent_name,
                        description=description,
                        duration_ms=duration_ms,
                        tokens=total_tokens,
                        tool_result=tool_result,
                    ):
                        yield event

                    screenshot_base64 = extract_screenshot(tool_result)
                    yield ToolResultEvent(
                        tool="Task",
                        result=tool_result.text,
                        tool_call_id=task_call.id,
                        is_error=tool_result.is_error,
                        screenshot_base64=screenshot_base64,
                    )
                    yield StepCompleteEvent(
                        step_id=task_call.id,
                        status="error" if tool_result.is_error else "completed",
                        duration_ms=duration_ms,
                    )
                    async for hidden_event in drain_hidden_events():
                        yield hidden_event

                    group_results_by_id[task_call.id] = tool_result

            for task_call in group:
                results_by_id[task_call.id] = group_results_by_id[task_call.id]
            continue

        step_number += 1
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {"_raw": tool_call.function.arguments}
        args_dict = normalize_tool_call_event_args(str(tool_name or ""), args)

        is_task = tool_name == "Task"
        is_task_stream = False
        if is_task:
            task_tool = agent._tool_map.get("Task")
            is_task_stream = getattr(task_tool, "_is_streaming_task", False)

        task_subagent_name = "Task"
        task_description = str(tool_name or "")
        if is_task:
            task_subagent_name, task_description, task_source_prefix = resolve_task_metadata(
                tool_call.id,
                args_dict,
            )
        else:
            task_source_prefix = ""

        yield StepStartEvent(
            step_id=tool_call.id,
            title=task_description if is_task else str(tool_name),
            step_number=step_number,
        )
        yield ToolCallEvent(
            tool=str(tool_name),
            args=args_dict,
            tool_call_id=tool_call.id,
            display_name=task_description if is_task else str(tool_name),
        )

        if is_task:
            engine_state.running_tasks[tool_call.id] = RunningTaskState(
                tool_call_id=tool_call.id,
                subagent_name=task_subagent_name,
                description=task_description,
                source_prefix=task_source_prefix,
                started_at_monotonic=time.monotonic(),
            )
            async for event in task_lifecycle.emit_start(
                tool_call_id=tool_call.id,
                subagent_name=task_subagent_name,
                description=task_description,
            ):
                yield event
            async for hidden_event in drain_hidden_events():
                yield hidden_event

        step_start_time = time.time()

        if is_task_stream:
            out = await execute_task_streaming(
                tool_call,
                (task_subagent_name, task_description),
            )
            tool_result = out
        else:
            tool_result_task = asyncio.create_task(execute_tool_call(agent, tool_call))
            while True:
                async for usage_event in drain_usage_events():
                    yield usage_event

                if is_interrupt_requested():
                    tool_result_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await tool_result_task

                    step_duration_ms = (time.time() - step_start_time) * 1000
                    if is_task:
                        cancelled_result = make_cancelled_tool_message(tool_call)
                        results_by_id[tool_call.id] = cancelled_result
                        async for event in emit_cancelled_task(tool_call.id):
                            yield event
                    else:
                        yield StepCompleteEvent(
                            step_id=tool_call.id,
                            status="cancelled",
                            duration_ms=step_duration_ms,
                        )
                        results_by_id[tool_call.id] = make_cancelled_tool_message(tool_call)

                    for call in tool_calls[idx + 1 :]:
                        results_by_id[call.id] = make_cancelled_tool_message(call)

                    if await committer.commit_if_needed(results_by_id):
                        async for hidden_event in drain_hidden_events():
                            yield hidden_event

                    engine_state.stop_reason = "interrupted"
                    async for event in yield_interrupted_stop():
                        yield event
                    return

                done, _ = await asyncio.wait(
                    {tool_result_task},
                    timeout=usage_poll_interval_seconds,
                    return_when=asyncio.ALL_COMPLETED,
                )
                async for usage_event in drain_usage_events():
                    yield usage_event
                if done:
                    break

            tool_result = await tool_result_task

        step_duration_ms = (time.time() - step_start_time) * 1000
        results_by_id[tool_call.id] = tool_result
        async for hidden_event in drain_hidden_events():
            yield hidden_event

        if is_task:
            state = engine_state.running_tasks.pop(tool_call.id, None)
            total_tokens = state.tokens if state is not None else 0
            async for event in task_lifecycle.emit_stop(
                tool_call_id=tool_call.id,
                subagent_name=task_subagent_name,
                description=task_description,
                duration_ms=step_duration_ms,
                tokens=total_tokens,
                tool_result=tool_result,
            ):
                yield event

        screenshot_base64 = extract_screenshot(tool_result)
        diff_metadata = extract_diff_metadata(str(tool_name or ""), tool_result)
        yield ToolResultEvent(
            tool=str(tool_name),
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
        async for hidden_event in drain_hidden_events():
            yield hidden_event

        if str(tool_name) == "TodoWrite" and not tool_result.is_error:
            todos = extract_todos(tool_result)
            if todos:
                yield TodoUpdatedEvent(todos=todos)

        if str(tool_name) == "AskUserQuestion" and not tool_result.is_error:
            questions = extract_questions(tool_result)
            if questions:
                yield UserQuestionEvent(
                    questions=questions,
                    tool_call_id=tool_call.id,
                )
                if await committer.commit_if_needed(results_by_id):
                    async for hidden_event in drain_hidden_events():
                        yield hidden_event
                async for usage_event in drain_usage_events():
                    yield usage_event
                if await run_stop_hook("waiting_for_input"):
                    async for hidden_event in drain_hidden_events():
                        yield hidden_event
                    idx += 1
                    continue
                async for hidden_event in drain_hidden_events():
                    yield hidden_event
                yield StopEvent(reason="waiting_for_input")
                engine_state.stop_reason = "waiting_for_input"
                return

        if is_interrupt_requested():
            for call in tool_calls[idx + 1 :]:
                results_by_id[call.id] = make_cancelled_tool_message(call)
            if await committer.commit_if_needed(results_by_id):
                async for hidden_event in drain_hidden_events():
                    yield hidden_event
            engine_state.stop_reason = "interrupted"
            async for event in yield_interrupted_stop():
                yield event
            return

        idx += 1

    if await committer.commit_if_needed(results_by_id):
        async for hidden_event in drain_hidden_events():
            yield hidden_event
