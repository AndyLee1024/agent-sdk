from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from comate_agent_sdk.agent.events import (
    AgentEvent,
    PreCompactEvent,
    StepCompleteEvent,
    StepStartEvent,
    StopEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserQuestionEvent,
)
from comate_agent_sdk.agent.history import destroy_ephemeral_messages
from comate_agent_sdk.agent.llm import invoke_llm
from comate_agent_sdk.agent.runner import check_and_compact, generate_max_iterations_summary, precheck_and_compact
from comate_agent_sdk.agent.tool_exec import execute_tool_call, extract_screenshot
from comate_agent_sdk.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from comate_agent_sdk.system_tools.tool_result import is_tool_result_envelope

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


async def query_stream(
    agent: "AgentRuntime", message: str | list[ContentPartTextParam | ContentPartImageParam]
) -> AsyncIterator[AgentEvent]:
    """流式执行：发送消息并逐步产出事件。"""
    # Add the user message to context (supports both string and multi-modal content)
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

        # Check for thinking content and yield it
        if response.thinking:
            yield ThinkingEvent(content=response.thinking)

        # Add assistant message to history
        from comate_agent_sdk.llm.messages import AssistantMessage

        assistant_msg = AssistantMessage(
            content=response.content,
            tool_calls=response.tool_calls if response.tool_calls else None,
        )
        agent._context.add_message(assistant_msg)

        # If no tool calls, finish
        if not response.has_tool_calls:
            compacted, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
            if pre_compact_event:
                yield pre_compact_event
            for compaction_meta_event in compaction_meta_events:
                yield compaction_meta_event
            if response.content:
                yield TextEvent(content=response.content)
            yield StopEvent(reason="completed")
            return

        # Yield text content if present alongside tool calls
        if response.content:
            yield TextEvent(content=response.content)

        # Execute tool calls, yielding events for each.
        # Contiguous Task tool calls can run in parallel (streamed by completion order),
        # while ToolMessage writes remain in original order for reproducibility.
        tool_calls = response.tool_calls
        step_number = 0
        idx = 0
        while idx < len(tool_calls):
            tool_call = tool_calls[idx]
            tool_name = tool_call.function.name

            # Parallel Task group (contiguous only)
            if agent.task_parallel_enabled and tool_name == "Task":
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
                    args_by_id[tc.id] = args if isinstance(args, dict) else {"_raw": str(args)}

                    subagent_name = ""
                    description = ""
                    if isinstance(args, dict):
                        raw_name = args.get("subagent_type")
                        if isinstance(raw_name, str):
                            subagent_name = raw_name.strip()
                        raw_desc = args.get("description", "")
                        if isinstance(raw_desc, str):
                            description = raw_desc.strip()

                    if not subagent_name:
                        subagent_name = "Task"
                    if not description:
                        description = subagent_name

                    meta_by_id[tc.id] = (subagent_name, description)

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

                semaphore = asyncio.Semaphore(max(1, int(agent.task_parallel_max_concurrency)))

                async def _run(tc: ToolCall) -> tuple[ToolCall, ToolMessage, float]:
                    start = time.time()
                    async with semaphore:
                        tool_result = await execute_tool_call(agent, tc)
                    duration_ms = (time.time() - start) * 1000
                    return tc, tool_result, duration_ms

                futures = [asyncio.create_task(_run(tc)) for tc in group]
                results_by_id: dict[str, ToolMessage] = {}

                # Stream completion in completion order
                for fut in asyncio.as_completed(futures):
                    tc, tool_result, duration_ms = await fut
                    subagent_name, description = meta_by_id.get(tc.id, ("Task", "Task"))

                    stop_status: Literal["completed", "error", "timeout"] = "completed"
                    error_msg: str | None = None
                    if tool_result.is_error:
                        error_msg = tool_result.text
                        if "timeout" in tool_result.text.lower():
                            stop_status = "timeout"
                        else:
                            stop_status = "error"

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

                    results_by_id[tc.id] = tool_result

                # Write ToolMessage(s) to context in original order
                for tc in group:
                    tool_result = results_by_id[tc.id]
                    agent._context.add_message(tool_result)

                    if agent._context.has_pending_skill_items:
                        agent._context.flush_pending_skill_items()

                # 新增:预检查压缩
                compacted, pre_compact_event, compaction_meta_events = await precheck_and_compact(agent)
                if pre_compact_event:
                    yield pre_compact_event
                for compaction_meta_event in compaction_meta_events:
                    yield compaction_meta_event

                continue

            # Default: serial execution
            step_number += 1

            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": tool_call.function.arguments}

            yield StepStartEvent(
                step_id=tool_call.id,
                title=tool_name,
                step_number=step_number,
            )

            yield ToolCallEvent(
                tool=tool_name,
                args=args,
                tool_call_id=tool_call.id,
                display_name=tool_name,
            )

            step_start_time = time.time()
            tool_result = await execute_tool_call(agent, tool_call)
            agent._context.add_message(tool_result)

            # 新增:预检查压缩
            compacted, pre_compact_event, compaction_meta_events = await precheck_and_compact(agent)
            if pre_compact_event:
                yield pre_compact_event
            for compaction_meta_event in compaction_meta_events:
                yield compaction_meta_event

            if agent._context.has_pending_skill_items:
                agent._context.flush_pending_skill_items()

            screenshot_base64 = extract_screenshot(tool_result)
            yield ToolResultEvent(
                tool=tool_name,
                result=tool_result.text,
                tool_call_id=tool_call.id,
                is_error=tool_result.is_error,
                screenshot_base64=screenshot_base64,
            )

            step_duration_ms = (time.time() - step_start_time) * 1000
            yield StepCompleteEvent(
                step_id=tool_call.id,
                status="error" if tool_result.is_error else "completed",
                duration_ms=step_duration_ms,
            )

            # Check if this was AskUserQuestion - if so, yield UserQuestionEvent and stop
            if tool_name == "AskUserQuestion" and not tool_result.is_error:
                questions: list[dict] = []
                try:
                    payload = json.loads(tool_result.text)
                    if is_tool_result_envelope(payload):
                        data = payload.get("data", {})
                        if isinstance(data, dict):
                            raw_questions = data.get("questions", [])
                            if isinstance(raw_questions, list):
                                questions = [q for q in raw_questions if isinstance(q, dict)]
                except Exception:
                    questions = []

                if not questions and isinstance(args, dict):
                    # fallback for legacy behavior
                    raw_questions = args.get("questions", [])
                    if isinstance(raw_questions, list):
                        questions = [q for q in raw_questions if isinstance(q, dict)]

                if questions:
                    yield UserQuestionEvent(
                        questions=questions,
                        tool_call_id=tool_call.id,
                    )
                    yield StopEvent(reason="waiting_for_input")
                    return

            idx += 1

        # Check for compaction after tool execution
        compacted, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
        if pre_compact_event:
            yield pre_compact_event
        for compaction_meta_event in compaction_meta_events:
            yield compaction_meta_event

    # Max iterations reached - generate summary of what was accomplished
    summary = await generate_max_iterations_summary(agent)
    yield TextEvent(content=summary)
    yield StopEvent(reason="max_iterations")
