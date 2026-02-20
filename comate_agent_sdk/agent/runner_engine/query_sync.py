from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from comate_agent_sdk.agent.history import destroy_ephemeral_messages
from comate_agent_sdk.agent.llm import invoke_llm
from comate_agent_sdk.agent.tool_exec import execute_tool_call
from comate_agent_sdk.llm.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage

from .compaction import (
    check_and_compact,
    generate_max_iterations_summary,
    log_compaction_meta_events,
    precheck_and_compact,
)
from .lifecycle import fire_session_start_if_needed, fire_user_prompt_submit
from .stop_policy import should_continue_after_stop_block
from .subagent_hooks import fire_subagent_hook
from .tool_policy import enforce_ask_user_exclusive_policy, is_ask_user_question

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


async def run_query(agent: "AgentRuntime", message: str) -> str:
    await fire_session_start_if_needed(agent)
    agent.reset_stop_hook_state()
    agent._context.add_message(UserMessage(content=message))
    await fire_user_prompt_submit(agent, message)

    iterations = 0
    askuser_repair_attempted = False

    while True:
        if iterations >= agent.options.max_iterations:
            summary = await generate_max_iterations_summary(agent)
            if await should_continue_after_stop_block(
                agent,
                "max_iterations",
                drain_hidden_immediately=True,
            ):
                iterations = max(0, agent.options.max_iterations - 1)
                continue
            return summary

        iterations += 1
        destroy_ephemeral_messages(agent)

        _, precheck_event, precheck_meta_events = await precheck_and_compact(agent)
        if precheck_event:
            logger.info(f"Pre-compact event: {precheck_event}")
        log_compaction_meta_events(precheck_meta_events)

        response = await invoke_llm(agent)

        if not response.has_tool_calls:
            agent._context.add_message(
                AssistantMessage(content=response.content, tool_calls=None)
            )
            _, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
            if pre_compact_event:
                logger.info(f"Pre-compact event: {pre_compact_event}")
            log_compaction_meta_events(compaction_meta_events)
            if await should_continue_after_stop_block(
                agent,
                "completed",
                drain_hidden_immediately=True,
            ):
                continue
            return response.content or ""

        tool_calls = list(response.tool_calls or [])
        assistant_msg = AssistantMessage(content=response.content, tool_calls=tool_calls)
        ask_policy = await enforce_ask_user_exclusive_policy(
            agent,
            assistant_message=assistant_msg,
            tool_calls=tool_calls,
            askuser_repair_attempted=askuser_repair_attempted,
        )
        askuser_repair_attempted = ask_policy.askuser_repair_attempted
        if ask_policy.repaired:
            agent.drain_hidden_user_messages()
            continue

        tool_calls = ask_policy.tool_calls
        assistant_msg = AssistantMessage(content=response.content, tool_calls=tool_calls)
        expected_ids = [call.id for call in tool_calls if str(call.id).strip()]
        agent._context.begin_tool_barrier(expected_ids)

        results_by_id: dict[str, ToolMessage] = {}
        idx = 0
        while idx < len(tool_calls):
            tool_call = tool_calls[idx]
            tool_name = tool_call.function.name

            if agent.options.task_parallel_enabled and tool_name == "Task":
                group: list[ToolCall] = []
                while idx < len(tool_calls) and tool_calls[idx].function.name == "Task":
                    group.append(tool_calls[idx])
                    idx += 1

                semaphore = asyncio.Semaphore(max(1, int(agent.options.task_parallel_max_concurrency)))
                for task_call in group:
                    await fire_subagent_hook(agent, event_name="SubagentStart", tool_call=task_call)

                async def _run(task_call: ToolCall) -> tuple[str, ToolMessage]:
                    async with semaphore:
                        try:
                            return task_call.id, await execute_tool_call(agent, task_call)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            logger.warning(f"Subagent {task_call.id} failed: {exc}")
                            return task_call.id, ToolMessage(
                                tool_call_id=task_call.id,
                                content=f"[Subagent Error] {type(exc).__name__}: {exc}",
                            )

                tasks_map: dict[str, asyncio.Task[tuple[str, ToolMessage]]] = {}
                async with asyncio.TaskGroup() as tg:
                    for task_call in group:
                        tasks_map[task_call.id] = tg.create_task(_run(task_call))

                group_results = {task_id: task.result()[1] for task_id, task in tasks_map.items()}
                for task_call in group:
                    tool_result = group_results[task_call.id]
                    await fire_subagent_hook(
                        agent,
                        event_name="SubagentStop",
                        tool_call=task_call,
                        subagent_status="error" if tool_result.is_error else "completed",
                    )
                    results_by_id[task_call.id] = tool_result
                continue

            if tool_name == "Task":
                await fire_subagent_hook(agent, event_name="SubagentStart", tool_call=tool_call)

            tool_result = await execute_tool_call(agent, tool_call)
            results_by_id[tool_call.id] = tool_result
            if tool_name == "Task":
                await fire_subagent_hook(
                    agent,
                    event_name="SubagentStop",
                    tool_call=tool_call,
                    subagent_status="error" if tool_result.is_error else "completed",
                )
            idx += 1

        agent._context.add_messages_atomic([assistant_msg] + [results_by_id[call.id] for call in tool_calls])
        agent.drain_hidden_user_messages()

        if len(tool_calls) == 1 and is_ask_user_question(tool_calls[0]):
            tool_result = results_by_id.get(tool_calls[0].id)
            if tool_result is not None and not tool_result.is_error:
                if await should_continue_after_stop_block(
                    agent,
                    "waiting_for_input",
                    drain_hidden_immediately=True,
                ):
                    continue
                return tool_result.text

        _, pre_compact_event, compaction_meta_events = await check_and_compact(agent, response)
        if pre_compact_event:
            logger.info(f"Pre-compact event: {pre_compact_event}")
        log_compaction_meta_events(compaction_meta_events)
