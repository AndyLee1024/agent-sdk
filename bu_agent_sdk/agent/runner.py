from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from bu_agent_sdk.agent.history import destroy_ephemeral_messages
from bu_agent_sdk.agent.llm import invoke_llm
from bu_agent_sdk.agent.tool_exec import execute_tool_call
from bu_agent_sdk.context import SelectiveCompactionPolicy
from bu_agent_sdk.context.offload import OffloadPolicy
from bu_agent_sdk.llm.messages import AssistantMessage, ToolCall, ToolMessage, UserMessage
from bu_agent_sdk.llm.views import ChatInvokeCompletion

logger = logging.getLogger("bu_agent_sdk.agent")

if TYPE_CHECKING:
    from bu_agent_sdk.agent.core import Agent


async def generate_max_iterations_summary(agent: "Agent") -> str:
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


async def check_and_compact(agent: "Agent", response: ChatInvokeCompletion) -> bool:
    """检查 token 使用并在需要时压缩。"""
    if agent._compaction_service is None:
        return False

    # Update token usage tracking
    agent._compaction_service.update_usage(response.usage)

    # 检查是否需要压缩
    if not await agent._compaction_service.should_compact(agent.llm.model):
        return False

    # 获取压缩阈值
    threshold = await agent._compaction_service.get_threshold_for_model(agent.llm.model)

    # 使用 token usage 中的实际 total_tokens
    from bu_agent_sdk.agent.compaction.models import TokenUsage

    actual_tokens = TokenUsage.from_usage(response.usage).total_tokens

    # 创建/复用 OffloadPolicy
    offload_policy = None
    if agent.offload_enabled and agent._context_fs:
        offload_policy = agent.offload_policy or OffloadPolicy(
            enabled=True,
            token_threshold=agent.offload_token_threshold,
        )

    # 创建选择性压缩策略
    policy = SelectiveCompactionPolicy(
        threshold=threshold,
        llm=agent.llm,
        fallback_to_full_summary=True,
        fs=agent._context_fs,
        offload_policy=offload_policy,
        token_cost=agent._token_cost,
        level=agent._effective_level,
    )

    return await agent._context.auto_compact(
        policy=policy,
        current_total_tokens=actual_tokens,
    )


async def query(agent: "Agent", message: str) -> str:
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
            await check_and_compact(agent, response)
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
                continue

            # Default: serial execution
            tool_result = await execute_tool_call(agent, tool_call)
            agent._context.add_message(tool_result)

            # 检查是否有待注入的 Skill items（必须在 ToolMessage 之后注入）
            if agent._context.has_pending_skill_items:
                agent._context.flush_pending_skill_items()

            idx += 1

        # Check for compaction after tool execution
        await check_and_compact(agent, response)

    # Max iterations reached - generate summary of what was accomplished
    return await generate_max_iterations_summary(agent)
