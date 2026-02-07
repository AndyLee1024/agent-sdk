from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from comate_agent_sdk.llm.messages import AssistantMessage, BaseMessage, ToolMessage
from comate_agent_sdk.tools.decorator import Tool

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


def clear_history(agent: "AgentRuntime") -> None:
    """清空 message history 与 token usage，并重建 ContextIR header。"""
    agent._context.clear()
    agent._token_cost.clear_history()

    # 重建 IR header 各独立段
    resolved_prompt = agent._resolve_system_prompt()
    if resolved_prompt:
        agent._context.set_system_prompt(resolved_prompt, cache=True)

    # 重建 tool_strategy
    from comate_agent_sdk.agent.tool_strategy import generate_tool_strategy

    tool_strategy = generate_tool_strategy(agent.tools)
    if tool_strategy:
        agent._context.set_tool_strategy(tool_strategy)

    # 重建 agent_loop
    from comate_agent_sdk.agent.prompts import AGENT_LOOP_PROMPT

    agent._context.set_agent_loop(AGENT_LOOP_PROMPT, cache=False)

    # 重建 subagent_strategy（如果有 agents）
    if agent.agents:
        from comate_agent_sdk.subagent.prompts import generate_subagent_prompt

        agent._context.set_subagent_strategy(generate_subagent_prompt(agent.agents))

    # 重建 skill_strategy（如果有 skills）
    if agent.skills:
        from comate_agent_sdk.skill.prompts import generate_skill_prompt

        skill_prompt = generate_skill_prompt(agent.skills)
        if skill_prompt:
            agent._context.set_skill_strategy(skill_prompt)

    # 如果有 memory，重新加载
    if agent.memory:
        agent._setup_memory()


def load_history(agent: "AgentRuntime", messages: list[BaseMessage]) -> None:
    """加载 message history（保留 header），用于恢复对话。"""
    # 清空现有 conversation（保留 header）
    agent._context.conversation.items.clear()
    agent._token_cost.clear_history()

    # 逐条加载消息到 IR
    for msg in messages:
        agent._context.add_message(msg)


def destroy_ephemeral_messages(agent: "AgentRuntime") -> None:
    """销毁旧的 ephemeral tool 输出（保留每个 tool 最近 N 条）。"""
    # 落盘阈值/开关（按类型）
    policy = getattr(agent, "offload_policy", None)
    tool_call_enabled = True
    tool_result_enabled = True
    tool_call_offload_threshold = 400
    tool_result_offload_threshold = agent.offload_token_threshold

    if policy is not None:
        if not bool(getattr(policy, "enabled", True)):
            tool_call_enabled = False
            tool_result_enabled = False
        else:
            type_enabled = getattr(policy, "type_enabled", {}) or {}
            tool_call_enabled = bool(type_enabled.get("tool_call", tool_call_enabled))
            tool_result_enabled = bool(type_enabled.get("tool_result", tool_result_enabled))

            threshold_by_type = getattr(policy, "token_threshold_by_type", {}) or {}
            tool_call_offload_threshold = int(threshold_by_type.get("tool_call", tool_call_offload_threshold))
            tool_result_offload_threshold = int(threshold_by_type.get("tool_result", getattr(policy, "token_threshold", tool_result_offload_threshold)))

    # 构建每个工具的 keep_count 映射
    tool_keep_counts: dict[str, int] = {}
    for tool in (agent.tools or []):
        if tool.ephemeral:
            # 如果设置了 ephemeral_keep_recent，覆盖工具的默认值
            if agent.ephemeral_keep_recent is not None:
                keep_count = agent.ephemeral_keep_recent
            else:
                keep_count = tool.ephemeral if isinstance(tool.ephemeral, int) else 1
            tool_keep_counts[tool.name] = keep_count

    conversation = agent._context.conversation.items
    idx_by_id = {it.id: i for i, it in enumerate(conversation)}

    # 遍历所有 ephemeral items，卸载需要被销毁的
    for tool_name, keep in tool_keep_counts.items():
        # 获取该工具的所有 ephemeral items
        same_tool_items = [
            item
            for item in agent._context.conversation.items
            if item.item_type.value == "tool_result"
            and item.ephemeral
            and not item.destroyed
            and (item.tool_name or "") == tool_name
        ]

        # 卸载需要销毁的（不是最后 N 个的）
        if keep <= 0:
            items_to_destroy = same_tool_items
        elif len(same_tool_items) > keep:
            items_to_destroy = same_tool_items[:-keep]
        else:
            items_to_destroy = []

        if items_to_destroy:
            for item in items_to_destroy:
                # 使用 ContextFS 卸载（如果启用）
                if agent._context_fs:
                    try:
                        tm = item.message if isinstance(item.message, ToolMessage) else None

                        # 尝试落盘 tool_call（arguments 可能很大），与 tool_result 独立门控
                        idx = idx_by_id.get(item.id)
                        if tm is not None and idx is not None:
                            found_tc = None
                            found_assistant_item_id: str | None = None
                            for j in range(idx - 1, -1, -1):
                                prev = conversation[j]
                                msg = prev.message
                                if isinstance(msg, AssistantMessage) and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        if tc.id == tm.tool_call_id:
                                            found_tc = tc
                                            found_assistant_item_id = prev.id
                                            break
                                if found_tc is not None:
                                    break

                            if found_tc is not None:
                                args = found_tc.function.arguments
                                args_tokens = agent._context.token_counter.count(args)
                                if tool_call_enabled and args_tokens >= tool_call_offload_threshold:
                                    agent._context_fs.offload_tool_call(
                                        tool_call_id=found_tc.id,
                                        tool_name=found_tc.function.name,
                                        arguments=args,
                                        assistant_item_id=found_assistant_item_id,
                                        arguments_token_count=args_tokens,
                                    )

                        # 只对超过阈值的 tool_result 落盘；否则直接销毁以省 tokens
                        if tool_result_enabled and item.token_count >= tool_result_offload_threshold:
                            if tm is None:
                                raise ValueError("Expected ToolMessage for TOOL_RESULT item")

                            wr = agent._context_fs.offload_tool_result(
                                item,
                                tool_call_id=tm.tool_call_id,
                                tool_name=tm.tool_name,
                                result_token_count=item.token_count,
                            )

                            item.offload_path = wr.relative_path
                            item.offloaded = True

                            # 同步更新 message 对象（serializer 里会显示 Path）
                            item.message.offloaded = True
                            item.message.offload_path = str(
                                agent._context_fs.root_path / wr.relative_path
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to offload ephemeral item {item.id}: {e}"
                        )
                # 标记为 destroyed（serializer 会用占位符）
                item.destroyed = True
                if isinstance(item.message, ToolMessage):
                    item.message.destroyed = True

    # 不再需要调用 ContextIR.destroy_ephemeral_items，因为已直接操作 items


def build_tool_keep_counts(
    tools: list[Tool] | None, *, ephemeral_keep_recent: int | None
) -> dict[str, int]:
    """辅助函数：构建 tool_name -> keep_count 映射（用于调试/外部检查）。"""
    tool_keep_counts: dict[str, int] = {}
    for tool in tools or []:
        if tool.ephemeral:
            if ephemeral_keep_recent is not None:
                keep_count = ephemeral_keep_recent
            else:
                keep_count = tool.ephemeral if isinstance(tool.ephemeral, int) else 1
            tool_keep_counts[tool.name] = keep_count
    return tool_keep_counts
