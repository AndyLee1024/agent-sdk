from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bu_agent_sdk.llm.messages import BaseMessage, ToolMessage
from bu_agent_sdk.tools.decorator import Tool

logger = logging.getLogger("bu_agent_sdk.agent")

if TYPE_CHECKING:
    from bu_agent_sdk.agent.core import Agent


def clear_history(agent: "Agent") -> None:
    """清空 message history 与 token usage，并重建 ContextIR header。"""
    agent._context.clear()
    agent._token_cost.clear_history()

    # 重建 IR header 各独立段
    resolved_prompt = agent._resolve_system_prompt()
    if resolved_prompt:
        agent._context.set_system_prompt(resolved_prompt, cache=True)

    # 重建 tool_strategy
    from bu_agent_sdk.agent.tool_strategy import generate_tool_strategy

    tool_strategy = generate_tool_strategy(agent.tools)
    if tool_strategy:
        agent._context.set_tool_strategy(tool_strategy)

    # 重建 subagent_strategy（如果有 agents）
    if agent.agents:
        from bu_agent_sdk.subagent.prompts import generate_subagent_prompt

        agent._context.set_subagent_strategy(generate_subagent_prompt(agent.agents))

    # 重建 skill_strategy（如果有 skills）
    if agent.skills:
        from bu_agent_sdk.skill.prompts import generate_skill_prompt

        skill_prompt = generate_skill_prompt(agent.skills)
        if skill_prompt:
            agent._context.set_skill_strategy(skill_prompt)

    # 如果有 memory，重新加载
    if agent.memory:
        agent._setup_memory()


def load_history(agent: "Agent", messages: list[BaseMessage]) -> None:
    """加载 message history（保留 header），用于恢复对话。"""
    # 清空现有 conversation（保留 header）
    agent._context.conversation.items.clear()
    agent._token_cost.clear_history()

    # 逐条加载消息到 IR
    for msg in messages:
        agent._context.add_message(msg)


def destroy_ephemeral_messages(agent: "Agent") -> None:
    """销毁旧的 ephemeral tool 输出（保留每个 tool 最近 N 条）。"""
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
                        path = agent._context_fs.offload(item)
                        item.offload_path = path
                        item.offloaded = True
                        # 同步更新 message 对象
                        if isinstance(item.message, ToolMessage):
                            item.message.offloaded = True
                            item.message.offload_path = str(
                                agent._context_fs.root_path / path
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

