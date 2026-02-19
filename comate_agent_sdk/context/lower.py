"""Lowering 管道

将 ContextIR 的高层语义结构转换为 LLM API 所需的 list[BaseMessage] 格式。

数据流：
    ContextIR.lower()
    → LoweringPipeline.lower(context)
    → list[BaseMessage]
    → llm.ainvoke(messages=...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import (
    SystemMessage,
)

if TYPE_CHECKING:
    from comate_agent_sdk.context.ir import ContextIR
    from comate_agent_sdk.llm.messages import BaseMessage


class LoweringPipeline:
    """Lowering 管道：IR → API messages

    转换规则：
    1. Header → SystemMessage（拼接 system_prompt + subagent + skill 策略）
       cache=True 如果任一 header item 有 cache_hint
    2. Conversation items → 按顺序转为 BaseMessage
    """

    @staticmethod
    def lower(context: ContextIR) -> list[BaseMessage]:
        """将 ContextIR 转换为 API messages 格式

        Args:
            context: ContextIR 实例

        Returns:
            可直接传给 llm.ainvoke() 的消息列表
        """
        messages: list[BaseMessage] = []

        # Step 1: Header → SystemMessage
        header_text = LoweringPipeline._build_header_text(context)
        if header_text:
            # 判断是否需要 cache hint
            cache = any(item.cache_hint for item in context.header.items)
            messages.append(SystemMessage(content=header_text, cache=cache))

        # Step 1.5: Memory → UserMessage (在 SystemMessage 之后、conversation 之前)
        if context.memory_item and context.memory_item.message:
            messages.append(context.memory_item.message)

        # Step 2: Conversation items → BaseMessage
        for item in context.conversation.items:
            if item.destroyed:
                # 已销毁的条目仍然需要保留在列表中（serializer 会用 placeholder）
                if item.message is not None:
                    messages.append(item.message)
                continue

            if item.message is not None:
                messages.append(item.message)

        return messages

    @staticmethod
    def _build_header_text(context: ContextIR) -> str:
        """拼接 header 段的各部分文本

        顺序：system_prompt → agent_loop → tool_strategy → subagent_strategy → skill_strategy → system_env → git_env
        注意：memory 不再在 header 中，而是作为独立 UserMessage 注入
        """
        parts: list[str] = []

        # Header 段中的各类型按固定顺序拼接
        type_order = [
            ItemType.SYSTEM_PROMPT,
            ItemType.AGENT_LOOP,
            ItemType.TOOL_STRATEGY,
            ItemType.MCP_TOOL,
            ItemType.SUBAGENT_STRATEGY,
            ItemType.SKILL_STRATEGY,
            ItemType.SYSTEM_ENV,
            ItemType.GIT_ENV,
        ]

        for item_type in type_order:
            item = context.header.find_one_by_type(item_type)
            if item and item.content_text:
                parts.append(item.content_text)

        return "\n".join(parts) if parts else ""
