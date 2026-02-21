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

from comate_agent_sdk.context.header.order import (
    SESSION_STATE_ITEM_TYPES_IN_ORDER,
    STATIC_HEADER_ITEM_TYPES_IN_ORDER,
)
from comate_agent_sdk.llm.messages import (
    SystemMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from comate_agent_sdk.context.ir import ContextIR
    from comate_agent_sdk.llm.messages import BaseMessage


class LoweringPipeline:
    """Lowering 管道：IR → API messages

    转换规则：
    1. Static Header → SystemMessage
       cache=True 如果任一 static header item 有 cache_hint
    2. Memory → UserMessage(is_meta=True)
    3. Session State → UserMessage(is_meta=True)
    4. Conversation items → 按顺序转为 BaseMessage
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

        # Step 1: Static Header → SystemMessage
        header_text = LoweringPipeline._build_static_header_text(context)
        if header_text:
            # 判断是否需要 cache hint
            cache = any(item.cache_hint for item in context.header.items)
            messages.append(SystemMessage(content=header_text, cache=cache))

        # Step 2: Memory → UserMessage(is_meta=True)
        if context.memory_item and context.memory_item.message:
            messages.append(context.memory_item.message)

        # Step 3: Session State → UserMessage(is_meta=True)
        session_state_text = LoweringPipeline._build_session_state_text(context)
        if session_state_text:
            messages.append(UserMessage(content=session_state_text, is_meta=True))

        # Step 4: Conversation items → BaseMessage
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
    def _build_static_header_text(context: ContextIR) -> str:
        """拼接 static header 段的各部分文本。

        顺序：system_prompt → agent_loop → tool_strategy → subagent_strategy → skill_strategy
        """
        parts: list[str] = []

        for item_type in STATIC_HEADER_ITEM_TYPES_IN_ORDER:
            item = context.header.find_one_by_type(item_type)
            if item and item.content_text:
                parts.append(item.content_text)

        return "\n".join(parts) if parts else ""

    @staticmethod
    def _build_session_state_text(context: ContextIR) -> str:
        """拼接 session_state 段文本。

        顺序：system_env → git_env → mcp_tool → output_style
        """
        parts: list[str] = []

        for item_type in SESSION_STATE_ITEM_TYPES_IN_ORDER:
            item = context.session_state.find_one_by_type(item_type)
            if item and item.content_text:
                parts.append(item.content_text)

        return "\n".join(parts) if parts else ""
