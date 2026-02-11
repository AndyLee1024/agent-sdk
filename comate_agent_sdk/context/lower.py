"""Lowering 管道

将 ContextIR 的高层语义结构转换为 LLM API 所需的 list[BaseMessage] 格式。

数据流：
    ContextIR.lower()
    → LoweringPipeline.lower(context)
    → list[BaseMessage]
    → llm.ainvoke(messages=...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from comate_agent_sdk.context.items import ItemType, SegmentName
from comate_agent_sdk.context.reminder import ReminderPosition, SystemReminder
from comate_agent_sdk.llm.messages import (
    SystemMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from comate_agent_sdk.context.ir import ContextIR
    from comate_agent_sdk.llm.messages import BaseMessage

logger = logging.getLogger("comate_agent_sdk.context.lower")


class LoweringPipeline:
    """Lowering 管道：IR → API messages

    转换规则：
    1. Header → SystemMessage（拼接 system_prompt + subagent + skill 策略）
       cache=True 如果任一 header item 有 cache_hint
    2. Conversation items → 按顺序转为 BaseMessage
    3. 注入 system-reminders（按 position 插入 UserMessage(is_meta=True)）
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

        # Step 3: 注入 system-reminders
        messages = LoweringPipeline._inject_reminders(messages, context.reminders)

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

    @staticmethod
    def _inject_reminders(
        messages: list[BaseMessage],
        reminders: list[SystemReminder],
    ) -> list[BaseMessage]:
        """将 system-reminders 注入到消息列表中

        每个 reminder 根据其 position 属性插入到合适位置。
        注入为 UserMessage(content="<system-reminder>...</system-reminder>", is_meta=True)
        """
        if not reminders:
            return messages

        # 收集需要注入的 reminders
        to_inject: list[SystemReminder] = []
        for reminder in reminders:
            if reminder.should_inject():
                to_inject.append(reminder)

        if not to_inject:
            return messages

        # 按 position 分组
        by_position: dict[ReminderPosition, list[SystemReminder]] = {}
        for reminder in to_inject:
            by_position.setdefault(reminder.position, []).append(reminder)

        result = list(messages)

        # AFTER_SYSTEM: 在 SystemMessage 之后插入
        if ReminderPosition.AFTER_SYSTEM in by_position:
            insert_idx = 0
            for i, msg in enumerate(result):
                if isinstance(msg, SystemMessage):
                    insert_idx = i + 1
                    break
            for reminder in reversed(by_position[ReminderPosition.AFTER_SYSTEM]):
                result.insert(
                    insert_idx,
                    UserMessage(content=reminder.wrap_content(), is_meta=True),
                )

        # LAST_USER: 在最后一个 UserMessage 之后插入
        if ReminderPosition.LAST_USER in by_position:
            insert_idx = len(result)
            for i in range(len(result) - 1, -1, -1):
                if isinstance(result[i], UserMessage):
                    insert_idx = i + 1
                    break
            for reminder in reversed(by_position[ReminderPosition.LAST_USER]):
                result.insert(
                    insert_idx,
                    UserMessage(content=reminder.wrap_content(), is_meta=True),
                )

        # LAST_TOOL_RESULT: 在最后一个 ToolMessage 之后插入
        if ReminderPosition.LAST_TOOL_RESULT in by_position:
            from comate_agent_sdk.llm.messages import ToolMessage

            insert_idx = len(result)
            for i in range(len(result) - 1, -1, -1):
                if isinstance(result[i], ToolMessage):
                    insert_idx = i + 1
                    break
            for reminder in reversed(by_position[ReminderPosition.LAST_TOOL_RESULT]):
                result.insert(
                    insert_idx,
                    UserMessage(content=reminder.wrap_content(), is_meta=True),
                )

        # END: 追加到末尾
        if ReminderPosition.END in by_position:
            for reminder in by_position[ReminderPosition.END]:
                result.append(
                    UserMessage(content=reminder.wrap_content(), is_meta=True),
                )

        return result
