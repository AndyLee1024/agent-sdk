"""ContextIR 主类

结构化管理 Agent 上下文，替代扁平的 _messages: list[BaseMessage]。

核心职责：
1. 结构化管理 — 消息按语义类型分段（header / conversation）
2. 选择性压缩 — 按类型优先级逐步压缩
3. system-reminder — 动态注入，不污染核心上下文
4. 上下文预算 — 每类型可设 token 限额
5. 可观测性 — 所有 context 变异产生事件
6. Lowering — IR → list[BaseMessage] 转换
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from comate_agent_sdk.context.budget import BudgetConfig, BudgetStatus, TokenCounter
from comate_agent_sdk.context.items import (
    DEFAULT_PRIORITIES,
    ContextItem,
    ItemType,
    Segment,
    SegmentName,
)
from comate_agent_sdk.context.lower import LoweringPipeline
from comate_agent_sdk.context.observer import (
    ContextEvent,
    ContextEventBus,
    EventType,
)
from comate_agent_sdk.context.reminder import SystemReminder
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from comate_agent_sdk.context.compaction import SelectiveCompactionPolicy

logger = logging.getLogger("comate_agent_sdk.context.ir")

_HEADER_ITEM_ORDER: dict[ItemType, int] = {
    ItemType.SYSTEM_PROMPT: 0,
    ItemType.AGENT_LOOP: 1,
    ItemType.MEMORY: 2,
    ItemType.TOOL_STRATEGY: 3,
    ItemType.MCP_TOOL: 4,
    ItemType.SUBAGENT_STRATEGY: 5,
    ItemType.SKILL_STRATEGY: 6,
    ItemType.SYSTEM_ENV: 7,
    ItemType.GIT_ENV: 8,
}


# 消息类型 → ItemType 映射
_MESSAGE_TYPE_MAP: dict[type, ItemType] = {
    UserMessage: ItemType.USER_MESSAGE,
    AssistantMessage: ItemType.ASSISTANT_MESSAGE,
    ToolMessage: ItemType.TOOL_RESULT,
    DeveloperMessage: ItemType.DEVELOPER_MESSAGE,
    SystemMessage: ItemType.SYSTEM_PROMPT,
}


@dataclass
class ContextIR:
    """Context 中间表示

    将 Agent 的上下文从扁平 list[BaseMessage] 提升为结构化 IR。

    Segments:
        header: system_prompt + memory + subagent_strategy + skill_strategy
        conversation: 所有对话消息

    Attributes:
        budget: 预算配置
        token_counter: token 计数器
        event_bus: 事件总线
        reminders: 系统提醒列表
        _todo_state: TODO 状态存储 {todos: [...], updated_at: timestamp}
    """

    header: Segment = field(
        default_factory=lambda: Segment(name=SegmentName.HEADER)
    )
    conversation: Segment = field(
        default_factory=lambda: Segment(name=SegmentName.CONVERSATION)
    )
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    token_counter: TokenCounter = field(default_factory=TokenCounter)
    event_bus: ContextEventBus = field(default_factory=ContextEventBus)
    reminders: list[SystemReminder] = field(default_factory=list)
    _pending_skill_items: list[ContextItem] = field(default_factory=list)
    _todo_state: dict[str, Any] = field(default_factory=dict)
    _turn_number: int = 0
    _todo_turn_number_at_update: int = 0
    _todo_empty_reminder_start_turn: int = 0  # 空 todo 提醒开始的轮次

    # ===== Header 操作（幂等，重复调用会覆盖） =====

    def _ensure_header_item_position(self, item: ContextItem) -> None:
        """确保 header item 在预期顺序中的位置（用于幂等 setter 的插入顺序稳定）"""
        try:
            current_idx = self.header.items.index(item)
        except ValueError:
            return

        order = _HEADER_ITEM_ORDER.get(item.item_type)
        if order is None:
            return

        item_ref = self.header.items.pop(current_idx)

        insert_idx = len(self.header.items)
        for i, existing_item in enumerate(self.header.items):
            existing_order = _HEADER_ITEM_ORDER.get(existing_item.item_type)
            if existing_order is None:
                continue
            if existing_order > order:
                insert_idx = i
                break

        self.header.items.insert(insert_idx, item_ref)

    def set_system_prompt(self, prompt: str, cache: bool = True) -> None:
        """设置系统提示（幂等覆盖）

        Args:
            prompt: 系统提示文本
            cache: 是否建议缓存（如 Anthropic prompt cache）
        """
        existing = self.header.find_one_by_type(ItemType.SYSTEM_PROMPT)
        token_count = self.token_counter.count(prompt)

        if existing:
            existing.content_text = prompt
            existing.token_count = token_count
            existing.cache_hint = cache
        else:
            item = ContextItem(
                item_type=ItemType.SYSTEM_PROMPT,
                content_text=prompt,
                token_count=token_count,
                priority=DEFAULT_PRIORITIES[ItemType.SYSTEM_PROMPT],
                cache_hint=cache,
            )
            self.header.items.insert(0, item)  # system_prompt 总在最前面
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=ItemType.SYSTEM_PROMPT,
                item_id=item.id,
                detail="system_prompt set",
            ))

    def set_agent_loop(self, prompt: str, cache: bool = False) -> None:
        """设置 Agent 循环控制指令（幂等覆盖）

        插入位置：SYSTEM_PROMPT 之后、MEMORY 之前

        Args:
            prompt: Agent 循环控制指令文本
            cache: 是否建议缓存（如 Anthropic prompt cache）
        """
        existing = self.header.find_one_by_type(ItemType.AGENT_LOOP)
        token_count = self.token_counter.count(prompt)

        if existing:
            existing.content_text = prompt
            existing.token_count = token_count
            existing.cache_hint = cache
        else:
            item = ContextItem(
                item_type=ItemType.AGENT_LOOP,
                content_text=prompt,
                token_count=token_count,
                priority=DEFAULT_PRIORITIES[ItemType.AGENT_LOOP],
                cache_hint=cache,
            )
            # 在 SYSTEM_PROMPT 之后、MEMORY/TOOL_STRATEGY/SUBAGENT_STRATEGY/SKILL_STRATEGY 之前插入
            insert_idx = 0
            for i, existing_item in enumerate(self.header.items):
                if existing_item.item_type == ItemType.SYSTEM_PROMPT:
                    insert_idx = i + 1
                elif existing_item.item_type in (
                    ItemType.MEMORY,
                    ItemType.TOOL_STRATEGY,
                    ItemType.SUBAGENT_STRATEGY,
                    ItemType.SKILL_STRATEGY,
                ):
                    insert_idx = i
                    break
            self.header.items.insert(insert_idx, item)
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=ItemType.AGENT_LOOP,
                item_id=item.id,
                detail="agent_loop set",
            ))

    def set_memory(self, content: str, cache: bool = True) -> None:
        """设置 MEMORY 静态背景知识（幂等覆盖）

        Args:
            content: MEMORY 内容
            cache: 是否建议缓存（如 Anthropic prompt cache）
        """
        # 延迟导入避免循环依赖
        from comate_agent_sdk.agent.prompts import MEMORY_NOTICE

        # 包裹 <memory> 标签
        wrapped_content = f"<memory>\n{content}\n {MEMORY_NOTICE}\n</memory>"

        existing = self.header.find_one_by_type(ItemType.MEMORY)
        token_count = self.token_counter.count(wrapped_content)

        if existing:
            existing.content_text = wrapped_content
            existing.token_count = token_count
            existing.cache_hint = cache
        else:
            item = ContextItem(
                item_type=ItemType.MEMORY,
                content_text=wrapped_content,
                token_count=token_count,
                priority=DEFAULT_PRIORITIES[ItemType.MEMORY],
                cache_hint=cache,
            )
            # 在 system_prompt/agent_loop 之后、tool_strategy/subagent_strategy 之前插入
            insert_idx = len(self.header.items)  # 默认插入到末尾
            for i, existing_item in enumerate(self.header.items):
                if existing_item.item_type in (ItemType.SYSTEM_PROMPT, ItemType.AGENT_LOOP):
                    insert_idx = i + 1
                elif existing_item.item_type in (ItemType.TOOL_STRATEGY, ItemType.SUBAGENT_STRATEGY):
                    insert_idx = i
                    break
            self.header.items.insert(insert_idx, item)
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=ItemType.MEMORY,
                item_id=item.id,
                detail="memory set",
            ))

    def set_subagent_strategy(self, prompt: str) -> None:
        """设置 Subagent 策略提示（幂等覆盖）"""
        existing = self.header.find_one_by_type(ItemType.SUBAGENT_STRATEGY)
        token_count = self.token_counter.count(prompt)

        if existing:
            existing.content_text = prompt
            existing.token_count = token_count
        else:
            item = ContextItem(
                item_type=ItemType.SUBAGENT_STRATEGY,
                content_text=prompt,
                token_count=token_count,
                priority=DEFAULT_PRIORITIES[ItemType.SUBAGENT_STRATEGY],
            )
            # 在 system_prompt/agent_loop/memory/tool_strategy 之后、skill_strategy 之前插入
            insert_idx = 0
            for i, existing_item in enumerate(self.header.items):
                if existing_item.item_type in (ItemType.SYSTEM_PROMPT, ItemType.AGENT_LOOP, ItemType.MEMORY, ItemType.TOOL_STRATEGY):
                    insert_idx = i + 1
                elif existing_item.item_type == ItemType.SKILL_STRATEGY:
                    break
            self.header.items.insert(insert_idx, item)
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=ItemType.SUBAGENT_STRATEGY,
                item_id=item.id,
                detail="subagent_strategy set",
            ))

    def set_tool_strategy(self, prompt: str) -> None:
        """设置 Tool 策略提示（幂等覆盖）

        插入位置：MEMORY 之后、SUBAGENT_STRATEGY 之前
        """
        existing = self.header.find_one_by_type(ItemType.TOOL_STRATEGY)
        token_count = self.token_counter.count(prompt)

        if existing:
            existing.content_text = prompt
            existing.token_count = token_count
        else:
            item = ContextItem(
                item_type=ItemType.TOOL_STRATEGY,
                content_text=prompt,
                token_count=token_count,
                priority=DEFAULT_PRIORITIES[ItemType.TOOL_STRATEGY],
            )
            # 在 system_prompt/agent_loop/memory 之后、subagent_strategy/skill_strategy 之前插入
            insert_idx = 0
            for i, existing_item in enumerate(self.header.items):
                if existing_item.item_type in (ItemType.SYSTEM_PROMPT, ItemType.AGENT_LOOP, ItemType.MEMORY):
                    insert_idx = i + 1
                elif existing_item.item_type in (ItemType.SUBAGENT_STRATEGY, ItemType.SKILL_STRATEGY):
                    break
            self.header.items.insert(insert_idx, item)
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=ItemType.TOOL_STRATEGY,
                item_id=item.id,
                detail="tool_strategy set",
            ))

    def set_mcp_tools(self, overview_text: str, *, metadata: dict[str, Any] | None = None) -> None:
        """设置 MCP tools 概览（幂等覆盖）

        设计目标：
        - 将 MCP tool 的概览注入到 header（lowered 后进入 SystemMessage）
        - 结构化信息保存在 ContextItem.metadata（不直接进入 prompt，避免 token 爆炸）
        """
        existing = self.header.find_one_by_type(ItemType.MCP_TOOL)
        token_count = self.token_counter.count(overview_text)
        meta = metadata or {}

        if existing:
            existing.content_text = overview_text
            existing.token_count = token_count
            existing.metadata = meta
            self._ensure_header_item_position(existing)
            return

        item = ContextItem(
            item_type=ItemType.MCP_TOOL,
            content_text=overview_text,
            token_count=token_count,
            priority=DEFAULT_PRIORITIES[ItemType.MCP_TOOL],
            metadata=meta,
        )
        self.header.items.append(item)
        self._ensure_header_item_position(item)
        self.event_bus.emit(ContextEvent(
            event_type=EventType.ITEM_ADDED,
            item_type=ItemType.MCP_TOOL,
            item_id=item.id,
            detail="mcp_tools set",
        ))

    def remove_mcp_tools(self) -> None:
        """移除 MCP tools 概览。"""
        removed = self.header.remove_by_type(ItemType.MCP_TOOL)
        for item in removed:
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_REMOVED,
                item_type=ItemType.MCP_TOOL,
                item_id=item.id,
                detail="mcp_tools removed",
            ))

    def set_skill_strategy(self, prompt: str) -> None:
        """设置 Skill 策略提示（幂等覆盖）"""
        existing = self.header.find_one_by_type(ItemType.SKILL_STRATEGY)
        token_count = self.token_counter.count(prompt)

        if existing:
            existing.content_text = prompt
            existing.token_count = token_count
            self._ensure_header_item_position(existing)
        else:
            item = ContextItem(
                item_type=ItemType.SKILL_STRATEGY,
                content_text=prompt,
                token_count=token_count,
                priority=DEFAULT_PRIORITIES[ItemType.SKILL_STRATEGY],
            )
            self.header.items.append(item)
            self._ensure_header_item_position(item)
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=ItemType.SKILL_STRATEGY,
                item_id=item.id,
                detail="skill_strategy set",
            ))

    def set_system_env(self, content: str) -> None:
        """设置系统环境信息（幂等覆盖）

        插入位置：Header 段末尾（SKILL_STRATEGY 之后）
        """
        existing = self.header.find_one_by_type(ItemType.SYSTEM_ENV)
        token_count = self.token_counter.count(content)

        if existing:
            existing.content_text = content
            existing.token_count = token_count
            self._ensure_header_item_position(existing)
        else:
            item = ContextItem(
                item_type=ItemType.SYSTEM_ENV,
                content_text=content,
                token_count=token_count,
                priority=DEFAULT_PRIORITIES[ItemType.SYSTEM_ENV],
            )
            self.header.items.append(item)
            self._ensure_header_item_position(item)
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=ItemType.SYSTEM_ENV,
                item_id=item.id,
                detail="system_env set",
            ))

    def set_git_env(self, content: str) -> None:
        """设置 Git 状态信息（幂等覆盖）

        插入位置：SYSTEM_ENV 之后（Header 段最末尾）
        """
        existing = self.header.find_one_by_type(ItemType.GIT_ENV)
        token_count = self.token_counter.count(content)

        if existing:
            existing.content_text = content
            existing.token_count = token_count
            self._ensure_header_item_position(existing)
        else:
            item = ContextItem(
                item_type=ItemType.GIT_ENV,
                content_text=content,
                token_count=token_count,
                priority=DEFAULT_PRIORITIES[ItemType.GIT_ENV],
            )
            self.header.items.append(item)
            self._ensure_header_item_position(item)
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=ItemType.GIT_ENV,
                item_id=item.id,
                detail="git_env set",
            ))

    def remove_skill_strategy(self) -> None:
        """移除 Skill 策略提示"""
        removed = self.header.remove_by_type(ItemType.SKILL_STRATEGY)
        for item in removed:
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_REMOVED,
                item_type=ItemType.SKILL_STRATEGY,
                item_id=item.id,
                detail="skill_strategy removed",
            ))

    @property
    def system_prompt_text(self) -> str | None:
        """Header 各段拼接后的完整系统提示文本（与 lower 时一致）"""
        text = LoweringPipeline._build_header_text(self)
        return text if text else None

    # ===== Conversation 操作 =====

    def add_message(
        self,
        message: BaseMessage,
        *,
        item_type: ItemType | None = None,
        ephemeral: bool = False,
        tool_name: str | None = None,
        metadata: dict | None = None,
        cache_hint: bool = False,
    ) -> ContextItem:
        """添加消息到 conversation 段

        自动推断 ItemType：
        - UserMessage → USER_MESSAGE
        - AssistantMessage → ASSISTANT_MESSAGE
        - ToolMessage → TOOL_RESULT
        - DeveloperMessage → DEVELOPER_MESSAGE
        - SystemMessage → 调用 set_system_prompt() 处理

        Args:
            message: 消息对象
            item_type: 可选手动指定类型
            ephemeral: 是否临时内容
            tool_name: 工具名称（ToolMessage 时使用）
            metadata: 附加元数据
            cache_hint: 是否建议缓存

        Returns:
            创建的 ContextItem
        """
        # SystemMessage 特殊处理：走 header
        if isinstance(message, SystemMessage):
            self.set_system_prompt(
                message.text,
                cache=message.cache,
            )
            # 仍然返回一个 ContextItem 但不加入 conversation
            return ContextItem(
                item_type=ItemType.SYSTEM_PROMPT,
                message=message,
                content_text=message.text,
                token_count=self.token_counter.count(message.text),
                priority=DEFAULT_PRIORITIES[ItemType.SYSTEM_PROMPT],
                cache_hint=message.cache,
            )

        # 自动推断类型
        if item_type is None:
            item_type = _MESSAGE_TYPE_MAP.get(type(message), ItemType.USER_MESSAGE)

        if (
            item_type == ItemType.USER_MESSAGE
            and isinstance(message, UserMessage)
            and not bool(getattr(message, "is_meta", False))
        ):
            self._turn_number += 1

        # 提取文本内容用于 token 估算
        content_text = message.text if hasattr(message, "text") else ""

        # AssistantMessage 特殊处理：需要包括 tool_calls 的 tokens
        # 因为 tool_calls 也会被发送给 LLM，占用 prompt tokens
        if isinstance(message, AssistantMessage) and message.tool_calls:
            import json

            tool_calls_json = json.dumps(
                [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
                ensure_ascii=False,
            )
            # 如果有文本内容，拼接；否则只用 tool_calls
            content_text = content_text + "\n" + tool_calls_json if content_text else tool_calls_json

        token_count = self.token_counter.count(content_text)

        # ToolMessage 特殊属性
        is_tool_error = False
        item_metadata = dict(metadata or {})
        truncation_record = None
        if isinstance(message, ToolMessage):
            ephemeral = ephemeral or message.ephemeral
            tool_name = tool_name or message.tool_name
            is_tool_error = message.is_error
            if message.execution_meta:
                item_metadata["tool_execution_meta"] = message.execution_meta
            if message.raw_envelope:
                item_metadata["tool_raw_envelope"] = message.raw_envelope
            truncation_record = getattr(message, "truncation_record", None)

        item = ContextItem(
            item_type=item_type,
            message=message,
            content_text=content_text,
            token_count=token_count,
            priority=DEFAULT_PRIORITIES.get(item_type, 50),
            ephemeral=ephemeral,
            tool_name=tool_name,
            metadata=item_metadata,
            truncation_record=truncation_record,
            cache_hint=cache_hint,
            is_tool_error=is_tool_error,
            created_turn=self._turn_number,
        )

        self.conversation.items.append(item)

        self.event_bus.emit(ContextEvent(
            event_type=EventType.ITEM_ADDED,
            item_type=item_type,
            item_id=item.id,
            detail=f"message added: {item_type.value}",
        ))

        return item

    def set_turn_number(self, turn_number: int) -> None:
        """设置/纠正 turn_number（单调不减）。

        主要用于 ChatSession.resume() 从持久化恢复 turn_number。
        """
        try:
            value = int(turn_number)
        except Exception:
            return
        if value > self._turn_number:
            self._turn_number = value

    def get_turn_number(self) -> int:
        """获取当前 turn_number（真实用户输入轮次）。"""
        return int(self._turn_number)

    def cleanup_stale_error_items(self, max_turns: int = 10) -> list[str]:
        """清理超过指定轮次的失败工具调用项

        同时删除失败的 tool_result 及其关联的 AssistantMessage(保持消息配对完整性)

        Args:
            max_turns: 最大保留轮次(默认10轮后删除)

        Returns:
            被删除的 item ID 列表
        """
        current_turn = self._turn_number
        removed_ids: list[str] = []

        # 1. 收集需要删除的失败 tool_result
        error_tool_results: list[ContextItem] = []
        for item in self.conversation.items:
            if (item.item_type == ItemType.TOOL_RESULT
                and item.is_tool_error
                and (current_turn - item.created_turn) >= max_turns):
                error_tool_results.append(item)

        if not error_tool_results:
            return removed_ids

        # 2. 收集需要删除的关联 AssistantMessage
        error_tool_call_ids = set()
        for item in error_tool_results:
            if isinstance(item.message, ToolMessage):
                error_tool_call_ids.add(item.message.tool_call_id)

        assistant_items_to_remove: list[str] = []
        for item in self.conversation.items:
            if item.item_type != ItemType.ASSISTANT_MESSAGE:
                continue
            if not isinstance(item.message, AssistantMessage):
                continue
            if not item.message.tool_calls:
                continue
            # 检查是否所有 tool_calls 都是失败的
            all_failed = all(
                tc.id in error_tool_call_ids
                for tc in item.message.tool_calls
            )
            if all_failed:
                assistant_items_to_remove.append(item.id)

        # 3. 执行删除(先删除 tool_result,再删除 assistant)
        for item in error_tool_results:
            if self.conversation.remove_by_id(item.id):
                removed_ids.append(item.id)
                self.event_bus.emit(ContextEvent(
                    event_type=EventType.ITEM_REMOVED,
                    item_type=ItemType.TOOL_RESULT,
                    item_id=item.id,
                    detail=f"stale error tool_result removed (turn gap={current_turn - item.created_turn})",
                ))

        for item_id in assistant_items_to_remove:
            if self.conversation.remove_by_id(item_id):
                removed_ids.append(item_id)
                self.event_bus.emit(ContextEvent(
                    event_type=EventType.ITEM_REMOVED,
                    item_type=ItemType.ASSISTANT_MESSAGE,
                    item_id=item_id,
                    detail="associated assistant message removed (all tool_calls failed)",
                ))

        return removed_ids

    def add_skill_injection(
        self,
        skill_name: str,
        metadata_msg: UserMessage,
        prompt_msg: UserMessage,
    ) -> list[ContextItem]:
        """添加 Skill 注入（存入 pending，flush 后加入 conversation）

        Args:
            skill_name: Skill 名称
            metadata_msg: Skill 元数据消息
            prompt_msg: Skill 提示消息

        Returns:
            创建的 ContextItem 列表
        """
        metadata_item = ContextItem(
            item_type=ItemType.SKILL_METADATA,
            message=metadata_msg,
            content_text=metadata_msg.text,
            token_count=self.token_counter.count(metadata_msg.text),
            priority=DEFAULT_PRIORITIES[ItemType.SKILL_METADATA],
            metadata={"skill_name": skill_name},
        )
        prompt_item = ContextItem(
            item_type=ItemType.SKILL_PROMPT,
            message=prompt_msg,
            content_text=prompt_msg.text,
            token_count=self.token_counter.count(prompt_msg.text),
            priority=DEFAULT_PRIORITIES[ItemType.SKILL_PROMPT],
            metadata={"skill_name": skill_name},
        )

        self._pending_skill_items = [metadata_item, prompt_item]
        return [metadata_item, prompt_item]

    def flush_pending_skill_items(self) -> None:
        """将待注入的 Skill items 刷入 conversation"""
        if not self._pending_skill_items:
            return

        for item in self._pending_skill_items:
            self.conversation.items.append(item)
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=item.item_type,
                item_id=item.id,
                detail=f"skill item flushed: {item.item_type.value}",
            ))

        self._pending_skill_items = []

    @property
    def has_pending_skill_items(self) -> bool:
        """是否有待注入的 Skill items"""
        return len(self._pending_skill_items) > 0

    # ===== Reminder =====

    def register_reminder(self, reminder: SystemReminder) -> None:
        """注册系统提醒"""
        # 防止重名
        self.reminders = [r for r in self.reminders if r.name != reminder.name]
        self.reminders.append(reminder)
        self.event_bus.emit(ContextEvent(
            event_type=EventType.REMINDER_REGISTERED,
            detail=f"reminder registered: {reminder.name}",
        ))

    def remove_reminder(self, name: str) -> None:
        """移除系统提醒"""
        before_count = len(self.reminders)
        self.reminders = [r for r in self.reminders if r.name != name]
        if len(self.reminders) < before_count:
            self.event_bus.emit(ContextEvent(
                event_type=EventType.REMINDER_REMOVED,
                detail=f"reminder removed: {name}",
            ))

    def cleanup_one_shot_reminders(self) -> None:
        """清理已使用的一次性提醒"""
        self.reminders = [r for r in self.reminders if not r.one_shot]

    # ===== Ephemeral =====

    def destroy_ephemeral_items(
        self, tool_keep_counts: dict[str, int]
    ) -> list[str]:
        """销毁旧的 ephemeral items

        对每个工具名，保留最近 N 个 ephemeral 条目，销毁更早的。

        Args:
            tool_keep_counts: 工具名 → 保留数量

        Returns:
            被销毁的 item ID 列表
        """
        # 按工具名分组 ephemeral items
        ephemeral_by_tool: dict[str, list[ContextItem]] = {}
        for item in self.conversation.items:
            if item.item_type != ItemType.TOOL_RESULT:
                continue
            if not item.ephemeral or item.destroyed:
                continue
            tool_name = item.tool_name or ""
            ephemeral_by_tool.setdefault(tool_name, []).append(item)

        destroyed_ids: list[str] = []

        for tool_name, items in ephemeral_by_tool.items():
            keep_count = tool_keep_counts.get(tool_name, 1)
            items_to_destroy = items[:-keep_count] if keep_count > 0 else items

            for item in items_to_destroy:
                item.destroyed = True
                # 同步到底层消息
                if isinstance(item.message, ToolMessage):
                    item.message.destroyed = True
                destroyed_ids.append(item.id)

                self.event_bus.emit(ContextEvent(
                    event_type=EventType.ITEM_DESTROYED,
                    item_type=ItemType.TOOL_RESULT,
                    item_id=item.id,
                    detail=f"ephemeral destroyed: {tool_name} (keeping last {keep_count})",
                ))

        return destroyed_ids

    # ===== Budget =====

    @property
    def total_tokens(self) -> int:
        """当前总 token 数（估算值）"""
        return self.header.total_tokens + self.conversation.total_tokens

    def get_budget_status(self) -> BudgetStatus:
        """获取预算状态快照"""
        tokens_by_type: dict[ItemType, int] = {}

        for segment in (self.header, self.conversation):
            for item in segment.items:
                if item.destroyed:
                    continue
                current = tokens_by_type.get(item.item_type, 0)
                tokens_by_type[item.item_type] = current + item.token_count

        return BudgetStatus(
            total_tokens=self.total_tokens,
            header_tokens=self.header.total_tokens,
            conversation_tokens=self.conversation.total_tokens,
            tokens_by_type=tokens_by_type,
            total_limit=self.budget.total_limit,
            compact_threshold_ratio=self.budget.compact_threshold_ratio,
        )

    # ===== Lowering =====

    def lower(self) -> list[BaseMessage]:
        """将 IR 转换为 API messages 格式

        Returns:
            可直接传给 llm.ainvoke() 的消息列表
        """
        messages = LoweringPipeline.lower(self)
        # 清理一次性 reminders
        self.cleanup_one_shot_reminders()
        return messages

    # ===== Compaction =====

    async def auto_compact(
        self,
        policy: SelectiveCompactionPolicy,
        current_total_tokens: int | None = None,
    ) -> bool:
        """自动压缩

        Args:
            policy: 选择性压缩策略
            current_total_tokens: 实际 token 数（来自 LLM response.usage）。
                如果提供则使用实际值判断，否则使用 IR 内估算值。

        Returns:
            是否执行了压缩
        """
        check_tokens = current_total_tokens if current_total_tokens is not None else self.total_tokens

        if not policy.should_compact(check_tokens):
            return False

        result = await policy.compact(self)

        if result:
            self.event_bus.emit(ContextEvent(
                event_type=EventType.COMPACTION_PERFORMED,
                detail=f"auto_compact: {check_tokens} → {self.total_tokens} tokens",
            ))

        return result

    # ===== 序列化 =====

    def clear(self) -> None:
        """清空所有上下文"""
        self.header.items.clear()
        self.conversation.items.clear()
        self._pending_skill_items.clear()
        self.reminders.clear()
        self._todo_state.clear()

        self.event_bus.emit(ContextEvent(
            event_type=EventType.CONTEXT_CLEARED,
            detail="context cleared",
        ))

    @property
    def conversation_messages(self) -> list[BaseMessage]:
        """获取 conversation 段的所有底层消息"""
        return [
            item.message
            for item in self.conversation.items
            if item.message is not None
        ]

    def replace_conversation(self, items: list[ContextItem]) -> None:
        """替换整个 conversation 段

        Args:
            items: 新的 conversation 条目列表
        """
        self.conversation.items = list(items)
        self.event_bus.emit(ContextEvent(
            event_type=EventType.CONVERSATION_REPLACED,
            detail=f"conversation replaced with {len(items)} items",
        ))

    @property
    def is_empty(self) -> bool:
        """上下文是否为空（header 和 conversation 都没有内容）"""
        return not self.header.items and not self.conversation.items

    # ===== TODO 状态管理 =====

    def set_todo_state(self, todos: list[dict]) -> None:
        """设置 TODO 状态并注册 reminder

        Args:
            todos: TODO 列表（字典格式，来自 TodoWrite 工具）
        """
        import time
        from comate_agent_sdk.context.reminder import ReminderPosition

        active_todos = [
            t for t in todos
            if isinstance(t, dict) and t.get("status") in ("pending", "in_progress")
        ]

        self._todo_state = {
            "todos": active_todos,
            "updated_at": time.time(),
        }
        self._todo_turn_number_at_update = self._turn_number

        # 移除旧的 todo reminders
        self.remove_reminder("todo_list_empty")
        self.remove_reminder("todo_list_update")
        self.remove_reminder("todo_gentle_reminder")

        # 根据状态注册新 reminder
        if not active_todos:
            # 如果 todo 为空，注册"空列表"提醒
            # 每次进入空状态时，重置起始轮次（这样会在第一次立即提醒）
            self._todo_empty_reminder_start_turn = self._turn_number
            self.register_reminder(SystemReminder(
                name="todo_list_empty",
                content=(
                    "This is a reminder that your todo list is currently empty. "
                    "DO NOT mention this to the user explicitly because they are already aware. "
                    "If you are working on tasks that would benefit from a todo list please use the TodoWrite tool to create one. "
                    "If not, please feel free to ignore. "
                    "Again do not mention this message to the user."
                ),
                position=ReminderPosition.END,
                one_shot=False,
                condition=self._should_remind_empty_todo,  # 每 8 轮提醒一次
            ))
        else:
            # 如果 todo 非空，重置空提醒起始轮次（下次变空时重新计时）
            self._todo_empty_reminder_start_turn = 0
            # 注册"温和提醒"（不包含完整 JSON）
            self.register_reminder(SystemReminder(
                name="todo_gentle_reminder",
                content=(
                    "You have active TODO items. DO NOT mention this reminder to the user. "
                    "Continue working on the next TODO item(s) and keep the TODO list up to date via the TodoWrite tool."
                ),
                position=ReminderPosition.END,
                one_shot=False,  # 持续提醒直到下次更新
                condition=self._should_remind_todo,
            ))

        # 触发事件
        self.event_bus.emit(ContextEvent(
            event_type=EventType.TODO_STATE_UPDATED,
            detail=f"todo_state updated: {len(active_todos)} active items",
        ))

    def restore_todo_state(
        self,
        *,
        todos: list[dict],
        turn_number_at_update: int,
    ) -> None:
        """从持久化数据恢复 TODO 状态（用于 session resume）。

        与 set_todo_state() 的区别：
        - 不用当前 turn_number 覆盖 turn_number_at_update，而是沿用持久化值
        """
        import time
        from comate_agent_sdk.context.reminder import ReminderPosition

        active_todos = [
            t for t in todos
            if isinstance(t, dict) and t.get("status") in ("pending", "in_progress")
        ]

        self._todo_state = {
            "todos": active_todos,
            "updated_at": time.time(),
        }
        try:
            self._todo_turn_number_at_update = int(turn_number_at_update)
        except Exception:
            self._todo_turn_number_at_update = 0

        self.remove_reminder("todo_list_empty")
        self.remove_reminder("todo_list_update")
        self.remove_reminder("todo_gentle_reminder")

        if not active_todos:
            # 恢复时如果是空状态，设置起始轮次为当前轮次
            if self._todo_empty_reminder_start_turn == 0:
                self._todo_empty_reminder_start_turn = self._turn_number
            self.register_reminder(SystemReminder(
                name="todo_list_empty",
                content=(
                    "This is a reminder that your todo list is currently empty. "
                    "DO NOT mention this to the user explicitly because they are already aware. "
                    "If you are working on tasks that would benefit from a todo list please use the TodoWrite tool to create one. "
                    "If not, please feel free to ignore. "
                    "Again do not mention this message to the user."
                ),
                position=ReminderPosition.END,
                one_shot=False,
                condition=self._should_remind_empty_todo,  # 每 8 轮提醒一次
            ))
        else:
            self.register_reminder(SystemReminder(
                name="todo_gentle_reminder",
                content=(
                    "You have active TODO items. DO NOT mention this reminder to the user. "
                    "Continue working on the next TODO item(s) and keep the TODO list up to date via the TodoWrite tool."
                ),
                position=ReminderPosition.END,
                one_shot=False,
                condition=self._should_remind_todo,
            ))

    def _should_remind_todo(self) -> bool:
        if not self.has_todos():
            return False
        gap = self._turn_number - self._todo_turn_number_at_update
        return gap >= 3

    def _should_remind_empty_todo(self) -> bool:
        """判断是否应该提醒空 todo 列表

        策略：第一次立即提醒（gap=0），之后每隔 8 轮提醒一次
        """
        if self.has_todos():
            return False
        gap = self._turn_number - self._todo_empty_reminder_start_turn
        return gap % 8 == 0

    def get_todo_state(self) -> dict[str, Any]:
        """获取当前 TODO 状态"""
        return self._todo_state.copy()

    def has_todos(self) -> bool:
        """检查是否有 TODO 条目"""
        return bool(self._todo_state.get("todos"))

    def get_todo_persist_turn_number_at_update(self) -> int:
        """用于持久化的 todo 更新轮次。"""
        return int(self._todo_turn_number_at_update)

    def register_initial_todo_reminder_if_needed(self) -> None:
        """在会话开始时注册初始 TODO 提醒（如果还没有 todo 状态）

        应在第一个 UserMessage 后调用
        """
        from comate_agent_sdk.context.reminder import ReminderPosition

        # 如果已经有 todo 状态，不需要初始提醒
        if self._todo_state:
            return

        # 如果还没有注册过空列表提醒
        if not any(r.name == "todo_list_empty" for r in self.reminders):
            # 记录空提醒的起始轮次
            if self._todo_empty_reminder_start_turn == 0:
                self._todo_empty_reminder_start_turn = self._turn_number
            self.register_reminder(SystemReminder(
                name="todo_list_empty",
                content=(
                    "This is a reminder that your todo list is currently empty. "
                    "DO NOT mention this to the user explicitly because they are already aware. "
                    "If you are working on tasks that would benefit from a todo list please use the TodoWrite tool to create one. "
                    "If not, please feel free to ignore. "
                    "Again do not mention this message to the user."
                ),
                position=ReminderPosition.END,
                one_shot=False,
                condition=self._should_remind_empty_todo,  # 每 8 轮提醒一次
            ))
