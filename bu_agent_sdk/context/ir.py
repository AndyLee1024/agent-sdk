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

from bu_agent_sdk.context.budget import BudgetConfig, BudgetStatus, TokenCounter
from bu_agent_sdk.context.items import (
    DEFAULT_PRIORITIES,
    ContextItem,
    ItemType,
    Segment,
    SegmentName,
)
from bu_agent_sdk.context.lower import LoweringPipeline
from bu_agent_sdk.context.observer import (
    ContextEvent,
    ContextEventBus,
    EventType,
)
from bu_agent_sdk.context.reminder import SystemReminder
from bu_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from bu_agent_sdk.context.compaction import SelectiveCompactionPolicy

logger = logging.getLogger("bu_agent_sdk.context.ir")


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

    # ===== Header 操作（幂等，重复调用会覆盖） =====

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

    def set_memory(self, content: str, cache: bool = True) -> None:
        """设置 MEMORY 静态背景知识（幂等覆盖）

        Args:
            content: MEMORY 内容
            cache: 是否建议缓存（如 Anthropic prompt cache）
        """
        # 延迟导入避免循环依赖
        from bu_agent_sdk.agent.prompts import MEMORY_NOTICE

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
            # 在 system_prompt 之后、subagent_strategy 之前插入
            insert_idx = len(self.header.items)  # 默认插入到末尾
            for i, existing_item in enumerate(self.header.items):
                if existing_item.item_type == ItemType.SYSTEM_PROMPT:
                    insert_idx = i + 1
                elif existing_item.item_type == ItemType.SUBAGENT_STRATEGY:
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
            # 在 system_prompt/memory 之后、skill_strategy 之前插入
            insert_idx = 0
            for i, existing_item in enumerate(self.header.items):
                if existing_item.item_type in (ItemType.SYSTEM_PROMPT, ItemType.MEMORY):
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
            # 在 system_prompt/memory 之后、subagent_strategy/skill_strategy 之前插入
            insert_idx = 0
            for i, existing_item in enumerate(self.header.items):
                if existing_item.item_type in (ItemType.SYSTEM_PROMPT, ItemType.MEMORY):
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

    def set_skill_strategy(self, prompt: str) -> None:
        """设置 Skill 策略提示（幂等覆盖）"""
        existing = self.header.find_one_by_type(ItemType.SKILL_STRATEGY)
        token_count = self.token_counter.count(prompt)

        if existing:
            existing.content_text = prompt
            existing.token_count = token_count
        else:
            item = ContextItem(
                item_type=ItemType.SKILL_STRATEGY,
                content_text=prompt,
                token_count=token_count,
                priority=DEFAULT_PRIORITIES[ItemType.SKILL_STRATEGY],
            )
            self.header.items.append(item)  # skill_strategy 总在最后
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=ItemType.SKILL_STRATEGY,
                item_id=item.id,
                detail="skill_strategy set",
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
        if isinstance(message, ToolMessage):
            ephemeral = ephemeral or message.ephemeral
            tool_name = tool_name or message.tool_name

        item = ContextItem(
            item_type=item_type,
            message=message,
            content_text=content_text,
            token_count=token_count,
            priority=DEFAULT_PRIORITIES.get(item_type, 50),
            ephemeral=ephemeral,
            tool_name=tool_name,
            metadata=metadata or {},
            cache_hint=cache_hint,
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
        from bu_agent_sdk.context.reminder import ReminderPosition

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
                one_shot=False,  # 持续提醒直到创建 todo
            ))
        else:
            # 如果 todo 非空，注册"温和提醒"（不包含完整 JSON）
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
        from bu_agent_sdk.context.reminder import ReminderPosition

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
        from bu_agent_sdk.context.reminder import ReminderPosition

        # 如果已经有 todo 状态，不需要初始提醒
        if self._todo_state:
            return

        # 如果还没有注册过空列表提醒
        if not any(r.name == "todo_list_empty" for r in self.reminders):
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
            ))
