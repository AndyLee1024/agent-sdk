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

import json
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
from comate_agent_sdk.context.reminder_engine import (
    ReminderEngine,
    ReminderOrigin,
)
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
    ItemType.TOOL_STRATEGY: 2,
    ItemType.MCP_TOOL: 3,
    ItemType.SUBAGENT_STRATEGY: 4,
    ItemType.SKILL_STRATEGY: 5,
    ItemType.SYSTEM_ENV: 6,
    ItemType.GIT_ENV: 7,
}


# 消息类型 → ItemType 映射
_MESSAGE_TYPE_MAP: dict[type, ItemType] = {
    UserMessage: ItemType.USER_MESSAGE,
    AssistantMessage: ItemType.ASSISTANT_MESSAGE,
    ToolMessage: ItemType.TOOL_RESULT,
    DeveloperMessage: ItemType.DEVELOPER_MESSAGE,
}


@dataclass(slots=True)
class PendingHookInjection:
    """延迟注入的 hook meta message。"""

    content: str
    hook_name: str | None = None
    related_tool_call_id: str | None = None
    created_turn: int = 0


@dataclass
class ContextIR:
    """Context 中间表示

    将 Agent 的上下文从扁平 list[BaseMessage] 提升为结构化 IR。

    Segments:
        header: system_prompt + agent_loop + tool_strategy + subagent_strategy + skill_strategy
        conversation: 所有对话消息
        memory: 独立字段，lowering 时作为 UserMessage(is_meta=True) 注入

    Attributes:
        budget: 预算配置
        token_counter: token 计数器
        event_bus: 事件总线
        _reminder_engine: 统一 system-reminder 调度引擎
        _todo_state: TODO 状态存储 {todos: [...], updated_at: timestamp}
        _memory_item: Memory 独立存储（不在 header 或 conversation 中）
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
    _reminder_engine: ReminderEngine = field(default_factory=ReminderEngine)
    _pending_skill_items: list[ContextItem] = field(default_factory=list)
    _todo_state: dict[str, Any] = field(default_factory=dict)
    _turn_number: int = 0
    _memory_item: ContextItem | None = field(default=None, repr=False)
    _inflight_tool_call_ids: set[str] = field(default_factory=set, repr=False)
    _thinking_protected_assistant_ids: set[str] = field(default_factory=set, repr=False)
    """Tool loop 中含 thinking blocks 的 assistant item IDs，compaction 时需要保护。"""
    _pending_hook_injections: list[PendingHookInjection] = field(default_factory=list, repr=False)
    _flushed_hook_injection_texts: list[str] = field(default_factory=list, repr=False)

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

    def _set_header_item(
        self,
        item_type: ItemType,
        content: str,
        *,
        cache: bool = False,
        metadata: dict[str, Any] | None = None,
        insert_idx: int | None = None,
    ) -> None:
        """统一的 header item 设置方法（幂等覆盖）。

        Args:
            item_type: 要设置的 item 类型
            content: 内容文本
            cache: 是否建议缓存
            metadata: 附加元数据
            insert_idx: 指定插入位置；None 表示 append + _ensure_header_item_position
        """
        existing = self.header.find_one_by_type(item_type)
        token_count = self.token_counter.count(content)

        if existing:
            existing.content_text = content
            existing.token_count = token_count
            existing.cache_hint = cache
            if metadata is not None:
                existing.metadata = metadata
            self._ensure_header_item_position(existing)
            return

        item = ContextItem(
            item_type=item_type,
            content_text=content,
            token_count=token_count,
            priority=DEFAULT_PRIORITIES[item_type],
            cache_hint=cache,
            metadata=metadata or {},
        )
        if insert_idx is not None:
            self.header.items.insert(insert_idx, item)
        else:
            self.header.items.append(item)
            self._ensure_header_item_position(item)
        self.event_bus.emit(ContextEvent(
            event_type=EventType.ITEM_ADDED,
            item_type=item_type,
            item_id=item.id,
            detail=f"{item_type.value} set",
        ))

    def set_system_prompt(self, prompt: str, cache: bool = True) -> None:
        """设置系统提示（幂等覆盖）

        Args:
            prompt: 系统提示文本
            cache: 是否建议缓存（如 Anthropic prompt cache）
        """
        self._set_header_item(ItemType.SYSTEM_PROMPT, prompt, cache=cache, insert_idx=0)

    def set_agent_loop(self, prompt: str, cache: bool = False) -> None:
        """设置 Agent 循环控制指令（幂等覆盖）

        插入位置：SYSTEM_PROMPT 之后、TOOL_STRATEGY/SUBAGENT_STRATEGY/SKILL_STRATEGY 之前

        Args:
            prompt: Agent 循环控制指令文本
            cache: 是否建议缓存（如 Anthropic prompt cache）
        """
        insert_idx = 0
        for i, existing_item in enumerate(self.header.items):
            if existing_item.item_type == ItemType.SYSTEM_PROMPT:
                insert_idx = i + 1
            elif existing_item.item_type in (
                ItemType.TOOL_STRATEGY,
                ItemType.SUBAGENT_STRATEGY,
                ItemType.SKILL_STRATEGY,
            ):
                insert_idx = i
                break
        self._set_header_item(ItemType.AGENT_LOOP, prompt, cache=cache, insert_idx=insert_idx)

    def set_memory(self, content: str, cache: bool = True) -> None:
        """设置 MEMORY 静态背景知识（幂等覆盖）

        Memory 不进入 header，而是作为独立字段存储。
        lowering 时会被注入为 UserMessage(is_meta=True, content="<instructions>...")，
        位于 SystemMessage 之后、conversation items 之前。

        Args:
            content: MEMORY 内容（通常来自 CLAUDE.md / AGENTS.md 等仓库文件）
            cache: 是否建议缓存（如 Anthropic prompt cache）
        """
        # 延迟导入避免循环依赖
        from comate_agent_sdk.agent.prompts import MEMORY_NOTICE

        # 包裹 <instructions> 标签（替代 <memory>）
        wrapped_content = f"<user_instructions>\n{content}\n</user_instructions> \n{MEMORY_NOTICE}"

        token_count = self.token_counter.count(wrapped_content)

        # 创建 UserMessage(is_meta=True)
        message = UserMessage(content=wrapped_content, is_meta=True, cache=cache)

        if self._memory_item:
            # 幂等覆盖
            self._memory_item.content_text = wrapped_content
            self._memory_item.token_count = token_count
            self._memory_item.cache_hint = cache
            self._memory_item.message = message
        else:
            # 首次创建
            self._memory_item = ContextItem(
                item_type=ItemType.MEMORY,
                message=message,
                content_text=wrapped_content,
                token_count=token_count,
                priority=DEFAULT_PRIORITIES[ItemType.MEMORY],
                cache_hint=cache,
            )
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=ItemType.MEMORY,
                item_id=self._memory_item.id,
                detail="memory set (as independent UserMessage)",
            ))

    def set_subagent_strategy(self, prompt: str) -> None:
        """设置 Subagent 策略提示（幂等覆盖）

        插入位置：TOOL_STRATEGY 之后、SKILL_STRATEGY 之前
        """
        insert_idx = 0
        for i, existing_item in enumerate(self.header.items):
            if existing_item.item_type in (
                ItemType.SYSTEM_PROMPT,
                ItemType.AGENT_LOOP,
                ItemType.TOOL_STRATEGY,
            ):
                insert_idx = i + 1
            elif existing_item.item_type == ItemType.SKILL_STRATEGY:
                break
        self._set_header_item(ItemType.SUBAGENT_STRATEGY, prompt, insert_idx=insert_idx)

    def set_tool_strategy(self, prompt: str) -> None:
        """设置 Tool 策略提示（幂等覆盖）

        插入位置：AGENT_LOOP 之后、SUBAGENT_STRATEGY 之前
        """
        insert_idx = 0
        for i, existing_item in enumerate(self.header.items):
            if existing_item.item_type in (ItemType.SYSTEM_PROMPT, ItemType.AGENT_LOOP):
                insert_idx = i + 1
            elif existing_item.item_type in (
                ItemType.SUBAGENT_STRATEGY,
                ItemType.SKILL_STRATEGY,
            ):
                break
        self._set_header_item(ItemType.TOOL_STRATEGY, prompt, insert_idx=insert_idx)

    def set_mcp_tools(self, overview_text: str, *, metadata: dict[str, Any] | None = None) -> None:
        """设置 MCP tools 概览（幂等覆盖）

        设计目标：
        - 将 MCP tool 的概览注入到 header（lowered 后进入 SystemMessage）
        - 结构化信息保存在 ContextItem.metadata（不直接进入 prompt，避免 token 爆炸）
        """
        self._set_header_item(ItemType.MCP_TOOL, overview_text, metadata=metadata)

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
        self._set_header_item(ItemType.SKILL_STRATEGY, prompt)

    def set_system_env(self, content: str) -> None:
        """设置系统环境信息（幂等覆盖）

        插入位置：Header 段末尾（SKILL_STRATEGY 之后）
        """
        self._set_header_item(ItemType.SYSTEM_ENV, content)

    def set_git_env(self, content: str) -> None:
        """设置 Git 状态信息（幂等覆盖）

        插入位置：SYSTEM_ENV 之后（Header 段最末尾）
        """
        self._set_header_item(ItemType.GIT_ENV, content)

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
            self._reminder_engine.set_turn(self._turn_number)

        # 提取文本内容用于 token 估算
        content_text = message.text if hasattr(message, "text") else ""

        # AssistantMessage 特殊处理：需要包括 tool_calls 的 tokens
        # 因为 tool_calls 也会被发送给 LLM，占用 prompt tokens
        if isinstance(message, AssistantMessage) and message.tool_calls:
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

        # Tool barrier tracking:
        # - assistant(tool_calls) opens barrier
        # - tool_result closes corresponding ids and flushes deferred hook injections when barrier cleared
        if isinstance(message, AssistantMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.id:
                    self._inflight_tool_call_ids.add(tool_call.id)

            # Thinking protection: 如果包含 thinking blocks，保护这条 assistant message
            # 根据 Anthropic 约束：tool loop 中不能删除含 thinking signature 的 assistant message
            if self._contains_thinking_blocks(message):
                self._thinking_protected_assistant_ids.add(item.id)
                logger.debug(
                    f"Thinking protection enabled for assistant item {item.id} "
                    f"(tool_call_ids: {[tc.id for tc in message.tool_calls]})"
                )

        elif isinstance(message, ToolMessage):
            if message.tool_call_id:
                self._inflight_tool_call_ids.discard(message.tool_call_id)

            # 当 tool barrier 完全释放时，清空 thinking 保护
            if not self._inflight_tool_call_ids:
                if self._thinking_protected_assistant_ids:
                    logger.debug(
                        f"Tool barrier cleared, releasing thinking protection for "
                        f"{len(self._thinking_protected_assistant_ids)} assistant messages"
                    )
                self._thinking_protected_assistant_ids.clear()

            self._flush_pending_hook_injections_if_unblocked()
            self._flush_pending_skill_items_if_unblocked()

        # 防御性检查：保护集合不应超过合理大小
        if len(self._thinking_protected_assistant_ids) > 100:
            logger.warning(
                f"Thinking protection set unusually large: "
                f"{len(self._thinking_protected_assistant_ids)} items. Forcing cleanup."
            )
            self._thinking_protected_assistant_ids.clear()

        return item

    @property
    def has_tool_barrier(self) -> bool:
        """当前是否处于 tool barrier 中。

        Tool barrier 表示 committed history 中存在尚未闭合的 tool_call_id。
        也可由 runner 在执行工具前通过 begin_tool_barrier() 显式开启，
        用于防止注入消息插入到 tool_calls 与 tool_results 之间。
        """
        return bool(self._inflight_tool_call_ids)

    def begin_tool_barrier(self, tool_call_ids: list[str]) -> None:
        """显式开启 tool barrier（KISS：仅设置 inflight ids）。

        说明：
        - runner 在执行工具前可调用此方法，把本轮预期的 tool_call_id
          预先写入 inflight 集合，从而让 hook/runtime 注入在 barrier 期间被延迟。
        - barrier 会在对应 ToolMessage 落库（add_message）后逐个释放；
          当 inflight 为空时，延迟注入会自动 flush。
        """
        normalized = {str(tool_call_id).strip() for tool_call_id in tool_call_ids if str(tool_call_id).strip()}
        if not normalized:
            return
        self._inflight_tool_call_ids = set(normalized)

    def clear_tool_barrier(self, *, flush: bool = True) -> None:
        """强制清理 tool barrier（主要用于策略重试/回滚）。

        Args:
            flush: True 表示清理后立即 flush pending injections；
                False 表示丢弃 pending injections（仅清理未提交的注入，不影响已落库 messages）。
        """
        self._inflight_tool_call_ids.clear()
        self._thinking_protected_assistant_ids.clear()
        if not flush:
            self._pending_hook_injections.clear()
            return
        self._flush_pending_hook_injections_if_unblocked()
        self._flush_pending_skill_items_if_unblocked()

    @staticmethod
    def _contains_thinking_blocks(message: AssistantMessage) -> bool:
        """检查 AssistantMessage 是否包含 thinking blocks。"""
        if not isinstance(message.content, list):
            return False
        return any(
            getattr(part, "type", None) in ("thinking", "redacted_thinking")
            for part in message.content
        )

    def add_messages_atomic(self, messages: list[BaseMessage]) -> None:
        """原子追加多条 messages（commit 段内禁止 await，保证不被 interleave）。"""
        for message in messages:
            self.add_message(message)

    def add_hook_hidden_user_message(
        self,
        content: str,
        *,
        hook_name: str | None = None,
        related_tool_call_id: str | None = None,
    ) -> None:
        """添加 hook hidden 用户消息，若处于 tool barrier 则延迟注入。"""
        text = content.strip()
        if not text:
            return

        if self._inflight_tool_call_ids:
            self._pending_hook_injections.append(
                PendingHookInjection(
                    content=text,
                    hook_name=hook_name,
                    related_tool_call_id=related_tool_call_id,
                    created_turn=self._turn_number,
                )
            )
            return

        self._append_hook_injection_message(
            text=text,
            hook_name=hook_name,
            related_tool_call_id=related_tool_call_id,
            created_turn=self._turn_number,
        )

    def _append_hook_injection_message(
        self,
        *,
        text: str,
        hook_name: str | None,
        related_tool_call_id: str | None,
        created_turn: int,
    ) -> None:
        """将 hook 注入消息立即落盘，并登记为已 flush 文本供 runtime 发事件。"""
        metadata = {
            "origin": "hook",
            "hook_name": hook_name,
            "related_tool_call_id": related_tool_call_id,
            "created_turn": created_turn,
        }
        self.add_message(
            UserMessage(content=text, is_meta=True),
            metadata=metadata,
        )
        self._flushed_hook_injection_texts.append(text)

    def _flush_pending_hook_injections_if_unblocked(self) -> None:
        """tool barrier 解除时，批量刷入延迟的 hook 注入。"""
        if self._inflight_tool_call_ids:
            return
        if not self._pending_hook_injections:
            return

        pending = list(self._pending_hook_injections)
        self._pending_hook_injections.clear()
        for injection in pending:
            self._append_hook_injection_message(
                text=injection.content,
                hook_name=injection.hook_name,
                related_tool_call_id=injection.related_tool_call_id,
                created_turn=injection.created_turn,
            )

    def pop_flushed_hook_injection_texts(self) -> list[str]:
        """弹出已刷入 context 的 hook 注入文本（供 UI hidden event 使用）。"""
        if not self._flushed_hook_injection_texts:
            return []
        texts = list(self._flushed_hook_injection_texts)
        self._flushed_hook_injection_texts.clear()
        return texts

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
            self._reminder_engine.set_turn(self._turn_number)

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

        self._pending_skill_items.extend([metadata_item, prompt_item])
        return [metadata_item, prompt_item]

    def flush_pending_skill_items(self) -> None:
        """将待注入的 Skill items 刷入 conversation（tool barrier 内会延迟）。"""
        self._flush_pending_skill_items_if_unblocked()

    def _flush_pending_skill_items_if_unblocked(self) -> None:
        """tool barrier 解除时，批量刷入 Skill pending items。"""
        if self._inflight_tool_call_ids:
            return
        if not self._pending_skill_items:
            return

        pending = list(self._pending_skill_items)
        self._pending_skill_items = []
        for item in pending:
            self.conversation.items.append(item)
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_ADDED,
                item_type=item.item_type,
                item_id=item.id,
                detail=f"skill item flushed: {item.item_type.value}",
            ))

    # ===== Reminder Engine =====

    def set_plan_mode(self, enabled: bool) -> None:
        """设置 plan mode 提醒开关。"""
        self._reminder_engine.set_plan_mode(enabled)

    def record_tool_event(
        self,
        *,
        tool_name: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """记录工具事件，驱动 reminder 调度状态。"""
        self._reminder_engine.record_tool_event(
            tool_name=tool_name,
            turn=self._turn_number,
            payload=payload,
        )

    def inject_due_reminders(self) -> ContextItem | None:
        """把当前轮次到期的 reminder 注入为一条 meta UserMessage。"""
        if self.has_tool_barrier:
            return None

        reminders = self._reminder_engine.collect_due_reminders(turn=self._turn_number)
        if not reminders:
            return None

        reminder_text = ReminderEngine.render_merged_message(reminders)
        if not reminder_text:
            return None

        rule_ids = [r.rule_id for r in reminders]
        for item in reversed(self.conversation.items):
            if item.destroyed:
                continue
            if not self._is_system_reminder_item(item):
                continue
            same_turn = int(item.metadata.get("reminder_turn", -1)) == self._turn_number
            existing_rule_ids = item.metadata.get("reminder_rule_ids", [])
            same_rules = isinstance(existing_rule_ids, list) and existing_rule_ids == rule_ids
            if same_turn and same_rules:
                return None
            break

        metadata: dict[str, Any] = {
            "origin": ReminderOrigin.SYSTEM_REMINDER.value,
            "reminder_rule_ids": rule_ids,
            "reminder_tools": [r.tool_name for r in reminders],
            "reminder_turn": self._turn_number,
        }
        return self.add_message(
            UserMessage(content=reminder_text, is_meta=True),
            item_type=ItemType.SYSTEM_REMINDER,
            metadata=metadata,
        )

    def _is_system_reminder_item(self, item: ContextItem) -> bool:
        if item.item_type == ItemType.SYSTEM_REMINDER:
            return True

        origin = str(item.metadata.get("origin", "")).strip()
        if origin == ReminderOrigin.SYSTEM_REMINDER.value:
            return True

        message = item.message
        if not isinstance(message, UserMessage) or not bool(getattr(message, "is_meta", False)):
            return False

        text = (message.text or "").strip()
        return "<system-reminder>" in text and "</system-reminder>" in text

    def purge_system_reminders(
        self,
        *,
        include_persistent: bool = True,
    ) -> int:
        """统一清理 conversation 中的 system reminder。

        注意：
        - 当前仅清理 system-reminder（origin/tag/item_type 命中）
        - TODO: 后续可扩展为对其他 hook/meta 消息的统一治理策略
        """
        removed_ids: list[str] = []
        kept: list[ContextItem] = []

        for item in self.conversation.items:
            if include_persistent and self._is_system_reminder_item(item):
                removed_ids.append(item.id)
                continue
            kept.append(item)

        if not removed_ids:
            return 0

        self.conversation.items = kept
        for item_id in removed_ids:
            self.event_bus.emit(ContextEvent(
                event_type=EventType.ITEM_REMOVED,
                item_type=ItemType.SYSTEM_REMINDER,
                item_id=item_id,
                detail="system reminder purged",
            ))
        return len(removed_ids)

    # ===== Budget =====

    @property
    def total_tokens(self) -> int:
        """当前总 token 数（估算值）"""
        memory_tokens = self._memory_item.token_count if self._memory_item else 0
        return self.header.total_tokens + self.conversation.total_tokens + memory_tokens

    def get_budget_status(self) -> BudgetStatus:
        """获取预算状态快照"""
        tokens_by_type: dict[ItemType, int] = {}

        for segment in (self.header, self.conversation):
            for item in segment.items:
                if item.destroyed:
                    continue
                current = tokens_by_type.get(item.item_type, 0)
                tokens_by_type[item.item_type] = current + item.token_count

        # 统计 memory_item
        if self._memory_item and not self._memory_item.destroyed:
            current = tokens_by_type.get(self._memory_item.item_type, 0)
            tokens_by_type[self._memory_item.item_type] = current + self._memory_item.token_count

        memory_tokens = self._memory_item.token_count if self._memory_item else 0

        return BudgetStatus(
            total_tokens=self.total_tokens,
            header_tokens=self.header.total_tokens,
            conversation_tokens=self.conversation.total_tokens + memory_tokens,
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
        return LoweringPipeline.lower(self)

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
        self._pending_hook_injections.clear()
        self._inflight_tool_call_ids.clear()
        self._thinking_protected_assistant_ids.clear()
        self._flushed_hook_injection_texts.clear()
        self._reminder_engine.clear()
        self._todo_state.clear()
        self._memory_item = None

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
    def memory_item(self) -> ContextItem | None:
        """获取 Memory item（只读）"""
        return self._memory_item

    # ===== TODO 状态管理 =====

    @staticmethod
    def _normalize_active_todos(todos: list[dict]) -> list[dict]:
        return [
            todo
            for todo in todos
            if isinstance(todo, dict) and todo.get("status") in ("pending", "in_progress")
        ]

    def set_todo_state(self, todos: list[dict]) -> None:
        """设置 TODO 状态并更新 ReminderEngine

        Args:
            todos: TODO 列表（字典格式，来自 TodoWrite 工具）
        """
        import time

        active_todos = self._normalize_active_todos(todos)

        self._todo_state = {
            "todos": active_todos,
            "updated_at": time.time(),
        }

        self._reminder_engine.update_todo_state(
            active_count=len(active_todos),
            turn=self._turn_number,
            update_last_write_turn=True,
        )

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

        active_todos = self._normalize_active_todos(todos)

        self._todo_state = {
            "todos": active_todos,
            "updated_at": time.time(),
        }

        self._reminder_engine.update_todo_state(
            active_count=len(active_todos),
            turn=self._turn_number,
            update_last_write_turn=False,
        )
        self._reminder_engine.restore_todo_write_turn(turn_number_at_update)


    def get_todo_state(self) -> dict[str, Any]:
        """获取当前 TODO 状态"""
        return self._todo_state.copy()

    def has_todos(self) -> bool:
        """检查是否有 TODO 条目"""
        return bool(self._todo_state.get("todos"))

    def get_todo_persist_turn_number_at_update(self) -> int:
        """用于持久化的 todo 更新轮次。"""
        return int(self._reminder_engine.state.last_todowrite_turn)
