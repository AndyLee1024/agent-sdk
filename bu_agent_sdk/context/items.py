"""Context IR 核心数据类型

定义上下文中每个条目的语义类型、压缩优先级和段落结构。
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from bu_agent_sdk.llm.messages import BaseMessage


class ItemType(Enum):
    """上下文条目的语义类型"""

    # Header 段（lowered 为 SystemMessage 的一部分）
    SYSTEM_PROMPT = "system_prompt"
    AGENT_LOOP = "agent_loop"
    MEMORY = "memory"
    TOOL_STRATEGY = "tool_strategy"
    SUBAGENT_STRATEGY = "subagent_strategy"
    SKILL_STRATEGY = "skill_strategy"
    SYSTEM_ENV = "system_env"
    GIT_ENV = "git_env"

    # Conversation 段
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_RESULT = "tool_result"

    # Skill 注入
    SKILL_METADATA = "skill_metadata"
    SKILL_PROMPT = "skill_prompt"

    # 特殊
    COMPACTION_SUMMARY = "compaction_summary"
    OFFLOAD_PLACEHOLDER = "offload_placeholder"
    SYSTEM_REMINDER = "system_reminder"
    DEVELOPER_MESSAGE = "developer_message"


# 压缩优先级：数值越低越先压缩，数值越高越不应被压缩
DEFAULT_PRIORITIES: dict[ItemType, int] = {
    ItemType.TOOL_RESULT: 10,            # 最先压缩
    ItemType.SKILL_PROMPT: 20,
    ItemType.ASSISTANT_MESSAGE: 30,
    ItemType.SKILL_METADATA: 40,
    ItemType.USER_MESSAGE: 50,
    ItemType.DEVELOPER_MESSAGE: 60,
    ItemType.SYSTEM_REMINDER: 70,
    ItemType.OFFLOAD_PLACEHOLDER: 75,   # 块级卸载占位符（默认不再压缩）
    ItemType.COMPACTION_SUMMARY: 80,     # 不压缩
    ItemType.SKILL_STRATEGY: 90,
    ItemType.TOOL_STRATEGY: 93,          # 在 SUBAGENT 和 SKILL 之间
    ItemType.SUBAGENT_STRATEGY: 95,
    ItemType.SYSTEM_ENV: 100,            # 永不压缩
    ItemType.GIT_ENV: 100,               # 永不压缩
    ItemType.MEMORY: 100,                # 永不压缩
    ItemType.AGENT_LOOP: 100,            # 永不压缩
    ItemType.SYSTEM_PROMPT: 100,         # 永不压缩
}


def _generate_item_id() -> str:
    """生成短 ID（uuid 前 8 位）"""
    return uuid.uuid4().hex[:8]


@dataclass
class ContextItem:
    """上下文中的单个条目

    每个 ContextItem 代表一条语义化的上下文信息，
    既可对应一条 BaseMessage，也可以是纯结构信息（如 header 段）。

    Attributes:
        id: 唯一标识（uuid[:8]）
        item_type: 语义类型
        message: 底层消息对象（conversation 段有值，header 段为 None）
        content_text: 纯文本（用于 token 估算）
        token_count: 估算 token 数
        priority: 压缩优先级（低=先压缩）
        ephemeral: 是否是临时内容（大型输出等）
        destroyed: 是否已被销毁
        tool_name: 工具名称（仅 TOOL_RESULT 类型使用）
        created_at: 创建时间戳
        metadata: 附加元数据
        cache_hint: 是否建议缓存（如 Anthropic prompt cache）
    """

    id: str = field(default_factory=_generate_item_id)
    item_type: ItemType = ItemType.USER_MESSAGE
    message: BaseMessage | None = None
    content_text: str = ""
    token_count: int = 0
    priority: int = 50
    ephemeral: bool = False
    destroyed: bool = False
    tool_name: str | None = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_hint: bool = False
    offload_path: str | None = None  # 卸载文件路径（相对路径）
    offloaded: bool = False          # 是否已卸载


class SegmentName(Enum):
    """段落名称"""
    HEADER = "header"
    CONVERSATION = "conversation"


@dataclass
class Segment:
    """上下文段落

    将 ContextItem 按语义分组管理。

    Attributes:
        name: 段落名称
        items: 段落中的条目列表
    """

    name: SegmentName
    items: list[ContextItem] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """段落总 token 数"""
        return sum(item.token_count for item in self.items if not item.destroyed)

    def find_by_type(self, item_type: ItemType) -> list[ContextItem]:
        """查找指定类型的所有条目"""
        return [item for item in self.items if item.item_type == item_type]

    def find_one_by_type(self, item_type: ItemType) -> ContextItem | None:
        """查找指定类型的第一个条目（用于 header 段的幂等操作）"""
        for item in self.items:
            if item.item_type == item_type:
                return item
        return None

    def remove_by_type(self, item_type: ItemType) -> list[ContextItem]:
        """移除指定类型的所有条目，返回被移除的条目"""
        removed = [item for item in self.items if item.item_type == item_type]
        self.items = [item for item in self.items if item.item_type != item_type]
        return removed

    def remove_by_id(self, item_id: str) -> ContextItem | None:
        """按 ID 移除条目"""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                return self.items.pop(i)
        return None
