"""上下文信息数据模型

用于 /context 命令展示当前会话的上下文使用情况。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.context.items import ItemType
    from comate_agent_sdk.context.budget import BudgetStatus
    from comate_agent_sdk.context.ir import ContextIR


@dataclass(frozen=True)
class ContextCategoryInfo:
    """单个上下文类别的统计信息

    Attributes:
        label: 展示名称（如 "System Prompt", "Messages"）
        token_count: 该类别的 token 数
        item_count: 该类别的条目数
        item_types: 该类别包含的 ItemType 元组
    """
    label: str
    token_count: int
    item_count: int = 0
    item_types: tuple[ItemType, ...] = ()


@dataclass(frozen=True)
class ContextInfo:
    """完整的上下文信息快照

    Attributes:
        model_name: 使用的模型名称
        context_limit: 模型的上下文窗口大小（tokens）
        compact_threshold: 压缩触发阈值（tokens）
        compact_threshold_ratio: 压缩阈值比例（0.0-1.0）
        total_tokens: 总 token 数（header + conversation）
        header_tokens: Header 段 token 数
        conversation_tokens: Conversation 段 token 数
        tool_definitions_tokens: Tool JSON schema 估算 token 数
        used_tokens_message_only: 仅消息上下文占用（IR 估算）
        used_tokens_with_tools: 消息 + 工具定义占用（IR 估算）
        next_step_estimated_tokens: 下一次调用估算占用（主口径）
        last_step_reported_tokens: 上一次调用 provider 回传总量（对账口径）
        categories: 各类别统计列表
        compaction_enabled: 是否启用压缩
    """
    model_name: str
    context_limit: int
    compact_threshold: int
    compact_threshold_ratio: float
    total_tokens: int
    header_tokens: int
    conversation_tokens: int
    tool_definitions_tokens: int = 0
    used_tokens_message_only: int = 0
    used_tokens_with_tools: int = 0
    next_step_estimated_tokens: int = 0
    last_step_reported_tokens: int = 0
    categories: list[ContextCategoryInfo] = ()  # type: ignore
    compaction_enabled: bool = True

    @property
    def used_tokens(self) -> int:
        """兼容字段：返回消息 + 工具定义占用。"""
        return self.used_tokens_with_tools

    @property
    def primary_used_tokens(self) -> int:
        """主展示口径：next-step 估算（缺失时回退到消息+工具）。"""
        if self.next_step_estimated_tokens > 0:
            return self.next_step_estimated_tokens
        return self.used_tokens_with_tools

    @property
    def free_tokens(self) -> int:
        """到达压缩阈值前的剩余空间"""
        return max(0, self.compact_threshold - self.primary_used_tokens)

    @property
    def buffer_tokens(self) -> int:
        """压缩阈值到上下文限制之间的缓冲区"""
        return max(0, self.context_limit - self.compact_threshold)

    @property
    def utilization_percent(self) -> float:
        """上下文利用率百分比"""
        if self.context_limit <= 0:
            return 0.0
        return (self.primary_used_tokens / self.context_limit) * 100


def _build_categories(
    tokens_by_type: dict[ItemType, int],
    context: ContextIR,
) -> list[ContextCategoryInfo]:
    """根据 ItemType token 统计构建类别聚合信息

    Args:
        tokens_by_type: 各 ItemType 的 token 统计
        context: ContextIR 实例（用于获取 item_count）

    Returns:
        聚合后的类别列表（只包含有内容的类别）
    """
    from comate_agent_sdk.context.items import ItemType

    # 类别聚合规则：display_label -> ItemType list
    category_mapping = {
        "System Prompt": [ItemType.SYSTEM_PROMPT],
        "Agent Loop": [ItemType.AGENT_LOOP],
        "Memory": [ItemType.MEMORY],
        "Subagent Strategy": [ItemType.SUBAGENT_STRATEGY],
        "Skill Strategy": [ItemType.SKILL_STRATEGY],
        "System Env": [ItemType.SYSTEM_ENV],
        "Git Env": [ItemType.GIT_ENV],
        "Messages": [ItemType.USER_MESSAGE, ItemType.ASSISTANT_MESSAGE],
        "Tool Results": [ItemType.TOOL_RESULT],
        "Skills": [ItemType.SKILL_METADATA, ItemType.SKILL_PROMPT],
        "Compaction Summary": [ItemType.COMPACTION_SUMMARY],
        "System Reminder": [ItemType.SYSTEM_REMINDER],
    }

    categories = []

    for label, item_types in category_mapping.items():
        # 计算该类别的总 token 数
        total_tokens = sum(tokens_by_type.get(t, 0) for t in item_types)

        # 跳过空类别
        if total_tokens == 0:
            continue

        # 计算条目数（遍历 header 和 conversation segment 的所有 item）
        item_count = 0
        for segment in [context.header, context.conversation]:
            for item in segment.items:
                if item.item_type in item_types:
                    item_count += 1

        # 额外检查 memory_item（独立字段）
        if context.memory_item and context.memory_item.item_type in item_types:
            item_count += 1

        categories.append(ContextCategoryInfo(
            label=label,
            token_count=total_tokens,
            item_count=item_count,
            item_types=tuple(item_types),
        ))

    # 按 token 数降序排列
    categories.sort(key=lambda c: c.token_count, reverse=True)

    return categories
