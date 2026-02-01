"""选择性压缩模块

按类型优先级逐步压缩（tool_result 先压缩，system_prompt 永不压缩），
而非全量替换。当选择性压缩不足时，回退到 CompactionService 全量摘要。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from bu_agent_sdk.context.items import DEFAULT_PRIORITIES, ItemType

if TYPE_CHECKING:
    from bu_agent_sdk.context.ir import ContextIR
    from bu_agent_sdk.llm.base import BaseChatModel
    from bu_agent_sdk.context.fs import ContextFileSystem
    from bu_agent_sdk.context.offload import OffloadPolicy
    from bu_agent_sdk.tokens import TokenCost
    from bu_agent_sdk.agent.llm_levels import LLMLevel

logger = logging.getLogger("bu_agent_sdk.context.compaction")


class CompactionStrategy(Enum):
    """压缩策略"""

    NONE = "none"            # 不压缩
    DROP = "drop"            # 直接丢弃
    TRUNCATE = "truncate"    # 保留最近 N 个
    SUMMARIZE = "summarize"  # LLM 摘要（暂未实现，留接口）


@dataclass
class TypeCompactionRule:
    """单个类型的压缩规则

    Attributes:
        strategy: 压缩策略
        keep_recent: TRUNCATE 策略保留的最近条目数
    """

    strategy: CompactionStrategy = CompactionStrategy.NONE
    keep_recent: int = 5


# 默认规则
DEFAULT_COMPACTION_RULES: dict[str, TypeCompactionRule] = {
    ItemType.TOOL_RESULT.value: TypeCompactionRule(
        strategy=CompactionStrategy.TRUNCATE, keep_recent=3
    ),
    ItemType.SKILL_PROMPT.value: TypeCompactionRule(
        strategy=CompactionStrategy.DROP
    ),
    ItemType.SKILL_METADATA.value: TypeCompactionRule(
        strategy=CompactionStrategy.DROP
    ),
    ItemType.ASSISTANT_MESSAGE.value: TypeCompactionRule(
        strategy=CompactionStrategy.TRUNCATE, keep_recent=5
    ),
    ItemType.USER_MESSAGE.value: TypeCompactionRule(
        strategy=CompactionStrategy.TRUNCATE, keep_recent=5
    ),
    # 以下类型永不压缩
    ItemType.SYSTEM_PROMPT.value: TypeCompactionRule(
        strategy=CompactionStrategy.NONE
    ),
    ItemType.MEMORY.value: TypeCompactionRule(
        strategy=CompactionStrategy.NONE
    ),
    ItemType.SUBAGENT_STRATEGY.value: TypeCompactionRule(
        strategy=CompactionStrategy.NONE
    ),
    ItemType.COMPACTION_SUMMARY.value: TypeCompactionRule(
        strategy=CompactionStrategy.NONE
    ),
}


@dataclass
class SelectiveCompactionPolicy:
    """选择性压缩策略

    按 DEFAULT_PRIORITIES 从低到高逐类型压缩，
    每压缩一种类型后检查是否已降到阈值以下。
    如果选择性压缩不够，回退到全量摘要。

    Attributes:
        threshold: token 阈值（超过此值触发压缩）
        rules: 每类型的压缩规则
        llm: 用于全量摘要回退的 LLM
        fallback_to_full_summary: 选择性压缩不够时是否回退全量摘要
    """

    threshold: int = 0
    rules: dict[str, TypeCompactionRule] = field(
        default_factory=lambda: dict(DEFAULT_COMPACTION_RULES)
    )
    llm: BaseChatModel | None = None
    fallback_to_full_summary: bool = True
    fs: ContextFileSystem | None = None
    offload_policy: OffloadPolicy | None = None
    token_cost: TokenCost | None = None
    level: LLMLevel | None = None

    def should_compact(self, total_tokens: int) -> bool:
        """检查是否需要压缩"""
        if self.threshold <= 0:
            return False
        return total_tokens >= self.threshold

    async def compact(self, context: ContextIR) -> bool:
        """执行选择性压缩

        按 DEFAULT_PRIORITIES 从低到高逐类型压缩。
        每压缩一种类型后检查总 token 数是否降到阈值以下。

        Args:
            context: ContextIR 实例

        Returns:
            是否成功将 token 数压缩到阈值以下
        """
        if self.threshold <= 0:
            return False

        initial_tokens = context.total_tokens
        logger.info(
            f"开始选择性压缩: current={initial_tokens}, threshold={self.threshold}"
        )

        # 按优先级排序：低优先级（先压缩）在前
        sorted_types = sorted(
            DEFAULT_PRIORITIES.items(),
            key=lambda x: x[1],
        )

        compacted_any = False

        for item_type, _ in sorted_types:
            rule = self.rules.get(item_type.value)
            if rule is None or rule.strategy == CompactionStrategy.NONE:
                continue

            # 获取该类型的所有 conversation items
            items = context.conversation.find_by_type(item_type)
            if not items:
                continue

            tokens_before = context.total_tokens

            if rule.strategy == CompactionStrategy.DROP:
                # 直接移除所有该类型的条目
                removed = context.conversation.remove_by_type(item_type)
                if removed:
                    compacted_any = True
                    logger.info(
                        f"DROP {item_type.value}: 移除 {len(removed)} 条, "
                        f"释放 ~{tokens_before - context.total_tokens} tokens"
                    )

            elif rule.strategy == CompactionStrategy.TRUNCATE:
                # 保留最近 N 个，删除更早的
                if rule.keep_recent > 0 and len(items) <= rule.keep_recent:
                    continue

                if rule.keep_recent <= 0:
                    items_to_remove = items
                else:
                    items_to_remove = items[:-rule.keep_recent]
                for item in items_to_remove:
                    # 跳过已 destroyed 的（已被 ephemeral 处理）
                    if item.destroyed:
                        continue
                    # 检查是否需要卸载
                    if self._should_offload(item):
                        self.fs.offload(item)
                    context.conversation.remove_by_id(item.id)
                compacted_any = True
                logger.info(
                    f"TRUNCATE {item_type.value}: 移除 {len(items_to_remove)} 条 "
                    f"(保留最近 {rule.keep_recent}), "
                    f"释放 ~{tokens_before - context.total_tokens} tokens"
                )

            # 检查是否已降到阈值以下
            current_tokens = context.total_tokens
            if current_tokens < self.threshold:
                logger.info(
                    f"选择性压缩完成: {initial_tokens} → {current_tokens} tokens "
                    f"(阈值={self.threshold})"
                )
                return True

        # 选择性压缩不够，检查是否需要回退全量摘要
        current_tokens = context.total_tokens
        if current_tokens >= self.threshold and self.fallback_to_full_summary:
            logger.info(
                f"选择性压缩不足 ({initial_tokens} → {current_tokens}), "
                f"回退到全量摘要"
            )
            return await self._fallback_full_summary(context)

        if compacted_any:
            logger.info(
                f"选择性压缩部分完成: {initial_tokens} → {current_tokens} tokens "
                f"(未达阈值 {self.threshold})"
            )

        return compacted_any

    def _should_offload(self, item) -> bool:
        """判断是否需要卸载

        Args:
            item: ContextItem 实例

        Returns:
            是否应该卸载此条目
        """
        if not self.fs or not self.offload_policy:
            return False
        if not self.offload_policy.enabled:
            return False
        # 已压缩的 summary 不卸载
        if item.item_type == ItemType.COMPACTION_SUMMARY:
            return False
        # 检查类型是否启用
        type_enabled = self.offload_policy.type_enabled.get(
            item.item_type.value, False
        )
        if not type_enabled:
            return False
        # 检查 token 阈值
        return item.token_count >= self.offload_policy.token_threshold

    async def _fallback_full_summary(self, context: ContextIR) -> bool:
        """回退到全量摘要

        复用现有 CompactionService.compact() 逻辑。
        """
        from bu_agent_sdk.agent.compaction import CompactionService
        from bu_agent_sdk.llm.messages import UserMessage

        if self.llm is None:
            logger.warning("无法执行全量摘要回退：未提供 LLM")
            return False

        # 获取当前所有 conversation messages（排除 Skill 注入项）
        #
        # Skill 的详细指令来自运行时注入，允许随时重复加载。
        # 为避免在全量摘要中固化 Skill prompt，摘要时应跳过这些注入项。
        conversation_messages = [
            item.message
            for item in context.conversation.items
            if item.message is not None
            and item.item_type not in (ItemType.SKILL_PROMPT, ItemType.SKILL_METADATA)
        ]

        if not conversation_messages:
            return False

        service = CompactionService(llm=self.llm, token_cost=self.token_cost)
        result = await service.compact(
            conversation_messages,
            self.llm,
            level=self.level,
        )

        if result.compacted and result.summary:
            # 用摘要替换整个 conversation
            from bu_agent_sdk.context.items import ContextItem

            summary_item = ContextItem(
                item_type=ItemType.COMPACTION_SUMMARY,
                message=UserMessage(content=result.summary),
                content_text=result.summary,
                token_count=context.token_counter.count(result.summary),
                priority=DEFAULT_PRIORITIES[ItemType.COMPACTION_SUMMARY],
            )
            context.replace_conversation([summary_item])
            logger.info(
                f"全量摘要完成: 新 token 数 ~{summary_item.token_count}"
            )
            return True

        return False
