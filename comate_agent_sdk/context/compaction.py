"""选择性压缩模块。

按类型优先级逐步压缩（tool_result 先压缩，system_prompt 永不压缩），
并在选择性压缩后始终执行全量摘要。
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

from comate_agent_sdk.context.items import DEFAULT_PRIORITIES, ItemType

if TYPE_CHECKING:
    from comate_agent_sdk.agent.llm_levels import LLMLevel
    from comate_agent_sdk.context.fs import ContextFileSystem
    from comate_agent_sdk.context.ir import ContextIR
    from comate_agent_sdk.context.items import ContextItem
    from comate_agent_sdk.context.offload import OffloadPolicy
    from comate_agent_sdk.llm.base import BaseChatModel
    from comate_agent_sdk.tokens import TokenCost

logger = logging.getLogger("comate_agent_sdk.context.compaction")


@dataclass(frozen=True)
class _ToolCallInfo:
    tool_call_id: str
    tool_name: str


@dataclass(frozen=True)
class _ToolInteractionBlock:
    start_idx: int
    end_idx: int
    assistant_item_id: str
    tool_calls: list[_ToolCallInfo]
    tool_result_items: list["ContextItem"]


@dataclass
class _CompactionStats:
    tool_blocks_kept: int = 0
    tool_blocks_dropped: int = 0
    tool_calls_truncated: int = 0
    tool_results_truncated: int = 0


@dataclass(frozen=True)
class CompactionMetaRecord:
    phase: Literal["selective_start", "selective_done", "summary_start", "summary_done", "rollback"]
    tokens_before: int
    tokens_after: int
    tool_blocks_kept: int
    tool_blocks_dropped: int
    tool_calls_truncated: int
    tool_results_truncated: int
    reason: str


class CompactionStrategy(Enum):
    """压缩策略。"""

    NONE = "none"            # 不压缩
    DROP = "drop"            # 直接丢弃
    TRUNCATE = "truncate"    # 保留最近 N 个
    SUMMARIZE = "summarize"  # LLM 摘要（暂未实现，留接口）


@dataclass
class TypeCompactionRule:
    """单个类型的压缩规则。"""

    strategy: CompactionStrategy = CompactionStrategy.NONE
    keep_recent: int = 5


# 默认规则
DEFAULT_COMPACTION_RULES: dict[str, TypeCompactionRule] = {
    ItemType.TOOL_RESULT.value: TypeCompactionRule(
        strategy=CompactionStrategy.TRUNCATE,
        keep_recent=5,
    ),
    ItemType.SKILL_PROMPT.value: TypeCompactionRule(
        strategy=CompactionStrategy.DROP
    ),
    ItemType.SKILL_METADATA.value: TypeCompactionRule(
        strategy=CompactionStrategy.DROP
    ),
    ItemType.ASSISTANT_MESSAGE.value: TypeCompactionRule(
        strategy=CompactionStrategy.TRUNCATE,
        keep_recent=5,
    ),
    ItemType.USER_MESSAGE.value: TypeCompactionRule(
        strategy=CompactionStrategy.TRUNCATE,
        keep_recent=5,
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
    ItemType.OFFLOAD_PLACEHOLDER.value: TypeCompactionRule(
        strategy=CompactionStrategy.NONE
    ),
}


@dataclass
class SelectiveCompactionPolicy:
    """选择性压缩策略。"""

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
    source_prefix: str | None = None

    tool_blocks_keep_recent: int = 8
    tool_call_threshold: int = 500
    tool_result_threshold: int = 600
    preview_tokens: int = 200
    dialogue_rounds_keep_min: int = 15
    summary_retry_attempts: int = 2
    error_item_max_turns: int = 10  # 失败工具项最大保留轮次

    meta_records: list[CompactionMetaRecord] = field(default_factory=list, init=False)

    def should_compact(self, total_tokens: int) -> bool:
        """检查是否需要压缩。"""
        if self.threshold <= 0:
            return False
        return total_tokens >= self.threshold

    async def compact(self, context: "ContextIR") -> bool:
        """执行选择性压缩 + 全量摘要（事务化）。"""
        if self.threshold <= 0:
            return False

        # 在选择性压缩前先清理过期的失败工具调用
        if self.error_item_max_turns > 0:
            removed_error_ids = context.cleanup_stale_error_items(
                max_turns=self.error_item_max_turns
            )
            if removed_error_ids:
                logger.info(
                    f"清理过期失败工具项: 移除 {len(removed_error_ids)} 条"
                )

        self.meta_records = []
        initial_tokens = context.total_tokens
        conversation_snapshot = copy.deepcopy(context.conversation.items)
        stats = _CompactionStats()

        logger.info(
            f"开始选择性压缩: current={initial_tokens}, threshold={self.threshold}"
        )
        self._append_meta(
            phase="selective_start",
            tokens_before=initial_tokens,
            tokens_after=initial_tokens,
            stats=stats,
            reason="started",
        )

        try:
            sorted_types = sorted(
                DEFAULT_PRIORITIES.items(),
                key=lambda x: x[1],
            )
            compacted_any = False
            protected_ids = self._collect_recent_round_protected_ids(context)

            for item_type, _ in sorted_types:
                rule = self.rules.get(item_type.value)
                if rule is None or rule.strategy == CompactionStrategy.NONE:
                    continue

                items = context.conversation.find_by_type(item_type)
                if not items:
                    continue

                tokens_before = context.total_tokens

                if rule.strategy == CompactionStrategy.DROP:
                    removed = context.conversation.remove_by_type(item_type)
                    if removed:
                        compacted_any = True
                        logger.info(
                            f"DROP {item_type.value}: 移除 {len(removed)} 条, "
                            f"释放 ~{tokens_before - context.total_tokens} tokens"
                        )

                elif rule.strategy == CompactionStrategy.TRUNCATE:
                    if item_type == ItemType.TOOL_RESULT:
                        removed_blocks = await self._truncate_tool_blocks(
                            context=context,
                            keep_recent=self.tool_blocks_keep_recent,
                            stats=stats,
                        )
                        if removed_blocks > 0:
                            compacted_any = True
                            logger.info(
                                f"TRUNCATE tool_blocks: 移除 {removed_blocks} 个块 "
                                f"(保留最近 {self.tool_blocks_keep_recent}), "
                                f"释放 ~{tokens_before - context.total_tokens} tokens"
                            )
                    else:
                        candidates = items
                        if item_type in (ItemType.USER_MESSAGE, ItemType.ASSISTANT_MESSAGE):
                            candidates = [it for it in items if it.id not in protected_ids]

                        if not candidates:
                            continue

                        if rule.keep_recent > 0 and len(candidates) <= rule.keep_recent:
                            continue

                        if rule.keep_recent <= 0:
                            items_to_remove = candidates
                        else:
                            items_to_remove = candidates[:-rule.keep_recent]

                        removed_count = 0
                        for item in items_to_remove:
                            if item.destroyed:
                                continue
                            if self._should_offload(item):
                                self.fs.offload(item)
                            if context.conversation.remove_by_id(item.id) is not None:
                                removed_count += 1

                        if removed_count > 0:
                            compacted_any = True
                            logger.info(
                                f"TRUNCATE {item_type.value}: 移除 {removed_count} 条 "
                                f"(保留最近 {rule.keep_recent}, 轮次保底={self.dialogue_rounds_keep_min}), "
                                f"释放 ~{tokens_before - context.total_tokens} tokens"
                            )

                current_tokens = context.total_tokens
                if current_tokens < self.threshold:
                    logger.info(
                        f"选择性压缩达到阈值: {initial_tokens} → {current_tokens} tokens "
                        f"(阈值={self.threshold})"
                    )
                    break

            selective_tokens = context.total_tokens
            self._append_meta(
                phase="selective_done",
                tokens_before=initial_tokens,
                tokens_after=selective_tokens,
                stats=stats,
                reason="done",
            )

            if self.fallback_to_full_summary:
                logger.info(
                    f"选择性压缩完成 ({initial_tokens} → {selective_tokens})，执行 LLM Summary"
                )
                self._append_meta(
                    phase="summary_start",
                    tokens_before=selective_tokens,
                    tokens_after=selective_tokens,
                    stats=stats,
                    reason="always_run",
                )

                summary_success, summary_reason = await self._fallback_full_summary_with_retry(context)
                if not summary_success:
                    raise RuntimeError(f"summary_failed_or_empty:{summary_reason}")

                final_tokens = context.total_tokens
                self._append_meta(
                    phase="summary_done",
                    tokens_before=selective_tokens,
                    tokens_after=final_tokens,
                    stats=stats,
                    reason="success",
                )
                return True

            return compacted_any

        except Exception as exc:
            context.conversation.items = conversation_snapshot
            rollback_tokens = context.total_tokens
            self._append_meta(
                phase="rollback",
                tokens_before=initial_tokens,
                tokens_after=rollback_tokens,
                stats=stats,
                reason=str(exc),
            )
            logger.warning(f"压缩失败，已回滚: {exc}", exc_info=True)
            return False

    async def _truncate_tool_blocks(
        self,
        context: "ContextIR",
        *,
        keep_recent: int,
        stats: _CompactionStats | None = None,
    ) -> int:
        """按“最近 keep_recent 块保留，旧块整块删除”处理工具交互历史。"""
        blocks = self._extract_tool_blocks(context)
        if not blocks:
            if stats is not None:
                stats.tool_blocks_kept = 0
                stats.tool_blocks_dropped = 0
            return 0

        blocks_sorted = sorted(blocks, key=lambda b: b.start_idx)
        if keep_recent <= 0:
            kept_blocks: list[_ToolInteractionBlock] = []
            dropped_blocks = blocks_sorted
        else:
            kept_blocks = blocks_sorted[-keep_recent:]
            dropped_blocks = blocks_sorted[:-keep_recent]

        if stats is not None:
            stats.tool_blocks_kept = len(kept_blocks)
            stats.tool_blocks_dropped = len(dropped_blocks)

        for block in kept_blocks:
            calls_count, results_count = self._truncate_tool_block_fields(context, block)
            if stats is not None:
                stats.tool_calls_truncated += calls_count
                stats.tool_results_truncated += results_count

        removed = 0
        for block in sorted(dropped_blocks, key=lambda b: b.start_idx, reverse=True):
            del context.conversation.items[block.start_idx:block.end_idx + 1]
            removed += 1

        return removed

    def _truncate_tool_block_fields(
        self,
        context: "ContextIR",
        block: _ToolInteractionBlock,
    ) -> tuple[int, int]:
        """对保留工具块内字段做阈值截断。"""
        from comate_agent_sdk.llm.messages import AssistantMessage, ToolMessage

        tool_calls_truncated = 0
        tool_results_truncated = 0

        assistant_item = context.conversation.items[block.start_idx]
        assistant_msg = assistant_item.message
        if isinstance(assistant_msg, AssistantMessage) and assistant_msg.tool_calls:
            truncation_details: list[dict[str, object]] = []
            for tool_call in assistant_msg.tool_calls:
                arguments = tool_call.function.arguments or ""
                arguments_tokens = context.token_counter.count(arguments)
                if arguments_tokens <= self.tool_call_threshold:
                    continue
                tool_call.function.arguments = self._truncate_text(
                    text=arguments,
                    original_tokens=arguments_tokens,
                )
                truncation_details.append(
                    {
                        "field": "tool_call.arguments",
                        "tool_call_id": tool_call.id,
                        "original_tokens": arguments_tokens,
                        "kept_tokens": self.preview_tokens,
                        "threshold": self.tool_call_threshold,
                    }
                )
                tool_calls_truncated += 1

            if truncation_details:
                self._merge_truncation_metadata(assistant_item, truncation_details)
                self._refresh_assistant_item_tokens(context, assistant_item)

        for result_item in block.tool_result_items:
            result_msg = result_item.message
            if not isinstance(result_msg, ToolMessage):
                continue
            result_text = result_msg.text
            result_tokens = context.token_counter.count(result_text)
            if result_tokens <= self.tool_result_threshold:
                continue

            result_msg.content = self._truncate_text(
                text=result_text,
                original_tokens=result_tokens,
            )
            result_item.content_text = result_msg.text
            result_item.token_count = context.token_counter.count(result_item.content_text)
            self._merge_truncation_metadata(
                result_item,
                [
                    {
                        "field": "tool_result.content",
                        "tool_call_id": result_msg.tool_call_id,
                        "original_tokens": result_tokens,
                        "kept_tokens": self.preview_tokens,
                        "threshold": self.tool_result_threshold,
                    }
                ],
            )
            tool_results_truncated += 1

        return tool_calls_truncated, tool_results_truncated

    def _truncate_text(self, *, text: str, original_tokens: int) -> str:
        preview = self._take_first_tokens(text=text, max_tokens=self.preview_tokens, total_tokens=original_tokens)
        return f"{preview}\n[TRUNCATED original~{original_tokens} tokens]"

    def _take_first_tokens(self, *, text: str, max_tokens: int, total_tokens: int) -> str:
        if not text:
            return text
        if max_tokens <= 0:
            return ""
        try:
            import tiktoken

            encoder = tiktoken.get_encoding("cl100k_base")
            token_ids = encoder.encode(text)
            return encoder.decode(token_ids[:max_tokens])
        except Exception:
            token_base = max(total_tokens, 1)
            ratio = min(max_tokens / token_base, 1.0)
            keep_chars = max(1, int(len(text) * ratio))
            return text[:keep_chars]

    def _refresh_assistant_item_tokens(self, context: "ContextIR", item: "ContextItem") -> None:
        from comate_agent_sdk.llm.messages import AssistantMessage

        message = item.message
        if not isinstance(message, AssistantMessage):
            return

        content_text = message.text
        if message.tool_calls:
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
            content_text = content_text + "\n" + tool_calls_json if content_text else tool_calls_json

        item.content_text = content_text
        item.token_count = context.token_counter.count(content_text)

    def _merge_truncation_metadata(
        self,
        item: "ContextItem",
        details: list[dict[str, object]],
    ) -> None:
        existing = item.metadata.get("truncation")
        detail_list: list[dict[str, object]]
        if isinstance(existing, dict):
            old_details = existing.get("details")
            if isinstance(old_details, list):
                detail_list = [d for d in old_details if isinstance(d, dict)]
            else:
                detail_list = []
        else:
            detail_list = []

        detail_list.extend(details)
        item.metadata["truncated"] = True
        item.metadata["truncation"] = {
            "details": detail_list,
        }

    def _collect_recent_round_protected_ids(self, context: "ContextIR") -> set[str]:
        from comate_agent_sdk.llm.messages import UserMessage

        items = context.conversation.items
        round_ranges: list[tuple[int, int]] = []
        round_start: int | None = None

        for idx, item in enumerate(items):
            message = item.message
            if isinstance(message, UserMessage) and not bool(getattr(message, "is_meta", False)):
                if round_start is not None:
                    round_ranges.append((round_start, idx - 1))
                round_start = idx

        if round_start is not None:
            round_ranges.append((round_start, len(items) - 1))

        if not round_ranges:
            return set()

        protected_ranges = round_ranges[-self.dialogue_rounds_keep_min:]
        protected_ids: set[str] = set()
        for start, end in protected_ranges:
            for item in items[start:end + 1]:
                if item.item_type in (ItemType.USER_MESSAGE, ItemType.ASSISTANT_MESSAGE):
                    protected_ids.add(item.id)

        return protected_ids

    def _append_meta(
        self,
        *,
        phase: Literal["selective_start", "selective_done", "summary_start", "summary_done", "rollback"],
        tokens_before: int,
        tokens_after: int,
        stats: _CompactionStats,
        reason: str,
    ) -> None:
        self.meta_records.append(
            CompactionMetaRecord(
                phase=phase,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tool_blocks_kept=stats.tool_blocks_kept,
                tool_blocks_dropped=stats.tool_blocks_dropped,
                tool_calls_truncated=stats.tool_calls_truncated,
                tool_results_truncated=stats.tool_results_truncated,
                reason=reason,
            )
        )

    def _extract_tool_blocks(self, context: "ContextIR") -> list[_ToolInteractionBlock]:
        """从 conversation 中提取工具交互块（assistant tool_calls + tool_results）。"""
        from comate_agent_sdk.llm.messages import AssistantMessage, ToolMessage, UserMessage

        items = context.conversation.items
        blocks: list[_ToolInteractionBlock] = []
        i = 0
        while i < len(items):
            item = items[i]
            msg = item.message
            if not isinstance(msg, AssistantMessage) or not msg.tool_calls:
                i += 1
                continue

            tool_calls: list[_ToolCallInfo] = []
            for tc in msg.tool_calls:
                tool_calls.append(
                    _ToolCallInfo(
                        tool_call_id=tc.id,
                        tool_name=tc.function.name,
                    )
                )

            call_ids = {tc.tool_call_id for tc in tool_calls}

            found: set[str] = set()
            tool_result_items: list["ContextItem"] = []
            end_idx = i

            for j in range(i + 1, len(items)):
                next_msg = items[j].message

                # 避免跨越到下一个 tool_calls 起点
                if isinstance(next_msg, AssistantMessage) and next_msg.tool_calls:
                    break

                # 避免跨越到下一轮真实用户输入
                if isinstance(next_msg, UserMessage) and not bool(getattr(next_msg, "is_meta", False)):
                    break

                if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id in call_ids:
                    tool_result_items.append(items[j])
                    found.add(next_msg.tool_call_id)
                    end_idx = j
                    if len(found) >= len(call_ids):
                        break

            blocks.append(
                _ToolInteractionBlock(
                    start_idx=i,
                    end_idx=end_idx,
                    assistant_item_id=item.id,
                    tool_calls=tool_calls,
                    tool_result_items=tool_result_items,
                )
            )
            i = end_idx + 1

        return blocks

    def _should_offload(self, item: "ContextItem") -> bool:
        """判断是否需要卸载。"""
        if not self.fs or not self.offload_policy:
            return False
        if not self.offload_policy.enabled:
            return False
        if item.item_type == ItemType.COMPACTION_SUMMARY:
            return False
        type_enabled = self.offload_policy.type_enabled.get(
            item.item_type.value,
            False,
        )
        if not type_enabled:
            return False
        threshold_by_type = getattr(self.offload_policy, "token_threshold_by_type", {}) or {}
        threshold = int(
            threshold_by_type.get(item.item_type.value, self.offload_policy.token_threshold)
        )
        return item.token_count >= threshold

    async def _fallback_full_summary_with_retry(self, context: "ContextIR") -> tuple[bool, str]:
        """带重试的全量摘要回退。"""
        max_attempts = max(1, int(self.summary_retry_attempts) + 1)
        last_reason = "unknown"
        for attempt in range(1, max_attempts + 1):
            success, reason = await self._fallback_full_summary_once(context)
            if success:
                return True, "success"
            last_reason = reason
            logger.warning(
                f"全量摘要失败 (attempt {attempt}/{max_attempts}): reason={reason}"
            )
        return False, last_reason

    async def _fallback_full_summary_once(self, context: "ContextIR") -> tuple[bool, str]:
        """单次全量摘要执行。"""
        from comate_agent_sdk.agent.compaction import CompactionService
        from comate_agent_sdk.context.items import ContextItem
        from comate_agent_sdk.llm.messages import UserMessage

        if self.llm is None:
            logger.warning("无法执行全量摘要回退：未提供 LLM")
            return False, "llm_missing"

        conversation_messages = [
            item.message
            for item in context.conversation.items
            if item.message is not None
        ]
        if not conversation_messages:
            return False, "no_messages"

        usage_source = "compaction"
        if self.source_prefix:
            usage_source = f"{self.source_prefix}:compaction"

        service = CompactionService(
            llm=self.llm,
            token_cost=self.token_cost,
            usage_source=usage_source,
        )
        try:
            result = await service.compact(
                conversation_messages,
                self.llm,
                level=self.level,
            )
        except Exception as exc:
            return False, f"summary_exception:{type(exc).__name__}"

        if not result.compacted or not result.summary:
            reason = result.failure_reason or "compact_false"
            if result.stop_reason:
                reason = f"{reason}|stop_reason={result.stop_reason}"
            if result.failure_detail:
                reason = f"{reason}|{result.failure_detail}"
            return False, reason

        summary_item = ContextItem(
            item_type=ItemType.COMPACTION_SUMMARY,
            message=UserMessage(content=result.summary),
            content_text=result.summary,
            token_count=context.token_counter.count(result.summary),
            priority=DEFAULT_PRIORITIES[ItemType.COMPACTION_SUMMARY],
        )
        context.replace_conversation([summary_item])
        logger.info(f"全量摘要完成: 新 token 数 ~{summary_item.token_count}")
        return True, "success"
