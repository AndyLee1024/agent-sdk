"""ContextUsageTracker: 统一 context token usage 追踪。

替代 ContextTokenAccounting + CompactionService._last_usage，
作为唯一的"当前 context 已用 token"权威来源。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("comate_agent_sdk.context.usage_tracker")

_DEFAULT_CONTEXT_WINDOW = 128_000
_DEFAULT_THRESHOLD_RATIO = 0.85
_PRECHECK_BUFFER = 500


@dataclass
class ContextUsageTracker:
    """统一 context token usage 追踪器。

    持有最近一次 API 返回的 total_tokens（context_usage），
    并据此计算压缩阈值判断和 precheck 增量估算。

    Attributes:
        context_usage: 最近 API 返回的 total_tokens（0 = 尚未收到首次响应）。
        context_window: 模型 context window 大小（token 数）。
        threshold_ratio: 触发 compaction 的利用率阈值（0.0-1.0）。
    """

    context_usage: int = 0
    context_window: int = _DEFAULT_CONTEXT_WINDOW
    threshold_ratio: float = _DEFAULT_THRESHOLD_RATIO
    _ir_total_at_last_report: int = field(default=0, repr=False)

    PRECHECK_BUFFER: int = field(default=_PRECHECK_BUFFER, repr=False, init=False)

    def observe_response(self, total_tokens: int, ir_total: int) -> None:
        """记录一次 LLM 响应的 usage。

        Args:
            total_tokens: API 返回的 total_tokens。
            ir_total: 本次响应时 ContextIR 的 total_tokens（用于后续 precheck delta 计算）。
        """
        self.context_usage = max(0, int(total_tokens))
        self._ir_total_at_last_report = max(0, int(ir_total))
        logger.debug(
            f"context usage updated: reported={self.context_usage}, "
            f"ir_snapshot={self._ir_total_at_last_report}, "
            f"threshold={self.threshold}"
        )

    def should_compact_post_response(self) -> bool:
        """判断 LLM 响应后是否需要压缩（基于 API 报告的 total_tokens）。

        Returns:
            True 当 context_usage >= threshold 且已收到过 API 响应。
        """
        if self.context_usage <= 0:
            return False
        return self.context_usage >= self.threshold

    def should_compact_precheck(self, ir_total: int) -> bool:
        """判断下一次 invoke_llm 前是否需要压缩（增量估算）。

        Args:
            ir_total: 当前 ContextIR 的 total_tokens。

        Returns:
            True 当估算 token 数 >= threshold。
        """
        return self.estimate_precheck(ir_total) >= self.threshold

    def estimate_precheck(self, ir_total: int) -> int:
        """估算下一次 invoke_llm 时的 context token 数（含安全缓冲）。

        当 context_usage == 0（首轮，尚未收到 API 报告）时，
        回退到 ir_total + PRECHECK_BUFFER。

        Args:
            ir_total: 当前 ContextIR 的 total_tokens。

        Returns:
            估算 token 数（用于 precheck 阈值比较和 PreCompactEvent 展示）。
        """
        if self.context_usage <= 0:
            return ir_total + self.PRECHECK_BUFFER
        ir_delta = max(0, ir_total - self._ir_total_at_last_report)
        return self.context_usage + ir_delta + self.PRECHECK_BUFFER

    def reset_after_compaction(self) -> None:
        """compaction 完成后重置，使下次 precheck 回退到 IR total 估算。"""
        self.context_usage = 0
        self._ir_total_at_last_report = 0
        logger.debug("context usage tracker reset after compaction")

    @property
    def threshold(self) -> int:
        """触发 compaction 的 token 数阈值。"""
        return int(self.context_window * self.threshold_ratio)

    @property
    def remaining_tokens(self) -> int:
        """剩余可用 token 数（基于最近报告的 context_usage）。"""
        return max(0, self.context_window - self.context_usage)

    @property
    def utilization_ratio(self) -> float:
        """当前 context 利用率（0.0-1.0）。"""
        if self.context_window <= 0:
            return 0.0
        return self.context_usage / self.context_window
