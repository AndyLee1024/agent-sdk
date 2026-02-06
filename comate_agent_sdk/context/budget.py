"""上下文预算控制模块

每类型可设 token 限额，auto-compact 基于 budget 阈值触发。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from comate_agent_sdk.context.items import ItemType

logger = logging.getLogger("comate_agent_sdk.context.budget")


@dataclass
class BudgetConfig:
    """上下文预算配置

    Attributes:
        total_limit: 全局 token 上限（0=不限）
        type_limits: 每类型 token 上限，如 {"tool_result": 30000}
        compact_threshold_ratio: 触发压缩的利用率比例
    """

    total_limit: int = 0
    type_limits: dict[str, int] = field(default_factory=dict)
    compact_threshold_ratio: float = 0.80


class TokenCounter:
    """Token 计数器

    优先使用 tiktoken(cl100k_base)，回退到 len(text) // 3
    """

    def __init__(self) -> None:
        self._encoder = None
        self._initialized = False

    def _init_encoder(self) -> None:
        """延迟初始化 tiktoken encoder"""
        if self._initialized:
            return
        self._initialized = True
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.debug(
                "tiktoken 未安装，使用 len(text)//3 估算 token 数。"
                "安装 tiktoken 可获得更精确的估算：uv add tiktoken"
            )

    def count(self, text: str) -> int:
        """估算文本的 token 数

        Args:
            text: 要估算的文本

        Returns:
            估算的 token 数
        """
        if not text:
            return 0

        self._init_encoder()

        if self._encoder is not None:
            return len(self._encoder.encode(text))

        # 回退：粗略估算（中文约 1 字 = 1-2 tokens，英文约 4 字符 = 1 token）
        return max(1, len(text) // 3)


@dataclass
class BudgetStatus:
    """预算状态快照

    Attributes:
        total_tokens: 总 token 数
        header_tokens: Header 段 token 数
        conversation_tokens: Conversation 段 token 数
        tokens_by_type: 各类型 token 数
        total_limit: 总限额
        compact_threshold_ratio: 压缩阈值比例
    """

    total_tokens: int = 0
    header_tokens: int = 0
    conversation_tokens: int = 0
    tokens_by_type: dict[ItemType, int] = field(default_factory=dict)
    total_limit: int = 0
    compact_threshold_ratio: float = 0.80

    @property
    def utilization_ratio(self) -> float:
        """当前利用率（0.0-1.0），若无限额则返回 0.0"""
        if self.total_limit <= 0:
            return 0.0
        return self.total_tokens / self.total_limit

    @property
    def is_over_threshold(self) -> bool:
        """是否超过压缩阈值"""
        if self.total_limit <= 0:
            return False
        return self.utilization_ratio >= self.compact_threshold_ratio
