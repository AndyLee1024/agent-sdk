"""上下文预算控制模块

每类型可设 token 限额，auto-compact 基于 budget 阈值触发。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

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
    compact_threshold_ratio: float = 0.75


class TokenCounter:
    """Token 计数器

    优先使用 tiktoken(cl100k_base)，回退到 len(text) // 3
    """

    def __init__(self) -> None:
        self._encoder = None
        self._openai_model_encoders: dict[str, Any] = {}
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

    def _fallback_count(self, text: str) -> int:
        return max(1, len(text) // 3)

    def _count_with_encoder(self, text: str, encoder: Any | None) -> int:
        if encoder is None:
            return self._fallback_count(text)
        try:
            return len(encoder.encode(text))
        except Exception:
            return self._fallback_count(text)

    def _get_openai_model_encoder(self, model: str) -> Any | None:
        """获取 OpenAI 模型专用 encoder（失败则回退 default）。"""
        if not model:
            return self._encoder

        cached = self._openai_model_encoders.get(model)
        if cached is not None:
            return cached

        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(model)
            self._openai_model_encoders[model] = enc
            return enc
        except Exception:
            # Unknown model / tiktoken mapping missing -> 回退 cl100k_base
            return self._encoder

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
            return self._count_with_encoder(text, self._encoder)

        # 回退：粗略估算（中文约 1 字 = 1-2 tokens，英文约 4 字符 = 1 token）
        return self._fallback_count(text)

    def count_for_model(
        self,
        text: str,
        *,
        provider: str | None,
        model: str | None,
    ) -> int:
        """按 provider/model 估算文本 token 数。

        说明：
        - OpenAI：优先使用 encoding_for_model(model)
        - 其他 provider：当前使用通用 cl100k_base 估算
        """
        if not text:
            return 0

        self._init_encoder()

        if (provider or "").strip().lower() == "openai":
            encoder = self._get_openai_model_encoder((model or "").strip())
            return self._count_with_encoder(text, encoder)

        return self.count(text)



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
