"""上下文预算控制模块

每类型可设 token 限额，auto-compact 基于 budget 阈值触发。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from comate_agent_sdk.context.items import ItemType

logger = logging.getLogger("comate_agent_sdk.context.budget")

if TYPE_CHECKING:
    from comate_agent_sdk.llm.base import BaseChatModel
    from comate_agent_sdk.llm.messages import BaseMessage


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

    async def count_messages_for_model(
        self,
        messages: list["BaseMessage"],
        *,
        llm: "BaseChatModel | None",
        timeout_ms: int = 300,
    ) -> int:
        """按 provider/model 估算整段 messages token 数（失败自动回退）。"""
        if not messages:
            return 0

        if llm is None:
            return self._fallback_count_messages(messages)

        provider = (getattr(llm, "provider", "") or "").strip().lower()

        if provider == "openai":
            count = self._count_openai_messages(messages, llm)
            if count is not None:
                return count
            return self._fallback_count_messages(messages)

        if provider == "anthropic":
            count = await self._count_anthropic_messages(messages, llm, timeout_ms=timeout_ms)
            if count is not None:
                return count
            return self._fallback_count_messages(messages)

        if provider == "google":
            count = await self._count_google_messages(messages, llm, timeout_ms=timeout_ms)
            if count is not None:
                return count
            return self._fallback_count_messages(messages)

        # 回退：粗略估算（中文约 1 字 = 1-2 tokens，英文约 4 字符 = 1 token）
        return self._fallback_count_messages(messages)

    def _fallback_count_messages(self, messages: list["BaseMessage"]) -> int:
        total = 0
        for msg in messages:
            text = getattr(msg, "text", "")
            if isinstance(text, str) and text:
                total += self.count(text)

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                try:
                    tool_calls_json = json.dumps(
                        [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                        ensure_ascii=False,
                    )
                    total += self.count(tool_calls_json)
                except Exception:
                    pass

        return max(total, 1)

    def _count_openai_messages(
        self,
        messages: list["BaseMessage"],
        llm: "BaseChatModel",
    ) -> int | None:
        """OpenAI 本地估算：序列化后按模型 encoder 计数。"""
        try:
            from comate_agent_sdk.llm.openai.serializer import OpenAIMessageSerializer

            serialized = OpenAIMessageSerializer.serialize_messages(messages)
            payload = json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))
            return self.count_for_model(
                payload,
                provider="openai",
                model=str(getattr(llm, "model", "")),
            )
        except Exception as e:
            logger.debug(f"OpenAI message token count fallback: {e}")
            return None

    async def _count_anthropic_messages(
        self,
        messages: list["BaseMessage"],
        llm: "BaseChatModel",
        *,
        timeout_ms: int,
    ) -> int | None:
        """Anthropic 估算：优先 SDK count_tokens。"""
        try:
            from comate_agent_sdk.llm.anthropic.serializer import AnthropicMessageSerializer

            serialized_messages, system_prompt = AnthropicMessageSerializer.serialize_messages(messages)
            client = llm.get_client()  # type: ignore[attr-defined]

            kwargs: dict[str, Any] = {
                "model": str(getattr(llm, "model", "")),
                "messages": serialized_messages,
                "timeout": max(1, int(timeout_ms)) / 1000.0,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            resp = await asyncio.wait_for(
                client.messages.count_tokens(**kwargs),
                timeout=max(1, int(timeout_ms)) / 1000.0 + 0.05,
            )
            input_tokens = int(getattr(resp, "input_tokens", 0) or 0)
            if input_tokens > 0:
                return input_tokens
        except Exception as e:
            logger.warning(f"Anthropic token count failed, fallback to cl100k: {e}")
        return None

    async def _count_google_messages(
        self,
        messages: list["BaseMessage"],
        llm: "BaseChatModel",
        *,
        timeout_ms: int,
    ) -> int | None:
        """Google 估算：优先 SDK count_tokens。"""
        try:
            from comate_agent_sdk.llm.google.serializer import GoogleMessageSerializer

            include_system_in_user = bool(getattr(llm, "include_system_in_user", False))
            formatted_messages, system_message = GoogleMessageSerializer.serialize_messages(
                messages,
                include_system_in_user=include_system_in_user,
            )
            if system_message:
                # 与模型实际调用保持一致：若未嵌入用户消息，补一条 system 文本估算
                from google.genai.types import Content, Part

                formatted_messages = list(formatted_messages)
                formatted_messages.insert(0, Content(role="user", parts=[Part.from_text(text=system_message)]))

            timeout_s = max(1, int(timeout_ms)) / 1000.0
            client = llm.get_client()  # type: ignore[attr-defined]
            resp = await asyncio.wait_for(
                client.aio.models.count_tokens(
                    model=str(getattr(llm, "model", "")),
                    contents=formatted_messages,
                ),
                timeout=timeout_s + 0.05,
            )
            total_tokens = int(getattr(resp, "total_tokens", 0) or 0)
            if total_tokens > 0:
                return total_tokens
        except Exception as e:
            logger.warning(f"Google token count failed, fallback to cl100k: {e}")
        return None


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
