from dataclasses import dataclass

from comate_agent_sdk.llm.anthropic.like import ChatAnthropicLike
from comate_agent_sdk.llm.messages import (
    ContentPartRedactedThinkingParam,
    ContentPartTextParam,
    ContentPartThinkingParam,
)
from comate_agent_sdk.llm.views import ChatInvokeUsage
from comate_agent_sdk.tokens.custom_pricing import (
    resolve_max_input_tokens_from_custom_pricing,
)


@dataclass
class ChatMiniMax(ChatAnthropicLike):
    """MiniMax Anthropic-compatible provider.

    MiniMax-specific quirks vs standard Claude:
    - thinking blocks lack ``signature`` field → use duck-typing instead of isinstance
    - no prompt-caching beta header
    """

    prompt_cache_beta: str | None = None  # MiniMax 不支持 Anthropic prompt-caching beta

    @property
    def provider(self) -> str:
        return "minimax"

    def get_context_window(self) -> int | None:
        """Resolve MiniMax context window from custom pricing data."""
        return resolve_max_input_tokens_from_custom_pricing(str(self.model))

    def _extract_thinking(self, response):
        """MiniMax thinking blocks have no ``signature``, so isinstance(ThinkingBlock) fails.
        Duck-type via block.type instead."""
        thinking_parts, redacted_parts = [], []
        for block in response.content:
            t = getattr(block, "type", None)
            if t == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif t == "redacted_thinking":
                redacted_parts.append(getattr(block, "data", "") or "")
        return (
            "\n".join(thinking_parts) or None,
            "\n".join(redacted_parts) or None,
        )

    def _get_usage(self, response) -> ChatInvokeUsage | None:
        """MiniMax usage: true prompt = input + cache_read + cache_creation."""
        u = response.usage
        cache_read = getattr(u, "cache_read_input_tokens", None) or 0
        cache_creation = getattr(u, "cache_creation_input_tokens", None) or 0
        input_tokens = u.input_tokens
        output_tokens = u.output_tokens
        total_prompt = input_tokens + cache_read + cache_creation
        return ChatInvokeUsage(
            prompt_tokens=total_prompt,
            completion_tokens=output_tokens,
            total_tokens=total_prompt + output_tokens,
            prompt_cached_tokens=cache_read or None,
            prompt_cache_creation_tokens=cache_creation or None,
            prompt_image_tokens=None,
        )

    @staticmethod
    def _build_structured_content(response):
        """Same duck-typing fix; fall back to empty string for missing ``signature``."""
        if not response.content:
            return None
        blocks = []
        for block in response.content:
            t = getattr(block, "type", None)
            if t == "text":
                blocks.append(ContentPartTextParam(text=getattr(block, "text", "")))
            elif t == "thinking":
                blocks.append(
                    ContentPartThinkingParam(
                        thinking=getattr(block, "thinking", ""),
                        signature=getattr(block, "signature", "") or "",
                    )
                )
            elif t == "redacted_thinking":
                blocks.append(
                    ContentPartRedactedThinkingParam(data=getattr(block, "data", ""))
                )
        return blocks or None
