"""DeepSeek LLM integration with thinking mode support.

DeepSeek uses OpenAI-compatible API with custom `reasoning_content` field
for thinking mode. This module maps `reasoning_content` to the unified
thinking infrastructure (ContentPartThinkingParam, raw_content_blocks).
"""

from comate_agent_sdk.llm.deepseek.chat import ChatDeepSeek

__all__ = ["ChatDeepSeek"]
