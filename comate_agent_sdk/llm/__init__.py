"""
LLM abstraction layer with type-safe tool calling support.

This module provides a unified interface for chat models across different providers
(OpenAI, Anthropic, Google) with first-class support for tool calling.
"""

from typing import TYPE_CHECKING

# Auto-load .env file for API keys
from dotenv import load_dotenv

load_dotenv()

# Core types - always imported
from comate_agent_sdk.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    DeveloperMessage,
    Function,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from comate_agent_sdk.llm.messages import (
    ContentPartImageParam as ContentImage,
)
from comate_agent_sdk.llm.messages import (
    ContentPartRedactedThinkingParam as ContentRedactedThinking,
)
from comate_agent_sdk.llm.messages import (
    ContentPartRefusalParam as ContentRefusal,
)
from comate_agent_sdk.llm.messages import (
    ContentPartTextParam as ContentText,
)
from comate_agent_sdk.llm.messages import (
    ContentPartThinkingParam as ContentThinking,
)
from comate_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage

# Type stubs for lazy imports
if TYPE_CHECKING:
    from comate_agent_sdk.llm.anthropic.chat import ChatAnthropic
    from comate_agent_sdk.llm.aws.chat_anthropic import ChatAnthropicBedrock
    from comate_agent_sdk.llm.aws.chat_bedrock import ChatAWSBedrock
    from comate_agent_sdk.llm.azure.chat import ChatAzureOpenAI
    from comate_agent_sdk.llm.cerebras.chat import ChatCerebras
    from comate_agent_sdk.llm.deepseek.chat import ChatDeepSeek
    from comate_agent_sdk.llm.google.chat import ChatGoogle
    from comate_agent_sdk.llm.groq.chat import ChatGroq
    from comate_agent_sdk.llm.comate_agent_sdk.chat import ChatBrowserUse
    from comate_agent_sdk.llm.mistral.chat import ChatMistral
    from comate_agent_sdk.llm.oci_raw.chat import ChatOCIRaw
    from comate_agent_sdk.llm.ollama.chat import ChatOllama
    from comate_agent_sdk.llm.openai.chat import ChatOpenAI
    from comate_agent_sdk.llm.openrouter.chat import ChatOpenRouter
    from comate_agent_sdk.llm.vercel.chat import ChatVercel

    # Type stubs for model instances - enables IDE autocomplete
    openai_gpt_4o: ChatOpenAI
    openai_gpt_4o_mini: ChatOpenAI
    openai_gpt_4_1_mini: ChatOpenAI
    openai_o1: ChatOpenAI
    openai_o1_mini: ChatOpenAI
    openai_o1_pro: ChatOpenAI
    openai_o3: ChatOpenAI
    openai_o3_mini: ChatOpenAI
    openai_o3_pro: ChatOpenAI
    openai_o4_mini: ChatOpenAI
    openai_gpt_5: ChatOpenAI
    openai_gpt_5_mini: ChatOpenAI
    openai_gpt_5_nano: ChatOpenAI

    azure_gpt_4o: ChatAzureOpenAI
    azure_gpt_4o_mini: ChatAzureOpenAI
    azure_gpt_4_1_mini: ChatAzureOpenAI
    azure_o1: ChatAzureOpenAI
    azure_o1_mini: ChatAzureOpenAI
    azure_o1_pro: ChatAzureOpenAI
    azure_o3: ChatAzureOpenAI
    azure_o3_mini: ChatAzureOpenAI
    azure_o3_pro: ChatAzureOpenAI
    azure_gpt_5: ChatAzureOpenAI
    azure_gpt_5_mini: ChatAzureOpenAI

    google_gemini_2_0_flash: ChatGoogle
    google_gemini_2_0_pro: ChatGoogle
    google_gemini_2_5_pro: ChatGoogle
    google_gemini_2_5_flash: ChatGoogle
    google_gemini_2_5_flash_lite: ChatGoogle

# Models are imported on-demand via __getattr__

# Lazy imports mapping for heavy chat models
_LAZY_IMPORTS = {
    "ChatAnthropic": ("comate_agent_sdk.llm.anthropic.chat", "ChatAnthropic"),
    "ChatAnthropicBedrock": (
        "comate_agent_sdk.llm.aws.chat_anthropic",
        "ChatAnthropicBedrock",
    ),
    "ChatAWSBedrock": ("comate_agent_sdk.llm.aws.chat_bedrock", "ChatAWSBedrock"),
    "ChatAzureOpenAI": ("comate_agent_sdk.llm.azure.chat", "ChatAzureOpenAI"),
    "ChatBrowserUse": ("comate_agent_sdk.llm.comate_agent_sdk.chat", "ChatBrowserUse"),
    "ChatCerebras": ("comate_agent_sdk.llm.cerebras.chat", "ChatCerebras"),
    "ChatDeepSeek": ("comate_agent_sdk.llm.deepseek.chat", "ChatDeepSeek"),
    "ChatGoogle": ("comate_agent_sdk.llm.google.chat", "ChatGoogle"),
    "ChatGroq": ("comate_agent_sdk.llm.groq.chat", "ChatGroq"),
    "ChatMistral": ("comate_agent_sdk.llm.mistral.chat", "ChatMistral"),
    "ChatOCIRaw": ("comate_agent_sdk.llm.oci_raw.chat", "ChatOCIRaw"),
    "ChatOllama": ("comate_agent_sdk.llm.ollama.chat", "ChatOllama"),
    "ChatOpenAI": ("comate_agent_sdk.llm.openai.chat", "ChatOpenAI"),
    "ChatOpenRouter": ("comate_agent_sdk.llm.openrouter.chat", "ChatOpenRouter"),
    "ChatVercel": ("comate_agent_sdk.llm.vercel.chat", "ChatVercel"),
}

# Cache for model instances - only created when accessed
_model_cache: dict[str, "BaseChatModel"] = {}


def __getattr__(name: str):
    """Lazy import mechanism for heavy chat model imports and model instances."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            from importlib import import_module

            module = import_module(module_path)
            attr = getattr(module, attr_name)
            return attr
        except ImportError as e:
            raise ImportError(f"Failed to import {name} from {module_path}: {e}") from e

    # Check cache first for model instances
    if name in _model_cache:
        return _model_cache[name]

    # Try to get model instances from models module on-demand
    try:
        from comate_agent_sdk.llm.models import __getattr__ as models_getattr

        attr = models_getattr(name)
        # Cache in our clean cache dict
        _model_cache[name] = attr
        return attr
    except (AttributeError, ImportError):
        pass

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Message types
    "BaseMessage",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "DeveloperMessage",
    # Tool calling types
    "ToolCall",
    "Function",
    "ToolDefinition",
    "ToolChoice",
    # Response types
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
    # Content parts with better names
    "ContentText",
    "ContentRefusal",
    "ContentImage",
    "ContentThinking",
    "ContentRedactedThinking",
    # Chat models
    "BaseChatModel",
    "ChatOpenAI",
    "ChatBrowserUse",
    "ChatDeepSeek",
    "ChatGoogle",
    "ChatAnthropic",
    "ChatAnthropicBedrock",
    "ChatAWSBedrock",
    "ChatGroq",
    "ChatMistral",
    "ChatAzureOpenAI",
    "ChatOCIRaw",
    "ChatOllama",
    "ChatOpenRouter",
    "ChatVercel",
    "ChatCerebras",
]
