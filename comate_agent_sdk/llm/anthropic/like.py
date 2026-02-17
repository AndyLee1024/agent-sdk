from dataclasses import dataclass

from comate_agent_sdk.llm.anthropic.chat import ChatAnthropic


@dataclass
class ChatAnthropicLike(ChatAnthropic):
    """Base class for providers using the Anthropic API protocol but not strict Claude.

    Analogous to ChatOpenAILike — a thin shell that acts as a protocol marker.
    Provider-specific quirks (e.g. missing signature, custom fields) belong
    in each concrete subclass, not here.
    """

    model: str
