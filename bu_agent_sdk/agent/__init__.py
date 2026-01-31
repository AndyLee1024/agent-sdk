"""
Agent module for running agentic loops with tool calling.
"""

from bu_agent_sdk.agent.compaction import (
    CompactionConfig,
    CompactionResult,
    CompactionService,
)
from bu_agent_sdk.agent.events import (
    AgentEvent,
    FinalResponseEvent,
    SessionInitEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from bu_agent_sdk.agent.chat_session import (
    ChatSession,
    ChatSessionClosedError,
    ChatSessionError,
)
from bu_agent_sdk.agent.service import Agent, TaskComplete

__all__ = [
    "Agent",
    "TaskComplete",
    # Chat session
    "ChatSession",
    "ChatSessionError",
    "ChatSessionClosedError",
    # Events
    "AgentEvent",
    "FinalResponseEvent",
    "SessionInitEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    # Compaction
    "CompactionConfig",
    "CompactionResult",
    "CompactionService",
]
