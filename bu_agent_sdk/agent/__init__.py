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
    SessionInitEvent,
    StopEvent,
    SubagentStartEvent,
    SubagentStopEvent,
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
from bu_agent_sdk.agent.options import ComateAgentOptions
from bu_agent_sdk.agent.service import Agent
from bu_agent_sdk.agent.prompts import MEMORY_NOTICE

__all__ = [
    "Agent",
    "ComateAgentOptions",
    # Chat session
    "ChatSession",
    "ChatSessionError",
    "ChatSessionClosedError",
    # Events
    "AgentEvent",
    "SessionInitEvent",
    "StopEvent",
    "SubagentStartEvent",
    "SubagentStopEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    # Compaction
    "CompactionConfig",
    "CompactionResult",
    "CompactionService",

    ## prompts
    "MEMORY_NOTICE",
]
