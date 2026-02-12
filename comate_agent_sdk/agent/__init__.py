"""
Agent module for running agentic loops with tool calling.
"""

from comate_agent_sdk.agent.compaction import (
    CompactionConfig,
    CompactionResult,
    CompactionService,
)
from comate_agent_sdk.agent.events import (
    AgentEvent,
    CompactionMetaEvent,
    SessionInitEvent,
    StopEvent,
    PreCompactEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageDeltaEvent,
    UserQuestionEvent,
)
from comate_agent_sdk.agent.chat_session import (
    ChatSession,
    ChatSessionClosedError,
    ChatSessionError,
)
from comate_agent_sdk.agent.interrupt import SessionRunController
from comate_agent_sdk.agent.options import AgentConfig
from comate_agent_sdk.agent.service import Agent, AgentRuntime, AgentTemplate
from comate_agent_sdk.agent.prompts import MEMORY_NOTICE

__all__ = [
    "Agent",
    "AgentTemplate",
    "AgentRuntime",
    "AgentConfig",
    # Chat session
    "ChatSession",
    "ChatSessionError",
    "ChatSessionClosedError",
    "SessionRunController",
    # Events
    "AgentEvent",
    "SessionInitEvent",
    "StopEvent",
    "UsageDeltaEvent",
    "SubagentProgressEvent",
    "SubagentStartEvent",
    "SubagentStopEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "UserQuestionEvent",
    "PreCompactEvent",
    "CompactionMetaEvent",
    # Compaction
    "CompactionConfig",
    "CompactionResult",
    "CompactionService",

    # prompts
    "MEMORY_NOTICE",
]
