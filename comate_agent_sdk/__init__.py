"""
A framework for building agentic applications with LLMs.
"""

from comate_agent_sdk.agent import (
    Agent,
    AgentConfig,
    AgentRuntime,
    AgentTemplate,
    ChatSession,
    SubagentProgressEvent,
    SessionInitEvent,
    UsageDeltaEvent,
)
from comate_agent_sdk.context import MemoryConfig
from comate_agent_sdk.observability import Laminar, observe, observe_debug
from comate_agent_sdk.tools import Depends, tool
from comate_agent_sdk.skill import (
    SkillDefinition,
    apply_skill_context,
    create_skill_tool,
    discover_skills,
)
from comate_agent_sdk.subagent import (
    AgentDefinition,
    SubagentEvent,
    SubagentResult,
    SubagentStartEvent,
    SubagentStopEvent,
)
from comate_agent_sdk.mcp import create_sdk_mcp_server, mcp_tool

__all__ = [
    "Agent",
    "AgentTemplate",
    "AgentRuntime",
    "AgentConfig",
    "ChatSession",
    "SessionInitEvent",
    "UsageDeltaEvent",
    "SubagentProgressEvent",
    "Laminar",
    "observe",
    "observe_debug",
    "tool",
    "Depends",
    # Memory support
    "MemoryConfig",
    # Subagent support
    "AgentDefinition",
    "SubagentResult",
    "SubagentStartEvent",
    "SubagentEvent",
    "SubagentStopEvent",
    # Skill support
    "SkillDefinition",
    "discover_skills",
    "create_skill_tool",
    "apply_skill_context",
    # MCP support
    "create_sdk_mcp_server",
    "mcp_tool",
]
