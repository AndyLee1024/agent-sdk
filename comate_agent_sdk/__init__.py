"""
A framework for building agentic applications with LLMs.

Example:
    from comate_agent_sdk import Agent
    from comate_agent_sdk.agent import ComateAgentOptions
    from comate_agent_sdk.llm import ChatOpenAI
    from comate_agent_sdk.tools import tool

    @tool("Add two numbers")
    async def add(a: int, b: int) -> int:
        return a + b

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        options=ComateAgentOptions(tools=[add]),
    )

    result = await agent.query("What is 2 + 3?")
"""

from comate_agent_sdk.agent import Agent, ChatSession, SessionInitEvent
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
    "ChatSession",
    "SessionInitEvent",
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
