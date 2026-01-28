"""
A framework for building agentic applications with LLMs.

Example:
    from bu_agent_sdk import Agent
    from bu_agent_sdk.llm import ChatOpenAI
    from bu_agent_sdk.tools import tool

    @tool("Add two numbers")
    async def add(a: int, b: int) -> int:
        return a + b

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        tools=[add],
    )

    result = await agent.query("What is 2 + 3?")
"""

from bu_agent_sdk.agent import Agent
from bu_agent_sdk.observability import Laminar, observe, observe_debug
from bu_agent_sdk.skill import (
    SkillDefinition,
    apply_skill_context,
    create_skill_tool,
    discover_skills,
)
from bu_agent_sdk.subagent import (
    AgentDefinition,
    SubagentCompleteEvent,
    SubagentEvent,
    SubagentResult,
    SubagentStartEvent,
)

__all__ = [
    "Agent",
    "Laminar",
    "observe",
    "observe_debug",
    # Subagent support
    "AgentDefinition",
    "SubagentResult",
    "SubagentStartEvent",
    "SubagentEvent",
    "SubagentCompleteEvent",
    # Skill support
    "SkillDefinition",
    "discover_skills",
    "create_skill_tool",
    "apply_skill_context",
]
