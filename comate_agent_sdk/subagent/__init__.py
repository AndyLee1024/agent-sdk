"""
Subagent 模块 - 提供 Subagent 功能
"""

from comate_agent_sdk.subagent.builtin import get_builtin_agent_names, get_builtin_agents
from comate_agent_sdk.subagent.events import (
    SubagentEvent,
    SubagentStartEvent,
    SubagentStopEvent,
)
from comate_agent_sdk.subagent.loader import discover_subagents
from comate_agent_sdk.subagent.models import AgentDefinition, SubagentResult
from comate_agent_sdk.subagent.prompts import generate_subagent_prompt
from comate_agent_sdk.subagent.task_tool import create_task_tool

__all__ = [
    "AgentDefinition",
    "SubagentResult",
    "discover_subagents",
    "get_builtin_agents",
    "get_builtin_agent_names",
    "generate_subagent_prompt",
    "create_task_tool",
    "SubagentStartEvent",
    "SubagentEvent",
    "SubagentStopEvent",
]
