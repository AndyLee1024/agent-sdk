"""
Subagent 模块 - 提供 Subagent 功能
"""

from bu_agent_sdk.subagent.events import (
    SubagentEvent,
    SubagentStartEvent,
    SubagentStopEvent,
)
from bu_agent_sdk.subagent.loader import discover_subagents
from bu_agent_sdk.subagent.models import AgentDefinition, SubagentResult
from bu_agent_sdk.subagent.prompts import generate_subagent_prompt
from bu_agent_sdk.subagent.task_tool import create_task_tool

__all__ = [
    "AgentDefinition",
    "SubagentResult",
    "discover_subagents",
    "generate_subagent_prompt",
    "create_task_tool",
    "SubagentStartEvent",
    "SubagentEvent",
    "SubagentStopEvent",
]
