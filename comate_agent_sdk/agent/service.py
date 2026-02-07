"""Agent public service exports."""

from comate_agent_sdk.agent.core import (
    Agent as _AgentTemplateAlias,
    AgentRuntime as _AgentRuntime,
    AgentTemplate as _AgentTemplate,
)
from comate_agent_sdk.agent.system_prompt import SystemPromptConfig as _SystemPromptConfig
from comate_agent_sdk.agent.system_prompt import SystemPromptType

Agent = _AgentTemplateAlias
AgentTemplate = _AgentTemplate
AgentRuntime = _AgentRuntime
SystemPromptConfig = _SystemPromptConfig

# 兼容性：保持 pickle / 反射里显示的 module 路径不变
Agent.__module__ = __name__
AgentTemplate.__module__ = __name__
AgentRuntime.__module__ = __name__
SystemPromptConfig.__module__ = __name__

__all__ = [
    "Agent",
    "AgentTemplate",
    "AgentRuntime",
    "SystemPromptConfig",
    "SystemPromptType",
]
