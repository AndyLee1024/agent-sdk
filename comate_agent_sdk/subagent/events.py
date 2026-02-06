"""
Subagent 事件类型定义
"""

from dataclasses import dataclass

from comate_agent_sdk.agent.events import AgentEvent, SubagentStartEvent, SubagentStopEvent


@dataclass
class SubagentEvent:
    """Subagent 内部事件包装

    用于将 Subagent 的事件传播到父 Agent
    """

    subagent_name: str  # Subagent 名称
    event: AgentEvent  # 原始 Agent 事件
