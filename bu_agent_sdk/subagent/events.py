"""
Subagent 事件类型定义
"""

from dataclasses import dataclass

from bu_agent_sdk.agent.events import AgentEvent
from bu_agent_sdk.subagent.models import SubagentResult


@dataclass
class SubagentStartEvent:
    """Subagent 开始执行事件"""

    subagent_name: str  # Subagent 名称
    task: str  # 任务描述


@dataclass
class SubagentEvent:
    """Subagent 内部事件包装

    用于将 Subagent 的事件传播到父 Agent
    """

    subagent_name: str  # Subagent 名称
    event: AgentEvent  # 原始 Agent 事件


@dataclass
class SubagentCompleteEvent:
    """Subagent 执行完成事件"""

    subagent_name: str  # Subagent 名称
    result: SubagentResult  # 执行结果
