"""SDK 内置 Subagent 注册"""
from comate_agent_sdk.subagent.models import AgentDefinition

# ========== 在这里导入并注册内置 subagent ==========
from comate_agent_sdk.subagent.builtin.explorer import ExplorerAgent
from comate_agent_sdk.subagent.builtin.plan import PlanAgent

BUILTIN_AGENTS: list[AgentDefinition] = [
    ExplorerAgent,
    PlanAgent,
]


def get_builtin_agents() -> list[AgentDefinition]:
    """返回所有内置 subagent 定义（副本）"""
    return list(BUILTIN_AGENTS)


def get_builtin_agent_names() -> set[str]:
    """返回所有内置 subagent 名称集合（用于冲突检测）"""
    return {a.name for a in BUILTIN_AGENTS}
