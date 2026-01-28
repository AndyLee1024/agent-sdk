"""Skill Execution Context 修改"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bu_agent_sdk.agent.service import Agent
    from bu_agent_sdk.skill.models import SkillDefinition

logger = logging.getLogger(__name__)


def apply_skill_context(agent: "Agent", skill_def: "SkillDefinition") -> None:
    """应用 Skill 的 execution context 修改（持久化）

    Skill 的工具权限和模型修改会一直生效，不会自动退出。

    Args:
        agent: Agent 实例
        skill_def: Skill 定义
    """
    # 1. 应用 Skill 工具权限
    if skill_def.allowed_tools:
        allowed_set = set(skill_def.allowed_tools)
        # 过滤工具列表
        agent.tools = [t for t in agent.tools if t.name in allowed_set]
        # 过滤工具映射
        agent._tool_map = {k: v for k, v in agent._tool_map.items() if k in allowed_set}
        logger.info(f"Skill '{skill_def.name}': restricted to {len(agent.tools)} tools (persistent)")

    # 2. 应用 Skill 模型切换
    if skill_def.model and skill_def.model != "inherit":
        from bu_agent_sdk.subagent.task_tool import resolve_model

        agent.llm = resolve_model(skill_def.model, agent.llm)
        logger.info(f"Skill '{skill_def.name}': switched to model {skill_def.model} (persistent)")
