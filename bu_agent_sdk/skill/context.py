"""Skill Execution Context 修改"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bu_agent_sdk.agent.service import Agent
    from bu_agent_sdk.skill.models import SkillDefinition

logger = logging.getLogger(__name__)


def apply_skill_context(agent: "Agent", skill_def: "SkillDefinition") -> None:
    """应用 Skill 的 execution context 修改（持久化）

    Skill 的工具权限修改会一直生效，不会自动退出。
    注意: 模型切换功能已弃用，配置会被忽略。

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

    # 2. [已弃用] Skill 模型切换功能
    # 注意: SKILL 不再支持修改模型配置，该字段已被忽略
    # 如需使用不同模型，请使用 Subagent 功能
    if skill_def.model and skill_def.model != "inherit":
        logger.warning(
            f"Skill '{skill_def.name}': model 配置已弃用并被忽略 (配置值: {skill_def.model}). "
            "如需使用不同模型，请使用 Subagent 功能。"
        )
