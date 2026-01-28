"""Skill Meta-Tool 实现"""

import logging

from bu_agent_sdk.skill.models import SkillDefinition
from bu_agent_sdk.tools import Tool, tool

logger = logging.getLogger(__name__)


def create_skill_tool(skills: list[SkillDefinition]) -> Tool:
    """创建 Skill meta-tool

    这个工具的描述会动态包含所有可用 skills 的列表
    """
    # Token budget 限制（参考 Claude Code 默认 15000 字符）
    TOKEN_BUDGET = 15000

    # 过滤掉 disable_model_invocation=True 的 skills
    active_skills = [s for s in skills if not s.disable_model_invocation]

    # 格式化 skill 列表
    skill_list = []
    for skill in active_skills:
        skill_list.append(f'  - "{skill.name}": {skill.description}')

    skills_str = "\n".join(skill_list)

    # 构建工具描述
    description = f"""Execute a skill within the main conversation.

Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke skills using this tool with the skill name only (no arguments)
- When you invoke a skill, the skill's prompt will expand and provide detailed instructions
- Example: skill_name="explain-code"
- You can activate multiple different skills in the same conversation

Important:
- Only use skills listed below
- DO NOT activate the same skill twice (prevents infinite recursion)
- You CAN activate multiple different skills (e.g., first "skill-a", then "skill-b")
- Skill modifications (tool permissions, model) will remain active after invocation

Available skills:
{skills_str}
"""

    # Token budget 检查
    if len(description) > TOKEN_BUDGET:
        logger.warning(f"Skill tool description exceeds token budget ({len(description)} > {TOKEN_BUDGET})")
        # 截断：只保留前 N 个 skills
        max_skills = max(1, len(active_skills) * TOKEN_BUDGET // len(description))
        active_skills = active_skills[:max_skills]
        # 重新生成 skill_list
        skill_list = [f'  - "{skill.name}": {skill.description}' for skill in active_skills]
        skills_str = "\n".join(skill_list)
        description = f"""Execute a skill within the main conversation.

Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke skills using this tool with the skill name only (no arguments)
- When you invoke a skill, the skill's prompt will expand and provide detailed instructions
- You can activate multiple different skills in the same conversation

Important:
- Only use skills listed below
- DO NOT activate the same skill twice (prevents infinite recursion)
- You CAN activate multiple different skills

Available skills (truncated):
{skills_str}
"""

    @tool(description, name="Skill")
    async def Skill(skill_name: str) -> str:
        """调用 Skill

        Args:
            skill_name: Skill 名称
        """
        # 实际执行逻辑在 Agent._execute_skill_call() 中
        return f"Skill '{skill_name}' loaded"

    return Skill
