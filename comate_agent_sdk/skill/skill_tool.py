"""Skill Meta-Tool 实现

设计决策：
- Tool description 只包含简洁的功能说明
- Skills 列表和详细使用规则放在 system_prompt 中（通过 generate_skill_prompt）
- 这样做的好处是避免每次 API 调用都发送大量重复的使用指南
"""

import logging

from comate_agent_sdk.skill.models import SkillDefinition
from comate_agent_sdk.tools import Tool, tool

logger = logging.getLogger(__name__)


def create_skill_tool(skills: list[SkillDefinition]) -> Tool:
    """创建 Skill meta-tool（简洁版）

    Tool description 只包含简洁的功能说明。
    Skills 列表和详细使用规则通过 generate_skill_prompt() 注入到 system_prompt。

    Args:
        skills: Skill 定义列表（用于日志记录，实际列表在 system_prompt 中）

    Returns:
        Skill 工具实例
    """
    # 过滤掉 disable_model_invocation=True 的 skills（仅用于日志）
    active_skills = [s for s in skills if not s.disable_model_invocation]
    logger.debug(f"Creating Skill tool with {len(active_skills)} active skill(s)")

    skill_usage_rule = """Load best practices for specific task types before execution.

When to use:
- Before creating documents (docx, pptx, xlsx, pdf)
- Before complex operations where domain-specific guidance exists
- When available skills are listed in system prompt

Workflow: Call Skill first → read returned instructions → execute task following best practices
Available skills are listed in the system prompt under <skills>.
    """

    # 简洁的工具描述（详细使用规则在 system_prompt 中）
    description = "Load task-specific best practices before execution. Check <skills> for list. "

    @tool(description, name="Skill", usage_rules=skill_usage_rule)
    async def Skill(skill_name: str) -> str:
        """调用 Skill

        Args:
            skill_name: Skill 名称
        """
        # 实际执行逻辑在 Agent._execute_skill_call() 中
        return f"Skill '{skill_name}' loaded"

    return Skill
