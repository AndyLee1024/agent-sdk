"""Skill 系统（兼容 Claude Code SKILL.md 格式）"""

from comate_agent_sdk.skill.context import apply_skill_context
from comate_agent_sdk.skill.loader import discover_skills
from comate_agent_sdk.skill.models import SkillDefinition
from comate_agent_sdk.skill.prompts import generate_skill_prompt
from comate_agent_sdk.skill.skill_tool import create_skill_tool

__all__ = [
    "SkillDefinition",
    "discover_skills",
    "create_skill_tool",
    "apply_skill_context",
    "generate_skill_prompt",
]
