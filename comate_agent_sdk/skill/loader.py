"""Skill 自动发现和加载"""

import logging
from pathlib import Path

from comate_agent_sdk.skill.models import SkillDefinition

logger = logging.getLogger(__name__)


def discover_skills(project_root: Path | None = None) -> list[SkillDefinition]:
    """自动发现所有 Skill

    扫描路径（优先级从高到低）：
    1. {project_root}/.agent/skills/
    2. ~/.agent/skills/

    支持两种格式：
    - 单个目录（包含 SKILL.md + 可选的 scripts/references/assets）

    注意：
    - Project skills 会覆盖同名的 user skills（项目级优先）
    - 跳过 disable_model_invocation=True 的 skills（在 create_skill_tool 中过滤）
    """
    if project_root is None:
        project_root = Path.cwd()

    user_dir = Path.home() / ".agent" / "skills"
    project_dir = project_root / ".agent" / "skills"

    skills: list[SkillDefinition] = []
    seen_names: set[str] = set()

    # 优先加载 project skills（会覆盖同名 user skills）
    for skill_dir in [project_dir, user_dir]:
        if not skill_dir.exists():
            continue

        # 扫描所有子目录
        for item in sorted(skill_dir.iterdir()):
            if not item.is_dir():
                continue

            try:
                # 尝试加载 SKILL.md
                skill = SkillDefinition.from_directory(item)

                # 跳过重复名称（project 优先）
                if skill.name in seen_names:
                    logger.info(f"Skipping duplicate skill '{skill.name}' from {item}")
                    continue

                # 注意：不在这里跳过 disable_model_invocation 的 skills
                # 因为我们需要先加载所有 skills，然后在 create_skill_tool() 中过滤
                # 这样用户可以在代码中显式启用它们

                skills.append(skill)
                seen_names.add(skill.name)
                logger.info(f"Loaded skill '{skill.name}' from {item}")

            except Exception as e:
                logger.warning(f"Failed to load skill from {item}: {e}")

    return skills
