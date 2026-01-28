"""Skill 系统提示模板

参考 Subagent 的架构模式（prompts.py），将 skills 列表和使用规则放入 system_prompt，
而不是全部塞进 Tool description，实现关注点分离。
"""

from bu_agent_sdk.skill.models import SkillDefinition

SKILL_STRATEGY_PROMPT = """
## Skill 工具使用指南

你可以使用 Skill 工具来执行预定义的技能。每个 skill 是存储在 `SKILL.md` 文件中的一组本地指令。

### 可用 Skills

{skill_list}

### 使用方式

- Discovery: The list above is the skills available in this session (name + description + file path). Skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, open its `SKILL.md`. Read only enough to follow the workflow.
  2) If `SKILL.md` points to extra folders such as `references/`, load only the specific files needed for the request; don't bulk-load everything.
  3) If `scripts/` exist, prefer running or patching them instead of retyping large code blocks.
  4) If `assets/` or templates exist, reuse them instead of recreating from scratch.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Keep context small: summarize long sections instead of pasting them; only load extra files when needed.
  - Avoid deep reference-chasing: prefer opening only files directly linked from `SKILL.md` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue.
"""


def generate_skill_prompt(skills: list[SkillDefinition]) -> str:
    """生成 Skill 策略提示

    Args:
        skills: Skill 定义列表

    Returns:
        格式化的 Skill 策略提示文本
    """
    # 过滤掉 disable_model_invocation=True 的 skills
    active_skills = [s for s in skills if not s.disable_model_invocation]

    if not active_skills:
        return ""

    # 格式化 skill 列表（包含文件路径以便模型定位源文件）
    skill_list = []
    for skill in active_skills:
        path_str = str(skill.base_dir / "SKILL.md") if skill.base_dir else "N/A"
        skill_list.append(f"- **{skill.name}**: {skill.description} (file: {path_str})")

    skills_str = "\n".join(skill_list)
    return SKILL_STRATEGY_PROMPT.format(skill_list=skills_str)
