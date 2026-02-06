"""Skill execution context（已弃用）

历史上 Skill 支持通过 frontmatter（例如 allowed-tools）修改 Agent 的运行时环境。
该能力已移除：Skill 只负责提供一组可被模型加载的本地指令，不再对工具权限/模型配置产生副作用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.agent.service import Agent
    from comate_agent_sdk.skill.models import SkillDefinition


def apply_skill_context(agent: "Agent", skill_def: "SkillDefinition") -> None:
    """应用 Skill 的 execution context 修改（已弃用，当前为 no-op）。

    保留该函数仅用于向后兼容外部 import。
    """
    _ = agent
    _ = skill_def
    return

