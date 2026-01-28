"""
Subagent 自动发现和加载
"""

import logging
from pathlib import Path

from bu_agent_sdk.subagent.models import AgentDefinition


def discover_subagents(
    project_root: Path | None = None,
) -> list[AgentDefinition]:
    """自动发现并加载所有 Subagent 定义

    优先级规则：
    - user 和 project 都存在 → 只用 project（完全忽略 user）
    - 只有 user 存在 → 用 user
    - 只有 project 存在 → 用 project

    Args:
        project_root: 项目根目录，默认使用 cwd

    Returns:
        发现的 AgentDefinition 列表
    """
    if project_root is None:
        project_root = Path.cwd()

    user_dir = Path.home() / ".agent" / "subagents"
    project_dir = project_root / ".agent" / "subagents"

    # 检查是否存在 .md 文件（不只是目录）
    user_exists = user_dir.exists() and any(user_dir.glob("*.md"))
    project_exists = project_dir.exists() and any(project_dir.glob("*.md"))

    # 优先级决策：都存在时只用 project
    if project_exists:
        target_dir = project_dir
        source = "project"
    elif user_exists:
        target_dir = user_dir
        source = "user"
    else:
        return []  # 都不存在

    agents: list[AgentDefinition] = []
    for md_file in sorted(target_dir.glob("*.md")):
        try:
            agent = AgentDefinition.from_file(md_file)
            agents.append(agent)
            logging.info(f"Loaded {source} subagent: {agent.name} from {md_file}")
        except Exception as e:
            logging.warning(f"Failed to load {md_file}: {e}")

    return agents
