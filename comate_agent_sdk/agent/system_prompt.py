from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from comate_agent_sdk.agent.prompts import SDK_DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger("comate_agent_sdk.agent.system_prompt")


GENERAL_ROLE = "general"
SOFTWARE_ENGINEERING_ROLE = "software_engineering"
DEFAULT_AGENT_ROLE = GENERAL_ROLE

_ROLE_OPENING_LINES: dict[str, str] = {
    GENERAL_ROLE: (
        "You are Comate CLI, an interactive general AI agent running on a user's computer that helps users as a general AI agent."
    ),
    SOFTWARE_ENGINEERING_ROLE: (
        "You are Comate CLI, an interactive general AI agent running on a user's computer that helps users with software engineering tasks."
    ),
}


def _normalize_role(role: str | None) -> str:
    if role is None:
        return DEFAULT_AGENT_ROLE

    normalized = role.strip().lower()
    if not normalized:
        return DEFAULT_AGENT_ROLE

    if normalized not in _ROLE_OPENING_LINES:
        logger.warning(f"Unknown role '{role}', fallback to '{GENERAL_ROLE}'")
        return GENERAL_ROLE

    return normalized


def _replace_first_non_empty_line(text: str, replacement: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip():
            lines[i] = replacement
            return "\n".join(lines)
    return replacement


def build_default_system_prompt(role: str | None = None) -> str:
    resolved_role = _normalize_role(role)
    opening_line = _ROLE_OPENING_LINES[resolved_role]
    return _replace_first_non_empty_line(SDK_DEFAULT_SYSTEM_PROMPT, opening_line)


@dataclass
class SystemPromptConfig:
    """System prompt 配置

    Attributes:
        content: 系统提示内容
        mode:
            - "override": 完全覆盖 SDK 默认 prompt
            - "append": 追加到 SDK 默认 prompt 之后
    """

    content: str
    mode: Literal["override", "append"] = "override"


# 支持 str（向后兼容，等同于 override）或 SystemPromptConfig
SystemPromptType = str | SystemPromptConfig | None


def resolve_system_prompt(system_prompt: SystemPromptType, *, role: str | None = None) -> str:
    """解析 system_prompt 配置，返回最终的系统提示文本

    逻辑：
    - None → 带 role 的 SDK 默认 prompt
    - str → 完全覆盖（向后兼容）
    - SystemPromptConfig(mode="override") → 完全覆盖
    - SystemPromptConfig(mode="append") → 带 role 的 SDK 默认 + 用户内容
    """
    if system_prompt is None:
        return build_default_system_prompt(role=role)

    if isinstance(system_prompt, str):
        # 向后兼容：str 等同于 override
        return system_prompt

    # SystemPromptConfig
    if system_prompt.mode == "override":
        return system_prompt.content

    default_prompt = build_default_system_prompt(role=role)
    return f"{default_prompt}\n\n{system_prompt.content}"
