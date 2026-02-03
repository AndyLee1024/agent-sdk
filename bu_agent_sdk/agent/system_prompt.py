from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bu_agent_sdk.agent.prompts import SDK_DEFAULT_SYSTEM_PROMPT


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


def resolve_system_prompt(system_prompt: SystemPromptType) -> str:
    """解析 system_prompt 配置，返回最终的系统提示文本

    逻辑：
    - None → SDK 默认 prompt
    - str → 完全覆盖（向后兼容）
    - SystemPromptConfig(mode="override") → 完全覆盖
    - SystemPromptConfig(mode="append") → SDK 默认 + 用户内容
    """
    if system_prompt is None:
        return SDK_DEFAULT_SYSTEM_PROMPT

    if isinstance(system_prompt, str):
        # 向后兼容：str 等同于 override
        return system_prompt

    # SystemPromptConfig
    if system_prompt.mode == "override":
        return system_prompt.content

    return f"{SDK_DEFAULT_SYSTEM_PROMPT}\n\n{system_prompt.content}"

