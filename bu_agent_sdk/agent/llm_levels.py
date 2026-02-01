from __future__ import annotations

import logging
import os
from typing import Literal

from bu_agent_sdk.llm.anthropic.chat import ChatAnthropic
from bu_agent_sdk.llm.base import BaseChatModel

logger = logging.getLogger(__name__)

LLMLevel = Literal["LOW", "MID", "HIGH"]
ALL_LEVELS: tuple[LLMLevel, ...] = ("LOW", "MID", "HIGH")


def _default_levels() -> dict[LLMLevel, BaseChatModel]:
    # 默认按 Anthropic 三档（可通过 env 覆盖）
    return {
        "LOW": ChatAnthropic(model="claude-haiku-4-5"),
        "MID": ChatAnthropic(model="claude-sonnet-4-5"),
        "HIGH": ChatAnthropic(model="claude-opus-4-5"),
    }


def _parse_env_value(level: LLMLevel, raw: str) -> BaseChatModel:
    value = raw.strip()
    if not value:
        raise ValueError(f"环境变量 BU_AGENT_SDK_LLM_{level} 不能为空")

    if ":" not in value:
        raise ValueError(
            f"环境变量 BU_AGENT_SDK_LLM_{level} 必须为 'provider:model' 格式，例如 'openai:gpt-5-mini'，实际为：{value}"
        )

    provider, model = value.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise ValueError(
            f"环境变量 BU_AGENT_SDK_LLM_{level} 必须为 'provider:model' 格式，例如 'openai:gpt-5-mini'，实际为：{value}"
        )

    base_url = os.getenv(f"BU_AGENT_SDK_LLM_{level}_BASE_URL")

    if provider == "openai":
        from bu_agent_sdk.llm.openai.chat import ChatOpenAI

        websocket_base_url = os.getenv(f"BU_AGENT_SDK_LLM_{level}_WEBSOCKET_BASE_URL")
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
        )
    if provider == "anthropic":
        from bu_agent_sdk.llm.anthropic.chat import ChatAnthropic as _ChatAnthropic

        return _ChatAnthropic(model=model, base_url=base_url)
    if provider == "google":
        from bu_agent_sdk.llm.google.chat import ChatGoogle

        http_options = {"base_url": base_url} if base_url else None
        return ChatGoogle(model=model, http_options=http_options)

    raise ValueError(
        f"不支持的 provider：{provider}（来自 BU_AGENT_SDK_LLM_{level}）。当前支持：openai/anthropic/google"
    )


def resolve_llm_levels(
    *,
    explicit: dict[str, BaseChatModel] | None,
) -> dict[LLMLevel, BaseChatModel]:
    """解析三档 LLM。

    优先级（已与你确认）：
    1) 若 explicit 非 None：使用 explicit 覆盖默认（不会读取 env）
    2) 若 explicit 为 None：读取 env 覆盖默认
    """
    base = _default_levels()

    if explicit is not None:
        # 允许 partial 覆盖，缺失项使用默认
        merged: dict[LLMLevel, BaseChatModel] = dict(base)
        for k, v in explicit.items():
            if k in ALL_LEVELS:
                merged[k] = v
        return merged

    # env override
    for level in ALL_LEVELS:
        env_key = f"BU_AGENT_SDK_LLM_{level}"
        raw = os.getenv(env_key)
        if raw is None:
            continue
        base[level] = _parse_env_value(level, raw)

    return base
