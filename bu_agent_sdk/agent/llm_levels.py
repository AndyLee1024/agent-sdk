from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

from bu_agent_sdk.llm.anthropic.chat import ChatAnthropic
from bu_agent_sdk.llm.base import BaseChatModel

if TYPE_CHECKING:
    from bu_agent_sdk.agent.settings import SettingsConfig

logger = logging.getLogger(__name__)

LLMLevel = Literal["LOW", "MID", "HIGH"]
ALL_LEVELS: tuple[LLMLevel, ...] = ("LOW", "MID", "HIGH")


def _default_levels() -> dict[LLMLevel, BaseChatModel]:
    # 默认按 Anthropic 三档（可通过 env / settings 覆盖）
    return {
        "LOW": ChatAnthropic(model="claude-haiku-4-5"),
        "MID": ChatAnthropic(model="claude-sonnet-4-5"),
        "HIGH": ChatAnthropic(model="claude-opus-4-5"),
    }


def _parse_provider_model(source_label: str, raw: str) -> tuple[str, str]:
    """解析 'provider:model' 格式字符串，返回 (provider, model)"""
    value = raw.strip()
    if not value:
        raise ValueError(f"{source_label} 不能为空")

    if ":" not in value:
        raise ValueError(
            f"{source_label} 必须为 'provider:model' 格式，例如 'openai:gpt-5-mini'，实际为：{value}"
        )

    provider, model = value.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise ValueError(
            f"{source_label} 必须为 'provider:model' 格式，例如 'openai:gpt-5-mini'，实际为：{value}"
        )

    return provider, model


def _build_model(
    level: LLMLevel,
    provider: str,
    model: str,
    base_url: str | None,
    *,
    websocket_base_url: str | None = None,
) -> BaseChatModel:
    """根据 provider 创建对应 ChatModel 实例"""
    if provider == "openai":
        from bu_agent_sdk.llm.openai.chat import ChatOpenAI

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
        f"不支持的 provider：{provider}（level={level}）。当前支持：openai/anthropic/google"
    )


def _parse_env_value(level: LLMLevel, raw: str) -> BaseChatModel:
    provider, model = _parse_provider_model(f"环境变量 BU_AGENT_SDK_LLM_{level}", raw)
    base_url = os.getenv(f"BU_AGENT_SDK_LLM_{level}_BASE_URL")
    websocket_base_url = os.getenv(f"BU_AGENT_SDK_LLM_{level}_WEBSOCKET_BASE_URL")
    return _build_model(level, provider, model, base_url, websocket_base_url=websocket_base_url)


def _parse_settings_value(level: LLMLevel, raw: str, base_url: str | None) -> BaseChatModel:
    """解析 settings.json 中的 llm_levels 值"""
    provider, model = _parse_provider_model(f"settings.json llm_levels.{level}", raw)
    return _build_model(level, provider, model, base_url)


def resolve_llm_levels(
    *,
    explicit: dict[str, BaseChatModel] | None,
    settings: "SettingsConfig | None" = None,
) -> dict[LLMLevel, BaseChatModel]:
    """解析三档 LLM。

    优先级：
    1) explicit 非 None：使用 explicit 覆盖默认（不读取 settings/env）
    2) settings 非 None 且有 llm_levels：使用 settings 覆盖默认（不读取 env）
    3) 读取 env 覆盖默认
    """
    base = _default_levels()

    if explicit is not None:
        # 最高优先级：代码显式传入（允许 partial 覆盖）
        merged: dict[LLMLevel, BaseChatModel] = dict(base)
        for k, v in explicit.items():
            if k in ALL_LEVELS:
                merged[k] = v
        return merged

    if settings is not None and settings.llm_levels:
        # 第二优先级：settings.json
        for level in ALL_LEVELS:
            raw = settings.llm_levels.get(level)
            if raw is None:
                continue
            base_url = (settings.llm_levels_base_url or {}).get(level)
            base[level] = _parse_settings_value(level, raw, base_url)
        return base

    # 最低优先级：环境变量
    for level in ALL_LEVELS:
        env_key = f"BU_AGENT_SDK_LLM_{level}"
        raw = os.getenv(env_key)
        if raw is None:
            continue
        base[level] = _parse_env_value(level, raw)

    return base
