from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

from comate_agent_sdk.llm.anthropic.chat import ChatAnthropic
from comate_agent_sdk.llm.base import BaseChatModel

if TYPE_CHECKING:
    from comate_agent_sdk.agent.settings import SettingsConfig

logger = logging.getLogger(__name__)

LLMLevel = Literal["LOW", "MID", "HIGH"]
ALL_LEVELS: tuple[LLMLevel, ...] = ("LOW", "MID", "HIGH")

# provider → 其标准 env var 名称映射，用于判断是否让 SDK 自动读 env
_PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}


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
    api_key: str | None = None,
) -> BaseChatModel:
    """根据 provider 创建对应 ChatModel 实例

    api_key 优先级：provider 标准 env var 已设时传 None（让 SDK 自动读），否则使用传入值。
    """
    # 检查 API key 配置
    env_var_name = _PROVIDER_ENV_VARS.get(provider)
    env_api_key = os.getenv(env_var_name) if env_var_name else None

    if not env_api_key and not api_key:
        logger.warning(
            f"⚠️  {provider.upper()} API key 未配置（level={level}）！\n"
            f"  解决方式：\n"
            f"  1. 设置环境变量：export {env_var_name}=your-api-key\n"
            f"  2. 或在 .agent/settings.json 中添加：\n"
            f"     \"llm_levels_api_key\": {{\"{level}\": \"your-api-key\"}}"
        )

    # 若 provider 的标准 env var 已设置，优先让 SDK 自动读取，忽略 settings 传入的 api_key
    effective_api_key = None if env_api_key else api_key

    if provider == "openai":
        from comate_agent_sdk.llm.openai.chat import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            api_key=effective_api_key,
        )
    if provider == "anthropic":
        from comate_agent_sdk.llm.anthropic.chat import ChatAnthropic as _ChatAnthropic

        return _ChatAnthropic(model=model, base_url=base_url, api_key=effective_api_key)
    if provider == "google":
        from comate_agent_sdk.llm.google.chat import ChatGoogle

        http_options = {"base_url": base_url} if base_url else None
        return ChatGoogle(model=model, http_options=http_options, api_key=effective_api_key)
    if provider == "deepseek":
        from comate_agent_sdk.llm.deepseek.chat import ChatDeepSeek

        return ChatDeepSeek(model=model, base_url=base_url, api_key=effective_api_key)
    if provider == "minimax":
        from comate_agent_sdk.llm.minimax.chat import ChatMiniMax

        resolved_url = base_url or os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic")
        return ChatMiniMax(model=model, base_url=resolved_url, api_key=effective_api_key)

    raise ValueError(
        f"不支持的 provider：{provider}（level={level}）。当前支持：openai/anthropic/google/deepseek/minimax"
    )


def _parse_env_value(level: LLMLevel, raw: str) -> BaseChatModel:
    provider, model = _parse_provider_model(f"环境变量 COMATE_AGENT_SDK_LLM_{level}", raw)
    base_url = os.getenv(f"COMATE_AGENT_SDK_LLM_{level}_BASE_URL")
    websocket_base_url = os.getenv(f"COMATE_AGENT_SDK_LLM_{level}_WEBSOCKET_BASE_URL")
    return _build_model(level, provider, model, base_url, websocket_base_url=websocket_base_url)


def _parse_settings_value(level: LLMLevel, raw: str, base_url: str | None, api_key: str | None = None) -> BaseChatModel:
    """解析 settings.json 中的 llm_levels 值"""
    provider, model = _parse_provider_model(f"settings.json llm_levels.{level}", raw)
    return _build_model(level, provider, model, base_url, api_key=api_key)


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
            api_key = (settings.llm_levels_api_key or {}).get(level)
            base[level] = _parse_settings_value(level, raw, base_url, api_key)
        return base

    # 最低优先级：环境变量
    for level in ALL_LEVELS:
        env_key = f"COMATE_AGENT_SDK_LLM_{level}"
        raw = os.getenv(env_key)
        if raw is None:
            continue
        base[level] = _parse_env_value(level, raw)

    return base
