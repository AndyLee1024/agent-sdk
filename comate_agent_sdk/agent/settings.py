"""Settings 配置文件加载

从 ~/.agent/settings.json（user级）和 {project_root}/.agent/settings.json（project级）
加载 LLM 配置，支持分层覆盖。

优先级（从高到低）：
    代码参数 llm_levels= > project/.agent/settings.json > ~/.agent/settings.json > 环境变量 > 默认值
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger("comate_agent_sdk.agent.settings")

SettingSource = Literal["user", "project"]
DEFAULT_SETTING_SOURCES: tuple[SettingSource, ...] = ("user", "project")

# 用户级配置路径
USER_SETTINGS_PATH: Path = Path.home() / ".agent" / "settings.json"
USER_AGENTS_MD_PATH: Path = Path.home() / ".agent" / "AGENTS.md"


@dataclass
class SettingsConfig:
    """从 settings.json 加载的配置

    Attributes:
        llm_levels: LLM 三档配置，格式为 {"LOW": "provider:model", ...}
        llm_levels_base_url: 各档 base_url 覆盖，值为 None 表示不覆盖
        llm_levels_api_key: 各档 api_key 覆盖，值为 None 表示不覆盖（优先级低于 env var）
    """

    llm_levels: dict[str, str] | None = None
    llm_levels_base_url: dict[str, str | None] | None = None
    llm_levels_api_key: dict[str, str | None] | None = None


def load_settings_file(path: Path) -> SettingsConfig | None:
    """从 settings.json 文件加载配置

    文件不存在或解析失败时返回 None，不抛异常。

    Args:
        path: settings.json 的绝对路径

    Returns:
        解析后的 SettingsConfig，或 None
    """
    resolved = path.expanduser()
    if not resolved.exists():
        logger.debug(f"settings.json 不存在，跳过: {resolved}")
        return None

    if not resolved.is_file():
        logger.warning(f"settings.json 路径不是文件: {resolved}")
        return None

    try:
        raw = resolved.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"读取/解析 settings.json 失败 {resolved}: {e}")
        return None

    if not isinstance(data, dict):
        logger.warning(f"settings.json 内容不是 JSON 对象: {resolved}")
        return None

    llm_levels: dict[str, str] | None = None
    llm_levels_base_url: dict[str, str | None] | None = None
    llm_levels_api_key: dict[str, str | None] | None = None

    # 解析 llm_levels
    raw_levels = data.get("llm_levels")
    if raw_levels is not None:
        if not isinstance(raw_levels, dict):
            logger.warning(f"settings.json llm_levels 不是对象，跳过: {resolved}")
        else:
            llm_levels = {}
            for k, v in raw_levels.items():
                if not isinstance(v, str):
                    logger.warning(f"settings.json llm_levels.{k} 不是字符串，跳过该项")
                    continue
                llm_levels[k] = v

    # 解析 llm_levels_base_url
    raw_base_urls = data.get("llm_levels_base_url")
    if raw_base_urls is not None:
        if not isinstance(raw_base_urls, dict):
            logger.warning(f"settings.json llm_levels_base_url 不是对象，跳过: {resolved}")
        else:
            llm_levels_base_url = {}
            for k, v in raw_base_urls.items():
                if v is not None and not isinstance(v, str):
                    logger.warning(f"settings.json llm_levels_base_url.{k} 不是字符串或 null，跳过该项")
                    continue
                llm_levels_base_url[k] = v

    # 解析 llm_levels_api_key
    raw_api_keys = data.get("llm_levels_api_key")
    if raw_api_keys is not None:
        if not isinstance(raw_api_keys, dict):
            logger.warning(f"settings.json llm_levels_api_key 不是对象，跳过: {resolved}")
        else:
            llm_levels_api_key = {}
            for k, v in raw_api_keys.items():
                if v is not None and not isinstance(v, str):
                    logger.warning(f"settings.json llm_levels_api_key.{k} 不是字符串或 null，跳过该项")
                    continue
                llm_levels_api_key[k] = v

    if llm_levels is None and llm_levels_base_url is None and llm_levels_api_key is None:
        return None

    return SettingsConfig(llm_levels=llm_levels, llm_levels_base_url=llm_levels_base_url, llm_levels_api_key=llm_levels_api_key)


def resolve_settings(
    sources: tuple[SettingSource, ...] | None,
    project_root: Path | None,
) -> SettingsConfig | None:
    """按优先级加载并合并 settings

    合并策略：project 完全覆盖 user（project 定义了 llm_levels 就忽略 user 的 llm_levels）。

    Args:
        sources: 要加载的配置源，None 或空 tuple 表示不加载任何配置
        project_root: 项目根目录，None 时使用 cwd

    Returns:
        合并后的 SettingsConfig，或 None（无有效配置）
    """
    if not sources:
        logger.debug("setting_sources 为空，不加载 settings")
        return None

    root = (project_root or Path.cwd()).expanduser().resolve()

    user_settings: SettingsConfig | None = None
    project_settings: SettingsConfig | None = None

    if "user" in sources:
        user_settings = load_settings_file(USER_SETTINGS_PATH)
        if user_settings:
            logger.debug(f"加载 user settings: {USER_SETTINGS_PATH}")

    if "project" in sources:
        project_path = root / ".agent" / "settings.json"
        project_settings = load_settings_file(project_path)
        if project_settings:
            logger.debug(f"加载 project settings: {project_path}")

    # 合并：project 完全覆盖 user
    if project_settings is None:
        result = user_settings  # 可能也是 None
    elif user_settings is None:
        result = project_settings
    else:
        # 两者都有：project 字段非 None 时完全覆盖 user 对应字段
        result = SettingsConfig(
            llm_levels=project_settings.llm_levels if project_settings.llm_levels is not None else user_settings.llm_levels,
            llm_levels_base_url=project_settings.llm_levels_base_url if project_settings.llm_levels_base_url is not None else user_settings.llm_levels_base_url,
            llm_levels_api_key=project_settings.llm_levels_api_key if project_settings.llm_levels_api_key is not None else user_settings.llm_levels_api_key,
        )

    # 输出配置加载日志
    if result is not None:
        logger.info("✅ settings.json 已加载")
        if result.llm_levels:
            logger.info(f"  - llm_levels: {list(result.llm_levels.keys())}")
        if result.llm_levels_base_url:
            logger.info(f"  - llm_levels_base_url: 已配置 {len(result.llm_levels_base_url)} 个档位")
        if result.llm_levels_api_key:
            logger.info(f"  - llm_levels_api_key: 已配置 {len(result.llm_levels_api_key)} 个档位")
        else:
            logger.debug("  ℹ️  llm_levels_api_key 未配置，将使用环境变量")

    return result


def discover_agents_md(project_root: Path | None) -> list[Path]:
    """发现项目中的 AGENTS.md 文件

    搜索位置（按此顺序）：
    - {project_root}/AGENTS.md
    - {project_root}/.agent/AGENTS.md

    Args:
        project_root: 项目根目录，None 时使用 cwd

    Returns:
        存在的 AGENTS.md 文件路径列表（按上述顺序去重）
    """
    # 注意：不要在这里调用 resolve()，避免在 macOS 上把 /var/... 解析成 /private/var/...，
    # 导致调用方（尤其是测试）拿到的路径与输入不一致。
    root = (project_root or Path.cwd()).expanduser()
    candidates = [
        root / "AGENTS.md",
        root / ".agent" / "AGENTS.md",
    ]

    found: list[Path] = []
    for p in candidates:
        if p.is_file():
            found.append(p)
            logger.debug(f"发现 AGENTS.md: {p}")

    if not found:
        logger.debug(f"未在 {root} 中发现 AGENTS.md")

    return found


def discover_user_agents_md() -> list[Path]:
    """发现 user 级别的 AGENTS.md（~/.agent/AGENTS.md）

    Returns:
        包含 ~/.agent/AGENTS.md 的单元素列表（如果存在），否则空列表
    """
    if USER_AGENTS_MD_PATH.is_file():
        logger.debug(f"发现 user AGENTS.md: {USER_AGENTS_MD_PATH}")
        return [USER_AGENTS_MD_PATH]
    return []
