from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Mapping

from comate_agent_sdk.agent.hooks.models import (
    HookConfig,
    HookHandlerSpec,
    HookMatcherGroup,
    normalize_hook_event_name,
    resolve_project_root,
)

logger = logging.getLogger("comate_agent_sdk.agent.hooks")

HookSettingSource = Literal["user", "project", "local"]

USER_SETTINGS_PATH = Path.home() / ".agent" / "settings.json"


def _read_json_file(path: Path) -> dict[str, Any] | None:
    resolved = path.expanduser()
    if not resolved.exists() or not resolved.is_file():
        return None
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"读取 hooks settings 失败 {resolved}: {exc}")
        return None
    if not isinstance(data, dict):
        logger.warning(f"hooks settings 根节点不是对象，跳过: {resolved}")
        return None
    return data


def _normalize_timeout(raw_timeout: Any) -> int:
    if isinstance(raw_timeout, int) and raw_timeout > 0:
        return raw_timeout
    return 10


def load_hook_config_from_settings_dict(
    settings: Mapping[str, Any] | None,
    *,
    source: str = "",
) -> HookConfig:
    config = HookConfig()
    if not settings:
        return config

    raw_hooks = settings.get("hooks")
    if raw_hooks is None:
        return config
    if not isinstance(raw_hooks, dict):
        logger.warning(f"hooks 不是对象，跳过: {source}")
        return config

    for raw_event_name, raw_groups in raw_hooks.items():
        if not isinstance(raw_event_name, str):
            continue

        event_name = normalize_hook_event_name(raw_event_name)
        if event_name is None:
            logger.warning(f"未知 hooks 事件名 '{raw_event_name}'，跳过: {source}")
            continue

        if not isinstance(raw_groups, list):
            logger.warning(f"hooks.{raw_event_name} 不是数组，跳过: {source}")
            continue

        parsed_groups: list[HookMatcherGroup] = []
        for idx, raw_group in enumerate(raw_groups):
            if not isinstance(raw_group, dict):
                logger.warning(f"hooks.{raw_event_name}[{idx}] 不是对象，跳过: {source}")
                continue

            raw_matcher = raw_group.get("matcher", "*")
            matcher = raw_matcher if isinstance(raw_matcher, str) else "*"
            if matcher == "":
                matcher = "*"

            raw_handlers = raw_group.get("hooks")
            if not isinstance(raw_handlers, list):
                logger.warning(f"hooks.{raw_event_name}[{idx}].hooks 不是数组，跳过: {source}")
                continue

            handlers: list[HookHandlerSpec] = []
            for h_idx, raw_handler in enumerate(raw_handlers):
                if not isinstance(raw_handler, dict):
                    logger.warning(
                        f"hooks.{raw_event_name}[{idx}].hooks[{h_idx}] 不是对象，跳过: {source}"
                    )
                    continue

                raw_type = raw_handler.get("type")
                if raw_type != "command":
                    if raw_type == "python":
                        logger.warning(
                            f"python hooks 不支持从 settings 加载，跳过 hooks.{raw_event_name}[{idx}].hooks[{h_idx}]"
                        )
                    else:
                        logger.warning(
                            f"未知 hook type={raw_type!r}，跳过 hooks.{raw_event_name}[{idx}].hooks[{h_idx}]"
                        )
                    continue

                command = raw_handler.get("command")
                if not isinstance(command, str) or not command.strip():
                    logger.warning(
                        f"command hook 缺少有效 command，跳过 hooks.{raw_event_name}[{idx}].hooks[{h_idx}]"
                    )
                    continue

                handlers.append(
                    HookHandlerSpec(
                        type="command",
                        command=command,
                        timeout=_normalize_timeout(raw_handler.get("timeout")),
                        source=source,
                        name=f"{event_name}[{idx}].command[{h_idx}]",
                    )
                )

            if handlers:
                parsed_groups.append(
                    HookMatcherGroup(
                        matcher=matcher,
                        hooks=handlers,
                        source=source,
                    )
                )

        if parsed_groups:
            config.events.setdefault(event_name, []).extend(parsed_groups)

    return config


def merge_hook_configs(*configs: HookConfig) -> HookConfig:
    merged = HookConfig()
    for cfg in configs:
        merged.extend(cfg)
    return merged


def load_hook_config_from_sources(
    *,
    project_root: Path | None,
    sources: tuple[HookSettingSource, ...] | None,
) -> HookConfig:
    if not sources:
        return HookConfig()

    root = resolve_project_root(project_root)
    layer_paths: list[Path] = []

    if "user" in sources:
        layer_paths.append(USER_SETTINGS_PATH)
    if "project" in sources:
        layer_paths.append(root / ".agent" / "settings.json")
    if "local" in sources:
        layer_paths.append(root / ".agent" / "settings.local.json")

    configs: list[HookConfig] = []
    for path in layer_paths:
        data = _read_json_file(path)
        if data is None:
            continue
        cfg = load_hook_config_from_settings_dict(data, source=str(path))
        configs.append(cfg)

    return merge_hook_configs(*configs)
