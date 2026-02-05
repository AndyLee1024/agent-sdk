from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bu_agent_sdk.mcp.types import McpServerConfig, McpServersInput

logger = logging.getLogger("bu_agent_sdk.mcp.config")


def _default_user_mcp_path() -> Path:
    return Path.home() / ".agent" / ".mcp.json"


def _default_project_mcp_path(project_root: Path | None) -> Path:
    root = (project_root or Path.cwd()).expanduser().resolve()
    return root / ".agent" / ".mcp.json"


def _normalize_servers_payload(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None

    # 支持 {"servers": {...}} 和直接 {...}
    if "servers" in data and isinstance(data.get("servers"), dict):
        payload = data.get("servers")
        return payload if isinstance(payload, dict) else None

    return data


def load_mcp_servers_from_path(path: Path) -> dict[str, McpServerConfig]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {}
    if not resolved.is_file():
        logger.warning(f".mcp.json 路径不是文件：{resolved}")
        return {}

    try:
        raw = resolved.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:
        logger.warning(f"读取/解析 .mcp.json 失败：{resolved}：{e}")
        return {}

    payload = _normalize_servers_payload(data)
    if payload is None:
        logger.warning(f".mcp.json 内容不是对象：{resolved}")
        return {}

    servers: dict[str, McpServerConfig] = {}
    for alias, cfg in payload.items():
        if not isinstance(alias, str) or not alias.strip():
            continue
        if not isinstance(cfg, dict):
            logger.warning(f".mcp.json server 配置不是对象，跳过：{alias}")
            continue

        # sdk server 不能从文件中反序列化（instance 无法表达）
        if cfg.get("type") == "sdk":
            logger.warning(f".mcp.json 不支持 type='sdk'（只能代码注入），跳过：{alias}")
            continue

        servers[alias] = cfg  # type: ignore[assignment]

    return servers


def resolve_mcp_servers(
    mcp_servers: McpServersInput,
    *,
    project_root: Path | None,
) -> dict[str, McpServerConfig]:
    """解析 Agent.mcp_servers 输入为标准 dict[alias, config]。"""
    if mcp_servers is None:
        user_path = _default_user_mcp_path()
        project_path = _default_project_mcp_path(project_root)

        merged = load_mcp_servers_from_path(user_path)
        merged.update(load_mcp_servers_from_path(project_path))

        return merged

    if isinstance(mcp_servers, (str, Path)):
        return load_mcp_servers_from_path(Path(mcp_servers))

    if isinstance(mcp_servers, dict):
        # 显式 {} 表示禁用/不配置任何 server
        return mcp_servers

    logger.warning(f"不支持的 mcp_servers 类型：{type(mcp_servers).__name__}")
    return {}

