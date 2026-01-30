"""Memory 配置和文件加载

Memory 是静态背景知识，从指定文件加载后注入到 SystemMessage 中。
与 CLAUDE.md 类似，但作为独立的 header 段管理。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("bu_agent_sdk.context.memory")


@dataclass
class MemoryConfig:
    """Memory 配置

    Attributes:
        files: 文件路径列表（支持 str 或 Path）
        max_tokens: token 上限，超出时 warning（不截断）
        cache: 是否建议 prompt cache
    """

    files: list[str | Path]
    max_tokens: int = 2000
    cache: bool = True


def load_memory_content(config: MemoryConfig) -> str | None:
    """加载 Memory 内容

    读取所有文件，用 \\n\\n 拼接为单个字符串。
    文件不存在或为空时记录 warning，但不报错。

    Args:
        config: MemoryConfig 配置

    Returns:
        拼接后的 memory 内容，如果所有文件都不存在或为空则返回 None
    """
    parts: list[str] = []

    for file_path in config.files:
        path = Path(file_path).expanduser()

        if not path.exists():
            logger.warning(f"Memory 文件不存在: {path}")
            continue

        if not path.is_file():
            logger.warning(f"Memory 路径不是文件: {path}")
            continue

        try:
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                logger.warning(f"Memory 文件为空: {path}")
                continue
            parts.append(content.strip())
        except Exception as e:
            logger.warning(f"读取 Memory 文件失败 {path}: {e}")
            continue

    if not parts:
        return None

    return "\n\n".join(parts)
