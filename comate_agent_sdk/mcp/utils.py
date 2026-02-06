from __future__ import annotations

import re

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def sanitize_tool_name(value: str, *, max_len: int = 80) -> str:
    """将任意字符串规范化为 LLM tool name 友好的形式。

    约束：
    - 只保留 [A-Za-z0-9_]
    - 其他字符替换为 '_'
    - 连续 '_' 压缩
    - 截断到 max_len
    """
    if not value:
        return "unknown"

    cleaned = _SAFE_NAME_RE.sub("_", value.strip())
    cleaned = _MULTI_UNDERSCORE_RE.sub("_", cleaned).strip("_")
    if not cleaned:
        cleaned = "unknown"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("_")
    return cleaned or "unknown"

