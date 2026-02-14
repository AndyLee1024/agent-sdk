from __future__ import annotations

from prompt_toolkit.utils import get_cwidth

_ELLIPSIS = "…"


def fit_single_line(content: str, width: int) -> str:
    """Truncate string by terminal display width, handling wide chars correctly."""
    max_width = max(width, 8)
    if get_cwidth(content) <= max_width:
        return content
    if max_width <= 1:
        return _ELLIPSIS

    result: list[str] = []
    used_width = 0
    ellipsis_width = get_cwidth(_ELLIPSIS)
    target_width = max_width - ellipsis_width

    for char in content:
        char_width = get_cwidth(char)
        if used_width + char_width > target_width:
            break
        result.append(char)
        used_width += char_width

    return "".join(result) + _ELLIPSIS


