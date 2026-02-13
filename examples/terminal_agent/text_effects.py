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


def lerp_rgb(
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, ratio))
    red = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * clamped)
    green = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * clamped)
    blue = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * clamped)
    return red, green, blue


def sweep_gradient_fragments(content: str, frame: int) -> list[tuple[str, str]]:
    if not content:
        return [("", " ")]

    base_rgb = (96, 124, 156)
    mid_rgb = (118, 195, 225)
    high_rgb = (218, 246, 255)

    total = len(content)
    window = max(4, total // 6)
    cycle = max(total + window * 2, 20)
    center = (frame % cycle) - window

    fragments: list[tuple[str, str]] = []
    for idx, char in enumerate(content):
        distance = abs(idx - center)
        if distance <= window:
            glow = 1.0 - (distance / window)
            if glow >= 0.6:
                red, green, blue = lerp_rgb(mid_rgb, high_rgb, (glow - 0.6) / 0.4)
            else:
                red, green, blue = lerp_rgb(base_rgb, mid_rgb, glow / 0.6)
        else:
            red, green, blue = base_rgb
        fragments.append((f"fg:#{red:02x}{green:02x}{blue:02x} bold", char))
    return fragments

