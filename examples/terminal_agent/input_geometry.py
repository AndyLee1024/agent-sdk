from __future__ import annotations

from prompt_toolkit.utils import get_cwidth


def compute_visual_line_ranges(text: str, max_cols: int) -> list[tuple[int, int]]:
    """
    Compute visual line ranges [start, end) by wrapping `text` at `max_cols`
    display cells, respecting explicit newlines.
    """
    if max_cols <= 0:
        max_cols = 1

    ranges: list[tuple[int, int]] = []
    start = 0
    col = 0
    for idx, ch in enumerate(text):
        if ch == "\n":
            ranges.append((start, idx))
            start = idx + 1
            col = 0
            continue

        ch_width = get_cwidth(ch)
        if ch_width <= 0:
            ch_width = 1
        if ch_width > max_cols:
            ch_width = max_cols

        if col + ch_width > max_cols:
            ranges.append((start, idx))
            start = idx
            col = 0

        col += ch_width

    ranges.append((start, len(text)))
    return ranges


def visual_col_for_index(
    text: str,
    start: int,
    end: int,
    max_cols: int,
    index: int,
) -> int:
    if max_cols <= 0:
        max_cols = 1
    if index < start:
        return 0
    if index > end:
        index = end
    col = 0
    for ch in text[start:index]:
        ch_width = get_cwidth(ch)
        if ch_width <= 0:
            ch_width = 1
        if ch_width > max_cols:
            ch_width = max_cols
        if col + ch_width > max_cols:
            break
        col += ch_width
    return col


def index_for_visual_col(
    text: str,
    start: int,
    end: int,
    max_cols: int,
    target_col: int,
) -> int:
    if max_cols <= 0:
        max_cols = 1
    if target_col <= 0:
        return start
    if target_col > max_cols:
        target_col = max_cols
    col = 0
    for idx in range(start, end):
        ch = text[idx]
        ch_width = get_cwidth(ch)
        if ch_width <= 0:
            ch_width = 1
        if ch_width > max_cols:
            ch_width = max_cols
        if col + ch_width > target_col:
            return idx
        col += ch_width
    return end

