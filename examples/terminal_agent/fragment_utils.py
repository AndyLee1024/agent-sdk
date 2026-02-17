from __future__ import annotations

from typing import Iterable


def clip_fragments(
    fragments: list[tuple[str, str]],
    max_len: int,
) -> list[tuple[str, str]]:
    if max_len <= 0:
        return []

    remaining = max_len
    clipped: list[tuple[str, str]] = []
    for style, text in fragments:
        if remaining <= 0:
            break
        if len(text) <= remaining:
            clipped.append((style, text))
            remaining -= len(text)
            continue
        clipped.append((style, text[:remaining]))
        break

    return clipped
