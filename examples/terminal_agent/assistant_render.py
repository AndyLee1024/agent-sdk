from __future__ import annotations

import re
import unicodedata

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding

from terminal_agent.message_style import print_assistant_gap, print_assistant_prefix_line

_COMPLEX_MARKDOWN_PATTERN = re.compile(
    r"(^\s*[-*]\s+|^\s*\d+\.\s|^#{1,6}\s|```|`|^\s*>\s)",
    re.MULTILINE,
)
_LEADING_NOISE_CHARS = frozenset(
    {
        "\ufeff",  # BOM
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\u2060",  # word joiner
        "\u00ad",  # soft hyphen
    }
)


def _is_noise_char(ch: str) -> bool:
    if not ch:
        return True
    if ch.isspace() or ch in _LEADING_NOISE_CHARS:
        return True
    category = unicodedata.category(ch)
    # C*: control/format classes; M*: combining marks (often invisible without base)
    if category.startswith("C") or category in {"Mn", "Me"}:
        return True
    return False


def _strip_leading_noise(text: str) -> str:
    """Strip leading whitespace and invisible formatting noise."""
    idx = 0
    total = len(text)
    while idx < total:
        ch = text[idx]
        if _is_noise_char(ch):
            idx += 1
            continue
        break
    return text[idx:]


def _has_visible_text(text: str) -> bool:
    for ch in text:
        if not _is_noise_char(ch):
            return True
    return False


class AssistantStreamRenderer:
    """Render assistant output with buffered final display (no live race)."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._chunks: list[str] = []
        self._trim_leading_newlines_pending = True

    def start_turn(self) -> None:
        self._chunks = []
        self._trim_leading_newlines_pending = True

    def append_text(self, text: str) -> None:
        if self._trim_leading_newlines_pending:
            text = _strip_leading_noise(text)
            if not text:
                return
            self._trim_leading_newlines_pending = False
        self._chunks.append(text)

    def _extract_first_visible_line(self, content: str) -> tuple[str, str]:
        lines = content.splitlines()
        if not lines:
            return "", ""

        first = _strip_leading_noise(lines[0])
        idx = 0
        while idx < len(lines) and not _has_visible_text(first):
            idx += 1
            if idx < len(lines):
                first = _strip_leading_noise(lines[idx])

        if not _has_visible_text(first):
            return "", ""

        remainder = "\n".join(lines[idx + 1 :]).lstrip("\r\n")
        return first, remainder

    def finalize_markdown(self) -> None:
        content = _strip_leading_noise("".join(self._chunks))
        if not content:
            return

        first, remainder = self._extract_first_visible_line(content)
        if not first:
            return

        # Use soft wrapping to keep prefix and first visible content on one visual line.
        print_assistant_prefix_line(self._console, first)

        if not remainder:
            print_assistant_gap(self._console)
            return

        if _COMPLEX_MARKDOWN_PATTERN.search(content):
            self._console.print(
                Padding(
                    Markdown(remainder, code_theme="monokai", hyperlinks=True),
                    (0, 2, 0, 2),
                )
            )
            print_assistant_gap(self._console)
            return

        for line in remainder.splitlines():
            self._console.print(line, style="white")
        print_assistant_gap(self._console)
