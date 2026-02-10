from __future__ import annotations

import re
import unicodedata
from typing import Literal

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text

from terminal_agent.message_style import ASSISTANT_PREFIX, ASSISTANT_PREFIX_STYLE

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
_BLOCK_MARKDOWN_PREFIX_RE = re.compile(
    r"^\s*(#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s+|```|~~~)",
    re.MULTILINE,
)
_AMBIGUOUS_BLOCK_PREFIX_RE = re.compile(r"^\s*(#{1,6}|[-*+]|>|\d+\.|`{1,2}|~{1,2})$")
_SegmentMode = Literal["prefixed_plain_first_line", "markdown_only"]


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


def _first_visible_line(content: str) -> str:
    lines = content.splitlines()
    if not lines:
        return ""
    for line in lines:
        candidate = _strip_leading_noise(line)
        if candidate:
            return candidate
    return ""


def _split_first_line(content: str) -> tuple[str, str]:
    if not content:
        return "", ""
    if "\n" not in content:
        return content, ""
    first_line, remainder = content.split("\n", 1)
    return first_line.rstrip("\r"), remainder


def _is_block_markdown_start(first_line: str) -> bool:
    if not first_line:
        return False
    return _BLOCK_MARKDOWN_PREFIX_RE.match(first_line) is not None


def _is_ambiguous_block_start(first_line: str) -> bool:
    if not first_line:
        return False
    return _AMBIGUOUS_BLOCK_PREFIX_RE.match(first_line) is not None


class AssistantStreamRenderer:
    """Turn-scoped assistant renderer that exposes renderables for a parent Live layout."""

    def __init__(self) -> None:
        self._trim_segment_leading_noise_pending = True
        self._segment_chunks: list[str] = []
        self._segment_mode: _SegmentMode | None = None
        self._blocks: list[RenderableType] = []
        self._last_block_external = False
        self._pending_gap_before_next_segment = False

    def _reset_segment_state(self) -> None:
        self._trim_segment_leading_noise_pending = True
        self._segment_chunks = []
        self._segment_mode = None

    def start_turn(self) -> None:
        self._blocks = []
        self._last_block_external = False
        self._pending_gap_before_next_segment = False
        self._reset_segment_state()

    def _resolve_segment_mode(self, content: str) -> _SegmentMode | None:
        first_line = _first_visible_line(content)
        if not first_line:
            return None
        if _is_ambiguous_block_start(first_line):
            return None
        if _is_block_markdown_start(first_line):
            return "markdown_only"
        return "prefixed_plain_first_line"

    @staticmethod
    def _build_segment_renderable(mode: _SegmentMode, content: str) -> RenderableType:
        if mode == "markdown_only":
            return Markdown(content, code_theme="monokai", hyperlinks=True)

        first_line, remainder = _split_first_line(content)
        prefixed = Text()
        prefixed.append(f"{ASSISTANT_PREFIX} ", style=ASSISTANT_PREFIX_STYLE)
        prefixed.append(first_line)

        if not remainder:
            return prefixed
        return Group(
            prefixed,
            Markdown(remainder, code_theme="monokai", hyperlinks=True),
        )

    @staticmethod
    def _is_gap_block(renderable: RenderableType) -> bool:
        return isinstance(renderable, Text) and not renderable.plain

    def _append_gap_if_needed(self) -> None:
        if self._blocks and not self._is_gap_block(self._blocks[-1]):
            self._blocks.append(Text(""))

    def _render_unresolved_segment_if_needed(self) -> None:
        if not self._segment_chunks or self._segment_mode is not None:
            return
        # Segment ended before ambiguous markdown prefix was disambiguated.
        # Fall back to visible plain-text prefixed rendering to avoid content loss.
        self._segment_mode = "prefixed_plain_first_line"

    def _flush_active_segment(self, *, inter_block_gap: bool) -> bool:
        self._render_unresolved_segment_if_needed()
        content = "".join(self._segment_chunks)
        mode = self._segment_mode
        self._reset_segment_state()
        if not content or mode is None:
            return False

        self._blocks.append(self._build_segment_renderable(mode, content))
        self._last_block_external = False
        if inter_block_gap:
            self._append_gap_if_needed()
        return True

    def append_external_lines(self, lines: list[tuple[str, str]]) -> None:
        if not lines:
            return
        self._flush_active_segment(inter_block_gap=True)
        if not self._last_block_external:
            self._append_gap_if_needed()
        for content, style in lines:
            self._blocks.append(Text(content, style=style))
        self._last_block_external = True

    def append_text(self, text: str) -> None:
        if self._trim_segment_leading_noise_pending:
            text = _strip_leading_noise(text)
            if not text:
                return
            self._trim_segment_leading_noise_pending = False
        if not text:
            return

        if not self._segment_chunks and self._pending_gap_before_next_segment:
            self._append_gap_if_needed()
            self._pending_gap_before_next_segment = False

        self._segment_chunks.append(text)
        content = "".join(self._segment_chunks)
        if self._segment_mode is None:
            self._segment_mode = self._resolve_segment_mode(content)
        self._last_block_external = False

    def insert_gap_before_next_segment(self) -> None:
        if self._segment_chunks:
            return
        if not self._blocks:
            return
        self._pending_gap_before_next_segment = True

    def flush_line_for_external_event(self) -> None:
        self._flush_active_segment(inter_block_gap=True)

    def finalize_turn(self) -> None:
        self._flush_active_segment(inter_block_gap=False)
        self._pending_gap_before_next_segment = False

    def renderable(self) -> RenderableType:
        blocks = list(self._blocks)
        content = "".join(self._segment_chunks)
        if content and self._segment_mode is not None:
            blocks.append(self._build_segment_renderable(self._segment_mode, content))
        if not blocks:
            return Text("")
        return Group(*blocks)
