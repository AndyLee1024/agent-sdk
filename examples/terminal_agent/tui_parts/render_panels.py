from __future__ import annotations

from typing import Any

from prompt_toolkit.utils import get_cwidth
from rich.console import Console

from terminal_agent.animations import PULSE_COLORS, PULSE_GLYPHS, _cyan_sweep_text
from terminal_agent.fragment_utils import clip_fragments
from terminal_agent.text_effects import fit_single_line

console = Console()


class RenderPanelsMixin:
    def _terminal_width(self) -> int:
        if self._app is None:
            return 100
        try:
            return max(int(self._app.output.get_size().columns), 40)
        except Exception:
            return 100

    def _status_text(self) -> list[tuple[str, str]]:
        width = self._terminal_width()
        base = self._status_bar.footer_status_text()
        mode = "idle"
        if self._busy:
            mode = "running"
        elif self._waiting_for_input:
            mode = "waiting_input"

        fragments: list[tuple[str, str]] = [
            ("class:status", f"[{mode}] {base}"),
        ]
        git_fragments = self._status_bar.git_diff_fragments()
        if git_fragments:
            fragments.append(("class:status", " "))
            fragments.extend(git_fragments)

        full_text = "".join(text for _, text in fragments)
        clipped_text = fit_single_line(full_text, width - 1)
        return clip_fragments(fragments, len(clipped_text))

    def _loading_text(self) -> list[tuple[str, str]]:
        # Running tools: multi-line tool panel with breathing dot.
        if self._renderer.has_running_tools():
            width = self._terminal_width()
            show_flash_line = (
                self._tool_result_flash_active and self._tool_panel_max_lines >= 2
            )
            # 动态计算所需行数，但至少保留配置值作为最小值
            required_lines = self._renderer.compute_required_tool_panel_lines()
            base_max = self._tool_panel_max_lines
            tool_max = max(required_lines, base_max)
            if show_flash_line:
                tool_max = max(tool_max + 1, self._tool_panel_max_lines)
            entries = self._renderer.tool_panel_entries(max_lines=max(1, tool_max))
            if not entries:
                return [("", " ")]

            breath_phase = (self._loading_frame // 3) % 2
            dot_style = "fg:#6B7280 bold" if breath_phase == 0 else "fg:#9CA3AF bold"
            primary_style = "fg:#D1D5DB"
            nested_style = "fg:#9CA3AF"

            fragments: list[tuple[str, str]] = []
            if show_flash_line and self._tool_result_animator.is_active:
                renderable = self._tool_result_animator.renderable()
                fragments.extend(self._rich_text_to_pt_fragments(renderable))
                fragments.append(("", "\n"))

            last_index = len(entries) - 1
            for idx, (indent, line) in enumerate(entries):
                if indent < 0:
                    clipped = fit_single_line(line, width - 1)
                    fragments.append((nested_style, clipped))
                elif indent == 0:
                    clipped = fit_single_line(line, max(width - 2, 8))
                    fragments.append((dot_style, "● "))
                    fragments.append((primary_style, clipped))
                else:
                    padding = "  " * indent
                    clipped = fit_single_line(line, max(width - get_cwidth(padding), 8))
                    fragments.append((nested_style, padding))
                    fragments.append((nested_style, clipped))

                if idx != last_index:
                    fragments.append(("", "\n"))

            return fragments

        # 优先使用动画器的渲染（流光走字 + 随机 geek 术语）
        if self._tool_result_animator.is_active:
            renderable = self._tool_result_animator.renderable()
            return self._rich_text_to_pt_fragments(renderable)

        if self._animator.is_active:
            renderable = self._animator.renderable()
            return self._rich_text_to_pt_fragments(renderable)

        # 获取语义化的 loading 状态
        loading_state = self._renderer.loading_state()
        text = loading_state.text.strip()
        if not text:
            if self._busy:
                return self._animated_loading_fragments(self._fallback_loading_phrase)
            return [("", " ")]

        return self._animated_loading_fragments(text)

    def _animated_loading_fragments(self, phrase: str) -> list[tuple[str, str]]:
        """用与 SubmissionAnimator 一致的流光走字 + 脉冲 glyph 渲染 loading 文案."""
        width = self._terminal_width()
        clipped = fit_single_line(phrase, width - 4)  # 留出 glyph + 空格

        frame = self._loading_frame
        pulse_idx = frame % len(PULSE_COLORS)
        glyph = PULSE_GLYPHS[pulse_idx]
        glyph_color = PULSE_COLORS[pulse_idx]

        # 构建 Rich Text（与 SubmissionAnimator.renderable 相同结构）
        from rich.text import Text as RichText

        dot = RichText(f"{glyph} ", style=f"bold {glyph_color}")
        sweep = _cyan_sweep_text(clipped, frame=frame)
        combined = RichText.assemble(dot, sweep)
        return self._rich_text_to_pt_fragments(combined)

    def _loading_height(self) -> int:
        if self._animator.is_active:
            return 1
        if self._renderer.has_running_tools():
            show_flash_line = (
                self._tool_result_flash_active and self._tool_panel_max_lines >= 2
            )
            # 动态计算所需行数，但至少保留配置值作为最小值
            required_lines = self._renderer.compute_required_tool_panel_lines()
            base_max = self._tool_panel_max_lines
            tool_max = max(required_lines, base_max)
            if show_flash_line:
                tool_max = max(tool_max + 1, self._tool_panel_max_lines)
            tool_lines = self._renderer.tool_panel_entries(max_lines=max(1, tool_max))
            total = len(tool_lines) + (1 if show_flash_line else 0)
            return max(1, total)
        if self._tool_result_animator.is_active:
            return 1
        if self._renderer.loading_state().text.strip():
            return 1
        return 1

    _DIFF_PANEL_MAX_VISIBLE = 20

    def _diff_panel_text(self) -> list[tuple[str, str]]:
        import re

        diff_lines = self._renderer.latest_diff_lines
        if not diff_lines:
            return [("", " ")]

        width = self._terminal_width()
        total = len(diff_lines)
        viewport = self._DIFF_PANEL_MAX_VISIBLE - 1  # reserve 1 for header
        offset = self._diff_panel_scroll
        max_scroll = max(0, total - viewport)
        scroll_info = ""
        if total > viewport:
            scroll_info = f" [{offset + 1}-{min(offset + viewport, total)}/{total}] ↑↓"

        title = (
            " Diff Preview (Ctrl+O to close, use arrow keys (↑↓) to scroll) "
            f"{scroll_info} "
        )
        pad_len = max(width - len(title) - 2, 0)
        left_pad = pad_len // 2
        right_pad = pad_len - left_pad
        header = f"{'─' * left_pad}{title}{'─' * right_pad}"

        fragments: list[tuple[str, str]] = [("fg:#6B7280", header)]

        hunk_pattern = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
        # To track line numbers correctly, we must parse from the beginning
        old_line = 0
        new_line = 0
        visible_start = offset
        visible_end = offset + viewport

        for idx, line in enumerate(diff_lines):
            # Always parse hunk headers and file headers to track line numbers
            if line.startswith("---") or line.startswith("+++"):
                if visible_start <= idx < visible_end:
                    fragments.append(("", "\n"))
                    fragments.append(("fg:#6B7280", line))
                continue

            m = hunk_pattern.match(line)
            if m:
                old_line = int(m.group(1))
                new_line = int(m.group(2))
                if visible_start <= idx < visible_end:
                    fragments.append(("", "\n"))
                    fragments.append(("fg:#00BCD4", line))
                continue

            if visible_start <= idx < visible_end:
                fragments.append(("", "\n"))
                if line.startswith("-"):
                    prefix = f"{old_line:>4} "
                    prefix_width = get_cwidth(prefix)
                    content_width = max(0, width - prefix_width)
                    fitted = fit_single_line(line, content_width)
                    padding = " " * (content_width - get_cwidth(fitted))

                    fragments.append(("fg:#ffffff bg:#4a0202", prefix))
                    fragments.append(("fg:#ffffff bg:#4a0202", fitted + padding))
                elif line.startswith("+"):
                    prefix = f"{new_line:>4} "
                    prefix_width = get_cwidth(prefix)
                    content_width = max(0, width - prefix_width)
                    fitted = fit_single_line(line, content_width)
                    padding = " " * (content_width - get_cwidth(fitted))

                    fragments.append(("fg:#c6c6c6 bg:#2e502e", prefix))
                    fragments.append(("fg:#ffffff bg:#2e502e", fitted + padding))
                else:
                    fragments.append(("fg:#6B7280", f"{new_line:>4} "))
                    fragments.append(("fg:#6B7280", line))

            # Always update line counters
            if line.startswith("-"):
                old_line += 1
            elif line.startswith("+"):
                new_line += 1
            else:
                old_line += 1
                new_line += 1

        return fragments

    def _diff_panel_height(self) -> int:
        diff_lines = self._renderer.latest_diff_lines
        if not diff_lines:
            return 0
        viewport = self._DIFF_PANEL_MAX_VISIBLE - 1
        content_lines = min(len(diff_lines), viewport)
        return content_lines + 1  # +1 for header

    def _todo_text(self) -> list[tuple[str, str]]:
        lines = self._renderer.todo_panel_lines(max_lines=self._todo_panel_max_lines)
        if not lines:
            return [("", " ")]

        width = self._terminal_width()
        fragments: list[tuple[str, str]] = []
        last_index = len(lines) - 1
        for idx, line in enumerate(lines):
            clipped = fit_single_line(line, width - 1)
            style = "fg:#A7F3D0" if idx == 0 else "fg:#CBD5E1"
            if "✓" in line:
                style = "fg:#86EFAC"
            elif "◉" in line:
                style = "fg:#FDE68A"
            fragments.append((style, clipped))
            if idx != last_index:
                fragments.append(("", "\n"))
        return fragments

    def _todo_height(self) -> int:
        lines = self._renderer.todo_panel_lines(max_lines=self._todo_panel_max_lines)
        return max(1, len(lines))

    def _rich_text_to_pt_fragments(self, renderable: Any) -> list[tuple[str, str]]:
        """将 Rich Text 转换为 prompt_toolkit 的 fragments 格式."""
        from rich.segment import Segment

        fragments: list[tuple[str, str]] = []
        for segment in renderable.__rich_console__(console, console.options):
            if isinstance(segment, Segment):
                text = segment.text
                style = segment.style
                if style:
                    # 将 Rich style 转换为 prompt_toolkit style
                    pt_style = self._rich_style_to_pt(style)
                    fragments.append((pt_style, text))
                else:
                    fragments.append(("", text))
        return fragments if fragments else [("", " ")]

    def _rich_style_to_pt(self, rich_style: Any) -> str:
        """将 Rich style 转换为 prompt_toolkit style 字符串."""
        parts: list[str] = []
        if rich_style.bold:
            parts.append("bold")
        if rich_style.italic:
            parts.append("italic")
        if rich_style.underline:
            parts.append("underline")
        if rich_style.strike:
            parts.append("strike")
        if rich_style.dim:
            parts.append("dim")

        # 前景色
        if rich_style.color and rich_style.color.triplet:
            parts.append(f"fg:{rich_style.color.triplet.hex}")
        elif rich_style.color and rich_style.color.type.name == "STANDARD":
            parts.append(f"fg:ansi{rich_style.color.number}")
        elif rich_style.color and rich_style.color.type.name == "EIGHT_BIT":
            parts.append(f"fg:ansi{rich_style.color.number}")

        # 背景色
        if rich_style.bgcolor and rich_style.bgcolor.triplet:
            parts.append(f"bg:{rich_style.bgcolor.triplet.hex}")
        elif rich_style.bgcolor and rich_style.bgcolor.type.name == "STANDARD":
            parts.append(f"bg:ansi{rich_style.bgcolor.number}")
        elif rich_style.bgcolor and rich_style.bgcolor.type.name == "EIGHT_BIT":
            parts.append(f"bg:ansi{rich_style.bgcolor.number}")

        return " ".join(parts) if parts else ""
