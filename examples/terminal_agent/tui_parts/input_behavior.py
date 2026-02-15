from __future__ import annotations

from bisect import bisect_right
from typing import Any

from prompt_toolkit.document import Document

from terminal_agent.input_geometry import (
    compute_visual_line_ranges,
    index_for_visual_col,
    visual_col_for_index,
)
from terminal_agent.question_view import QuestionAction
from terminal_agent.tui_parts.ui_mode import UIMode


class InputBehaviorMixin:
    def _should_handle_history(self) -> bool:
        """检查是否应该处理历史浏览(而不是补全导航)

        Returns:
            True 表示可以浏览历史,False 表示应该优先处理补全
        """
        buffer = self._input_area.buffer

        # 如果有活动的补全菜单,不应该浏览历史
        if buffer.complete_state is not None and buffer.complete_state.completions:
            return False

        # 如果在补全上下文中(输入了 / 或 @),不应该浏览历史
        document = buffer.document
        if self._completion_context_active(
            document.text_before_cursor,
            document.text_after_cursor,
        ):
            return False

        return True

    def _completion_context_active(
        self,
        text_before_cursor: str,
        text_after_cursor: str,
    ) -> bool:
        if text_after_cursor.strip():
            return False
        if text_before_cursor.startswith("/") and " " not in text_before_cursor:
            return True
        return self._mention_completer.extract_context(text_before_cursor) is not None

    def _move_completion_selection(self, buffer: Any, *, backward: bool) -> bool:
        """尝试在补全菜单中移动选择项

        行为:
        - 如果补全菜单已显示,在菜单中导航
        - 如果在补全上下文中(输入 / 或 @)但菜单未显示,触发补全并选中第一项
        - 否则返回 False,允许调用者处理其他行为(如历史浏览)

        Args:
            buffer: 当前的 Buffer 对象
            backward: True 表示向上导航,False 表示向下导航

        Returns:
            True 表示已处理补全导航,False 表示未处理
        """
        complete_state = buffer.complete_state

        # 情况1: 已有补全菜单且有补全项
        if complete_state is not None and complete_state.completions:
            if backward:
                buffer.complete_previous()
            else:
                buffer.complete_next()
            return True

        # 情况2: 在补全上下文中但菜单未显示
        document = buffer.document
        if self._completion_context_active(
            document.text_before_cursor,
            document.text_after_cursor,
        ):
            # 启动补全并选择第一项(关键改动:select_first=True)
            buffer.start_completion(select_first=True)
            return True

        return False

    def _enter_question_mode(self, questions: list[dict[str, Any]]) -> None:
        if not self._question_ui.set_questions(questions):
            return
        self._ui_mode = UIMode.QUESTION
        self._sync_focus_for_mode()
        self._refresh_layers()

    def _exit_question_mode(self) -> None:
        self._ui_mode = UIMode.NORMAL
        self._question_ui.clear()
        self._sync_focus_for_mode()

    def _sync_focus_for_mode(self) -> None:
        if self._app is None:
            return
        if self._ui_mode == UIMode.QUESTION:
            self._app.layout.focus(self._question_ui.focus_target())
            return
        if self._ui_mode == UIMode.SELECTION:
            self._app.layout.focus(self._selection_ui.focus_target())
            return
        self._app.layout.focus(self._input_area.window)

    def _handle_question_action(self, action: QuestionAction | None) -> None:
        if action is None:
            return
        if action.kind == "cancel":
            self._submit_question_reply(action.message, cancelled=True)
            return
        if action.kind == "submit":
            self._submit_question_reply(action.message, cancelled=False)

    def _submit_question_reply(self, message: str, *, cancelled: bool) -> None:
        normalized = message.strip()
        if not normalized:
            return
        self._exit_question_mode()
        self._waiting_for_input = False
        self._pending_questions = None
        if cancelled:
            self._renderer.append_system_message("用户取消问答，已发送拒绝回答消息。")
        self._refresh_layers()
        self._schedule_background(self._submit_user_message(normalized))

    def _clear_input_area(self) -> None:
        self._clear_paste_state()
        self._last_input_len = 0
        self._last_input_text = ""

        buffer = self._input_area.buffer
        buffer.cancel_completion()
        self._suppress_input_change_hook = True
        try:
            buffer.set_document(Document("", cursor_position=0), bypass_readonly=True)
        finally:
            self._suppress_input_change_hook = False

    def _next_paste_token(self) -> str:
        self._paste_token_seq += 1
        return f"paste_{self._paste_token_seq}"

    def _clear_paste_state(self) -> None:
        self._active_paste_token = None
        self._paste_placeholder_text = None
        self._paste_payload_by_token.clear()

    @staticmethod
    def _find_inserted_segment(
        previous_text: str,
        current_text: str,
    ) -> tuple[int, int, str] | None:
        if len(current_text) <= len(previous_text):
            return None

        prefix = 0
        max_prefix = min(len(previous_text), len(current_text))
        while (
            prefix < max_prefix
            and previous_text[prefix] == current_text[prefix]
        ):
            prefix += 1

        previous_remaining = len(previous_text) - prefix
        current_remaining = len(current_text) - prefix
        suffix = 0
        max_suffix = min(previous_remaining, current_remaining)
        while (
            suffix < max_suffix
            and previous_text[len(previous_text) - 1 - suffix]
            == current_text[len(current_text) - 1 - suffix]
        ):
            suffix += 1

        start = prefix
        end = len(current_text) - suffix
        if end <= start:
            return None
        return start, end, current_text[start:end]

    def _resolve_submit_texts(self, raw_text: str) -> tuple[str, str]:
        stripped = raw_text.strip()
        token = self._active_paste_token
        placeholder = self._paste_placeholder_text
        if token is not None and placeholder is not None and placeholder in raw_text:
            payload = self._paste_payload_by_token.get(token)
            if payload is not None:
                submit_text = raw_text.replace(placeholder, payload, 1).strip()
                # 在 history 中展示真实发送内容，避免显示占位符文本。
                return submit_text, submit_text
        return stripped, stripped

    def _handle_large_paste(self, buffer: Any) -> bool:
        if self._suppress_input_change_hook:
            return False
        if self._busy:
            return False

        text = str(buffer.text)
        previous_text = self._last_input_text
        self._last_input_len = len(text)
        self._last_input_text = text

        placeholder = self._paste_placeholder_text
        if self._active_paste_token is not None and placeholder and placeholder in text:
            return False

        if self._active_paste_token is not None and self._paste_placeholder_text is not None:
            self._clear_paste_state()

        threshold = max(1, int(self._paste_threshold_chars))
        segment = self._find_inserted_segment(previous_text, text)
        if segment is None:
            return False
        start, end, inserted_text = segment
        if len(inserted_text) < threshold:
            return False

        token = self._next_paste_token()
        self._active_paste_token = token
        self._paste_payload_by_token[token] = inserted_text
        placeholder = f"[Pasted Content {len(inserted_text)} chars]"
        self._paste_placeholder_text = placeholder
        replaced_text = f"{text[:start]}{placeholder}{text[end:]}"

        self._suppress_input_change_hook = True
        try:
            buffer.cancel_completion()
            buffer.set_document(
                Document(replaced_text, cursor_position=start + len(placeholder)),
                bypass_readonly=True,
            )
            self._last_input_len = len(replaced_text)
            self._last_input_text = replaced_text
        finally:
            self._suppress_input_change_hook = False

        self._invalidate()
        return True

    def _move_cursor_visual(self, buffer: Any, *, backward: bool) -> bool:
        text = str(buffer.text)
        if not text:
            return False

        width = self._terminal_width()
        max_cols = max(1, width - self._input_prompt_width)
        ranges = compute_visual_line_ranges(text, max_cols=max_cols)
        if len(ranges) <= 1:
            return False

        cursor_index = int(buffer.cursor_position)
        starts = [start for start, _end in ranges]
        row = max(0, bisect_right(starts, cursor_index) - 1)
        row_start, row_end = ranges[row]

        current_col = visual_col_for_index(
            text,
            row_start,
            row_end,
            max_cols,
            cursor_index,
        )

        target_row = row - 1 if backward else row + 1
        if target_row < 0 or target_row >= len(ranges):
            return False

        target_start, target_end = ranges[target_row]
        target_index = index_for_visual_col(
            text, target_start, target_end, max_cols, current_col
        )
        buffer.cursor_position = target_index
        return True

    def _submit_from_input(self) -> None:
        if self._busy:
            return
        if self._ui_mode != UIMode.NORMAL:
            return

        raw_text = self._input_area.text
        display_text, submit_text = self._resolve_submit_texts(raw_text)
        if not submit_text.strip():
            return

        if self._input_area.buffer.complete_state is not None:
            self._input_area.buffer.cancel_completion()
        self._clear_input_area()
        if display_text.lstrip().startswith("/"):
            self._schedule_background(self._execute_command(display_text.strip()))
            return

        self._schedule_background(
            self._submit_user_message(
                submit_text,
                display_text=display_text,
            )
        )
