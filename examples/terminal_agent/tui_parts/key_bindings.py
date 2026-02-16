from __future__ import annotations

import time

from prompt_toolkit.filters import Condition, has_completions, has_focus
from prompt_toolkit.key_binding import KeyBindings

from terminal_agent.slash_commands import parse_slash_command_call
from terminal_agent.tui_parts.ui_mode import UIMode


class KeyBindingsMixin:
    def _build_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        normal_mode = Condition(lambda: self._ui_mode == UIMode.NORMAL)
        question_mode = Condition(lambda: self._ui_mode == UIMode.QUESTION)
        selection_mode = Condition(lambda: self._ui_mode == UIMode.SELECTION)

        # Selection menu key bindings
        @bindings.add("enter", filter=selection_mode)
        def _selection_enter(event) -> None:
            del event
            result = self._selection_ui.confirm()
            self._handle_selection_result(result)
            self._invalidate()

        @bindings.add("up", filter=selection_mode)
        def _selection_up(event) -> None:
            del event
            self._selection_ui.move_selection(-1)
            self._invalidate()

        @bindings.add("down", filter=selection_mode)
        def _selection_down(event) -> None:
            del event
            self._selection_ui.move_selection(1)
            self._invalidate()

        @bindings.add("k", filter=selection_mode)
        def _selection_k(event) -> None:
            del event
            self._selection_ui.move_selection(-1)
            self._invalidate()

        @bindings.add("j", filter=selection_mode)
        def _selection_j(event) -> None:
            del event
            self._selection_ui.move_selection(1)
            self._invalidate()

        @bindings.add("escape", filter=selection_mode)
        def _selection_cancel(event) -> None:
            del event
            result = self._selection_ui.cancel()
            self._handle_selection_result(result)
            self._invalidate()

        @bindings.add("enter", filter=question_mode)
        def _question_enter(event) -> None:
            del event
            action = self._question_ui.handle_enter()
            self._handle_question_action(action)
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("up", filter=question_mode)
        def _question_up(event) -> None:
            del event
            self._question_ui.move_option(-1)
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("down", filter=question_mode)
        def _question_down(event) -> None:
            del event
            self._question_ui.move_option(1)
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("left", filter=question_mode)
        def _question_prev(event) -> None:
            del event
            self._question_ui.prev_question()
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("right", filter=question_mode)
        def _question_next(event) -> None:
            del event
            self._question_ui.next_question()
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add(
            "space",
            filter=question_mode & ~has_focus(self._question_ui.custom_input_window),
        )
        def _question_toggle(event) -> None:
            del event
            self._question_ui.toggle_current_selection()
            self._invalidate()

        @bindings.add("tab", filter=question_mode)
        def _question_submit_tab(event) -> None:
            del event
            self._question_ui.focus_submit()
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("escape", filter=question_mode)
        def _question_cancel(event) -> None:
            del event
            self._handle_question_action(self._question_ui.handle_escape())
            self._sync_focus_for_mode()
            self._invalidate()

        @bindings.add("enter", filter=normal_mode, eager=True)
        def _enter(event) -> None:
            buffer = event.current_buffer
            cs = buffer.complete_state

            # 菜单打开时：先接受补全
            if cs is not None and cs.completions:
                completion = cs.current_completion or cs.completions[0]
                buffer.apply_completion(completion)
                buffer.cancel_completion()

                # ✅ 决策：slash 补全 -> 立即提交；mention 补全 -> 只填入不提交
                text_now = buffer.text
                doc_now = buffer.document

                is_mention = (
                    self._mention_completer.extract_context(doc_now.text_before_cursor)
                    is not None
                )
                parsed_slash = parse_slash_command_call(text_now)

                if (parsed_slash is not None) and (not is_mention):
                    self._submit_from_input()
                return

            # 没有菜单：正常提交
            self._submit_from_input()

        @bindings.add("tab", filter=normal_mode)
        def _tab(event) -> None:
            buffer = event.current_buffer
            cs = buffer.complete_state
            # Tab：如果有补全项 -> 接受当前项并关闭菜单
            if cs is not None and cs.completions:
                completion = cs.current_completion or cs.completions[0]
                buffer.apply_completion(completion)
                buffer.cancel_completion()
                return
            # 否则触发补全菜单
            buffer.start_completion(select_first=False)

        @bindings.add("s-tab", filter=normal_mode & has_completions)
        def _active_prev(event) -> None:
            event.current_buffer.complete_previous()

        @bindings.add("up", filter=normal_mode)
        def _active_prev_up(event) -> None:
            buffer = event.current_buffer
            if self._move_completion_selection(buffer, backward=True):
                return
            if self._move_cursor_visual(buffer, backward=True):
                self._invalidate()
                return
            if self._should_handle_history():
                buffer.auto_up(count=1)

        @bindings.add("down", filter=normal_mode)
        def _active_next_down(event) -> None:
            buffer = event.current_buffer
            if self._move_completion_selection(buffer, backward=False):
                return
            if self._move_cursor_visual(buffer, backward=False):
                self._invalidate()
                return
            if self._should_handle_history():
                buffer.auto_down(count=1)

        @bindings.add("escape", filter=normal_mode)
        def _esc(event) -> None:
            buffer = event.current_buffer
            now = time.monotonic()
            if buffer.complete_state is not None:
                buffer.cancel_completion()
                self._esc_press_count = 0
                self._esc_last_pressed_at = now
                return

            if now - self._esc_last_pressed_at > self._esc_clear_window_seconds:
                self._esc_press_count = 0
            self._esc_press_count += 1
            self._esc_last_pressed_at = now

            if self._esc_press_count >= 2:
                self._esc_press_count = 0
                self._clear_input_area()
                self._invalidate()
                return
            if self._app is not None:
                self._app.layout.focus(self._input_area.window)

        @bindings.add("c-c")
        def _interrupt_or_exit(event) -> None:
            del event
            if self._busy and self._stream_task is not None:
                now = time.monotonic()
                interrupted_once = (
                    self._interrupt_requested_at is not None
                    and (now - self._interrupt_requested_at)
                    <= self._interrupt_force_window_seconds
                )

                if interrupted_once:
                    self._stream_task.cancel()
                    self._session.run_controller.clear()
                    self._interrupt_requested_at = None
                    self._renderer.interrupt_turn()
                    self._renderer.append_system_message("已强制中断当前任务，可继续输入。")
                    self._set_busy(False)
                    self._waiting_for_input = False
                    self._pending_questions = None
                    self._exit_question_mode()
                    self._refresh_layers()
                    return

                self._session.run_controller.interrupt(reason="user")
                self._interrupt_requested_at = now
                self._renderer.append_system_message(
                    "已发送中断信号。再次按 Ctrl+C 将强制中断。"
                )
                self._refresh_layers()
                return
            now = time.monotonic()
            if now - self._ctrl_c_last_pressed_at > self._ctrl_c_exit_window_seconds:
                self._ctrl_c_press_count = 0
            self._ctrl_c_press_count += 1
            self._ctrl_c_last_pressed_at = now

            if self._ctrl_c_press_count >= 2:
                self._ctrl_c_press_count = 0
                self._exit_app()
                return

            self._clear_input_area()
            self._invalidate()

        @bindings.add("c-d")
        def _exit(event) -> None:
            del event
            self._exit_app()

        @bindings.add("c-o", filter=normal_mode)
        def _toggle_diff_panel(event) -> None:
            del event
            self._diff_panel_visible = not self._diff_panel_visible
            self._diff_panel_scroll = 0
            self._invalidate()

        @bindings.add("c-t", filter=normal_mode)
        def _toggle_thinking(event) -> None:
            del event
            self._show_thinking = not self._show_thinking
            status = "开启" if self._show_thinking else "关闭"
            self._renderer.append_system_message(f"Thinking 显示已{status}")
            self._invalidate()

        diff_panel_open = Condition(lambda: self._diff_panel_visible)

        @bindings.add("up", filter=normal_mode & diff_panel_open)
        def _diff_scroll_up(event) -> None:
            del event
            if self._diff_panel_scroll > 0:
                self._diff_panel_scroll -= 1
                self._invalidate()

        @bindings.add("down", filter=normal_mode & diff_panel_open)
        def _diff_scroll_down(event) -> None:
            del event
            diff_lines = self._renderer.latest_diff_lines
            if diff_lines:
                max_scroll = max(0, len(diff_lines) - (self._DIFF_PANEL_MAX_VISIBLE - 1))
                if self._diff_panel_scroll < max_scroll:
                    self._diff_panel_scroll += 1
                    self._invalidate()

        return bindings
