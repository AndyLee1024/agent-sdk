"""Terminal Agent TUI implementation.

This module was split out from `terminal_agent.app` to keep the entrypoint small.
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from bisect import bisect_right
from collections.abc import Callable
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.completion import (
    merge_completers,
)
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, has_focus
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea
from rich.console import Console

from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import AssistantMessage
from comate_agent_sdk.agent import ChatSession
from comate_agent_sdk.agent.llm_levels import ALL_LEVELS

from terminal_agent.animations import (
    DEFAULT_STATUS_PHRASES,
    PULSE_COLORS,
    PULSE_GLYPHS,
    _cyan_sweep_text,
    StreamAnimationController,
    SubmissionAnimator,
)
from terminal_agent.env_utils import read_env_float, read_env_int
from terminal_agent.event_renderer import EventRenderer
from terminal_agent.history_printer import (
    print_history_group_sync,
    render_history_group,
)
from terminal_agent.input_geometry import (
    compute_visual_line_ranges,
    index_for_visual_col,
    visual_col_for_index,
)
from terminal_agent.markdown_render import render_markdown_to_plain
from terminal_agent.mention_completer import LocalFileMentionCompleter
from terminal_agent.models import HistoryEntry
from terminal_agent.question_view import AskUserQuestionUI, QuestionAction
from terminal_agent.selection_menu import (
    SelectionMenuUI,
    SelectionResult,
    build_model_level_options,
)
from terminal_agent.slash_commands import (
    SLASH_COMMAND_SPECS,
    SlashCommandCompleter,
    SlashCommandSpec,
    parse_slash_command_call,
)
from terminal_agent.status_bar import StatusBar
from terminal_agent.text_effects import fit_single_line

console = Console()
logger = logging.getLogger(__name__)
logging.getLogger("comate_agent_sdk.system_tools.tools").setLevel(logging.ERROR)
class UIMode(Enum):
    NORMAL = "normal"
    QUESTION = "question"
    SELECTION = "selection"  # 通用选择菜单模式（如模型切换）


class TerminalAgentTUI:
    def __init__(
        self,
        session: ChatSession,
        status_bar: StatusBar,
        renderer: EventRenderer,
    ) -> None:
        self._session = session
        self._status_bar = status_bar
        self._renderer = renderer

        self._tool_panel_max_lines = read_env_int(
            "AGENT_SDK_TUI_TOOL_PANEL_MAX_LINES",
            4,
        )
        self._todo_panel_max_lines = read_env_int(
            "AGENT_SDK_TUI_TODO_PANEL_MAX_LINES",
            6,
        )

        self._busy = False
        self._waiting_for_input = False
        self._pending_questions: list[dict[str, Any]] | None = None
        self._ui_mode = UIMode.NORMAL
        self._input_read_only = Condition(lambda: self._busy)

        self._slash_completer = SlashCommandCompleter(SLASH_COMMAND_SPECS)
        self._mention_completer = LocalFileMentionCompleter(Path.cwd())
        self._input_completer = merge_completers(
            [self._slash_completer, self._mention_completer],
            deduplicate=True,
        )
        self._slash_lookup: dict[str, SlashCommandSpec] = {}
        for spec in SLASH_COMMAND_SPECS:
            self._slash_lookup[spec.name] = spec
            for alias in spec.aliases:
                self._slash_lookup[alias] = spec
        self._slash_handlers: dict[str, Callable[[str], Any]] = {
            "help": self._slash_help,
            "model": self._slash_model,
            "session": self._slash_session,
            "usage": self._slash_usage,
            "context": self._slash_context,
            "exit": self._slash_exit,
        }
        self._loading_frame = 0
        self._fallback_loading_phrase = (
            random.choice(DEFAULT_STATUS_PHRASES)
            if DEFAULT_STATUS_PHRASES
            else "Thinking…"
        )
        self._fallback_phrase_refresh_at = 0.0

        self._tool_result_flash_seconds = read_env_float(
            "AGENT_SDK_TUI_TOOL_RESULT_FLASH_SECONDS",
            0.55,
        )
        self._tool_result_flash_gen = 0
        self._tool_result_flash_until_monotonic: float | None = None
        self._tool_result_flash_active = False

        # 初始化提交动画控制器
        self._animator = SubmissionAnimator()
        self._animation_controller = StreamAnimationController(self._animator)
        self._tool_result_animator = SubmissionAnimator()

        self._closing = False
        self._printed_history_index = 0
        self._render_dirty = True
        self._last_loading_line = ""

        self._app: Application[None] | None = None
        self._stream_task: asyncio.Task[Any] | None = None
        self._ui_tick_task: asyncio.Task[None] | None = None
        self._interrupt_requested_at: float | None = None
        self._interrupt_force_window_seconds = 1.5

        self._esc_last_pressed_at: float = 0.0
        self._esc_press_count: int = 0
        esc_window_ms = read_env_int("AGENT_SDK_TUI_ESC_CLEAR_WINDOW_MS", 700)
        self._esc_clear_window_seconds = esc_window_ms / 1000.0

        self._ctrl_c_last_pressed_at: float = 0.0
        self._ctrl_c_press_count: int = 0
        ctrl_c_window_ms = read_env_int("AGENT_SDK_TUI_CTRL_C_EXIT_WINDOW_MS", 700)
        self._ctrl_c_exit_window_seconds = ctrl_c_window_ms / 1000.0

        self._paste_threshold_chars = read_env_int(
            "AGENT_SDK_TUI_PASTE_PLACEHOLDER_THRESHOLD_CHARS",
            500,
        )
        self._paste_placeholder_text: str | None = None
        self._active_paste_token: str | None = None
        self._paste_payload_by_token: dict[str, str] = {}
        self._paste_token_seq = 0
        self._suppress_input_change_hook = False
        self._last_input_len = 0
        self._last_input_text = ""

        self._input_prompt_text = "> "
        self._input_prompt_width = max(1, get_cwidth(self._input_prompt_text))

        def _input_line_prefix(
            _line_number: int,
            wrap_count: int,
        ) -> list[tuple[str, str]]:
            if wrap_count <= 0:
                return [("class:input.prompt", self._input_prompt_text)]
            return [("class:input.prompt", " " * self._input_prompt_width)]

        self._input_area = TextArea(
            text="",
            multiline=True,
            prompt="",
            wrap_lines=True,
            dont_extend_height=True,
            completer=self._input_completer,
            complete_while_typing=False,  # 通过 Tab/上下键手动触发补全
            history=InMemoryHistory(),
            read_only=self._input_read_only,
            style="class:input.line",
            get_line_prefix=_input_line_prefix,
        )
        # Fill the entire input area with styled spaces to avoid VT100
        # erase-to-end-of-line resetting the background to the terminal default.
        # (prompt_toolkit renderer resets attributes before erase_end_of_line.)
        self._input_area.window.char = " "

        @self._input_area.buffer.on_text_changed.add_handler
        def _trigger_completion(_buffer) -> None:
            if self._handle_large_paste(_buffer):
                return
            if self._busy:
                return
            doc = self._input_area.buffer.document
            if self._completion_context_active(
                doc.text_before_cursor,
                doc.text_after_cursor,
            ):
                # 输入 / 或 @ 时自动弹出（不会选中第一项）
                self._input_area.buffer.start_completion(select_first=False)

        self._question_ui = AskUserQuestionUI()
        self._selection_ui = SelectionMenuUI()
        self._todo_control = FormattedTextControl(text=self._todo_text)
        self._loading_control = FormattedTextControl(text=self._loading_text)
        self._status_control = FormattedTextControl(text=self._status_text)

        self._todo_window = Window(
            content=self._todo_control,
            height=self._todo_height,
            dont_extend_height=True,
            style="class:loading",
        )
        self._todo_container = ConditionalContainer(
            content=self._todo_window,
            filter=Condition(lambda: self._renderer.has_active_todos()),
        )

        self._loading_window = Window(
            content=self._loading_control,
            height=self._loading_height,
            dont_extend_height=True,
            style="class:loading",
        )
        self._status_window = Window(
            content=self._status_control,
            height=1,
            dont_extend_height=True,
            style="class:status",
        )
   
        # 补全菜单：在底部区域显示（覆盖 statusline）
        self._completion_menu = CompletionsMenu(
            max_height=8,
            scroll_offset=0,
            extra_filter=(
                Condition(lambda: self._ui_mode == UIMode.NORMAL)
                & has_focus(self._input_area)
            ),
        )

        self._completion_visible = (
            Condition(lambda: self._ui_mode == UIMode.NORMAL)
            & has_focus(self._input_area)
            & has_completions
        )
        self._bottom_container = ConditionalContainer(
            content=self._completion_menu,
            filter=self._completion_visible,
            alternative_content=self._status_window,
        )

        self._input_container = ConditionalContainer(
            content=self._input_area,
            filter=Condition(lambda: self._ui_mode == UIMode.NORMAL),
        )
        self._question_container = ConditionalContainer(
            content=self._question_ui.container,
            filter=Condition(lambda: self._ui_mode == UIMode.QUESTION),
        )
        self._selection_container = ConditionalContainer(
            content=self._selection_ui.container,
            filter=Condition(lambda: self._ui_mode == UIMode.SELECTION),
        )

        self._main_container = HSplit(
            [
                self._todo_container,
                self._loading_window,
                Window(height=1, char=" ", style="class:input.pad"),
                self._input_container,
                self._question_container,
                self._selection_container,
                Window(height=1, char=" ", style="class:input.pad"),
                self._bottom_container,
            ]
        )

        self._root = FloatContainer(
            content=self._main_container,
            floats=[],
        )

        self._layout = Layout(self._root, focused_element=self._input_area.window)
        self._bindings = self._build_key_bindings()
        self._style = PTStyle.from_dict(
            {
                "": "bg:#1f232a #e5e9f0",
                "history": "bg:#1a1e24 #d8dee9",
                "input.pad": "fg:default bg:default",
                "input.prompt": "bg:default #f2f4f8",
                "input.line": "bg:default #f2f4f8",
                "input-line": "bg:default #f2f4f8",
                "status": "bg:#2d3138 #c3ccd8",
                "question.tabs": "bg:#30353f #c7d2fe",
                "question.tabs.nav": "bg:#30353f #93c5fd",
                "question.tab": "bg:#374151 #cbd5e1",
                "question.tab.submit": "bg:#14532d #86efac bold",
                "question.tab.active": "bg:#1d4ed8 #eff6ff bold",
                "question.divider": "fg:#4b5563",
                "question.body": "bg:#252a33 #d8dee9",
                "question.title": "fg:#f8fafc bold",
                "question.hint": "fg:#94a3b8",
                "question.option": "fg:#dbeafe",
                "question.option.cursor": "bg:#1e3a8a #f8fafc bold",
                "question.option.selected": "fg:#93c5fd bold",
                "question.option.description": "fg:#9ca3af",
                "question.custom_input": "bg:#0f172a #f8fafc",
                "question.custom_input.border": "fg:#334155",
                "question.preview.title": "fg:#e2e8f0 bold",
                "question.preview.question": "fg:#bfdbfe",
                "question.preview.answer": "fg:#f1f5f9",
                "selection.title": "bg:#2d3138 #c3ccd8 bold",
                "selection.divider": "fg:#4b5563",
                "selection.body": "bg:#252a33 #d8dee9",
                "selection.option": "fg:#dbeafe",
                "selection.option.selected": "bg:#1e3a8a #f8fafc bold",
                "selection.description": "fg:#9ca3af",
                "selection.description.selected": "fg:#93c5fd",
                "selection.hint": "fg:#94a3b8",
                "completion-menu": "bg:#2d3441 #d8dee9",
                "completion-menu.completion.current": "bg:#5e81ac #eceff4",
                "completion-menu.meta.completion": "bg:#2d3441 #81a1c1",
                "completion-menu.meta.completion.current": "bg:#5e81ac #e5e9f0",
                "loading": "bg:#1a1e24",
            }
        )

        self._app = Application(
            layout=self._layout,
            key_bindings=self._bindings,
            style=self._style,
            full_screen=False,
            mouse_support=False,
        )

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        self._render_dirty = True
        self._invalidate()

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
                self._renderer.append_system_message("已发送中断信号。再次按 Ctrl+C 将强制中断。")
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

        return bindings

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

    def add_resume_history(self, mode: str) -> None:
        if mode != "resume":
            return

        items = self._session._agent._context.conversation.items
        history = [
            item
            for item in items
            if item.item_type in (ItemType.USER_MESSAGE, ItemType.ASSISTANT_MESSAGE)
        ]
        if not history:
            return

        self._renderer.append_system_message(f"history loaded: {len(history)} messages")
        for item in history:
            if item.item_type == ItemType.USER_MESSAGE:
                content = str(item.content_text or "").strip()
                if content:
                    self._renderer.seed_user_message(content)
                continue

            # Handle tool calls restoration
            message = getattr(item, "message", None)
            if isinstance(message, AssistantMessage) and message.tool_calls:
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    args_str = tc.function.arguments
                    # Parse arguments as dict for summarize_tool_args
                    try:
                        import json
                        args_dict = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args_dict = {"_raw": args_str}
                    args_summary = self._summarize_tool_args(tool_name, args_dict)
                    self._renderer.restore_tool_call(
                        tool_call_id=tc.id,
                        tool_name=tool_name,
                        args_summary=args_summary,
                    )

            assistant_text = self._extract_assistant_text(item).strip()
            if assistant_text:
                self._renderer.append_assistant_message(assistant_text)
            else:
                self._renderer.append_system_message("(tool call only message)")

        self._drain_history_sync()

    def _summarize_tool_args(self, tool_name: str, args: dict[str, Any]) -> str:
        """Summarize tool arguments for display in history."""
        from terminal_agent.tool_view import summarize_tool_args
        summary = summarize_tool_args(tool_name, args, self._renderer._project_root).strip()
        return summary

    @staticmethod
    def _extract_assistant_text(item: Any) -> str:
        content = item.content_text or ""
        if content:
            return str(content)
        message = getattr(item, "message", None)
        if message is None:
            return ""
        msg_content = getattr(message, "content", "")
        if isinstance(msg_content, str):
            return msg_content
        if isinstance(msg_content, list):
            text_parts: list[str] = []
            for part in msg_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            return "".join(text_parts)
        return ""

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

    async def _submit_user_message(
        self,
        text: str,
        *,
        display_text: str | None = None,
    ) -> None:
        if self._busy:
            self._renderer.append_system_message("当前已有任务在运行，请稍候。", is_error=True)
            return

        if self._ui_mode == UIMode.QUESTION:
            self._exit_question_mode()

        self._session.run_controller.clear()
        self._interrupt_requested_at = None

        self._set_busy(True)
        self._waiting_for_input = False
        self._pending_questions = None
        # 本轮默认 loading 文案：避免出现 Working… 这种突兀 fallback
        self._fallback_loading_phrase = (
            random.choice(DEFAULT_STATUS_PHRASES)
            if DEFAULT_STATUS_PHRASES
            else "Thinking…"
        )
        self._fallback_phrase_refresh_at = time.monotonic() + 3.0  # 每 3 秒允许换一次

        self._renderer.start_turn()
        self._renderer.seed_user_message(
            display_text if display_text is not None else text
        )
        self._refresh_layers()

        # 启动提交动画
        await self._animation_controller.start()

        waiting_for_input = False
        questions: list[dict[str, Any]] | None = None

        stream_task = asyncio.create_task(
            self._consume_stream(text),
            name="terminal-tui-stream",
        )
        self._stream_task = stream_task
        try:
            waiting_for_input, questions = await stream_task
        except asyncio.CancelledError:
            logger.debug("stream task cancelled")
        except Exception as exc:
            logger.exception("stream failed")
            self._renderer.append_system_message(f"流式处理失败: {exc}", is_error=True)
        finally:
            self._stream_task = None
            self._interrupt_requested_at = None
            self._set_busy(False)
            await self._status_bar.refresh()

        if waiting_for_input:
            self._waiting_for_input = True
            self._pending_questions = questions
            if questions:
                self._enter_question_mode(questions)
            else:
                self._renderer.append_system_message("请输入对上述问题的回答后回车提交。")
        else:
            self._waiting_for_input = False
            self._pending_questions = None
            self._exit_question_mode()

        self._refresh_layers()

    async def _consume_stream(
        self,
        text: str,
    ) -> tuple[bool, list[dict[str, Any]] | None]:
        waiting_for_input = False
        questions: list[dict[str, Any]] | None = None

        async for event in self._session.query_stream(text):
            self._maybe_cancel_tool_result_flash(event)
            # 将事件传递给动画控制器以控制动画生命周期
            await self._animation_controller.on_event(event)
            is_waiting, new_questions = self._renderer.handle_event(event)
            self._maybe_flash_tool_result(event)
            if is_waiting:
                waiting_for_input = True
                if new_questions is not None:
                    questions = new_questions
            self._refresh_layers()

        self._renderer.finalize_turn()
        await self._animation_controller.shutdown()
        return waiting_for_input, questions

    def _maybe_cancel_tool_result_flash(self, event: object) -> None:
        if not self._tool_result_flash_active:
            return

        from comate_agent_sdk.agent.events import StopEvent, TextEvent

        if isinstance(event, TextEvent) or isinstance(event, StopEvent):
            self._schedule_background(self._stop_tool_result_flash())

    def _maybe_flash_tool_result(self, event: object) -> None:
        from comate_agent_sdk.agent.events import ToolResultEvent

        if not isinstance(event, ToolResultEvent):
            return
        tool_name = str(event.tool or "").strip()
        if tool_name.lower() == "todowrite":
            return
        if self._renderer.has_running_tools():
            return
        self._trigger_tool_result_flash(
            tool_name=tool_name or "Tool",
            is_error=bool(event.is_error),
        )

    def _trigger_tool_result_flash(self, *, tool_name: str, is_error: bool) -> None:
        del is_error, tool_name
        phrase = (
            random.choice(DEFAULT_STATUS_PHRASES)
            if DEFAULT_STATUS_PHRASES
            else "Vibing..."
        )
        duration_seconds = max(0.15, float(self._tool_result_flash_seconds))
        self._tool_result_flash_gen += 1
        self._tool_result_flash_active = True
        self._tool_result_flash_until_monotonic = time.monotonic() + duration_seconds
        gen = self._tool_result_flash_gen
        self._schedule_background(self._start_tool_result_flash(gen=gen, hint=phrase))

    async def _start_tool_result_flash(self, *, gen: int, hint: str) -> None:
        if gen != self._tool_result_flash_gen:
            return
        await self._tool_result_animator.start()
        if gen != self._tool_result_flash_gen:
            return
        self._tool_result_animator.set_status_hint(hint)
        self._render_dirty = True
        self._invalidate()

    async def _stop_tool_result_flash(self) -> None:
        if not self._tool_result_flash_active:
            return
        self._tool_result_flash_gen += 1
        self._tool_result_flash_active = False
        self._tool_result_flash_until_monotonic = None
        self._tool_result_animator.set_status_hint(None)
        await self._tool_result_animator.stop()
        self._render_dirty = True
        self._invalidate()

    def _schedule_background(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)

        def _done(done_task: asyncio.Task[Any]) -> None:
            try:
                done_task.result()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("background task failed")

        task.add_done_callback(_done)

    async def _execute_command(self, command: str) -> None:
        parsed = parse_slash_command_call(command)
        normalized = command.strip()
        if parsed is None:
            self._renderer.append_system_message(
                f"Unknown command: {normalized}",
                is_error=True,
            )
            self._refresh_layers()
            return

        spec = self._slash_lookup.get(parsed.name)
        if spec is None:
            self._renderer.append_system_message(
                f"Unknown command: {normalized}",
                is_error=True,
            )
            self._refresh_layers()
            return

        handler = self._slash_handlers.get(spec.name)
        if handler is None:
            logger.error("slash handler missing for command: %s", spec.name)
            self._renderer.append_system_message(
                f"Unknown command: {parsed.raw_input}",
                is_error=True,
            )
            self._refresh_layers()
            return

        result = handler(parsed.args)
        if asyncio.iscoroutine(result):
            await result
        self._refresh_layers()

    def _slash_help(self, _args: str) -> None:
        lines = []
        for spec in SLASH_COMMAND_SPECS:
            alias_text = ""
            if spec.aliases:
                alias_text = f" ({', '.join(f'/{alias}' for alias in spec.aliases)})"
            lines.append(f"/{spec.name}{alias_text} - {spec.description}")
        self._renderer.append_system_message("\n".join(lines))

    def _slash_session(self, _args: str) -> None:
        self._renderer.append_system_message(f"Session ID: {self._session.session_id}")

    async def _slash_usage(self, _args: str) -> None:
        await self._append_usage_snapshot()

    async def _slash_context(self, _args: str) -> None:
        await self._append_context_snapshot()

    def _slash_exit(self, _args: str) -> None:
        self._exit_app()

    def _slash_model(self, args: str) -> None:
        """Handle /model command - switch model level."""
        # If args provided, try to use it directly
        if args.strip():
            level = args.strip().upper()
            if level in ALL_LEVELS:
                self._switch_model_level(level)
                return
            # Invalid level, show error and open menu
            self._renderer.append_system_message(
                f"Invalid model level: {args.strip()}. Use LOW, MID, or HIGH.",
                is_error=True,
            )

        # Open selection menu
        self._enter_selection_mode()

    def _switch_model_level(self, level: str) -> None:
        """Switch to the specified model level."""
        try:
            llm_level = level  # type: ignore[assignment]
            event = self._session.set_level(llm_level)

            # Get model names for display
            prev_model = event.previous_model or "unknown"
            new_model = event.new_model or "unknown"

            self._renderer.append_system_message(
                f"Model switched: {event.previous_level} → {event.new_level}\n"
                f"  ({prev_model} → {new_model})"
            )
            logger.info(f"Model level switched: {event}")

            # Update status bar model name - 使用 event 中的新模型名
            self._status_bar.set_model_name(new_model)
            self._invalidate()
        except Exception as e:
            logger.exception("Failed to switch model level")
            self._renderer.append_system_message(
                f"Failed to switch model: {e}",
                is_error=True,
            )

    def _update_status_bar_model(self) -> None:
        """Update status bar with current model name from session."""
        try:
            agent = getattr(self._session, "_agent", None)
            llm = getattr(agent, "llm", None)
            model = getattr(llm, "model", "")
            if model:
                self._status_bar.set_model_name(str(model))
                self._invalidate()
        except Exception:
            logger.exception("Failed to update status bar model name")

    def _enter_selection_mode(self) -> None:
        """Enter model selection menu mode."""
        # Get current level and llm_levels
        # Default to MID if level is not set
        current_level = "MID"
        llm_levels = None
        try:
            agent_level = self._session._agent.level
            if agent_level:
                current_level = agent_level
            llm_levels = self._session._agent.llm_levels
        except Exception:
            pass

        # Setup selection menu
        def on_confirm(value: str) -> None:
            self._exit_selection_mode()
            self._switch_model_level(value)
            self._refresh_layers()

        def on_cancel() -> None:
            self._exit_selection_mode()
            self._renderer.append_system_message("Model switch cancelled.")
            self._refresh_layers()

        title, options = build_model_level_options(
            current_level=current_level,
            llm_levels=llm_levels,
        )
        ok = self._selection_ui.set_options(
            title=title,
            options=options,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
        )
        if not ok:
            self._renderer.append_system_message(
                "No model options available.",
                is_error=True,
            )
            self._refresh_layers()
            return
        self._selection_ui.refresh()

        # Enter selection mode
        self._ui_mode = UIMode.SELECTION
        self._sync_focus_for_mode()
        self._invalidate()

    def _exit_selection_mode(self) -> None:
        """Exit selection menu mode."""
        self._ui_mode = UIMode.NORMAL
        self._selection_ui.clear()
        self._sync_focus_for_mode()
        self._invalidate()

    def _handle_selection_result(self, result: SelectionResult | None) -> None:
        """Handle selection result."""
        if result is None:
            return

        if not result.confirmed:
            self._exit_selection_mode()
            self._renderer.append_system_message("Model switch cancelled.")
            return

        # The callback handles the actual switch
        self._exit_selection_mode()

    async def _append_usage_snapshot(self) -> None:
        usage = await self._session.get_usage()
        include_cost = bool(getattr(self._session._agent, "include_cost", False))
        prompt_new_tokens = max(
            usage.total_prompt_tokens - usage.total_prompt_cached_tokens,
            0,
        )

        lines = [
            "Token Usage",
            f"- total: {usage.total_tokens:,}",
            f"- entries: {usage.entry_count}",
            f"- prompt: {usage.total_prompt_tokens:,}",
            f"- prompt_cached: {usage.total_prompt_cached_tokens:,}",
            f"- prompt_new: {prompt_new_tokens:,}",
            f"- completion: {usage.total_completion_tokens:,}",
        ]

        if include_cost:
            lines.extend(
                [
                    f"- prompt_cost: ${usage.total_prompt_cost:.4f}",
                    f"- completion_cost: ${usage.total_completion_cost:.4f}",
                    f"- total_cost: ${usage.total_cost:.4f}",
                ]
            )
        self._renderer.append_system_message("\n".join(lines))

    async def _append_context_snapshot(self) -> None:
        info = await self._session.get_context_info()
        utilization = float(getattr(info, "utilization_percent", 0.0))
        context_limit = getattr(info, "context_limit_tokens", None)
        used_tokens = getattr(info, "used_tokens", None)

        lines = [
            "Context Usage",
            f"- utilization: {utilization:.1f}%",
            f"- left: {max(0.0, 100.0 - utilization):.1f}%",
        ]
        if used_tokens is not None:
            lines.append(f"- used_tokens: {used_tokens}")
        if context_limit is not None:
            lines.append(f"- context_limit_tokens: {context_limit}")

        self._renderer.append_system_message("\n".join(lines))

    def _refresh_layers(self) -> None:
        self._sync_focus_for_mode()
        self._render_dirty = True

    async def _drain_history_async(self) -> None:
        pending = self._pending_history_entries()
        if not pending:
            return
        group = render_history_group(
            console,
            pending,
            terminal_width=self._terminal_width(),
            render_markdown_to_plain=render_markdown_to_plain,
        )
        if group is None:
            return

        def _print() -> None:
            width = self._terminal_width()
            out = sys.__stdout__ or sys.stdout
            scrollback_console = Console(
                file=out,
                force_terminal=True,
                width=width,
            )
            print_history_group_sync(scrollback_console, group)

        await run_in_terminal(_print, in_executor=False)

    def _drain_history_sync(self) -> None:
        pending = self._pending_history_entries()
        if not pending:
            return
        group = render_history_group(
            console,
            pending,
            terminal_width=self._terminal_width(),
            render_markdown_to_plain=render_markdown_to_plain,
        )
        if group is None:
            return
        print_history_group_sync(console, group)

    def _pending_history_entries(self) -> list[HistoryEntry]:
        entries = self._renderer.history_entries()
        if self._printed_history_index >= len(entries):
            return []
        pending = entries[self._printed_history_index :]
        self._printed_history_index = len(entries)
        return pending

    async def _ui_tick(self) -> None:
        try:
            while not self._closing:
                self._renderer.tick_progress()
                self._loading_frame += 2
                # busy 时每隔几秒换一个短语，让 loading 更“活”
                if self._busy and time.monotonic() >= self._fallback_phrase_refresh_at:
                    self._fallback_loading_phrase = (
                        random.choice(DEFAULT_STATUS_PHRASES)
                        if DEFAULT_STATUS_PHRASES
                        else "Thinking…"
                    )
                    self._fallback_phrase_refresh_at = time.monotonic() + 3.0
                    self._render_dirty = True

                await self._drain_history_async()

                # 检查动画器是否需要刷新
                if self._animator.consume_dirty():
                    self._render_dirty = True
                if self._tool_result_animator.consume_dirty():
                    self._render_dirty = True

                if self._tool_result_flash_active:
                    until = self._tool_result_flash_until_monotonic
                    if until is None or time.monotonic() >= until:
                        self._schedule_background(self._stop_tool_result_flash())
                    else:
                        self._render_dirty = True

                loading_line = self._renderer.loading_line().strip()
                loading_changed = loading_line != self._last_loading_line
                if loading_changed:
                    self._last_loading_line = loading_line
                    self._render_dirty = True

                if self._renderer.has_running_tools():
                    # Keep repainting for breathing animation.
                    self._render_dirty = True
                elif self._busy:
                    self._render_dirty = True


                if self._render_dirty:
                    self._invalidate()
                    self._render_dirty = False

                # 动态帧率：busy/动画时 12fps，idle 时 4fps
                fast = (
                    self._busy
                    or self._animator.is_active
                    or self._renderer.has_running_tools()
                )
                sleep_s = 1 / 12 if fast else 1 / 4
                await asyncio.sleep(sleep_s)
        except asyncio.CancelledError:
            return

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
        merged = f"[{mode}] {base}"
        return [("class:status", fit_single_line(merged, width - 1))]

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

    def _rich_text_to_pt_fragments(self, renderable) -> list[tuple[str, str]]:
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

    def _rich_style_to_pt(self, rich_style) -> str:
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

    def _invalidate(self) -> None:
        if self._app is None:
            return
        self._app.invalidate()

    def _exit_app(self) -> None:
        self._closing = True
        self._session.run_controller.clear()
        if self._stream_task is not None:
            self._stream_task.cancel()
        if self._app is not None:
            self._app.exit(result=None)

    async def run(self) -> None:
        if self._app is None:
            return

        self._refresh_layers()
        self._ui_tick_task = asyncio.create_task(
            self._ui_tick(),
            name="terminal-ui-tick",
        )
        try:
            # Ensure scrollback prints don't mix with the inline (full_screen=False) UI.
            with patch_stdout(raw=True):
                await self._app.run_async()
        finally:
            self._closing = True
            if self._ui_tick_task is not None:
                self._ui_tick_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._ui_tick_task
            self._renderer.close()
