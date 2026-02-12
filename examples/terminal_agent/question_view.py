from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import TextArea

_CANCEL_MESSAGE = "user reject answer this question."
_CHAT_MESSAGE = "Chat about this"
_PREVIEW_CUSTOM_INPUT_MAX_CHARS = 15


@dataclass(frozen=True, slots=True)
class OptionState:
    """单个候选项状态。"""

    label: str
    description: str


@dataclass(slots=True)
class QuestionState:
    """单个问题状态。"""

    question: str
    header: str
    options: list[OptionState]
    multi_select: bool
    selected_indices: set[int] = field(default_factory=set)
    custom_input: str = ""
    preset_answer: str = ""
    is_answered: bool = False


@dataclass(slots=True)
class QuestionUIState:
    """问答 UI 整体状态。"""

    questions: list[QuestionState] = field(default_factory=list)
    current_question_index: int = 0
    current_option_index: int = 0
    is_custom_input_active: bool = False
    is_preview_mode: bool = False
    preview_option_index: int = 0


@dataclass(frozen=True, slots=True)
class QuestionAction:
    """问答交互动作输出。"""

    kind: str
    message: str = ""


class AskUserQuestionUI:
    """AskUserQuestion 交互式 UI 组件。"""

    def __init__(self) -> None:
        self._state = QuestionUIState()

        self._question_tabs_control = FormattedTextControl(text=self._tabs_fragments)
        self._question_content_control = FormattedTextControl(text=self._question_content_fragments)
        self._options_control = FormattedTextControl(text=self._options_fragments, focusable=True)
        self._special_options_control = FormattedTextControl(text=self._special_options_fragments)
        self._preview_control = FormattedTextControl(text=self._preview_fragments)

        self._custom_input_area = TextArea(
            text="",
            multiline=False,
            prompt="    > ",
            wrap_lines=False,
            style="class:question.custom_input",
        )
        self._custom_input_area.window.style = "class:question.custom_input"

        @self._custom_input_area.buffer.on_text_changed.add_handler
        def _sync_custom_input(buffer) -> None:
            question = self._active_question()
            if question is None:
                return
            question.custom_input = buffer.text
            if not question.multi_select and question.custom_input.strip():
                question.selected_indices.clear()
            question.is_answered = self._question_is_answered(question)

        self._custom_input_area_container = ConditionalContainer(
            content=HSplit(
                [
                    Window(
                        content=FormattedTextControl(
                            text=[
                                (
                                    "class:question.custom_input.border",
                                    "    ┌──────────────────────────────────────────────┐",
                                )
                            ]
                        ),
                        dont_extend_height=True,
                        height=1,
                    ),
                    self._custom_input_area,
                    Window(
                        content=FormattedTextControl(
                            text=[
                                (
                                    "class:question.custom_input.border",
                                    "    └──────────────────────────────────────────────┘",
                                )
                            ]
                        ),
                        dont_extend_height=True,
                        height=1,
                    ),
                ]
            ),
            filter=Condition(self._show_custom_input),
        )

        self._question_content_window = Window(
            content=self._question_content_control,
            wrap_lines=True,
            dont_extend_height=False,
            style="class:question.body",
        )
        self._options_window = Window(
            content=self._options_control,
            wrap_lines=True,
            dont_extend_height=False,
            style="class:question.body",
        )
        self._special_options_window = Window(
            content=self._special_options_control,
            wrap_lines=True,
            dont_extend_height=False,
            style="class:question.body",
        )

        self._question_mode_container = HSplit(
            [
                self._question_content_window,
                self._options_window,
                self._custom_input_area_container,
                self._special_options_window,
            ]
        )
        self._preview_window = Window(
            content=self._preview_control,
            wrap_lines=True,
            dont_extend_height=False,
            style="class:question.body",
        )

        self._root = HSplit(
            [
                Window(
                    content=self._question_tabs_control,
                    height=1,
                    dont_extend_height=True,
                    style="class:question.tabs",
                ),
                Window(height=1, char="─", style="class:question.divider"),
                ConditionalContainer(
                    content=self._question_mode_container,
                    filter=Condition(lambda: not self._state.is_preview_mode),
                ),
                ConditionalContainer(
                    content=self._preview_window,
                    filter=Condition(lambda: self._state.is_preview_mode),
                ),
            ]
        )

    @property
    def container(self) -> HSplit:
        return self._root

    def has_questions(self) -> bool:
        return bool(self._state.questions)

    def set_questions(self, questions: list[dict[str, Any]] | None) -> bool:
        parsed_questions: list[QuestionState] = []
        for idx, raw in enumerate(questions or [], start=1):
            if not isinstance(raw, dict):
                continue
            question_text = str(raw.get("question", "")).strip()
            header = str(raw.get("header", f"Q{idx}")).strip()[:12] or f"Q{idx}"
            raw_options = raw.get("options", [])
            options: list[OptionState] = []
            if isinstance(raw_options, list):
                for option in raw_options:
                    if not isinstance(option, dict):
                        continue
                    label = str(option.get("label", "")).strip()
                    if not label:
                        continue
                    description = str(option.get("description", "")).strip()
                    options.append(OptionState(label=label, description=description))
            if not options:
                options.append(OptionState(label="Continue", description="Proceed with default decision."))
            parsed_questions.append(
                QuestionState(
                    question=question_text,
                    header=header,
                    options=options,
                    multi_select=bool(raw.get("multiSelect", False)),
                )
            )

        self._state.questions = parsed_questions
        self._state.current_question_index = 0
        self._state.current_option_index = 0
        self._state.is_custom_input_active = False
        self._state.is_preview_mode = False
        self._state.preview_option_index = 0
        self._sync_custom_input_buffer()
        return bool(self._state.questions)

    def clear(self) -> None:
        self._state = QuestionUIState()
        self._sync_custom_input_buffer()

    def focus_target(self) -> Any:
        if self._state.is_custom_input_active:
            return self._custom_input_area.window
        return self._options_window

    @property
    def custom_input_window(self) -> Window:
        return self._custom_input_area.window

    def move_option(self, delta: int) -> None:
        if not self._state.questions:
            return
        if self._state.is_preview_mode:
            self._state.preview_option_index = (self._state.preview_option_index + delta) % 2
            return

        question = self._active_question()
        if question is None:
            return

        max_index = len(question.options) + 1
        next_index = self._state.current_option_index + delta
        self._state.current_option_index = max(0, min(max_index, next_index))
        if self._state.current_option_index != len(question.options):
            self._deactivate_custom_input()

    def prev_question(self) -> None:
        self._switch_question(-1)

    def next_question(self) -> None:
        self._switch_question(1)

    def focus_submit(self) -> None:
        if not self._state.questions:
            return
        self._state.is_preview_mode = True
        self._state.preview_option_index = 0
        self._deactivate_custom_input()

    def toggle_current_selection(self) -> None:
        if not self._state.questions or self._state.is_preview_mode:
            return

        question = self._active_question()
        if question is None:
            return

        index = self._state.current_option_index
        if index >= len(question.options):
            return

        if question.multi_select:
            if index in question.selected_indices:
                question.selected_indices.remove(index)
            else:
                question.selected_indices.add(index)
            question.preset_answer = ""
        else:
            question.selected_indices = {index}
            question.custom_input = ""
            question.preset_answer = ""
            if self._state.current_question_index >= 0:
                self._sync_custom_input_buffer()
        question.is_answered = self._question_is_answered(question)

    def set_custom_input(self, text: str) -> None:
        question = self._active_question()
        if question is None:
            return
        question.custom_input = text
        if not question.multi_select and question.custom_input.strip():
            question.selected_indices.clear()
            question.preset_answer = ""
        question.is_answered = self._question_is_answered(question)
        if self._custom_input_area.text != text:
            self._custom_input_area.buffer.document = Document(text=text, cursor_position=len(text))

    def handle_enter(self) -> QuestionAction | None:
        if not self._state.questions:
            return None

        if self._state.is_preview_mode:
            if self._state.preview_option_index == 0:
                return QuestionAction(kind="submit", message=self.build_answers_message())
            self._state.is_preview_mode = False
            self._state.preview_option_index = 0
            self._deactivate_custom_input()
            return None

        question = self._active_question()
        if question is None:
            return None

        current_index = self._state.current_option_index
        custom_index = len(question.options)
        chat_index = len(question.options) + 1

        if current_index == custom_index:
            if not self._state.is_custom_input_active:
                self._state.is_custom_input_active = True
                if not question.multi_select:
                    question.selected_indices.clear()
                    question.preset_answer = ""
                self._sync_custom_input_buffer()
                return None
            if not question.custom_input.strip():
                return None
            if not question.multi_select:
                question.preset_answer = ""
            question.is_answered = True
            self._advance_question_or_preview()
            return None

        if current_index < len(question.options):
            if question.multi_select:
                if not question.selected_indices:
                    question.selected_indices.add(current_index)
            else:
                question.selected_indices = {current_index}
                question.custom_input = ""
                question.preset_answer = ""
                self._sync_custom_input_buffer()

            if not self._question_is_answered(question):
                return None
            question.is_answered = True
            self._advance_question_or_preview()
            return None

        if current_index == chat_index:
            question.preset_answer = _CHAT_MESSAGE
            if not question.multi_select:
                question.selected_indices.clear()
                question.custom_input = ""
                self._sync_custom_input_buffer()
            question.is_answered = True
            self._advance_question_or_preview()
            return None

        return None

    def handle_escape(self) -> QuestionAction:
        return QuestionAction(kind="cancel", message=_CANCEL_MESSAGE)

    def build_answers_message(self) -> str:
        lines = ["User answered Comate's questions:"]
        for idx, question in enumerate(self._state.questions, start=1):
            answer = self._question_answer_summary(question, for_preview=False)
            lines.append(f"- {idx}. {question.header}: {answer}")
        lines.append("")
        return "\n".join(lines)

    def _switch_question(self, delta: int) -> None:
        if not self._state.questions:
            return

        if self._state.is_preview_mode:
            self._state.is_preview_mode = False

        count = len(self._state.questions)
        next_index = (self._state.current_question_index + delta) % count
        self._state.current_question_index = next_index
        self._state.current_option_index = 0
        self._deactivate_custom_input()
        self._sync_custom_input_buffer()

    def _advance_question_or_preview(self) -> None:
        if self._state.current_question_index < len(self._state.questions) - 1:
            self._state.current_question_index += 1
            self._state.current_option_index = 0
            self._deactivate_custom_input()
            self._sync_custom_input_buffer()
            return
        self.focus_submit()

    def _active_question(self) -> QuestionState | None:
        if not self._state.questions:
            return None
        index = self._state.current_question_index
        if index < 0 or index >= len(self._state.questions):
            return None
        return self._state.questions[index]

    def _question_is_answered(self, question: QuestionState) -> bool:
        return (
            bool(question.selected_indices)
            or bool(question.custom_input.strip())
            or bool(question.preset_answer.strip())
        )

    @staticmethod
    def _truncate_preview_text(content: str, *, max_chars: int = _PREVIEW_CUSTOM_INPUT_MAX_CHARS) -> str:
        text = content.strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}..."

    def _question_answer_summary(self, question: QuestionState, *, for_preview: bool) -> str:
        selected_labels = [
            question.options[idx].label
            for idx in sorted(question.selected_indices)
            if 0 <= idx < len(question.options)
        ]
        custom_input = question.custom_input.strip()
        if custom_input:
            if for_preview:
                selected_labels.append(self._truncate_preview_text(custom_input))
            else:
                selected_labels.append(custom_input)
        preset_answer = question.preset_answer.strip()
        if preset_answer:
            selected_labels.append(preset_answer)
        if not selected_labels:
            return "未选择（默认决策）"
        return ", ".join(selected_labels)

    def _show_custom_input(self) -> bool:
        return self._state.is_custom_input_active and not self._state.is_preview_mode

    def _deactivate_custom_input(self) -> None:
        self._state.is_custom_input_active = False

    def _sync_custom_input_buffer(self) -> None:
        question = self._active_question()
        if question is None:
            if self._custom_input_area.text:
                self._custom_input_area.buffer.document = Document(text="", cursor_position=0)
            return
        text = question.custom_input
        if self._custom_input_area.text == text:
            return
        self._custom_input_area.buffer.document = Document(text=text, cursor_position=len(text))

    def _tabs_fragments(self) -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = [("class:question.tabs.nav", "←  ")]
        for idx, question in enumerate(self._state.questions):
            if idx > 0:
                fragments.append(("class:question.tabs", "  "))
            is_active = not self._state.is_preview_mode and idx == self._state.current_question_index
            is_answered = self._question_is_answered(question)
            marker = "☒" if is_answered else "☐"
            style = "class:question.tab.active" if is_active else "class:question.tab"
            fragments.append((style, f"{marker} {question.header}"))

        if self._state.questions:
            fragments.append(("class:question.tabs", "  "))
        submit_style = "class:question.tab.active" if self._state.is_preview_mode else "class:question.tab.submit"
        submit_marker = "✔" if all(self._question_is_answered(q) for q in self._state.questions) else "○"
        fragments.append((submit_style, f"{submit_marker} Submit"))
        fragments.append(("class:question.tabs.nav", "  →"))
        return fragments

    def _question_content_fragments(self) -> list[tuple[str, str]]:
        question = self._active_question()
        if question is None:
            return [("class:question.title", "")]
        title = question.question or f"请选择 {question.header}"
        return [
            ("class:question.title", title),
            ("", "\n"),
            (
                "class:question.hint",
                "多选: Space 选择 | Enter 确认下一题 | Tab 进入提交预览 | Esc 取消",
            ),
            ("", "\n"),
        ]

    def _options_fragments(self) -> list[tuple[str, str]]:
        question = self._active_question()
        if question is None:
            return [("class:question.body", "")]

        fragments: list[tuple[str, str]] = []
        for idx, option in enumerate(question.options):
            cursor = "->" if idx == self._state.current_option_index else "  "
            selected = idx in question.selected_indices
            if selected:
                marker = "☒" if question.multi_select else "●"
            else:
                marker = f"{idx + 1}."

            line_style = "class:question.option.cursor" if idx == self._state.current_option_index else "class:question.option"
            if selected:
                line_style = "class:question.option.selected"

            fragments.append((line_style, f"{cursor} {marker} {option.label}\n"))
            if option.description:
                fragments.append(("class:question.option.description", f"   {option.description}\n"))

        return fragments

    def _special_options_fragments(self) -> list[tuple[str, str]]:
        question = self._active_question()
        if question is None:
            return [("class:question.body", "")]

        custom_index = len(question.options)
        chat_index = len(question.options) + 1

        custom_cursor = "->" if self._state.current_option_index == custom_index else "  "
        custom_style = (
            "class:question.option.cursor"
            if self._state.current_option_index == custom_index
            else "class:question.option"
        )
        custom_suffix = " (editing)" if self._state.is_custom_input_active else ""

        chat_cursor = "->" if self._state.current_option_index == chat_index else "  "
        chat_selected = question.preset_answer.strip() == _CHAT_MESSAGE
        if self._state.current_option_index == chat_index:
            chat_style = "class:question.option.cursor"
        elif chat_selected:
            chat_style = "class:question.option.selected"
        else:
            chat_style = "class:question.option"
        chat_marker = "☒" if chat_selected else f"{chat_index + 1}."

        return [
            (custom_style, f"{custom_cursor} {custom_index + 1}. Type something.{custom_suffix}\n"),
            (chat_style, f"{chat_cursor} {chat_marker} Chat about this\n"),
        ]

    def _preview_fragments(self) -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = [
            ("class:question.preview.title", "Review your answers\n\n"),
        ]
        for question in self._state.questions:
            answer = self._question_answer_summary(question, for_preview=True)
            fragments.append(("class:question.preview.question", f"● {question.question or question.header}\n"))
            fragments.append(("class:question.preview.answer", f"  -> {answer}\n\n"))

        fragments.append(("class:question.preview.title", "Ready to submit your answers?\n\n"))

        submit_cursor = "->" if self._state.preview_option_index == 0 else "  "
        cancel_cursor = "->" if self._state.preview_option_index == 1 else "  "
        submit_style = (
            "class:question.option.cursor"
            if self._state.preview_option_index == 0
            else "class:question.option"
        )
        cancel_style = (
            "class:question.option.cursor"
            if self._state.preview_option_index == 1
            else "class:question.option"
        )

        fragments.append((submit_style, f"{submit_cursor} 1. Submit answers\n"))
        fragments.append((cancel_style, f"{cancel_cursor} 2. Cancel\n"))
        return fragments
