"""通用选择菜单 UI 组件。

提供简单的单选菜单界面，支持上下选择、Enter 确认、Esc 取消。
可用于模型切换、配置选择等二级选择场景。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl


@dataclass(frozen=True, slots=True)
class SelectionOption:
    """单个选项。"""

    value: str
    label: str
    description: str = ""


@dataclass(slots=True)
class SelectionMenuState:
    """选择菜单状态。"""

    title: str = ""
    options: list[SelectionOption] = field(default_factory=list)
    selected_index: int = 0
    is_confirmed: bool = False
    is_cancelled: bool = False


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """选择结果。"""

    confirmed: bool
    value: str | None = None
    label: str | None = None


class SelectionMenuUI:
    """通用选择菜单 UI 组件。

    适用于简单的单选场景，如：
    - 模型级别切换 (LOW/MID/HIGH)
    - 配置选项选择
    - 快捷操作菜单

    按键：
    - ↑/↓ 或 k/j: 移动选择
    - Enter: 确认选择
    - Esc: 取消
    """

    def __init__(self) -> None:
        self._state = SelectionMenuState()
        self._on_confirm: Callable[[str], None] | None = None
        self._on_cancel: Callable[[], None] | None = None

        self._title_control = FormattedTextControl(text=self._title_fragments)
        self._options_control = FormattedTextControl(text=self._options_fragments)

        self._title_window = Window(
            content=self._title_control,
            height=1,
            dont_extend_height=True,
            style="class:selection.title",
        )
        self._options_window = Window(
            content=self._options_control,
            wrap_lines=True,
            dont_extend_height=False,
            style="class:selection.body",
        )

        self._root = HSplit(
            [
                self._title_window,
                Window(height=1, char="─", style="class:selection.divider"),
                self._options_window,
            ]
        )

    @property
    def container(self) -> HSplit:
        return self._root

    def set_options(
        self,
        title: str,
        options: list[dict[str, str]],
        on_confirm: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> bool:
        """设置选项。

        Args:
            title: 菜单标题
            options: 选项列表，每个选项为 {"value": "xxx", "label": "显示文本", "description": "描述"}
            on_confirm: 确认回调，参数为选中的 value
            on_cancel: 取消回调

        Returns:
            是否成功设置（至少有一个有效选项）
        """
        parsed_options: list[SelectionOption] = []
        for opt in options:
            if not isinstance(opt, dict):
                continue
            value = str(opt.get("value", "")).strip()
            label = str(opt.get("label", "")).strip()
            if not value or not label:
                continue
            description = str(opt.get("description", "")).strip()
            parsed_options.append(SelectionOption(value=value, label=label, description=description))

        if not parsed_options:
            return False

        self._state.title = title
        self._state.options = parsed_options
        self._state.selected_index = 0
        self._state.is_confirmed = False
        self._state.is_cancelled = False
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel
        return True

    def clear(self) -> None:
        """清除状态。"""
        self._state = SelectionMenuState()
        self._on_confirm = None
        self._on_cancel = None

    def has_options(self) -> bool:
        """是否有选项。"""
        return bool(self._state.options)

    def move_selection(self, delta: int) -> None:
        """移动选择。"""
        if not self._state.options:
            return
        count = len(self._state.options)
        self._state.selected_index = (self._state.selected_index + delta) % count

    def confirm(self) -> SelectionResult | None:
        """确认当前选择。

        Returns:
            选择结果，如果没有选项则返回 None
        """
        if not self._state.options:
            return None

        option = self._state.options[self._state.selected_index]
        self._state.is_confirmed = True

        if self._on_confirm:
            self._on_confirm(option.value)

        return SelectionResult(confirmed=True, value=option.value, label=option.label)

    def cancel(self) -> SelectionResult:
        """取消选择。

        Returns:
            取消结果
        """
        self._state.is_cancelled = True

        if self._on_cancel:
            self._on_cancel()

        return SelectionResult(confirmed=False)

    def get_selected(self) -> SelectionOption | None:
        """获取当前选中的选项。"""
        if not self._state.options:
            return None
        return self._state.options[self._state.selected_index]

    def _title_fragments(self) -> list[tuple[str, str]]:
        if not self._state.title:
            return [("class:selection.title", "  Select an option")]
        return [("class:selection.title", f"  {self._state.title}")]

    def _options_fragments(self) -> list[tuple[str, str]]:
        if not self._state.options:
            return [("class:selection.body", "  No options available")]

        fragments: list[tuple[str, str]] = []
        for idx, option in enumerate(self._state.options):
            is_selected = idx == self._state.selected_index
            cursor = "▶" if is_selected else " "
            marker = "●" if is_selected else "○"

            line_style = "class:selection.option.selected" if is_selected else "class:selection.option"
            desc_style = "class:selection.description.selected" if is_selected else "class:selection.description"

            # 主选项行
            fragments.append((line_style, f"  {cursor} {marker} {option.label}"))
            fragments.append(("", "\n"))

            # 描述行（如果有）
            if option.description:
                fragments.append((desc_style, f"      {option.description}"))
                fragments.append(("", "\n"))

        # 底部提示
        fragments.append(("", "\n"))
        fragments.append(
            ("class:selection.hint", "  ↑/↓ or k/j: Move  Enter: Confirm  Esc: Cancel")
        )
        return fragments

    def refresh(self) -> None:
        """刷新显示（触发重绘）。"""
        self._title_control.text = self._title_fragments
        self._options_control.text = self._options_fragments

    def focus_target(self) -> Window:
        """获取焦点目标窗口。"""
        return self._options_window

    def __pt_formatted_text__(self) -> list[tuple[str, str]]:
        """支持直接作为 formatted text 使用。"""
        return self._options_fragments()


# 便捷函数：创建模型级别选择菜单
def create_model_level_menu(
    current_level: str | None,
    llm_levels: dict[str, Any] | None,
    on_confirm: Callable[[str], None],
    on_cancel: Callable[[], None],
) -> SelectionMenuUI:
    """创建模型级别选择菜单。

    Args:
        current_level: 当前级别 (LOW/MID/HIGH)
        llm_levels: LLM 级别配置字典，包含实际的模型实例
        on_confirm: 确认回调
        on_cancel: 取消回调

    Returns:
        配置好的 SelectionMenuUI 实例
    """
    ui = SelectionMenuUI()

    # 从 llm_levels 获取实际模型名称
    def get_model_name(level: str) -> str:
        if llm_levels and level in llm_levels:
            llm = llm_levels[level]
            model = getattr(llm, "model", None)
            if model:
                return str(model)
        return "unknown"

    options = [
        {
            "value": "LOW",
            "label": "LOW  - Fast & Cheap",
            "description": f"{get_model_name('LOW')}",
        },
        {
            "value": "MID",
            "label": "MID  - Balanced",
            "description": f"{get_model_name('MID')}",
        },
        {
            "value": "HIGH",
            "label": "HIGH - Best Quality",
            "description": f"{get_model_name('HIGH')}",
        },
    ]

    # 标记当前选中的级别
    if current_level:
        for opt in options:
            if opt["value"] == current_level.upper():
                opt["label"] = f"{opt['label']}  (current)"
                break

    ui.set_options(
        title="Select Model Level",
        options=options,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
    )
    return ui
