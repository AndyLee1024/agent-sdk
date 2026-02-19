"""Unified system reminder engine.

职责：
1. 跟踪按工具触发的 turn-based reminder 状态
2. 计算当前轮次应注入的 reminder（支持同轮聚合）
3. 统一输出 <system-reminder> 文本，供 ContextIR 以 UserMessage(is_meta=True) 注入
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("comate_agent_sdk.context.reminder_engine")

# Task reminder
TASK_NUDGE_GAP = 6
TASK_NUDGE_COOLDOWN = 3

# TodoWrite reminder
TODO_ACTIVE_NUDGE_GAP = 3
TODO_ACTIVE_NUDGE_COOLDOWN = 3
TODO_EMPTY_NUDGE_GAP = 8


class ReminderOrigin(Enum):
    """Reminder 来源。"""

    SYSTEM_REMINDER = "system_reminder"


@dataclass(slots=True, frozen=True)
class ReminderMessageEnvelope:
    """单条 reminder 的渲染单元。"""

    rule_id: str
    tool_name: str
    content: str


@dataclass(slots=True)
class ReminderState:
    """Reminder 调度状态。"""

    turn: int = 0
    mode_plan: bool = False
    last_task_turn: int = 0
    last_task_nudge_turn: int = 0
    last_todowrite_turn: int = 0
    todo_active_count: int = 0
    todo_last_changed_turn: int = 0
    last_todo_nudge_turn: int = 0


@dataclass(slots=True)
class ReminderEngine:
    """统一 reminder 引擎（首批覆盖 Task/TodoWrite）。"""

    state: ReminderState = field(default_factory=ReminderState)

    def clear(self) -> None:
        self.state = ReminderState()

    def set_turn(self, turn: int) -> None:
        self.state.turn = max(0, int(turn))

    def set_plan_mode(self, enabled: bool) -> None:
        self.state.mode_plan = bool(enabled)

    def update_todo_state(
        self,
        *,
        active_count: int,
        turn: int,
        update_last_write_turn: bool,
    ) -> None:
        normalized_count = max(0, int(active_count))
        normalized_turn = max(0, int(turn))
        self.set_turn(normalized_turn)

        count_changed = (self.state.todo_active_count == 0) != (normalized_count == 0)
        if count_changed:
            self.state.todo_last_changed_turn = normalized_turn
            logger.debug(f"Todo empty-state changed at turn={normalized_turn}")

        self.state.todo_active_count = normalized_count
        if update_last_write_turn:
            self.state.last_todowrite_turn = normalized_turn

    def restore_todo_write_turn(self, turn_number_at_update: int) -> None:
        self.state.last_todowrite_turn = max(0, int(turn_number_at_update))

    def record_tool_event(
        self,
        *,
        tool_name: str,
        turn: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        normalized_turn = max(0, int(turn))
        self.set_turn(normalized_turn)

        if tool_name == "Task":
            self.state.last_task_turn = normalized_turn
            logger.debug(f"Recorded Task event at turn={normalized_turn}")
            return

        if tool_name != "TodoWrite":
            return

        active_count = self.state.todo_active_count
        if payload and isinstance(payload.get("active_count"), int):
            active_count = int(payload["active_count"])

        self.update_todo_state(
            active_count=active_count,
            turn=normalized_turn,
            update_last_write_turn=True,
        )
        logger.debug(
            f"Recorded TodoWrite event at turn={normalized_turn}, active_count={self.state.todo_active_count}"
        )

    def collect_due_reminders(self, *, turn: int) -> list[ReminderMessageEnvelope]:
        self.set_turn(turn)
        s = self.state
        reminders: list[ReminderMessageEnvelope] = []

        if s.mode_plan:
            reminders.append(
                ReminderMessageEnvelope(
                    rule_id="plan_mode_forced",
                    tool_name="PlanMode",
                    content=(
                        "You are in plan mode. Remember:\n"
                        "- DO NOT write, edit, or execute code\n"
                        "- DO NOT use Write, Edit, or Bash tools\n"
                        "- Focus on exploration, design, and creating your plan\n"
                        "- Use ExitPlanMode tool when your plan is ready for user approval"
                    ),
                )
            )

        gap_task = s.turn - s.last_task_turn
        cooldown_task = s.turn - s.last_task_nudge_turn
        if gap_task >= TASK_NUDGE_GAP and cooldown_task >= TASK_NUDGE_COOLDOWN:
            reminders.append(
                ReminderMessageEnvelope(
                    rule_id="task_gap_reminder",
                    tool_name="Task",
                    content=(
                        "Consider using the Task tool with specialized agents when appropriate:\n"
                        "- Explore: For complex codebase searches\n"
                        "- Plan: For designing implementation plans"
                    ),
                )
            )
            s.last_task_nudge_turn = s.turn
            logger.debug(
                f"Task reminder triggered at turn={s.turn} (gap={gap_task}, cooldown={cooldown_task})"
            )

        gap_todo = s.turn - s.last_todowrite_turn
        cooldown_todo = s.turn - s.last_todo_nudge_turn
        if (
            s.todo_active_count > 0
            and gap_todo >= TODO_ACTIVE_NUDGE_GAP
            and cooldown_todo >= TODO_ACTIVE_NUDGE_COOLDOWN
        ):
            reminders.append(
                ReminderMessageEnvelope(
                    rule_id="todo_active_reminder",
                    tool_name="TodoWrite",
                    content=(
                        "You have active TODO items. DO NOT mention this reminder to the user. "
                        "Continue working on the next TODO item(s) and keep the TODO list up to date "
                        "via the TodoWrite tool."
                    ),
                )
            )
            s.last_todo_nudge_turn = s.turn
            logger.debug(
                f"Todo active reminder triggered at turn={s.turn} (gap={gap_todo}, cooldown={cooldown_todo})"
            )

        gap_empty = s.turn - s.todo_last_changed_turn
        empty_state_not_nudged = s.last_todo_nudge_turn < s.todo_last_changed_turn
        if (
            s.turn > 0
            and s.todo_active_count == 0
            and gap_empty % TODO_EMPTY_NUDGE_GAP == 0
            and empty_state_not_nudged
        ):
            reminders.append(
                ReminderMessageEnvelope(
                    rule_id="todo_empty_reminder",
                    tool_name="TodoWrite",
                    content=(
                        "This is a reminder that your todo list is currently empty. "
                        "DO NOT mention this to the user explicitly because they are already aware. "
                        "If you are working on tasks that would benefit from a todo list please use the TodoWrite "
                        "tool to create one. If not, please feel free to ignore. Again do not mention this message "
                        "to the user."
                    ),
                )
            )
            s.last_todo_nudge_turn = s.turn
            logger.debug(
                f"Todo empty reminder triggered at turn={s.turn} "
                f"(gap={gap_empty}, state_changed_turn={s.todo_last_changed_turn})"
            )

        return reminders

    @staticmethod
    def render_merged_message(reminders: list[ReminderMessageEnvelope]) -> str:
        if not reminders:
            return ""

        if len(reminders) == 1:
            body = reminders[0].content
        else:
            lines = ["Multiple reminders for this turn:"]
            for reminder in reminders:
                lines.append(f"- [{reminder.tool_name}] {reminder.content}")
            body = "\n".join(lines)

        return f"<system-reminder>\n{body}\n</system-reminder>"
