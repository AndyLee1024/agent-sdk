"""Turn-based nudge system — 节流提醒机制

将 subagent/todo 提醒逻辑从 ContextIR 抽取出来，实现纯函数 render_reminders() + 最小状态 NudgeState。

核心设计：
1. NudgeState：所有节流状态字段（turn、last_*_turn、cooldown 标记）
2. render_reminders()：纯函数，读取 NudgeState → 生成提醒文本（追加到最后一条 UserMessage）
3. update_nudge_*：状态更新函数，由 tool_exec.py 调用
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("comate_agent_sdk.context.nudge")

# Nudge 节流规则（后续可配置化）
SUBAGENT_NUDGE_GAP = 6  # subagent 使用间隔超过 6 轮时触发
SUBAGENT_NUDGE_COOLDOWN = 3  # 上次 nudge 后需等待 3 轮
TODO_ACTIVE_NUDGE_GAP = 3  # todo 更新后超过 3 轮触发温和提醒
TODO_ACTIVE_NUDGE_COOLDOWN = 3  # 上次 nudge 后需等待 3 轮
TODO_EMPTY_NUDGE_GAP = 8  # 空 todo 每 8 轮提醒一次


@dataclass
class NudgeState:
    """Turn-based nudge 状态

    职责：跟踪 subagent 使用、todo 写入、提醒 cooldown
    不包含：todo 列表内容（仍在 ContextIR._todo_state），只跟踪 turn 节流
    """

    turn: int = 0  # 当前轮次（与 ContextIR._turn_number 同步）
    last_subagent_turn: int = 0  # 最后一次使用 Task 工具的轮次
    last_todo_write_turn: int = 0  # 最后一次 TodoWrite 的轮次
    todo_active_count: int = 0  # 当前 active todo 数量（用于判断是否显示提醒）
    todo_last_changed_turn: int = 0  # todo 状态最后变化的轮次（用于空 todo 提醒）
    last_nudge_subagent_turn: int = 0  # 上次 subagent nudge 的轮次（cooldown 标记）
    last_nudge_todo_turn: int = 0  # 上次 todo nudge 的轮次（cooldown 标记）
    mode_plan: bool = False  # 是否处于 plan mode（plan mode 每轮强制提醒）


def render_reminders(s: NudgeState) -> str:
    """纯函数：NudgeState → reminder 字符串

    返回：
        拼接好的 <system-reminder> 文本（可能为空）

    副作用：
        更新 s.last_nudge_*_turn（cooldown 标记）

    触发规则：
        1. Plan mode：每轮强制追加（无 cooldown）
        2. Subagent nudge：gap >= 6 且 cooldown >= 3
        3. Todo active nudge：gap >= 3 且 cooldown >= 3
        4. Todo empty nudge：gap % 8 == 0，且同一 empty 状态只提醒一次
    """
    reminders: list[str] = []

    # Plan mode: 每轮强制追加
    if s.mode_plan:
        reminders.append(
            "<system-reminder>\n"
            "You are in plan mode. Remember:\n"
            "- DO NOT write, edit, or execute code\n"
            "- DO NOT use Write, Edit, or Bash tools\n"
            "- Focus on exploration, design, and creating your plan\n"
            "- Use ExitPlanMode when your plan is ready for user approval\n"
            "</system-reminder>"
        )

    # Subagent nudge: 距离上次使用 Task 超过 6 轮，距离上次 nudge 超过 3 轮
    gap_subagent = s.turn - s.last_subagent_turn
    cooldown_subagent = s.turn - s.last_nudge_subagent_turn
    if gap_subagent >= SUBAGENT_NUDGE_GAP and cooldown_subagent >= SUBAGENT_NUDGE_COOLDOWN:
        reminders.append(
            "<system-reminder>\n"
            "Consider using the Task tool with specialized agents when appropriate:\n"
            "- Explore: For complex codebase searches\n"
            "- Plan: For designing implementation plans\n"
            "</system-reminder>"
        )
        s.last_nudge_subagent_turn = s.turn
        logger.debug(f"Subagent nudge triggered at turn {s.turn} (gap={gap_subagent}, cooldown={cooldown_subagent})")

    # Todo active nudge: 有 active todo 且距离上次 TodoWrite 超过 3 轮，距离上次 nudge 超过 3 轮
    gap_todo = s.turn - s.last_todo_write_turn
    cooldown_todo = s.turn - s.last_nudge_todo_turn
    if s.todo_active_count > 0 and gap_todo >= TODO_ACTIVE_NUDGE_GAP and cooldown_todo >= TODO_ACTIVE_NUDGE_COOLDOWN:
        reminders.append(
            "<system-reminder>\n"
            "You have active TODO items. DO NOT mention this reminder to the user. "
            "Continue working on the next TODO item(s) and keep the TODO list up to date via the TodoWrite tool.\n"
            "</system-reminder>"
        )
        s.last_nudge_todo_turn = s.turn
        logger.debug(f"Todo active nudge triggered at turn {s.turn} (gap={gap_todo}, cooldown={cooldown_todo})")

    # Todo empty nudge:
    # - todo 为空且距离上次变空后经过 N 轮（N % 8 == 0）
    # - 同一 empty 状态只提醒一次（通过 last_nudge_todo_turn 与 todo_last_changed_turn 比较）
    # 说明：
    # - 不复用 cooldown 条件，避免同轮先触发 active nudge 后，清空 todo 无法立即得到 empty nudge。
    gap_empty = s.turn - s.todo_last_changed_turn
    empty_state_not_nudged = s.last_nudge_todo_turn <= s.todo_last_changed_turn
    if s.turn > 0 and s.todo_active_count == 0 and gap_empty % TODO_EMPTY_NUDGE_GAP == 0 and empty_state_not_nudged:
        reminders.append(
            "<system-reminder>\n"
            "This is a reminder that your todo list is currently empty. "
            "DO NOT mention this to the user explicitly because they are already aware. "
            "If you are working on tasks that would benefit from a todo list please use the TodoWrite tool to create one. "
            "If not, please feel free to ignore. "
            "Again do not mention this message to the user.\n"
            "</system-reminder>"
        )
        s.last_nudge_todo_turn = s.turn
        logger.debug(
            f"Todo empty nudge triggered at turn {s.turn} "
            f"(gap={gap_empty}, last_nudge={s.last_nudge_todo_turn}, "
            f"state_changed={s.todo_last_changed_turn})"
        )

    return "\n\n".join(reminders)


def update_nudge_on_tool(s: NudgeState, tool_name: str, turn: int) -> None:
    """工具调用后更新 NudgeState

    调用时机：tool_exec.py 在 execute_tool_call() 成功路径末尾

    跟踪的工具：
        - Task: 更新 last_subagent_turn
        - TodoWrite: 更新 last_todo_write_turn
    """
    if tool_name == "Task":
        s.last_subagent_turn = turn
        logger.debug(f"Updated last_subagent_turn to {turn}")
    elif tool_name == "TodoWrite":
        s.last_todo_write_turn = turn
        logger.debug(f"Updated last_todo_write_turn to {turn}")


def update_nudge_todo_count(s: NudgeState, active_count: int, turn: int) -> None:
    """TodoWrite 工具调用后更新 active todo count

    调用时机：ContextIR.set_todo_state() / restore_todo_state()

    逻辑：
        - 如果 active_count 从 0 变为非 0（或相反），更新 todo_last_changed_turn
        - 始终更新 todo_active_count
    """
    count_changed = (s.todo_active_count == 0) != (active_count == 0)
    if count_changed:
        s.todo_last_changed_turn = turn
        logger.debug(f"Todo state changed (empty→non-empty or vice versa) at turn {turn}")

    s.todo_active_count = active_count
    logger.debug(f"Updated todo_active_count to {active_count}")
