"""NudgeState 和 render_reminders 单元测试"""

from comate_agent_sdk.context.nudge import (
    NudgeState,
    render_reminders,
    update_nudge_on_tool,
    update_nudge_todo_count,
)


def test_render_reminders_empty_state():
    """测试空状态下不生成提醒"""
    s = NudgeState()
    result = render_reminders(s)
    assert result == ""


def test_render_reminders_plan_mode():
    """测试 plan mode 每轮强制提醒"""
    s = NudgeState(turn=1, mode_plan=True)

    result = render_reminders(s)
    assert "plan mode" in result
    assert "DO NOT write, edit, or execute code" in result


def test_render_reminders_subagent_nudge():
    """测试 subagent nudge 触发条件"""
    s = NudgeState(
        turn=10,
        last_subagent_turn=0,
        last_nudge_subagent_turn=0,
    )

    # gap=10 >= 6, cooldown=10 >= 3，应触发
    result = render_reminders(s)
    assert "Task tool" in result
    assert s.last_nudge_subagent_turn == 10  # 副作用：更新 cooldown

    # 下次调用，cooldown 不足，不应触发
    s.turn = 11
    result = render_reminders(s)
    assert "Task tool" not in result


def test_render_reminders_todo_active_nudge():
    """测试 active todo 提醒触发条件"""
    s = NudgeState(
        turn=5,
        last_todo_write_turn=1,
        todo_active_count=2,
        last_nudge_todo_turn=-100,  # 确保 cooldown 足够
    )

    # gap=4 >= 3, cooldown=105 >= 3，应触发
    result = render_reminders(s)
    assert "active TODO items" in result
    assert s.last_nudge_todo_turn == 5

    # gap 不足时不触发
    s.turn = 3
    s.last_todo_write_turn = 1
    s.last_nudge_todo_turn = -100
    result = render_reminders(s)
    assert "active TODO items" not in result


def test_render_reminders_todo_empty_nudge():
    """测试空 todo 提醒触发条件（同一 empty 状态仅一次）"""
    s = NudgeState(
        turn=9,
        todo_last_changed_turn=1,
        todo_active_count=0,
        last_nudge_todo_turn=-100,
    )

    # gap=8, 8 % 8 == 0，应触发
    result = render_reminders(s)
    assert "currently empty" in result
    assert s.last_nudge_todo_turn == 9

    # 同一 empty 状态，即使再次命中 8 轮倍数，也不应重复提醒
    s.turn = 17
    result = render_reminders(s)
    assert "currently empty" not in result

    # 发生状态变化后（例如再次变空），可以重新提醒
    s.todo_last_changed_turn = 17
    s.turn = 17
    result = render_reminders(s)
    assert "currently empty" in result

    # gap=9, 9 % 8 == 1，不触发
    s.turn = 10
    s.todo_last_changed_turn = 1
    s.last_nudge_todo_turn = -100
    result = render_reminders(s)
    assert "currently empty" not in result


def test_update_nudge_on_tool_task():
    """测试 Task 工具更新 subagent turn"""
    s = NudgeState()
    update_nudge_on_tool(s, "Task", 5)
    assert s.last_subagent_turn == 5


def test_update_nudge_on_tool_todo_write():
    """测试 TodoWrite 工具更新 todo write turn"""
    s = NudgeState()
    update_nudge_on_tool(s, "TodoWrite", 8)
    assert s.last_todo_write_turn == 8


def test_update_nudge_on_tool_other_tools():
    """测试其他工具不更新状态"""
    s = NudgeState()
    update_nudge_on_tool(s, "Read", 10)
    assert s.last_subagent_turn == 0
    assert s.last_todo_write_turn == 0


def test_update_nudge_todo_count_empty_to_active():
    """测试从空状态到有 todo 时更新 last_changed_turn"""
    s = NudgeState(turn=5, todo_active_count=0)
    update_nudge_todo_count(s, 2, 5)

    assert s.todo_active_count == 2
    assert s.todo_last_changed_turn == 5


def test_update_nudge_todo_count_active_to_empty():
    """测试从有 todo 到空状态时更新 last_changed_turn"""
    s = NudgeState(turn=10, todo_active_count=2)
    update_nudge_todo_count(s, 0, 10)

    assert s.todo_active_count == 0
    assert s.todo_last_changed_turn == 10


def test_update_nudge_todo_count_no_state_change():
    """测试 todo 数量变化但状态不变时不更新 last_changed_turn"""
    s = NudgeState(turn=15, todo_active_count=2, todo_last_changed_turn=5)
    update_nudge_todo_count(s, 3, 15)

    assert s.todo_active_count == 3
    assert s.todo_last_changed_turn == 5  # 不变（仍然非空）


def test_cooldown_prevents_duplicate_nudge():
    """测试 cooldown 阻止连续轮次重复提醒"""
    s = NudgeState(
        turn=10,
        last_subagent_turn=0,
        last_nudge_subagent_turn=8,  # 上次 nudge 是 turn 8
    )

    # cooldown = 10 - 8 = 2 < 3，不应触发
    result = render_reminders(s)
    assert "Task tool" not in result

    # cooldown = 11 - 8 = 3 >= 3，应触发
    s.turn = 11
    result = render_reminders(s)
    assert "Task tool" in result
