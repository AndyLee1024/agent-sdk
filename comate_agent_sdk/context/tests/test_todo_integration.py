"""TodoWrite 工具与 ContextIR 集成测试（适配 NudgeState 系统）"""

from comate_agent_sdk.context import ContextIR
from comate_agent_sdk.llm.messages import UserMessage


def test_set_todo_state_empty():
    """测试设置空 TODO 列表（检查 NudgeState 更新）"""
    ctx = ContextIR()
    ctx.set_todo_state([])

    # 应该更新 NudgeState
    assert ctx._nudge.todo_active_count == 0
    assert ctx._nudge.todo_last_changed_turn == 0


def test_set_todo_state_with_items():
    """测试设置非空 TODO 列表（检查 NudgeState 更新）"""
    ctx = ContextIR()
    ctx.set_turn_number(1)
    todos = [
        {"id": "1", "content": "Task 1", "status": "pending", "priority": "high"},
        {"id": "2", "content": "Task 2", "status": "in_progress", "priority": "medium"},
    ]
    ctx.set_todo_state(todos)

    # 应该更新 NudgeState
    assert ctx._nudge.todo_active_count == 2
    assert ctx._nudge.todo_last_changed_turn == 1


def test_initial_todo_empty_nudge():
    """测试初始空 todo 提醒（通过 lower() 检查）"""
    ctx = ContextIR()
    ctx.add_message(UserMessage(content="Hello"))
    ctx.set_todo_state([])

    # turn=1, todo_last_changed_turn=0, gap=1
    # 不满足 gap % 8 == 0，不应提醒
    msgs = ctx.lower()
    last_user_msg = next((m for m in reversed(msgs) if isinstance(m, UserMessage)), None)
    assert last_user_msg is not None
    assert "currently empty" not in last_user_msg.text

    # turn=1, 设置 todo_last_changed_turn=1 后，gap=0，满足 gap % 8 == 0
    # 同时重置 cooldown（确保不被 cooldown 阻挡）
    ctx._nudge.todo_last_changed_turn = 1
    ctx._nudge.last_nudge_todo_turn = -100  # 确保 cooldown 足够大
    msgs = ctx.lower()
    last_user_msg = next((m for m in reversed(msgs) if isinstance(m, UserMessage)), None)
    assert last_user_msg is not None
    assert "currently empty" in last_user_msg.text


def test_initial_todo_reminder_with_existing_state():
    """测试有 todo 状态时不提醒空列表"""
    ctx = ContextIR()
    ctx.add_message(UserMessage(content="Hello"))
    todos = [{"id": "1", "content": "Task 1", "status": "pending", "priority": "medium"}]
    ctx.set_todo_state(todos)

    # 有 active todo，不应有空提醒
    msgs = ctx.lower()
    last_user_msg = next((m for m in reversed(msgs) if isinstance(m, UserMessage)), None)
    assert last_user_msg is not None
    assert "currently empty" not in last_user_msg.text


def test_get_todo_state():
    """测试获取 TODO 状态"""
    ctx = ContextIR()
    todos = [{"id": "1", "content": "Task 1", "status": "pending", "priority": "low"}]
    ctx.set_todo_state(todos)

    state = ctx.get_todo_state()
    assert "todos" in state
    assert "updated_at" in state
    assert len(state["todos"]) == 1
    assert state["todos"][0]["id"] == "1"


def test_has_todos():
    """测试 has_todos 方法"""
    ctx = ContextIR()
    assert not ctx.has_todos()

    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "medium"}])
    assert ctx.has_todos()

    ctx.set_todo_state([])
    assert not ctx.has_todos()


def test_todo_reminder_replacement():
    """测试 TODO 状态变化时 NudgeState 更新"""
    ctx = ContextIR()

    # 初始为空
    ctx.set_todo_state([])
    assert ctx._nudge.todo_active_count == 0

    # 添加 todos
    ctx.set_turn_number(5)
    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "high"}])
    assert ctx._nudge.todo_active_count == 1
    assert ctx._nudge.todo_last_changed_turn == 5

    # 再次设为空
    ctx.set_turn_number(10)
    ctx.set_todo_state([])
    assert ctx._nudge.todo_active_count == 0
    assert ctx._nudge.todo_last_changed_turn == 10


def test_todo_state_cleared_on_context_clear():
    """测试 clear() 时清除 TODO 状态"""
    ctx = ContextIR()
    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "medium"}])

    ctx.clear()
    assert not ctx.has_todos()
    assert len(ctx._todo_state) == 0


def test_todo_event_emission():
    """测试 TODO 状态更新时发送事件"""
    from comate_agent_sdk.context.observer import EventType

    ctx = ContextIR()
    events = []

    def listener(event):
        events.append(event)

    ctx.event_bus.subscribe(listener)

    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "low"}])

    # 应该有至少一个 TODO_STATE_UPDATED 事件
    todo_events = [e for e in events if e.event_type == EventType.TODO_STATE_UPDATED]
    assert len(todo_events) == 1
    assert "1 active items" in todo_events[0].detail


def test_todo_gentle_reminder_condition_threshold():
    """测试温和提醒的触发阈值（gap >= 3 才注入到 lower()）"""
    ctx = ContextIR()
    ctx.add_message(UserMessage(content="Hello"))
    ctx.set_turn_number(1)
    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "high"}])

    # 确保 cooldown 不干扰测试（设置足够久以前的 nudge）
    ctx._nudge.last_nudge_todo_turn = -100

    # gap=2（turn=3, last_todo_write_turn=1），不应注入
    ctx.set_turn_number(3)
    ctx._nudge.turn = 3
    msgs = ctx.lower()
    last_user_msg = next((m for m in reversed(msgs) if isinstance(m, UserMessage)), None)
    assert last_user_msg is not None
    assert "active TODO items" not in last_user_msg.text

    # gap=3（turn=4, last_todo_write_turn=1），应注入（cooldown 也满足）
    ctx.set_turn_number(4)
    ctx._nudge.turn = 4
    msgs = ctx.lower()
    last_user_msg = next((m for m in reversed(msgs) if isinstance(m, UserMessage)), None)
    assert last_user_msg is not None
    assert "active TODO items" in last_user_msg.text
