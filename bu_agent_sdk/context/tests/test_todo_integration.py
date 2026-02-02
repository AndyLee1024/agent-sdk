"""TodoWrite 工具与 ContextIR 集成测试"""

import json
import pytest

from bu_agent_sdk.context import ContextIR


def test_set_todo_state_empty():
    """测试设置空 TODO 列表"""
    ctx = ContextIR()
    ctx.set_todo_state([])

    # 应该注册 "empty" reminder
    assert len(ctx.reminders) == 1
    assert ctx.reminders[0].name == "todo_list_empty"
    assert "currently empty" in ctx.reminders[0].content


def test_set_todo_state_with_items():
    """测试设置非空 TODO 列表"""
    ctx = ContextIR()
    todos = [
        {"id": "1", "content": "Task 1", "status": "pending", "priority": "high"},
        {"id": "2", "content": "Task 2", "status": "in_progress", "priority": "medium"},
    ]
    ctx.set_todo_state(todos)

    # 应该注册 "update" reminder
    assert len(ctx.reminders) == 1
    assert ctx.reminders[0].name == "todo_list_update"
    assert "todo list has changed" in ctx.reminders[0].content
    assert '"id": "1"' in ctx.reminders[0].content  # JSON 应该在 reminder 中


def test_initial_todo_reminder():
    """测试初始化提醒"""
    ctx = ContextIR()
    ctx.register_initial_todo_reminder_if_needed()

    # 应该注册一次
    assert len(ctx.reminders) == 1
    assert ctx.reminders[0].name == "todo_list_empty"

    # 再次调用不应重复注册
    ctx.register_initial_todo_reminder_if_needed()
    assert len(ctx.reminders) == 1


def test_initial_todo_reminder_with_existing_state():
    """测试有 todo 状态时不注册初始提醒"""
    ctx = ContextIR()
    todos = [{"id": "1", "content": "Task 1", "status": "pending", "priority": "medium"}]
    ctx.set_todo_state(todos)

    # 清除现有 reminders
    ctx.reminders.clear()

    # 调用初始提醒不应注册（因为已有 todo_state）
    ctx.register_initial_todo_reminder_if_needed()
    assert len(ctx.reminders) == 0


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
    """测试 TODO reminder 的替换逻辑"""
    ctx = ContextIR()

    # 初始为空
    ctx.set_todo_state([])
    assert len(ctx.reminders) == 1
    assert ctx.reminders[0].name == "todo_list_empty"

    # 添加 todos 后替换 reminder
    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "high"}])
    assert len(ctx.reminders) == 1
    assert ctx.reminders[0].name == "todo_list_update"

    # 再次设为空，应替换回 empty reminder
    ctx.set_todo_state([])
    assert len(ctx.reminders) == 1
    assert ctx.reminders[0].name == "todo_list_empty"


def test_todo_state_cleared_on_context_clear():
    """测试 clear() 时清除 TODO 状态"""
    ctx = ContextIR()
    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "medium"}])

    ctx.clear()
    assert not ctx.has_todos()
    assert len(ctx._todo_state) == 0


def test_todo_event_emission():
    """测试 TODO 状态更新时发送事件"""
    from bu_agent_sdk.context.observer import EventType

    ctx = ContextIR()
    events = []

    def listener(event):
        events.append(event)

    ctx.event_bus.subscribe(listener)

    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "low"}])

    # 应该有至少一个 TODO_STATE_UPDATED 事件
    todo_events = [e for e in events if e.event_type == EventType.TODO_STATE_UPDATED]
    assert len(todo_events) == 1
    assert "1 items" in todo_events[0].detail


def test_todo_json_in_reminder():
    """测试 reminder 内容中包含正确的 JSON 格式"""
    ctx = ContextIR()
    todos = [
        {"id": "1", "content": "Task 1", "status": "pending", "priority": "high"},
        {"id": "2", "content": "Task 2", "status": "completed", "priority": "low"},
    ]
    ctx.set_todo_state(todos)

    reminder = ctx.reminders[0]
    assert reminder.name == "todo_list_update"

    # 验证 JSON 格式正确
    assert '"id": "1"' in reminder.content
    assert '"content": "Task 1"' in reminder.content
    assert '"priority": "high"' in reminder.content
