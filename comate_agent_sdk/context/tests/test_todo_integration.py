"""TodoWrite 与 ContextIR reminder 状态联动测试。"""

from comate_agent_sdk.context import ContextIR
from comate_agent_sdk.llm.messages import UserMessage


def test_set_todo_state_updates_engine_state() -> None:
    ctx = ContextIR()
    ctx.set_turn_number(1)
    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "high"}])
    assert ctx._reminder_engine.state.todo_active_count == 1
    assert ctx._reminder_engine.state.todo_last_changed_turn == 1
    assert ctx._reminder_engine.state.last_todowrite_turn == 1


def test_empty_todo_reminder_is_injected_as_meta_message() -> None:
    ctx = ContextIR()
    ctx.add_message(UserMessage(content="hello"))  # turn=1
    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "high"}])
    ctx.set_turn_number(2)
    ctx.set_todo_state([])

    item = ctx.inject_due_reminders()
    assert item is not None
    assert item.item_type.value == "system_reminder"
    assert "currently empty" in item.content_text
    assert item.metadata.get("origin") == "system_reminder"


def test_active_todo_reminder_gap_threshold() -> None:
    ctx = ContextIR()
    ctx.add_message(UserMessage(content="hello"))  # turn=1
    ctx.set_todo_state([{"id": "1", "content": "Task 1", "status": "pending", "priority": "high"}])

    ctx.set_turn_number(3)
    assert ctx.inject_due_reminders() is None

    ctx.set_turn_number(4)
    item = ctx.inject_due_reminders()
    assert item is not None
    assert "active TODO items" in item.content_text
