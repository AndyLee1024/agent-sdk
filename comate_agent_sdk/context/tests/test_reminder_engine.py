"""ReminderEngine 单元测试。"""

from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.reminder_engine import ReminderEngine
from comate_agent_sdk.llm.messages import UserMessage


def test_collect_due_reminders_empty_state() -> None:
    engine = ReminderEngine()
    reminders = engine.collect_due_reminders(turn=1)
    assert reminders == []


def test_plan_mode_forced_reminder() -> None:
    engine = ReminderEngine()
    engine.set_plan_mode(True)
    reminders = engine.collect_due_reminders(turn=1)
    assert any(r.rule_id == "plan_mode_forced" for r in reminders)


def test_task_gap_reminder_with_cooldown() -> None:
    engine = ReminderEngine()
    reminders = engine.collect_due_reminders(turn=10)
    assert any(r.rule_id == "task_gap_reminder" for r in reminders)
    assert engine.state.last_task_nudge_turn == 10

    reminders = engine.collect_due_reminders(turn=11)
    assert not any(r.rule_id == "task_gap_reminder" for r in reminders)


def test_todo_active_and_empty_reminder() -> None:
    engine = ReminderEngine()
    engine.update_todo_state(active_count=2, turn=1, update_last_write_turn=True)
    reminders = engine.collect_due_reminders(turn=4)
    assert any(r.rule_id == "todo_active_reminder" for r in reminders)

    engine.update_todo_state(active_count=0, turn=9, update_last_write_turn=True)
    reminders = engine.collect_due_reminders(turn=9)
    assert any(r.rule_id == "todo_empty_reminder" for r in reminders)

    reminders = engine.collect_due_reminders(turn=17)
    assert not any(r.rule_id == "todo_empty_reminder" for r in reminders)


def test_context_ir_inject_and_purge_system_reminders() -> None:
    ctx = ContextIR()
    ctx.add_message(UserMessage(content="hello"))  # turn=1

    ctx.record_tool_event(tool_name="Task")
    ctx.set_turn_number(8)
    item = ctx.inject_due_reminders()
    assert item is not None
    assert item.item_type.value == "system_reminder"
    assert item.metadata.get("origin") == "system_reminder"

    removed = ctx.purge_system_reminders()
    assert removed == 1


def test_context_ir_inject_deduplicates_same_turn_same_rules() -> None:
    ctx = ContextIR()
    ctx.add_message(UserMessage(content="hello"))  # turn=1

    ctx.record_tool_event(tool_name="Task")
    ctx.set_turn_number(8)

    first = ctx.inject_due_reminders()
    second = ctx.inject_due_reminders()
    assert first is not None
    assert second is None
