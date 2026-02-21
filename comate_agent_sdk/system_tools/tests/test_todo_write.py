"""TodoWrite 工具集成测试"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from comate_agent_sdk.context import ContextIR
from comate_agent_sdk.system_tools.tools import TodoWrite, _TodoItem
from comate_agent_sdk.tools.system_context import bind_system_tool_context


def _parse(raw):
    if isinstance(raw, str) and raw.strip().startswith(("{", "[")):
        return json.loads(raw)
    return raw


@pytest.mark.anyio
async def test_todo_write_updates_context():
    """测试 TodoWrite 工具更新 ContextIR"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session_root = root / "sessions" / "test"
        ctx_ir = ContextIR()

        todos = [
            _TodoItem(id="1", content="Read docs", status="pending", priority="high"),
            _TodoItem(id="2", content="Write code", status="in_progress", priority="medium"),
        ]

        with bind_system_tool_context(
            project_root=root,
            session_id="test",
            session_root=session_root,
            agent_context=ctx_ir,
        ):
            raw = await TodoWrite.execute(todos=[t.model_dump(mode="json") for t in todos])

        result = _parse(raw)
        assert isinstance(result, dict)
        assert result["ok"] is True
        assert "Remember to keep using the TODO list" in (result.get("message") or "")

        todo_state = ctx_ir.reminder_engine.get_todo_state()
        assert "todos" in todo_state
        assert len(todo_state["todos"]) == 2
        assert todo_state["todos"][0]["id"] == "1"
        assert todo_state["todos"][0]["priority"] == "high"
        assert ctx_ir._reminder_engine.state.todo_active_count == 2
        assert ctx_ir._reminder_engine.state.last_todowrite_turn == 0

        todo_file = session_root / "todos.json"
        assert todo_file.exists()
        file_data = json.loads(todo_file.read_text(encoding="utf-8"))
        assert file_data.get("schema_version") == 2
        assert len(file_data["todos"]) == 2
        assert isinstance(file_data.get("turn_number_at_update", 0), int)


@pytest.mark.anyio
async def test_todo_write_empty_list():
    """测试 TodoWrite 工具写入空列表"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session_root = root / "sessions" / "test"
        ctx_ir = ContextIR()

        with bind_system_tool_context(
            project_root=root,
            session_id="test",
            session_root=session_root,
            agent_context=ctx_ir,
        ):
            raw = await TodoWrite.execute(todos=[])

        result = _parse(raw)
        assert result["ok"] is True
        assert ctx_ir._reminder_engine.state.todo_active_count == 0
        assert not ctx_ir.reminder_engine.has_active_todos()
        assert not (session_root / "todos.json").exists()


@pytest.mark.anyio
async def test_todo_write_without_agent_context():
    """测试 TodoWrite 在没有 agent_context 时仍能工作"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session_root = root / "sessions" / "test"

        todos = [
            _TodoItem(id="1", content="Task 1", status="pending", priority="low"),
        ]

        with bind_system_tool_context(
            project_root=root,
            session_id="test",
            session_root=session_root,
        ):
            raw = await TodoWrite.execute(todos=[t.model_dump(mode="json") for t in todos])

        result = _parse(raw)
        assert result["ok"] is True
        assert "TODO list" in (result.get("message") or "")

        todo_file = session_root / "todos.json"
        assert todo_file.exists()
        file_data = json.loads(todo_file.read_text(encoding="utf-8"))
        assert file_data.get("schema_version") == 2
        assert len(file_data.get("todos", [])) == 1


@pytest.mark.anyio
async def test_todo_write_with_priority_field():
    """测试 TodoWrite 支持 priority 字段"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session_root = root / "sessions" / "test"
        ctx_ir = ContextIR()

        todos = [
            _TodoItem(id="1", content="High priority task", status="pending", priority="high"),
            _TodoItem(id="2", content="Medium priority task", status="pending", priority="medium"),
            _TodoItem(id="3", content="Low priority task", status="pending", priority="low"),
        ]

        with bind_system_tool_context(
            project_root=root,
            session_id="test",
            session_root=session_root,
            agent_context=ctx_ir,
        ):
            raw = await TodoWrite.execute(todos=[t.model_dump(mode="json") for t in todos])

        result = _parse(raw)
        assert result["ok"] is True

        todo_state = ctx_ir.reminder_engine.get_todo_state()
        assert todo_state["todos"][0]["priority"] == "high"
        assert todo_state["todos"][1]["priority"] == "medium"
        assert todo_state["todos"][2]["priority"] == "low"

        todo_file = session_root / "todos.json"
        file_data = json.loads(todo_file.read_text(encoding="utf-8"))
        assert file_data.get("schema_version") == 2
        assert file_data["todos"][0]["priority"] == "high"


@pytest.mark.anyio
async def test_todo_write_deletes_file_when_all_completed():
    """测试 todos 全部 completed 时删除 todos.json"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session_root = root / "sessions" / "test"
        ctx_ir = ContextIR()

        with bind_system_tool_context(
            project_root=root,
            session_id="test",
            session_root=session_root,
            agent_context=ctx_ir,
        ):
            await TodoWrite.execute(todos=[
                {"id": "1", "content": "t1", "status": "pending", "priority": "low"},
            ])

            todo_file = session_root / "todos.json"
            assert todo_file.exists()

            await TodoWrite.execute(todos=[
                {"id": "1", "content": "t1", "status": "completed", "priority": "low"},
            ])

        assert not (session_root / "todos.json").exists()


@pytest.mark.anyio
async def test_todo_write_default_priority():
    """测试 priority 字段的默认值为 medium"""
    todo = _TodoItem(id="1", content="Test task", status="pending")
    assert todo.priority == "medium"


@pytest.mark.anyio
async def test_todo_item_validation():
    """测试 TodoItem 的字段验证"""
    with pytest.raises(Exception):
        _TodoItem(content="Test")

    with pytest.raises(Exception):
        _TodoItem(id="1", content="Test", status="invalid", priority="high")

    with pytest.raises(Exception):
        _TodoItem(id="1", content="Test", status="pending", priority="critical")

    todo = _TodoItem(id="1", content="Test", status="pending", priority="high")
    assert todo.id == "1"
    assert todo.status == "pending"
    assert todo.priority == "high"
