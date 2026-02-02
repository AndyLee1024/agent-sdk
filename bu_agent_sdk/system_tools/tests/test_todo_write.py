"""TodoWrite 工具集成测试"""

import json
import tempfile
from pathlib import Path

import pytest

from bu_agent_sdk.system_tools.tools import TodoWrite, _TodoItem
from bu_agent_sdk.tools.system_context import bind_system_tool_context
from bu_agent_sdk.context import ContextIR


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
            result = await TodoWrite.execute(todos=[t.model_dump(mode="json") for t in todos])

        # 检查返回值包含提醒文本
        assert isinstance(result, str)
        assert "Remember to keep using the TODO list" in result
        assert "TODO list updated successfully" in result

        # 检查 ContextIR 是否包含 todo 状态
        todo_state = ctx_ir.get_todo_state()
        assert "todos" in todo_state
        assert len(todo_state["todos"]) == 2
        assert todo_state["todos"][0]["id"] == "1"
        assert todo_state["todos"][0]["priority"] == "high"

        # 检查是否注册了 reminder
        assert any(r.name == "todo_list_update" for r in ctx_ir.reminders)

        # 检查文件是否被写入
        todo_file = session_root / "todos.json"
        assert todo_file.exists()
        file_data = json.loads(todo_file.read_text(encoding="utf-8"))
        assert len(file_data["todos"]) == 2


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
            result = await TodoWrite.execute(todos=[])

        # 检查 ContextIR 注册了 empty reminder
        assert any(r.name == "todo_list_empty" for r in ctx_ir.reminders)
        assert not ctx_ir.has_todos()


@pytest.mark.anyio
async def test_todo_write_without_agent_context():
    """测试 TodoWrite 在没有 agent_context 时仍能工作"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session_root = root / "sessions" / "test"

        todos = [
            _TodoItem(id="1", content="Task 1", status="pending", priority="low"),
        ]

        # 不传递 agent_context
        with bind_system_tool_context(
            project_root=root,
            session_id="test",
            session_root=session_root,
        ):
            result = await TodoWrite.execute(todos=[t.model_dump(mode="json") for t in todos])

        # 应该成功返回（只是不会更新 ContextIR）
        assert isinstance(result, str)
        assert "TODO list updated successfully" in result

        # 文件应该被写入
        todo_file = session_root / "todos.json"
        assert todo_file.exists()


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
            result = await TodoWrite.execute(todos=[t.model_dump(mode="json") for t in todos])

        # 检查返回的 JSON 包含 priority
        assert '"priority": "high"' in result
        assert '"priority": "medium"' in result
        assert '"priority": "low"' in result

        # 检查 ContextIR 中的状态
        todo_state = ctx_ir.get_todo_state()
        assert todo_state["todos"][0]["priority"] == "high"
        assert todo_state["todos"][1]["priority"] == "medium"
        assert todo_state["todos"][2]["priority"] == "low"


@pytest.mark.anyio
async def test_todo_write_default_priority():
    """测试 priority 字段的默认值为 medium"""
    # 测试 Pydantic 模型的默认值
    todo = _TodoItem(id="1", content="Test task", status="pending")
    assert todo.priority == "medium"


@pytest.mark.anyio
async def test_todo_item_validation():
    """测试 TodoItem 的字段验证"""
    # 测试缺少必填字段
    with pytest.raises(Exception):  # Pydantic ValidationError
        _TodoItem(content="Test")  # 缺少 id 和 status

    # 测试 status 必须是指定值
    with pytest.raises(Exception):
        _TodoItem(id="1", content="Test", status="invalid", priority="high")

    # 测试 priority 必须是指定值
    with pytest.raises(Exception):
        _TodoItem(id="1", content="Test", status="pending", priority="critical")

    # 测试正常情况
    todo = _TodoItem(id="1", content="Test", status="pending", priority="high")
    assert todo.id == "1"
    assert todo.status == "pending"
    assert todo.priority == "high"
