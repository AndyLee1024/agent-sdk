from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import MethodType

import pytest

from comate_agent_sdk import create_sdk_mcp_server, mcp_tool
from comate_agent_sdk.mcp.manager import McpManager
from comate_agent_sdk.mcp import manager as manager_module


def _build_calc_server():
    @mcp_tool(name="add", description="Add two numbers")
    async def add(a: float, b: float) -> str:
        return f"Sum: {a + b}"

    return create_sdk_mcp_server(name="calculator", tools=[add])


def test_mcp_manager_aclose_idempotent() -> None:
    async def _run() -> None:
        manager = McpManager({"calc": _build_calc_server()})

        await manager.start()
        assert any(t.name == "mcp__calc__add" for t in manager.tools)

        await manager.aclose()
        await manager.aclose()

        assert manager.tools == []
        assert manager.tool_infos == []

    asyncio.run(_run())


def test_mcp_manager_concurrent_start() -> None:
    async def _run() -> None:
        manager = McpManager({"calc": _build_calc_server()})

        await asyncio.gather(manager.start(), manager.start(), manager.start())
        assert any(t.name == "mcp__calc__add" for t in manager.tools)

        await manager.aclose()

    asyncio.run(_run())


def test_mcp_manager_context_enter_exit_in_same_task() -> None:
    async def _run() -> None:
        manager = McpManager({"fake": {"type": "http", "url": "http://example.test"}})  # type: ignore[arg-type]

        task_state: dict[str, asyncio.Task[None] | None] = {
            "enter": None,
            "exit": None,
        }

        @asynccontextmanager
        async def _fake_connect(_self: McpManager, _alias: str, _cfg: dict[str, str]):
            task_state["enter"] = asyncio.current_task()
            yield object()
            task_state["exit"] = asyncio.current_task()

        async def _fake_list_all_tools(_self: McpManager, _session: object):
            return []

        manager._connect = MethodType(_fake_connect, manager)  # type: ignore[method-assign]
        manager._list_all_tools = MethodType(_fake_list_all_tools, manager)  # type: ignore[method-assign]

        await manager.start()
        await manager.aclose()

        assert task_state["enter"] is not None
        assert task_state["exit"] is not None
        assert task_state["enter"] is task_state["exit"]

    asyncio.run(_run())


def test_mcp_manager_start_timeout_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "_START_TIMEOUT_S", 0.05)
    monkeypatch.setattr(manager_module, "_SHUTDOWN_TIMEOUT_S", 0.05)

    async def _never_init(self: McpManager) -> None:
        await asyncio.Event().wait()

    async def _run() -> None:
        manager = McpManager({"calc": _build_calc_server()})
        manager._run_lifecycle = MethodType(_never_init, manager)  # type: ignore[method-assign]

        with pytest.raises(asyncio.TimeoutError):
            await manager.start()

        assert manager._lifecycle_task is None
        assert manager.tools == []
        assert manager.tool_infos == []

    asyncio.run(_run())


def test_mcp_manager_start_propagates_init_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "_START_TIMEOUT_S", 0.1)
    monkeypatch.setattr(manager_module, "_SHUTDOWN_TIMEOUT_S", 0.1)

    async def _raise_during_init(self: McpManager) -> None:
        init_done = self._init_done
        err = RuntimeError("init boom")
        if init_done is not None and not init_done.done():
            init_done.set_exception(err)
        raise err

    async def _run() -> None:
        manager = McpManager({"calc": _build_calc_server()})
        manager._run_lifecycle = MethodType(_raise_during_init, manager)  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="init boom"):
            await manager.start()

        assert manager._lifecycle_task is None
        assert manager.tools == []
        assert manager.tool_infos == []

    asyncio.run(_run())


def test_mcp_manager_partial_server_failures_do_not_block_healthy_server() -> None:
    async def _run() -> None:
        servers = {
            "bad-http": {"type": "http", "url": ""},
            "calc": _build_calc_server(),
        }
        manager = McpManager(servers)  # type: ignore[arg-type]

        await manager.start()
        tool_names = [t.name for t in manager.tools]
        assert "mcp__calc__add" in tool_names

        await manager.aclose()

    asyncio.run(_run())
