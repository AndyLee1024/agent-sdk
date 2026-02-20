import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from comate_agent_sdk.agent.tool_exec import execute_tool_call
from comate_agent_sdk.llm.messages import Function, ToolCall
from comate_agent_sdk.system_tools.tool_result import err, ok
from comate_agent_sdk.tools import tool


class _FakeAgent:
    def __init__(self, tool_obj) -> None:
        self._tool_map = {tool_obj.name: tool_obj}
        self.options = SimpleNamespace(
            dependency_overrides={},
            project_root=Path.cwd(),
            offload_root_path=None,
            llm_levels=None,
            permission_mode="default",
            tool_approval_callback=None,
        )
        self._session_id = "s_test"
        self.name = None
        self._is_subagent = False
        self._subagent_source_prefix = None
        self._token_cost = None
        self._context = None

    async def run_hook_event(self, event_name: str, **kwargs):  # type: ignore[no-untyped-def]
        return None

    def add_hidden_user_message(self, content: str) -> None:
        _ = content


class TestToolExecFormatterIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_execute_tool_call_formats_envelope_result(self) -> None:
        @tool("Read tool for testing", name="Read")
        async def Read(file_path: str, offset_line: int = 0, limit_lines: int = 2) -> dict:
            return ok(
                data={
                    "content": "     1\ta\n     2\tb",
                    "total_lines": 100,
                    "lines_returned": 2,
                    "has_more": True,
                    "next_offset_line": 2,
                    "truncated": True,
                },
                meta={
                    "file_path": file_path,
                    "offset_line": offset_line,
                    "limit_lines": limit_lines,
                },
            )

        agent = _FakeAgent(Read)
        tool_call = ToolCall(
            id="tc_read",
            function=Function(
                name="Read",
                arguments=json.dumps(
                    {"file_path": "src/main.py", "offset_line": 0, "limit_lines": 2},
                    ensure_ascii=False,
                ),
            ),
        )

        message = await execute_tool_call(agent, tool_call)
        self.assertFalse(message.is_error)
        self.assertIsNotNone(message.raw_envelope)
        self.assertIsNotNone(message.execution_meta)
        self.assertIsNotNone(message.truncation_record)
        self.assertEqual(message.execution_meta.get("status"), "ok")
        self.assertTrue(message.execution_meta.get("truncation", {}).get("formatter_truncated"))
        self.assertIn("# Read: src/main.py", message.text)
        self.assertIn("Recommended next step", message.text)

    async def test_execute_tool_call_formats_error_envelope(self) -> None:
        @tool("Grep tool for testing", name="Grep")
        async def Grep(pattern: str) -> dict:
            return err(
                "INVALID_ARGUMENT",
                f"invalid pattern: {pattern}",
                field_errors=[{"field": "pattern", "message": "invalid regex"}],
            )

        agent = _FakeAgent(Grep)
        tool_call = ToolCall(
            id="tc_grep",
            function=Function(
                name="Grep",
                arguments=json.dumps({"pattern": "["}, ensure_ascii=False),
            ),
        )

        message = await execute_tool_call(agent, tool_call)
        self.assertTrue(message.is_error)
        self.assertIsNotNone(message.raw_envelope)
        self.assertEqual(message.execution_meta.get("status"), "error")
        self.assertEqual(message.execution_meta.get("error_code"), "INVALID_ARGUMENT")
        self.assertIn("# Grep Error", message.text)

    async def test_execute_tool_call_keeps_plain_string_result(self) -> None:
        @tool("Plain tool for testing", name="PlainTool")
        async def PlainTool(value: str) -> str:
            return f"plain:{value}"

        agent = _FakeAgent(PlainTool)
        tool_call = ToolCall(
            id="tc_plain",
            function=Function(
                name="PlainTool",
                arguments=json.dumps({"value": "ok"}, ensure_ascii=False),
            ),
        )

        message = await execute_tool_call(agent, tool_call)
        self.assertFalse(message.is_error)
        self.assertEqual(message.text, "plain:ok")
        self.assertIsNone(message.raw_envelope)
        self.assertIsNone(message.execution_meta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
