import asyncio
import unittest

from comate_agent_sdk.agent.history import destroy_ephemeral_messages
from comate_agent_sdk.context.compaction import SelectiveCompactionPolicy
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.llm.messages import AssistantMessage, Function, ToolCall, ToolMessage
from comate_agent_sdk.system_tools.output_formatter import OutputFormatter
from comate_agent_sdk.system_tools.tool_result import ok


class _FakeTool:
    def __init__(self, name: str, ephemeral: int | bool) -> None:
        self.name = name
        self.ephemeral = ephemeral


class _FakeAgent:
    def __init__(self, context: ContextIR) -> None:
        self.offload_policy = None
        self.offload_token_threshold = 1000
        self.tools = [_FakeTool("Read", 1)]
        self.ephemeral_keep_recent = None
        self._context = context
        self._context_fs = None


class TestThreeLayerIntegration(unittest.TestCase):
    def test_truncation_record_flows_through_all_layers(self) -> None:
        payload = ok(
            data={
                "content": "     1\tline1\n     2\tline2",
                "total_lines": 1200,
                "lines_returned": 2,
                "has_more": True,
                "next_offset_line": 2,
                "truncated": True,
            },
            meta={
                "file_path": "src/main.py",
                "offset_line": 0,
                "limit_lines": 2,
            },
        )
        formatted = OutputFormatter.format(
            tool_name="Read",
            tool_call_id="tc_flow",
            result_dict=payload,
        )
        self.assertIsNotNone(formatted.meta.truncation)

        ctx = ContextIR()
        ctx.add_message(
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc_flow",
                        type="function",
                        function=Function(name="Read", arguments="x " * 900),
                    )
                ],
            )
        )
        result_item = ctx.add_message(
            ToolMessage(
                tool_call_id="tc_flow",
                tool_name="Read",
                content=formatted.text,
                ephemeral=True,
                execution_meta=formatted.meta.to_dict(),
                truncation_record=formatted.meta.truncation,
            )
        )
        # 增加一个非保护项，确保 Ephemeral 选择时有可替代项
        ctx.add_message(
            ToolMessage(
                tool_call_id="tc_extra",
                tool_name="Read",
                content="large " * 800,
                ephemeral=True,
            )
        )

        self.assertIsNotNone(result_item.truncation_record)
        self.assertTrue(result_item.metadata["tool_execution_meta"]["truncation"]["formatter_truncated"])

        policy = SelectiveCompactionPolicy(threshold=1)
        asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=5))

        self.assertNotIn("[TRUNCATED original~", result_item.message.text)

        assistant_item = ctx.conversation.items[0]
        self.assertIsNotNone(assistant_item.truncation_record)
        assert assistant_item.truncation_record is not None
        self.assertGreater(len(assistant_item.truncation_record.compaction_details), 0)

        agent = _FakeAgent(ctx)
        destroy_ephemeral_messages(agent)
        self.assertFalse(result_item.destroyed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
