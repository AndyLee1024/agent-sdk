import asyncio
import unittest

from comate_agent_sdk.context.compaction import SelectiveCompactionPolicy
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.truncation import TruncationRecord
from comate_agent_sdk.llm.messages import AssistantMessage, Function, ToolCall, ToolMessage


class TestCompactionSkipFormatterTruncated(unittest.TestCase):
    @staticmethod
    def _text_with_min_tokens(ctx: ContextIR, min_tokens: int) -> str:
        text = "x"
        while ctx.token_counter.count(text) < min_tokens:
            text += " x"
        return text

    @staticmethod
    def _add_tool_block(
        ctx: ContextIR,
        *,
        call_id: str,
        result_text: str,
        truncation_record: TruncationRecord | None = None,
        destroyed: bool = False,
    ) -> None:
        ctx.add_message(
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        type="function",
                        function=Function(name="Read", arguments='{"file_path":"a.txt"}'),
                    )
                ],
            )
        )
        item = ctx.add_message(
            ToolMessage(
                tool_call_id=call_id,
                tool_name="Read",
                content=result_text,
                truncation_record=truncation_record,
            )
        )
        if destroyed:
            item.destroyed = True
            if isinstance(item.message, ToolMessage):
                item.message.destroyed = True

    def test_skip_formatter_truncated_tool_result(self) -> None:
        ctx = ContextIR()
        long_text = self._text_with_min_tokens(ctx, 700)
        self._add_tool_block(
            ctx,
            call_id="tc_skip",
            result_text=long_text,
            truncation_record=TruncationRecord(
                formatter_truncated=True,
                formatter_reason="line_limit",
            ),
        )

        policy = SelectiveCompactionPolicy(threshold=1)
        asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=5))

        result_item = ctx.conversation.items[-1]
        assert isinstance(result_item.message, ToolMessage)
        self.assertEqual(result_item.message.text, long_text)
        self.assertNotIn("truncation", result_item.metadata)

    def test_still_truncates_non_formatter_results(self) -> None:
        ctx = ContextIR()
        long_text = self._text_with_min_tokens(ctx, 700)
        self._add_tool_block(
            ctx,
            call_id="tc_truncate",
            result_text=long_text,
        )

        policy = SelectiveCompactionPolicy(threshold=1)
        asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=5))

        result_item = ctx.conversation.items[-1]
        assert isinstance(result_item.message, ToolMessage)
        self.assertIn("[TRUNCATED original~", result_item.message.text)
        self.assertIn("truncation", result_item.metadata)

    def test_destroyed_items_skipped(self) -> None:
        ctx = ContextIR()
        long_text = self._text_with_min_tokens(ctx, 700)
        self._add_tool_block(
            ctx,
            call_id="tc_destroyed",
            result_text=long_text,
            destroyed=True,
        )

        policy = SelectiveCompactionPolicy(threshold=1)
        asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=5))

        result_item = ctx.conversation.items[-1]
        assert isinstance(result_item.message, ToolMessage)
        self.assertEqual(result_item.message.text, long_text)
        self.assertNotIn("truncation", result_item.metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
