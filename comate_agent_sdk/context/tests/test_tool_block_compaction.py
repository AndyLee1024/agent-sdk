import asyncio
import unittest

from comate_agent_sdk.context.compaction import SelectiveCompactionPolicy
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    Function,
    ToolCall,
    ToolMessage,
)


class TestToolBlockCompaction(unittest.TestCase):
    def _add_tool_block(
        self,
        ctx: ContextIR,
        *,
        call_id: str,
        tool_name: str,
        args: str,
        result: str,
    ) -> None:
        ctx.add_message(
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        type="function",
                        function=Function(name=tool_name, arguments=args),
                    )
                ],
            )
        )
        ctx.add_message(
            ToolMessage(
                tool_call_id=call_id,
                tool_name=tool_name,
                content=result,
            )
        )

    def _build_text_with_exact_tokens(self, ctx: ContextIR, target_tokens: int) -> str:
        counter = ctx.token_counter
        patterns = [
            lambda n: "a " * n,
            lambda n: ("token " * n).strip(),
            lambda n: "a" * n,
            lambda n: ("x" * n) + " end",
        ]
        for pattern in patterns:
            low = 1
            high = max(2, target_tokens * 2)

            while counter.count(pattern(high)) < target_tokens and high < target_tokens * 200:
                low = high
                high *= 2

            start = max(1, low - 100)
            end = high + 100
            for size in range(start, end + 1):
                candidate = pattern(size)
                token_count = counter.count(candidate)
                if token_count == target_tokens:
                    return candidate
        raise AssertionError(f"无法构造精确 {target_tokens} tokens 的测试文本")

    def test_keep_latest_five_tool_blocks_and_drop_older_whole_blocks(self) -> None:
        ctx = ContextIR()
        for i in range(8):
            self._add_tool_block(
                ctx,
                call_id=f"call_{i}",
                tool_name="Read",
                args=f'{{"file_path":"file_{i}.txt"}}',
                result=f"result_{i}",
            )

        policy = SelectiveCompactionPolicy(threshold=1)
        removed = asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=5))
        self.assertEqual(removed, 3)

        assistant_call_ids = [
            item.message.tool_calls[0].id
            for item in ctx.conversation.items
            if item.item_type == ItemType.ASSISTANT_MESSAGE
            and item.message is not None
            and getattr(item.message, "tool_calls", None)
        ]
        tool_result_ids = [
            item.message.tool_call_id
            for item in ctx.conversation.items
            if item.item_type == ItemType.TOOL_RESULT and item.message is not None
        ]

        self.assertEqual(assistant_call_ids, [f"call_{i}" for i in range(3, 8)])
        self.assertEqual(tool_result_ids, [f"call_{i}" for i in range(3, 8)])
        self.assertEqual(
            [it for it in ctx.conversation.items if it.item_type == ItemType.OFFLOAD_PLACEHOLDER],
            [],
        )

    def test_tool_call_truncation_499_501_boundary(self) -> None:
        ctx = ContextIR()
        args_499 = self._build_text_with_exact_tokens(ctx, 499)
        args_501 = self._build_text_with_exact_tokens(ctx, 501)

        self._add_tool_block(
            ctx,
            call_id="call_499",
            tool_name="Read",
            args=args_499,
            result="ok-499",
        )
        self._add_tool_block(
            ctx,
            call_id="call_501",
            tool_name="Read",
            args=args_501,
            result="ok-501",
        )

        policy = SelectiveCompactionPolicy(threshold=1)
        removed = asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=5))
        self.assertEqual(removed, 0)

        assistant_items = [
            item
            for item in ctx.conversation.items
            if item.item_type == ItemType.ASSISTANT_MESSAGE and item.message is not None
        ]
        self.assertEqual(len(assistant_items), 2)

        first_args = assistant_items[0].message.tool_calls[0].function.arguments
        second_args = assistant_items[1].message.tool_calls[0].function.arguments

        self.assertEqual(first_args, args_499)
        self.assertNotIn("[TRUNCATED original~", first_args)
        self.assertIn("[TRUNCATED original~501 tokens]", second_args)
        self.assertTrue(second_args.split("\n")[0].strip())

    def test_tool_result_truncation_599_601_boundary(self) -> None:
        ctx = ContextIR()
        result_599 = self._build_text_with_exact_tokens(ctx, 599)
        result_601 = self._build_text_with_exact_tokens(ctx, 601)

        self._add_tool_block(
            ctx,
            call_id="result_599",
            tool_name="Bash",
            args='{"command":"echo short"}',
            result=result_599,
        )
        self._add_tool_block(
            ctx,
            call_id="result_601",
            tool_name="Bash",
            args='{"command":"echo long"}',
            result=result_601,
        )

        policy = SelectiveCompactionPolicy(threshold=1)
        removed = asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=5))
        self.assertEqual(removed, 0)

        result_items = [
            item
            for item in ctx.conversation.items
            if item.item_type == ItemType.TOOL_RESULT and item.message is not None
        ]
        self.assertEqual(len(result_items), 2)

        first_result = result_items[0].message.text
        second_result = result_items[1].message.text

        self.assertEqual(first_result, result_599)
        self.assertNotIn("[TRUNCATED original~", first_result)
        self.assertIn("[TRUNCATED original~601 tokens]", second_result)
        self.assertTrue(second_result.split("\n")[0].strip())


if __name__ == "__main__":
    unittest.main()
