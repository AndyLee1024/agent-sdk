import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from bu_agent_sdk.context.compaction import SelectiveCompactionPolicy
from bu_agent_sdk.context.fs import ContextFileSystem
from bu_agent_sdk.context.ir import ContextIR
from bu_agent_sdk.context.items import ItemType
from bu_agent_sdk.context.offload import OffloadPolicy
from bu_agent_sdk.llm.messages import (
    AssistantMessage,
    Function,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class TestToolBlockCompaction(unittest.TestCase):
    def _add_tool_block(self, ctx: ContextIR, *, call_id: str, tool_name: str, args: str, result: str) -> None:
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

    def test_truncate_tool_blocks_replaces_old_block_with_meta_placeholder(self) -> None:
        ctx = ContextIR()
        ctx.add_message(UserMessage(content="hello"))
        self._add_tool_block(
            ctx,
            call_id="call_1",
            tool_name="Read",
            args='{"file_path": "/tmp/a.txt"}',
            result="result-1",
        )
        self._add_tool_block(
            ctx,
            call_id="call_2",
            tool_name="Bash",
            args='{"command": "echo hi"}',
            result="result-2",
        )

        policy = SelectiveCompactionPolicy(threshold=1)
        removed = asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=1))
        self.assertEqual(removed, 1)

        # 最老的块应被替换为 OFFLOAD_PLACEHOLDER（且为 meta user message）
        placeholders = [it for it in ctx.conversation.items if it.item_type == ItemType.OFFLOAD_PLACEHOLDER]
        self.assertEqual(len(placeholders), 1)
        ph = placeholders[0]
        self.assertIsNotNone(ph.message)
        self.assertEqual(ph.message.role, "user")
        self.assertTrue(getattr(ph.message, "is_meta", False))

    def test_truncate_tool_blocks_offloads_call_and_result_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fs = ContextFileSystem(root_path=root, session_id="test")
            policy = SelectiveCompactionPolicy(
                threshold=1,
                fs=fs,
                offload_policy=OffloadPolicy(
                    enabled=True,
                    token_threshold=1,
                    token_threshold_by_type={"tool_call": 1, "tool_result": 1},
                    type_enabled={"tool_call": True, "tool_result": True},
                ),
            )

            ctx = ContextIR()
            self._add_tool_block(
                ctx,
                call_id="call_1",
                tool_name="Read",
                args=json.dumps({"file_path": "/tmp/a.txt", "api_key": "sk-1234567890123456789012345"}, ensure_ascii=False),
                result="Authorization: Bearer abcdefg",
            )
            self._add_tool_block(
                ctx,
                call_id="call_2",
                tool_name="Bash",
                args='{"command": "echo hi"}',
                result="ok",
            )

            removed = asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=1))
            self.assertEqual(removed, 1)

            placeholders = [it for it in ctx.conversation.items if it.item_type == ItemType.OFFLOAD_PLACEHOLDER]
            self.assertEqual(len(placeholders), 1)
            content = placeholders[0].content_text

            # 占位符应包含绝对路径，并且落盘文件应存在
            self.assertIn(str(root), content)

            # tool_call 文件应存在，并且 api_key 被脱敏
            call_path = root / "tool_call" / "Read" / "call_1.json"
            self.assertTrue(call_path.exists())
            call_json = json.loads(call_path.read_text(encoding="utf-8"))
            self.assertIn("***REDACTED***", call_json.get("arguments", ""))

            # tool_result 文件应存在，并且 Bearer token 被脱敏
            result_path = root / "tool_result" / "Read" / "call_1.json"
            self.assertTrue(result_path.exists())
            result_json = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertIn("***REDACTED***", result_json.get("content", ""))


if __name__ == "__main__":
    unittest.main()

