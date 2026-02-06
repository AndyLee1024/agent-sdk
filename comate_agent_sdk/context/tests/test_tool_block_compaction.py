import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from comate_agent_sdk.context.compaction import SelectiveCompactionPolicy
from comate_agent_sdk.context.fs import ContextFileSystem
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.context.offload import OffloadPolicy
from comate_agent_sdk.llm.messages import (
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
        """测试工具类型差异化压缩：Read 只保留 1 个，Bash 保留 3 个"""
        ctx = ContextIR()
        ctx.add_message(UserMessage(content="hello"))
        # 添加 2 个 Read 块（keep_recent=1，应移除 1 个）
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
            tool_name="Read",
            args='{"file_path": "/tmp/b.txt"}',
            result="result-2",
        )
        # 添加 1 个 Bash 块（keep_recent=3，不应移除）
        self._add_tool_block(
            ctx,
            call_id="call_3",
            tool_name="Bash",
            args='{"command": "echo hi"}',
            result="result-3",
        )

        policy = SelectiveCompactionPolicy(threshold=1)
        removed = asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=1))
        # Read 有 2 个块，keep_recent=1，应移除 1 个
        self.assertEqual(removed, 1)

        # 最老的 Read 块应被替换为 OFFLOAD_PLACEHOLDER（且为 meta user message）
        placeholders = [it for it in ctx.conversation.items if it.item_type == ItemType.OFFLOAD_PLACEHOLDER]
        self.assertEqual(len(placeholders), 1)
        ph = placeholders[0]
        self.assertIsNotNone(ph.message)
        self.assertEqual(ph.message.role, "user")
        self.assertTrue(getattr(ph.message, "is_meta", False))
        # 验证占位符包含 Read 工具信息
        self.assertIn("Read", ph.content_text)

    def test_truncate_tool_blocks_offloads_call_and_result_when_enabled(self) -> None:
        """测试工具交互块卸载：启用落盘时，应将工具调用和结果写入文件"""
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
            # 添加 2 个 Read 块（keep_recent=1，应移除第 1 个）
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
                tool_name="Read",
                args='{"file_path": "/tmp/b.txt"}',
                result="ok",
            )

            removed = asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=1))
            # Read 有 2 个块，keep_recent=1，应移除 1 个
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

    def test_truncate_tool_blocks_webfetch_all_removed(self) -> None:
        """测试 WebFetch 工具：keep_recent=0，应全部移除"""
        ctx = ContextIR()
        ctx.add_message(UserMessage(content="hello"))
        self._add_tool_block(
            ctx,
            call_id="call_1",
            tool_name="WebFetch",
            args='{"url": "https://example.com"}',
            result="page content 1",
        )
        self._add_tool_block(
            ctx,
            call_id="call_2",
            tool_name="WebFetch",
            args='{"url": "https://example.org"}',
            result="page content 2",
        )

        policy = SelectiveCompactionPolicy(threshold=1)
        removed = asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=1))
        # WebFetch keep_recent=0，两个块都应被移除
        self.assertEqual(removed, 2)

        placeholders = [it for it in ctx.conversation.items if it.item_type == ItemType.OFFLOAD_PLACEHOLDER]
        self.assertEqual(len(placeholders), 2)

    def test_truncate_tool_blocks_edit_preserves_more(self) -> None:
        """测试 Edit 工具：keep_recent=5，应保留更多"""
        ctx = ContextIR()
        ctx.add_message(UserMessage(content="hello"))
        # 添加 6 个 Edit 块
        for i in range(6):
            self._add_tool_block(
                ctx,
                call_id=f"call_{i}",
                tool_name="Edit",
                args=f'{{"file_path": "/tmp/file{i}.txt"}}',
                result=f"edited file {i}",
            )

        policy = SelectiveCompactionPolicy(threshold=1)
        removed = asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=1))
        # Edit keep_recent=5，6 个块应移除 1 个
        self.assertEqual(removed, 1)

    def test_truncate_tool_blocks_unknown_tool_uses_default(self) -> None:
        """测试未配置工具：使用默认规则 keep_recent=3"""
        ctx = ContextIR()
        ctx.add_message(UserMessage(content="hello"))
        # 添加 5 个未知工具块
        for i in range(5):
            self._add_tool_block(
                ctx,
                call_id=f"call_{i}",
                tool_name="UnknownTool",
                args=f'{{"param": {i}}}',
                result=f"result {i}",
            )

        policy = SelectiveCompactionPolicy(threshold=1)
        removed = asyncio.run(policy._truncate_tool_blocks(ctx, keep_recent=1))
        # 未知工具默认 keep_recent=3，5 个块应移除 2 个
        self.assertEqual(removed, 2)


if __name__ == "__main__":
    unittest.main()

