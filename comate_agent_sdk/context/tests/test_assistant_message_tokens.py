"""测试 AssistantMessage 的 token 统计是否包含 tool_calls"""

import unittest

from comate_agent_sdk.context.budget import BudgetConfig, TokenCounter
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import AssistantMessage, ToolCall, Function


class TestAssistantMessageTokenCounting(unittest.TestCase):
    """验证 AssistantMessage 的 token 统计逻辑"""

    def setUp(self):
        """创建测试用的 ContextIR"""
        self.budget = BudgetConfig(
            total_limit=100000,
            compact_threshold_ratio=0.8,
        )
        self.context = ContextIR(budget=self.budget)
        self.token_counter = TokenCounter()

    def test_assistant_message_with_only_text_counts_text_tokens(self):
        """纯文本 AssistantMessage 应该统计文本 tokens"""
        msg = AssistantMessage(
            content="Hello, this is a test message.",
            tool_calls=None,
        )

        item = self.context.add_message(msg)

        # 验证
        self.assertEqual(item.item_type, ItemType.ASSISTANT_MESSAGE)
        self.assertGreater(item.token_count, 0)
        # 应该大约 6-8 个 tokens (取决于 tokenizer)
        self.assertGreater(item.token_count, 5)
        self.assertLess(item.token_count, 15)

    def test_assistant_message_with_only_tool_calls_counts_tool_call_tokens(self):
        """只有 tool_calls 的 AssistantMessage 应该统计 tool_calls tokens"""
        msg = AssistantMessage(
            content=None,  # 没有文本内容
            tool_calls=[
                ToolCall(
                    id="call_abc123",
                    type="function",
                    function=Function(
                        name="WebFetch",
                        arguments='{"url": "https://example.com", "prompt": "Fetch this page"}',
                    ),
                )
            ],
        )

        item = self.context.add_message(msg)

        # 验证
        self.assertEqual(item.item_type, ItemType.ASSISTANT_MESSAGE)
        # 关键：token_count 必须 > 0！
        self.assertGreater(
            item.token_count,
            0,
            "AssistantMessage with tool_calls should have token_count > 0",
        )
        # tool_calls JSON 应该有约 30+ tokens
        self.assertGreater(item.token_count, 20)

    def test_assistant_message_with_text_and_tool_calls_counts_both(self):
        """包含文本和 tool_calls 的 AssistantMessage 应该统计两者"""
        msg = AssistantMessage(
            content="Let me fetch that for you.",
            tool_calls=[
                ToolCall(
                    id="call_xyz789",
                    type="function",
                    function=Function(
                        name="Bash",
                        arguments='{"command": "ls -la", "description": "List files"}',
                    ),
                )
            ],
        )

        item = self.context.add_message(msg)

        # 验证
        self.assertEqual(item.item_type, ItemType.ASSISTANT_MESSAGE)
        self.assertGreater(item.token_count, 0)
        # 应该包含文本 + tool_calls 的 tokens，约 40+ tokens
        self.assertGreater(item.token_count, 30)

    def test_assistant_message_with_multiple_tool_calls(self):
        """多个 tool_calls 的 AssistantMessage 应该统计所有 tool_calls"""
        msg = AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=Function(name="Read", arguments='{"file_path": "/path/to/file.py"}'),
                ),
                ToolCall(
                    id="call_2",
                    type="function",
                    function=Function(name="Grep", arguments='{"pattern": "class.*:", "path": "."}'),
                ),
                ToolCall(
                    id="call_3",
                    type="function",
                    function=Function(name="Bash", arguments='{"command": "git status"}'),
                ),
            ],
        )

        item = self.context.add_message(msg)

        # 验证
        self.assertEqual(item.item_type, ItemType.ASSISTANT_MESSAGE)
        self.assertGreater(item.token_count, 0)
        # 3 个 tool_calls 应该有约 60+ tokens
        self.assertGreater(item.token_count, 50)

    def test_budget_status_includes_assistant_message_tokens(self):
        """get_budget_status() 应该正确统计 AssistantMessage tokens"""
        # 添加一条只有 tool_calls 的 AssistantMessage
        msg = AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_test",
                    type="function",
                    function=Function(name="Read", arguments='{"file_path": "/test.py"}'),
                )
            ],
        )

        self.context.add_message(msg)

        # 获取预算状态
        status = self.context.get_budget_status()

        # 验证
        # 应该有 ASSISTANT_MESSAGE 类型的 tokens
        self.assertIn(ItemType.ASSISTANT_MESSAGE, status.tokens_by_type)
        assistant_tokens = status.tokens_by_type[ItemType.ASSISTANT_MESSAGE]
        self.assertGreater(
            assistant_tokens,
            0,
            "ASSISTANT_MESSAGE tokens should be > 0 in budget status",
        )

    def test_messages_category_appears_in_context_info(self):
        """
        验证即使 AssistantMessage 只有 tool_calls，Messages 类别也会显示

        这是修复的核心问题：之前 Messages 类别可能因为 token_count=0 而不显示
        """
        from comate_agent_sdk.llm.messages import UserMessage

        # 添加用户消息
        self.context.add_message(UserMessage(content="Hello"))

        # 添加只有 tool_calls 的 Assistant 消息
        self.context.add_message(
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=Function(name="Bash", arguments='{"command": "echo test"}'),
                    )
                ],
            )
        )

        # 获取预算状态
        status = self.context.get_budget_status()

        # 验证
        # 应该有 USER_MESSAGE 和 ASSISTANT_MESSAGE 的统计
        self.assertIn(ItemType.USER_MESSAGE, status.tokens_by_type)
        self.assertIn(ItemType.ASSISTANT_MESSAGE, status.tokens_by_type)

        # 两者的 tokens 都应该 > 0
        user_tokens = status.tokens_by_type[ItemType.USER_MESSAGE]
        assistant_tokens = status.tokens_by_type[ItemType.ASSISTANT_MESSAGE]

        self.assertGreater(user_tokens, 0)
        self.assertGreater(assistant_tokens, 0)

        # total_tokens 应该包含两者
        self.assertGreaterEqual(status.total_tokens, user_tokens + assistant_tokens)


if __name__ == "__main__":
    unittest.main()
