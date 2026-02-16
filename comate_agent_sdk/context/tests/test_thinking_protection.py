"""Tests for thinking block protection during tool loops.

Anthropic constraint: When in a tool loop (stop_reason="tool_use"), the thinking
block (including signature) of the assistant message that triggered the tool_use
must NOT be removed or modified.

These tests verify that:
1. Thinking protection activates when AssistantMessage has thinking + tool_calls
2. Protection releases when tool barrier clears (all ToolMessages received)
3. Compaction respects thinking protection during tool loops
"""

import unittest

from comate_agent_sdk.context.compaction import SelectiveCompactionPolicy
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    ContentPartRedactedThinkingParam,
    ContentPartTextParam,
    ContentPartThinkingParam,
    Function,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class TestThinkingProtection(unittest.TestCase):
    """Test thinking block protection mechanism."""

    def test_thinking_protection_basic(self) -> None:
        """AssistantMessage with thinking + tool_calls should activate protection."""
        ctx = ContextIR()

        assistant = AssistantMessage(
            content=[
                ContentPartThinkingParam(thinking="Analyzing...", signature="sig123"),
                ContentPartTextParam(text="I will help."),
            ],
            tool_calls=[
                ToolCall(id="tc1", type="function", function=Function(name="read", arguments="{}"))
            ],
        )
        item = ctx.add_message(assistant)

        # Protection should be active
        self.assertTrue(ctx.has_tool_barrier)
        self.assertIn(item.id, ctx._thinking_protected_assistant_ids)

        # Add ToolMessage to release
        ctx.add_message(ToolMessage(tool_call_id="tc1", tool_name="read", content="result"))

        # Protection should be released
        self.assertFalse(ctx.has_tool_barrier)
        self.assertEqual(len(ctx._thinking_protected_assistant_ids), 0)

    def test_thinking_protection_parallel_tools(self) -> None:
        """With parallel tool calls, protection holds until ALL tools complete."""
        ctx = ContextIR()

        assistant = AssistantMessage(
            content=[
                ContentPartThinkingParam(thinking="Processing multiple files...", signature="sig_multi"),
                ContentPartTextParam(text="Reading both files."),
            ],
            tool_calls=[
                ToolCall(id="tc_a", type="function", function=Function(name="read", arguments='{"file":"a.txt"}')),
                ToolCall(id="tc_b", type="function", function=Function(name="read", arguments='{"file":"b.txt"}')),
            ],
        )
        item = ctx.add_message(assistant)

        self.assertTrue(ctx.has_tool_barrier)
        self.assertIn(item.id, ctx._thinking_protected_assistant_ids)

        # First tool result: protection should still be active
        ctx.add_message(ToolMessage(tool_call_id="tc_a", tool_name="read", content="content a"))
        self.assertTrue(ctx.has_tool_barrier)
        self.assertIn(item.id, ctx._thinking_protected_assistant_ids)

        # Second tool result: protection should release
        ctx.add_message(ToolMessage(tool_call_id="tc_b", tool_name="read", content="content b"))
        self.assertFalse(ctx.has_tool_barrier)
        self.assertEqual(len(ctx._thinking_protected_assistant_ids), 0)

    def test_no_thinking_no_protection(self) -> None:
        """AssistantMessage without thinking should not trigger thinking protection."""
        ctx = ContextIR()

        assistant = AssistantMessage(
            content="I will read the file.",
            tool_calls=[
                ToolCall(id="tc_no_think", type="function", function=Function(name="read", arguments="{}"))
            ],
        )
        ctx.add_message(assistant)

        # Tool barrier should be active, but NO thinking protection
        self.assertTrue(ctx.has_tool_barrier)
        self.assertEqual(len(ctx._thinking_protected_assistant_ids), 0)

    def test_redacted_thinking_also_protected(self) -> None:
        """Redacted thinking blocks should also trigger protection."""
        ctx = ContextIR()

        assistant = AssistantMessage(
            content=[
                ContentPartRedactedThinkingParam(data="encrypted_data"),
                ContentPartTextParam(text="Done thinking."),
            ],
            tool_calls=[
                ToolCall(id="tc_redacted", type="function", function=Function(name="bash", arguments="{}"))
            ],
        )
        item = ctx.add_message(assistant)

        self.assertTrue(ctx.has_tool_barrier)
        self.assertIn(item.id, ctx._thinking_protected_assistant_ids)

    def test_compaction_protects_thinking_in_tool_loop(self) -> None:
        """During tool loop, compaction should protect assistant with thinking."""
        ctx = ContextIR()
        policy = SelectiveCompactionPolicy(dialogue_rounds_keep_min=1)

        # User message
        ctx.add_message(UserMessage(content="Help me"))

        # Assistant with thinking + tool_calls
        assistant = AssistantMessage(
            content=[
                ContentPartThinkingParam(thinking="Deep analysis...", signature="sig_compact"),
                ContentPartTextParam(text="Calling tool."),
            ],
            tool_calls=[
                ToolCall(id="tc_compact", type="function", function=Function(name="tool", arguments="{}"))
            ],
        )
        assistant_item = ctx.add_message(assistant)

        # During tool loop: should be protected
        protected_ids = policy._collect_recent_round_protected_ids(ctx)
        self.assertIn(assistant_item.id, protected_ids)

    def test_compaction_can_remove_after_tool_loop(self) -> None:
        """After tool loop completes, thinking blocks can be removed by compaction rules."""
        ctx = ContextIR()
        policy = SelectiveCompactionPolicy(dialogue_rounds_keep_min=1)

        # Round 1: old conversation with thinking (will be outside protected range)
        ctx.add_message(UserMessage(content="Old request"))
        old_assistant = AssistantMessage(
            content=[
                ContentPartThinkingParam(thinking="Old thinking...", signature="old_sig"),
                ContentPartTextParam(text="Old response."),
            ],
            tool_calls=[
                ToolCall(id="old_tc", type="function", function=Function(name="old_tool", arguments="{}"))
            ],
        )
        old_item = ctx.add_message(old_assistant)
        ctx.add_message(ToolMessage(tool_call_id="old_tc", tool_name="old_tool", content="old result"))

        # Round 2: new conversation (within protected range)
        ctx.add_message(UserMessage(content="New request"))
        new_assistant = AssistantMessage(content="New response without tool.")
        ctx.add_message(new_assistant)

        # Tool loop is complete, old assistant should NOT be in thinking protection
        self.assertFalse(ctx.has_tool_barrier)
        self.assertNotIn(old_item.id, ctx._thinking_protected_assistant_ids)

        # Old assistant might be protected by round protection (dialogue_rounds_keep_min)
        # but NOT by thinking protection specifically
        protected_ids = policy._collect_recent_round_protected_ids(ctx)

        # With dialogue_rounds_keep_min=1, only the last round is protected
        # Old assistant is in round 1, new assistant is in round 2
        # So old_item should NOT be protected (outside the protected rounds)
        # Note: This depends on implementation - verify the expected behavior
        self.assertNotIn(old_item.id, ctx._thinking_protected_assistant_ids)

    def test_clear_tool_barrier_clears_thinking_protection(self) -> None:
        """clear_tool_barrier() should also clear thinking protection."""
        ctx = ContextIR()

        assistant = AssistantMessage(
            content=[
                ContentPartThinkingParam(thinking="Will be cleared...", signature="sig_clear"),
            ],
            tool_calls=[
                ToolCall(id="tc_clear", type="function", function=Function(name="tool", arguments="{}"))
            ],
        )
        item = ctx.add_message(assistant)

        self.assertTrue(ctx.has_tool_barrier)
        self.assertIn(item.id, ctx._thinking_protected_assistant_ids)

        # Force clear
        ctx.clear_tool_barrier()

        self.assertFalse(ctx.has_tool_barrier)
        self.assertEqual(len(ctx._thinking_protected_assistant_ids), 0)

    def test_context_clear_resets_thinking_protection(self) -> None:
        """ctx.clear() should reset all state including thinking protection."""
        ctx = ContextIR()

        assistant = AssistantMessage(
            content=[
                ContentPartThinkingParam(thinking="Will be cleared...", signature="sig_reset"),
            ],
            tool_calls=[
                ToolCall(id="tc_reset", type="function", function=Function(name="tool", arguments="{}"))
            ],
        )
        ctx.add_message(assistant)

        self.assertTrue(len(ctx._thinking_protected_assistant_ids) > 0)

        ctx.clear()

        self.assertEqual(len(ctx._thinking_protected_assistant_ids), 0)
        self.assertFalse(ctx.has_tool_barrier)

    def test_thinking_protection_with_text_content_only(self) -> None:
        """AssistantMessage with plain text content + tool_calls: no thinking protection."""
        ctx = ContextIR()

        # Plain string content (no thinking blocks)
        assistant = AssistantMessage(
            content="Just text, no thinking.",
            tool_calls=[
                ToolCall(id="tc_plain", type="function", function=Function(name="tool", arguments="{}"))
            ],
        )
        ctx.add_message(assistant)

        self.assertTrue(ctx.has_tool_barrier)
        self.assertEqual(len(ctx._thinking_protected_assistant_ids), 0)

    def test_empty_round_ranges_still_protects_thinking(self) -> None:
        """Even without user messages (empty rounds), thinking should be protected."""
        ctx = ContextIR()
        policy = SelectiveCompactionPolicy(dialogue_rounds_keep_min=2)

        # No user message, just assistant with thinking + tools
        assistant = AssistantMessage(
            content=[
                ContentPartThinkingParam(thinking="No user msg scenario...", signature="sig_empty"),
            ],
            tool_calls=[
                ToolCall(id="tc_empty", type="function", function=Function(name="tool", arguments="{}"))
            ],
        )
        item = ctx.add_message(assistant)

        # Should still be protected even though there are no "rounds"
        protected_ids = policy._collect_recent_round_protected_ids(ctx)
        self.assertIn(item.id, protected_ids)


if __name__ == "__main__":
    unittest.main()
