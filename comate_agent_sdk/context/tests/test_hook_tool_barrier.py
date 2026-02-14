import unittest

from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import AssistantMessage, Function, ToolCall, ToolMessage, UserMessage


def _build_tool_call(tool_call_id: str, *, name: str = "Read") -> ToolCall:
    return ToolCall(
        id=tool_call_id,
        function=Function(name=name, arguments="{}"),
    )


class TestHookToolBarrier(unittest.TestCase):
    def test_hook_injection_is_deferred_until_tool_result(self) -> None:
        ctx = ContextIR()
        ctx.add_message(UserMessage(content="hello"))
        ctx.add_message(
            AssistantMessage(
                content=None,
                tool_calls=[_build_tool_call("tc1")],
            )
        )

        ctx.add_hook_hidden_user_message(
            "hook-context",
            hook_name="PreToolUse",
            related_tool_call_id="tc1",
        )
        self.assertEqual(ctx.pop_flushed_hook_injection_texts(), [])
        self.assertEqual(len(ctx.conversation_messages), 2)

        ctx.add_message(
            ToolMessage(
                tool_call_id="tc1",
                tool_name="Read",
                content="ok",
            )
        )

        messages = ctx.conversation_messages
        self.assertEqual([m.role for m in messages], ["user", "assistant", "tool", "user"])
        self.assertTrue(isinstance(messages[-1], UserMessage) and messages[-1].is_meta)

        last_item = ctx.conversation.items[-1]
        self.assertEqual(last_item.metadata.get("origin"), "hook")
        self.assertEqual(last_item.metadata.get("hook_name"), "PreToolUse")
        self.assertEqual(last_item.metadata.get("related_tool_call_id"), "tc1")
        self.assertEqual(ctx.pop_flushed_hook_injection_texts(), ["hook-context"])

    def test_hook_injection_waits_until_all_parallel_tools_finished(self) -> None:
        ctx = ContextIR()
        ctx.add_message(
            AssistantMessage(
                content=None,
                tool_calls=[_build_tool_call("tc1"), _build_tool_call("tc2")],
            )
        )
        ctx.add_message(
            ToolMessage(
                tool_call_id="tc1",
                tool_name="Read",
                content="first",
            )
        )

        ctx.add_hook_hidden_user_message(
            "after-all-tools",
            hook_name="PostToolUse",
            related_tool_call_id="tc1",
        )
        self.assertEqual([m.role for m in ctx.conversation_messages], ["assistant", "tool"])
        self.assertEqual(ctx.pop_flushed_hook_injection_texts(), [])

        ctx.add_message(
            ToolMessage(
                tool_call_id="tc2",
                tool_name="Read",
                content="second",
            )
        )
        self.assertEqual(
            [m.role for m in ctx.conversation_messages],
            ["assistant", "tool", "tool", "user"],
        )
        self.assertEqual(ctx.pop_flushed_hook_injection_texts(), ["after-all-tools"])

    def test_hook_injection_is_immediate_when_no_tool_barrier(self) -> None:
        ctx = ContextIR()
        ctx.add_hook_hidden_user_message("session-note", hook_name="SessionStart")

        messages = ctx.conversation_messages
        self.assertEqual(len(messages), 1)
        self.assertTrue(isinstance(messages[0], UserMessage))
        self.assertTrue(messages[0].is_meta)
        self.assertEqual(ctx.pop_flushed_hook_injection_texts(), ["session-note"])

    def test_skill_flush_is_deferred_until_tool_barrier_released(self) -> None:
        ctx = ContextIR()
        ctx.add_message(
            AssistantMessage(
                content=None,
                tool_calls=[_build_tool_call("tc1"), _build_tool_call("tc2")],
            )
        )
        ctx.add_skill_injection(
            skill_name="skill-a",
            metadata_msg=UserMessage(content="m-a", is_meta=False),
            prompt_msg=UserMessage(content="p-a", is_meta=True),
        )
        ctx.flush_pending_skill_items()
        self.assertEqual([m.role for m in ctx.conversation_messages], ["assistant"])

        ctx.add_message(
            ToolMessage(
                tool_call_id="tc1",
                tool_name="Read",
                content="first",
            )
        )
        ctx.flush_pending_skill_items()
        self.assertEqual([m.role for m in ctx.conversation_messages], ["assistant", "tool"])

        ctx.add_message(
            ToolMessage(
                tool_call_id="tc2",
                tool_name="Read",
                content="second",
            )
        )
        types = [item.item_type for item in ctx.conversation.items]
        self.assertEqual(
            types,
            [
                ItemType.ASSISTANT_MESSAGE,
                ItemType.TOOL_RESULT,
                ItemType.TOOL_RESULT,
                ItemType.SKILL_METADATA,
                ItemType.SKILL_PROMPT,
            ],
        )

    def test_multiple_skill_injections_queue_without_overwrite(self) -> None:
        ctx = ContextIR()
        ctx.add_message(
            AssistantMessage(
                content=None,
                tool_calls=[_build_tool_call("tc1")],
            )
        )
        ctx.add_skill_injection(
            skill_name="skill-a",
            metadata_msg=UserMessage(content="m-a", is_meta=False),
            prompt_msg=UserMessage(content="p-a", is_meta=True),
        )
        ctx.add_skill_injection(
            skill_name="skill-b",
            metadata_msg=UserMessage(content="m-b", is_meta=False),
            prompt_msg=UserMessage(content="p-b", is_meta=True),
        )
        ctx.add_message(
            ToolMessage(
                tool_call_id="tc1",
                tool_name="Read",
                content="ok",
            )
        )

        skill_items = [item for item in ctx.conversation.items if item.item_type in {ItemType.SKILL_METADATA, ItemType.SKILL_PROMPT}]
        self.assertEqual(len(skill_items), 4)
        self.assertEqual(
            [item.metadata.get("skill_name") for item in skill_items],
            ["skill-a", "skill-a", "skill-b", "skill-b"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
