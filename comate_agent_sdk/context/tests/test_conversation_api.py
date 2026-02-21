import unittest

from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.observer import EventType
from comate_agent_sdk.llm.messages import AssistantMessage, UserMessage


class TestConversationApi(unittest.TestCase):
    def test_clear_conversation_emits_replaced_event(self) -> None:
        ctx = ContextIR()
        ctx.add_message(UserMessage(content="hello"))
        ctx.add_message(AssistantMessage(content="world"))

        replaced_before = sum(
            1 for event in ctx.event_bus.event_log
            if event.event_type == EventType.CONVERSATION_REPLACED
        )
        ctx.clear_conversation()
        replaced_after = sum(
            1 for event in ctx.event_bus.event_log
            if event.event_type == EventType.CONVERSATION_REPLACED
        )

        self.assertEqual(replaced_after, replaced_before + 1)
        self.assertEqual(len(ctx.get_conversation_items_snapshot()), 0)

    def test_snapshot_isolated_from_internal_list(self) -> None:
        ctx = ContextIR()
        ctx.add_message(UserMessage(content="hello"))

        snapshot = ctx.get_conversation_items_snapshot()
        snapshot.clear()

        self.assertEqual(len(ctx.get_conversation_items_snapshot()), 1)

    def test_iter_conversation_items_uses_stable_snapshot(self) -> None:
        ctx = ContextIR()
        first_item = ctx.add_message(UserMessage(content="first"))
        iterator = ctx.iter_conversation_items()
        ctx.add_message(UserMessage(content="second"))

        iterated_ids = [item.id for item in iterator]
        current_ids = [item.id for item in ctx.get_conversation_items_snapshot()]

        self.assertEqual(iterated_ids, [first_item.id])
        self.assertEqual(len(current_ids), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
