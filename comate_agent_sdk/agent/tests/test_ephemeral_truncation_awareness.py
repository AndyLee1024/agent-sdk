import unittest
from types import SimpleNamespace

from comate_agent_sdk.agent.history import destroy_ephemeral_messages
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.truncation import TruncationRecord
from comate_agent_sdk.llm.messages import ToolMessage


class _FakeTool:
    def __init__(self, name: str, ephemeral: int | bool) -> None:
        self.name = name
        self.ephemeral = ephemeral


class _FakeAgent:
    def __init__(self, *, keep_recent: int) -> None:
        self.options = SimpleNamespace(
            offload_policy=None,
            offload_token_threshold=1000,
            tools=[_FakeTool("Read", keep_recent)],
            ephemeral_keep_recent=None,
        )
        self._context = ContextIR()
        self._context_fs = None


class TestEphemeralTruncationAwareness(unittest.TestCase):
    def _add_tool_result(
        self,
        agent: _FakeAgent,
        *,
        call_id: str,
        text: str,
        truncation_record: TruncationRecord | None,
    ) -> str:
        item = agent._context.add_message(
            ToolMessage(
                tool_call_id=call_id,
                tool_name="Read",
                content=text,
                ephemeral=True,
                truncation_record=truncation_record,
            )
        )
        return item.id

    def _count_destroyed(self, agent: _FakeAgent) -> int:
        return sum(1 for item in agent._context.conversation.items if item.destroyed)

    def _get_item_by_id(self, agent: _FakeAgent, item_id: str):
        for item in agent._context.conversation.items:
            if item.id == item_id:
                return item
        raise AssertionError(f"item not found: {item_id}")

    def test_small_formatter_truncated_item_survives(self) -> None:
        agent = _FakeAgent(keep_recent=2)
        small = "small " * 30
        large = "large " * 800

        protected_old_id = self._add_tool_result(
            agent,
            call_id="tc_1",
            text=small,
            truncation_record=TruncationRecord(
                formatter_truncated=True,
                formatter_reason="line_limit",
            ),
        )
        self._add_tool_result(
            agent,
            call_id="tc_2",
            text=large,
            truncation_record=None,
        )
        self._add_tool_result(
            agent,
            call_id="tc_3",
            text=large,
            truncation_record=None,
        )
        self._add_tool_result(
            agent,
            call_id="tc_4",
            text=small,
            truncation_record=TruncationRecord(
                formatter_truncated=True,
                formatter_reason="line_limit",
            ),
        )

        destroy_ephemeral_messages(agent)
        protected_old_item = self._get_item_by_id(agent, protected_old_id)

        self.assertFalse(protected_old_item.destroyed)
        self.assertEqual(self._count_destroyed(agent), 2)

    def test_large_formatter_truncated_item_still_destroyed(self) -> None:
        agent = _FakeAgent(keep_recent=1)
        large = "large " * 800
        small = "small " * 20

        large_truncated_id = self._add_tool_result(
            agent,
            call_id="tc_large",
            text=large,
            truncation_record=TruncationRecord(
                formatter_truncated=True,
                formatter_reason="line_limit",
            ),
        )
        self._add_tool_result(
            agent,
            call_id="tc_small",
            text=small,
            truncation_record=TruncationRecord(
                formatter_truncated=True,
                formatter_reason="line_limit",
            ),
        )

        destroy_ephemeral_messages(agent)
        item = self._get_item_by_id(agent, large_truncated_id)
        self.assertTrue(item.destroyed)

    def test_all_protected_still_destroys(self) -> None:
        agent = _FakeAgent(keep_recent=1)
        small = "small " * 20

        id1 = self._add_tool_result(
            agent,
            call_id="tc_1",
            text=small,
            truncation_record=TruncationRecord(formatter_truncated=True, formatter_reason="line_limit"),
        )
        id2 = self._add_tool_result(
            agent,
            call_id="tc_2",
            text=small,
            truncation_record=TruncationRecord(formatter_truncated=True, formatter_reason="line_limit"),
        )
        id3 = self._add_tool_result(
            agent,
            call_id="tc_3",
            text=small,
            truncation_record=TruncationRecord(formatter_truncated=True, formatter_reason="line_limit"),
        )

        destroy_ephemeral_messages(agent)
        destroyed = [self._get_item_by_id(agent, iid).destroyed for iid in [id1, id2, id3]]

        self.assertEqual(sum(1 for flag in destroyed if flag), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
