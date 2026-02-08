import asyncio
import json
import unittest
from types import SimpleNamespace

from comate_agent_sdk.context.compaction import (
    CompactionStrategy,
    SelectiveCompactionPolicy,
    TypeCompactionRule,
)
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    Function,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class _EmptySummaryLLM:
    model = "fake-model"

    async def ainvoke(self, *, messages, tools=None, tool_choice=None):
        return SimpleNamespace(content="", usage=None)


class _ExceptionSummaryLLM:
    model = "fake-model"

    async def ainvoke(self, *, messages, tools=None, tool_choice=None):
        raise RuntimeError("summary invocation failed")


class _SummaryFalsePolicy(SelectiveCompactionPolicy):
    async def _fallback_full_summary_with_retry(self, context: ContextIR) -> tuple[bool, str]:  # noqa: D401
        return False, "compact_false"


class _FlakySummaryLLM:
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, *, messages, tools=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(content="", usage=None, stop_reason="end_turn", thinking=None)
        return SimpleNamespace(content="<summary>ok summary</summary>", usage=None, stop_reason="end_turn", thinking=None)


class TestCompactionRoundGuardAndRollback(unittest.TestCase):
    def _add_tool_block(
        self,
        ctx: ContextIR,
        *,
        call_id: str,
        tool_name: str = "Read",
    ) -> None:
        ctx.add_message(
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        type="function",
                        function=Function(name=tool_name, arguments='{"path":"a.txt"}'),
                    )
                ],
            )
        )
        ctx.add_message(
            ToolMessage(
                tool_call_id=call_id,
                tool_name=tool_name,
                content=f"result-{call_id}",
            )
        )

    def _conversation_signature(self, ctx: ContextIR) -> str:
        rows = []
        for item in ctx.conversation.items:
            message_dump = item.message.model_dump(mode="python") if item.message is not None else None
            rows.append(
                {
                    "id": item.id,
                    "item_type": item.item_type.value,
                    "content_text": item.content_text,
                    "token_count": item.token_count,
                    "metadata": item.metadata,
                    "message": message_dump,
                }
            )
        return json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)

    def test_recent_12_rounds_are_protected_and_meta_not_counted(self) -> None:
        ctx = ContextIR()
        ctx.add_message(UserMessage(content="meta-header", is_meta=True))

        for i in range(1, 15):
            ctx.add_message(UserMessage(content=f"user-{i}", is_meta=False))
            ctx.add_message(UserMessage(content=f"meta-{i}", is_meta=True))
            ctx.add_message(AssistantMessage(content=f"assistant-{i}", tool_calls=None))

        rules = dict(SelectiveCompactionPolicy().rules)
        rules[ItemType.USER_MESSAGE.value] = TypeCompactionRule(
            strategy=CompactionStrategy.TRUNCATE,
            keep_recent=0,
        )
        rules[ItemType.ASSISTANT_MESSAGE.value] = TypeCompactionRule(
            strategy=CompactionStrategy.TRUNCATE,
            keep_recent=0,
        )

        policy = SelectiveCompactionPolicy(
            threshold=1,
            fallback_to_full_summary=False,
            rules=rules,
            dialogue_rounds_keep_min=12,
        )
        compacted = asyncio.run(policy.compact(ctx))
        self.assertTrue(compacted)

        real_users = [
            item.message.text
            for item in ctx.conversation.items
            if item.item_type == ItemType.USER_MESSAGE
            and isinstance(item.message, UserMessage)
            and not item.message.is_meta
        ]
        assistants = [
            item.message.text
            for item in ctx.conversation.items
            if item.item_type == ItemType.ASSISTANT_MESSAGE
            and isinstance(item.message, AssistantMessage)
        ]

        self.assertEqual(real_users, [f"user-{i}" for i in range(3, 15)])
        self.assertEqual(assistants, [f"assistant-{i}" for i in range(3, 15)])

    def test_rollback_on_summary_failed_result(self) -> None:
        ctx = ContextIR()
        for i in range(8):
            self._add_tool_block(ctx, call_id=f"call-{i}")
        before = self._conversation_signature(ctx)

        policy = _SummaryFalsePolicy(threshold=1, fallback_to_full_summary=True)
        compacted = asyncio.run(policy.compact(ctx))

        self.assertFalse(compacted)
        self.assertEqual(before, self._conversation_signature(ctx))
        self.assertEqual(policy.meta_records[-1].phase, "rollback")

    def test_rollback_on_empty_summary(self) -> None:
        ctx = ContextIR()
        for i in range(8):
            self._add_tool_block(ctx, call_id=f"call-{i}")
        before = self._conversation_signature(ctx)

        policy = SelectiveCompactionPolicy(
            threshold=1,
            llm=_EmptySummaryLLM(),
            fallback_to_full_summary=True,
        )
        compacted = asyncio.run(policy.compact(ctx))

        self.assertFalse(compacted)
        self.assertEqual(before, self._conversation_signature(ctx))
        self.assertEqual(policy.meta_records[-1].phase, "rollback")

    def test_rollback_on_summary_exception(self) -> None:
        ctx = ContextIR()
        for i in range(8):
            self._add_tool_block(ctx, call_id=f"call-{i}")
        before = self._conversation_signature(ctx)

        policy = SelectiveCompactionPolicy(
            threshold=1,
            llm=_ExceptionSummaryLLM(),
            fallback_to_full_summary=True,
        )
        compacted = asyncio.run(policy.compact(ctx))

        self.assertFalse(compacted)
        self.assertEqual(before, self._conversation_signature(ctx))
        self.assertEqual(policy.meta_records[-1].phase, "rollback")

    def test_summary_retry_recovers_from_first_empty_attempt(self) -> None:
        ctx = ContextIR()
        for i in range(8):
            self._add_tool_block(ctx, call_id=f"call-{i}")

        llm = _FlakySummaryLLM()
        policy = SelectiveCompactionPolicy(
            threshold=1,
            llm=llm,
            fallback_to_full_summary=True,
            summary_retry_attempts=2,
        )
        compacted = asyncio.run(policy.compact(ctx))

        self.assertTrue(compacted)
        self.assertGreaterEqual(llm.calls, 2)
        self.assertEqual(len(ctx.conversation.items), 1)
        self.assertEqual(ctx.conversation.items[0].item_type, ItemType.COMPACTION_SUMMARY)


if __name__ == "__main__":
    unittest.main()
