import asyncio
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.agent.chat_session import ChatSession
from comate_agent_sdk.agent.session_store import (
    ConversationState as _ConversationState,
    build_conversation_event as _build_conversation_event,
    events_jsonl_append as _events_jsonl_append,
    replay_conversation_events as _replay_conversation_events,
    replay_session_events as _replay_session_events,
)
from comate_agent_sdk.context.items import ContextItem, ItemType
from comate_agent_sdk.llm.messages import AssistantMessage, ToolMessage, UserMessage
from comate_agent_sdk.llm.views import ChatInvokeUsage
from comate_agent_sdk.tokens.views import TokenUsageEntry


class _FakeChatModel:
    def __init__(self) -> None:
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("This test should not call the LLM")


class TestChatSessionPersistence(unittest.TestCase):
    @staticmethod
    def _make_usage_entry(
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        level: str = "MID",
    ) -> TokenUsageEntry:
        return TokenUsageEntry(
            model=model,
            timestamp=datetime.now(),
            usage=ChatInvokeUsage(
                prompt_tokens=prompt_tokens,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            level=level,
            source="agent",
        )

    def test_chat_session_constructs_with_session_options_clone(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent = Agent(
                llm=_FakeChatModel(),  # type: ignore[arg-type]
                config=AgentConfig(
                    tools=(),
                    agents=(),
                    offload_enabled=False,
                    setting_sources=None,
                ),
            )

            self.assertIsNone(agent.config.session_id)
            self.assertIsNone(agent.config.offload_root_path)

            session = ChatSession(
                agent,
                session_id="s1",
                storage_root=root / "session",
            )

            self.assertEqual(session.session_id, "s1")
            self.assertEqual(session._agent.session_id, "s1")
            self.assertEqual(session._agent.offload_root_path, str(root / "session" / "offload"))

            # 模板 agent 不应被 ChatSession 构造污染
            self.assertIsNone(agent.config.session_id)
            self.assertIsNone(agent.config.offload_root_path)

    def test_delta_append_and_replay_basic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            offload_root = root / "offload"
            path = root / "context.jsonl"

            before = _ConversationState.capture([])
            items = [
                ContextItem(
                    id="u1",
                    item_type=ItemType.USER_MESSAGE,
                    message=UserMessage(content="hi"),
                    content_text="hi",
                    token_count=2,
                ),
                ContextItem(
                    id="a1",
                    item_type=ItemType.ASSISTANT_MESSAGE,
                    message=AssistantMessage(content="hello"),
                    content_text="hello",
                    token_count=5,
                ),
            ]

            evt = _build_conversation_event(
                session_id="s1",
                turn_number=1,
                before=before,
                after_items=items,
                offload_root=offload_root,
            )
            _events_jsonl_append(path, evt)

            turn, replayed = _replay_conversation_events(path=path, offload_root=offload_root)
            self.assertEqual(turn, 1)
            self.assertEqual([i.id for i in replayed], ["u1", "a1"])
            self.assertEqual([i.item_type for i in replayed], [ItemType.USER_MESSAGE, ItemType.ASSISTANT_MESSAGE])

            # 确认没有把 header/system prompt 写进文件结构里
            raw = path.read_text(encoding="utf-8").strip()
            data = json.loads(raw)
            self.assertNotIn("context", data)

    def test_tool_offload_path_persisted_relative_and_rehydrated_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            offload_root = root / "offload"
            offload_root.mkdir(parents=True, exist_ok=True)
            path = root / "context.jsonl"

            rel = "tool_result/read/t1.json"
            abs_path = str(offload_root / rel)

            tool_msg = ToolMessage(
                tool_call_id="tc1",
                tool_name="read",
                content="very large content",
                is_error=False,
                destroyed=True,
                offloaded=True,
                offload_path=abs_path,
            )
            item = ContextItem(
                id="t1",
                item_type=ItemType.TOOL_RESULT,
                message=tool_msg,
                content_text="very large content",
                token_count=1000,
                destroyed=True,
                offloaded=True,
                offload_path=rel,
                tool_name="read",
            )

            before = _ConversationState.capture([])
            evt = _build_conversation_event(
                session_id="s1",
                turn_number=1,
                before=before,
                after_items=[item],
                offload_root=offload_root,
            )
            _events_jsonl_append(path, evt)

            line = path.read_text(encoding="utf-8").strip()
            data = json.loads(line)
            adds = data["conversation"]["adds"]
            self.assertEqual(adds[0]["offload_path"], rel)
            self.assertEqual(adds[0]["message"]["offload_path"], rel)
            self.assertEqual(adds[0]["message"]["content"], "")

            _, replayed = _replay_conversation_events(path=path, offload_root=offload_root)
            self.assertEqual(len(replayed), 1)
            replay_item = replayed[0]
            self.assertEqual(replay_item.offload_path, rel)
            self.assertIsInstance(replay_item.message, ToolMessage)
            self.assertEqual(replay_item.message.offload_path, abs_path)
            self.assertTrue(replay_item.message.destroyed)
            self.assertTrue(replay_item.message.offloaded)

    def test_updates_and_removes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            offload_root = root / "offload"
            offload_root.mkdir(parents=True, exist_ok=True)
            path = root / "context.jsonl"

            u1 = ContextItem(
                id="u1",
                item_type=ItemType.USER_MESSAGE,
                message=UserMessage(content="q"),
                content_text="q",
                token_count=1,
            )
            t1_msg = ToolMessage(tool_call_id="tc1", tool_name="read", content="x", is_error=False)
            t1 = ContextItem(
                id="t1",
                item_type=ItemType.TOOL_RESULT,
                message=t1_msg,
                content_text="x",
                token_count=1,
                tool_name="read",
            )
            a1 = ContextItem(
                id="a1",
                item_type=ItemType.ASSISTANT_MESSAGE,
                message=AssistantMessage(content="a"),
                content_text="a",
                token_count=1,
            )

            before = _ConversationState.capture([])
            evt1 = _build_conversation_event(
                session_id="s1",
                turn_number=1,
                before=before,
                after_items=[u1, t1, a1],
                offload_root=offload_root,
            )
            _events_jsonl_append(path, evt1)

            # 第二轮：先捕获 before，再模拟 t1 被 destroyed + offloaded，同时删除 a1
            before2 = _ConversationState.capture([u1, t1, a1])

            rel = "tool_result/read/t1.json"
            t1.destroyed = True
            t1.offloaded = True
            t1.offload_path = rel
            t1_msg.destroyed = True
            t1_msg.offloaded = True
            t1_msg.offload_path = str(offload_root / rel)

            evt2 = _build_conversation_event(
                session_id="s1",
                turn_number=2,
                before=before2,
                after_items=[u1, t1],
                offload_root=offload_root,
            )
            _events_jsonl_append(path, evt2)

            turn, replayed = _replay_conversation_events(path=path, offload_root=offload_root)
            self.assertEqual(turn, 2)
            self.assertEqual([i.id for i in replayed], ["u1", "t1"])
            replay_t1 = replayed[1]
            self.assertTrue(replay_t1.destroyed)
            self.assertTrue(replay_t1.offloaded)
            self.assertEqual(replay_t1.offload_path, rel)
            self.assertIsInstance(replay_t1.message, ToolMessage)
            self.assertTrue(replay_t1.message.destroyed)
            self.assertTrue(replay_t1.message.offloaded)
            self.assertEqual(replay_t1.message.offload_path, str(offload_root / rel))

    def test_reset_on_reorder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            offload_root = root / "offload"
            path = root / "context.jsonl"

            a = ContextItem(
                id="a",
                item_type=ItemType.USER_MESSAGE,
                message=UserMessage(content="a"),
                content_text="a",
                token_count=1,
            )
            b = ContextItem(
                id="b",
                item_type=ItemType.ASSISTANT_MESSAGE,
                message=AssistantMessage(content="b"),
                content_text="b",
                token_count=1,
            )

            evt1 = _build_conversation_event(
                session_id="s1",
                turn_number=1,
                before=_ConversationState.capture([]),
                after_items=[a, b],
                offload_root=offload_root,
            )
            _events_jsonl_append(path, evt1)

            evt2 = _build_conversation_event(
                session_id="s1",
                turn_number=2,
                before=_ConversationState.capture([a, b]),
                after_items=[b, a],
                offload_root=offload_root,
            )
            self.assertEqual(evt2["op"], "conversation_reset")
            _events_jsonl_append(path, evt2)

            turn, replayed = _replay_conversation_events(path=path, offload_root=offload_root)
            self.assertEqual(turn, 2)
            self.assertEqual([i.id for i in replayed], ["b", "a"])

    def test_replay_session_events_includes_usage_delta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            offload_root = root / "offload"
            path = root / "context.jsonl"

            usage_entry = self._make_usage_entry(
                model="m1",
                prompt_tokens=12,
                completion_tokens=8,
            )
            _events_jsonl_append(
                path,
                {
                    "schema_version": "2.0",
                    "session_id": "s1",
                    "turn_number": 1,
                    "op": "usage_delta",
                    "usage": {
                        "adds": [usage_entry.model_dump(mode="json")]
                    },
                },
            )

            turn, replayed_items, replayed_usage = _replay_session_events(
                path=path,
                offload_root=offload_root,
            )
            self.assertEqual(turn, 1)
            self.assertEqual(replayed_items, [])
            self.assertEqual(len(replayed_usage), 1)
            self.assertEqual(replayed_usage[0].model, "m1")
            self.assertEqual(replayed_usage[0].usage.total_tokens, 20)

    def test_replay_session_events_usage_reset_clears_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            offload_root = root / "offload"
            path = root / "context.jsonl"

            entry_a = self._make_usage_entry(
                model="m1",
                prompt_tokens=10,
                completion_tokens=5,
            )
            entry_b = self._make_usage_entry(
                model="m2",
                prompt_tokens=6,
                completion_tokens=4,
            )
            entry_c = self._make_usage_entry(
                model="m3",
                prompt_tokens=3,
                completion_tokens=2,
            )

            _events_jsonl_append(
                path,
                {
                    "schema_version": "2.0",
                    "session_id": "s1",
                    "turn_number": 1,
                    "op": "usage_delta",
                    "usage": {
                        "adds": [
                            entry_a.model_dump(mode="json"),
                            entry_b.model_dump(mode="json"),
                        ]
                    },
                },
            )
            _events_jsonl_append(
                path,
                {
                    "schema_version": "2.0",
                    "session_id": "s1",
                    "turn_number": 2,
                    "op": "usage_reset",
                    "usage": {"entries": []},
                },
            )
            _events_jsonl_append(
                path,
                {
                    "schema_version": "2.0",
                    "session_id": "s1",
                    "turn_number": 3,
                    "op": "usage_delta",
                    "usage": {"adds": [entry_c.model_dump(mode="json")]},
                },
            )

            turn, _, replayed_usage = _replay_session_events(path=path, offload_root=offload_root)
            self.assertEqual(turn, 3)
            self.assertEqual(len(replayed_usage), 1)
            self.assertEqual(replayed_usage[0].model, "m3")
            self.assertEqual(replayed_usage[0].usage.total_tokens, 5)

    def test_chat_session_resume_restores_usage_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session_root = root / "session"
            path = session_root / "context.jsonl"
            usage_entry = self._make_usage_entry(
                model="m1",
                prompt_tokens=11,
                completion_tokens=9,
            )
            _events_jsonl_append(
                path,
                {
                    "schema_version": "2.0",
                    "session_id": "s1",
                    "turn_number": 1,
                    "op": "usage_delta",
                    "usage": {
                        "adds": [usage_entry.model_dump(mode="json")]
                    },
                },
            )

            agent = Agent(
                llm=_FakeChatModel(),  # type: ignore[arg-type]
                config=AgentConfig(
                    tools=(),
                    agents=(),
                    offload_enabled=False,
                    setting_sources=None,
                ),
            )
            session = ChatSession.resume(
                agent,
                session_id="s1",
                storage_root=session_root,
            )

            usage = asyncio.run(session.get_usage())
            self.assertEqual(usage.entry_count, 1)
            self.assertEqual(usage.total_prompt_tokens, 11)
            self.assertEqual(usage.total_completion_tokens, 9)
            self.assertEqual(usage.total_tokens, 20)

    def test_clear_history_writes_usage_reset_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session_root = root / "session"
            agent = Agent(
                llm=_FakeChatModel(),  # type: ignore[arg-type]
                config=AgentConfig(
                    tools=(),
                    agents=(),
                    offload_enabled=False,
                    setting_sources=None,
                ),
            )

            session = ChatSession(
                agent,
                session_id="s1",
                storage_root=session_root,
            )
            session.clear_history()

            lines = [
                json.loads(line)
                for line in (session_root / "context.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([line["op"] for line in lines], ["conversation_reset", "usage_reset"])
            self.assertEqual(lines[0]["turn_number"], 1)
            self.assertEqual(lines[1]["turn_number"], 1)


if __name__ == "__main__":
    unittest.main()
