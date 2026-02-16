from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path

from comate_agent_sdk.agent.events import AgentEvent, LLMSwitchedEvent, SessionInitEvent, StopEvent
from comate_agent_sdk.agent.llm_levels import LLMLevel
from comate_agent_sdk.agent.interrupt import SessionRunController
from comate_agent_sdk.agent.service import AgentTemplate, AgentRuntime
from comate_agent_sdk.agent.session_store import (
    PERSISTENCE_SCHEMA_VERSION,
    ConversationState,
    build_conversation_event,
    build_usage_event,
    default_session_root,
    events_jsonl_append,
    replay_session_events,
    rewrite_session_id_in_jsonl,
    slice_usage_delta,
)
from comate_agent_sdk.llm.messages import ContentPartImageParam, ContentPartTextParam
from comate_agent_sdk.tokens.views import UsageSummary

logger = logging.getLogger("comate_agent_sdk.agent.chat_session")

ChatMessage = str | list[ContentPartTextParam | ContentPartImageParam]
MessageSource = AsyncIterator[ChatMessage] | Iterator[ChatMessage] | Iterable[ChatMessage]


class ChatSessionError(Exception):
    pass


class ChatSessionClosedError(ChatSessionError):
    pass


# Backward-compatible aliases for test code that imports private names.
_ConversationState = ConversationState
_build_conversation_event = build_conversation_event
_events_jsonl_append = events_jsonl_append
_replay_conversation_events = None  # replaced; see session_store.replay_conversation_events
_replay_session_events = replay_session_events


class ChatSession:
    def __init__(
        self,
        template: AgentTemplate,
        *,
        runtime: AgentRuntime | None = None,
        session_id: str | None = None,
        storage_root: Path | None = None,
        message_source: MessageSource | None = None,
    ):
        resolved_session_id = session_id or (runtime.session_id if runtime is not None else None)
        self.session_id = resolved_session_id or str(uuid.uuid4())
        self._storage_root = storage_root or default_session_root(self.session_id)
        self._offload_root = self._storage_root / "offload"
        self._context_jsonl = self._storage_root / "context.jsonl"

        self._template = template
        # Backward-compatible alias for existing call sites/tests.
        self._template_agent = template
        if runtime is None:
            self._runtime = template.create_runtime(
                session_id=self.session_id,
                offload_root_path=str(self._offload_root),
            )
        else:
            if runtime.session_id != self.session_id:
                raise ChatSessionError(
                    f"runtime.session_id={runtime.session_id} does not match session_id={self.session_id}"
                )
            self._runtime = runtime
        # Backward-compatible alias for existing call sites/tests.
        self._agent = self._runtime

        self._closed = False
        self._turn_number = 0
        self._init_event_emitted = False

        self._events_started = False
        self._queue: asyncio.Queue[ChatMessage | None] = asyncio.Queue()
        self._message_source = message_source
        self.run_controller = SessionRunController()

    @classmethod
    def resume(
        cls,
        template: AgentTemplate,
        *,
        session_id: str,
        storage_root: Path | None = None,
        message_source: MessageSource | None = None,
    ) -> "ChatSession":
        runtime = template.create_runtime(
            session_id=session_id,
            offload_root_path=str((storage_root or default_session_root(session_id)) / "offload"),
        )
        session = cls(
            template,
            runtime=runtime,
            session_id=session_id,
            storage_root=storage_root,
            message_source=message_source,
        )

        turn_number, items, usage_entries = replay_session_events(
            path=session._context_jsonl,
            offload_root=session._offload_root,
        )
        session._turn_number = int(turn_number)
        if items:
            session._agent._context.conversation.items = items
        else:
            logger.info(f"No persisted conversation found for session_id={session_id}, starting empty")
        session._agent._token_cost.usage_history = usage_entries

        session._agent._context.set_turn_number(session._turn_number)

        todo_path = session._storage_root / "todos.json"
        if todo_path.exists():
            try:
                data = json.loads(todo_path.read_text(encoding="utf-8"))
                todos = data.get("todos", []) or []
                turn_number_at_update = int(data.get("turn_number_at_update", 0) or 0)
                session._agent._context.restore_todo_state(
                    todos=todos,
                    turn_number_at_update=turn_number_at_update,
                )
                logger.info(f"Restored {len(todos)} TODO items from todos.json")
            except Exception as e:
                logger.warning(f"Failed to restore TODO state for session_id={session_id}: {e}", exc_info=True)

        # MCP tools：resume 后标记 dirty，下一次 invoke_llm 前刷新
        try:
            session._agent.invalidate_mcp_tools(reason="session_resume")
        except Exception:
            pass

        return session

    def fork_session(
        self,
        *,
        storage_root: Path | None = None,
        message_source: MessageSource | None = None,
    ) -> "ChatSession":
        new_session_id = str(uuid.uuid4())
        src = self._storage_root
        dst = storage_root or default_session_root(new_session_id)

        if self._closed:
            raise ChatSessionClosedError("Cannot fork a closed session")

        if dst.exists():
            raise ChatSessionError(f"Fork destination already exists: {dst}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

        rewrite_session_id_in_jsonl(dst / "context.jsonl", session_id=new_session_id)

        offload_index = dst / "offload" / "index.json"
        if offload_index.exists():
            try:
                data = json.loads(offload_index.read_text(encoding="utf-8"))
                data["session_id"] = new_session_id
                offload_index.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                logger.warning(f"Failed to rewrite offload index session_id: {e}")

        return ChatSession.resume(
            self._template,
            session_id=new_session_id,
            storage_root=dst,
            message_source=message_source,
        )

    async def get_usage(
        self,
        *,
        model: str | None = None,
        source_prefix: str | None = None,
    ) -> UsageSummary:
        """获取会话的 token 使用统计。"""
        return await self._agent.get_usage(
            model=model,
            source_prefix=source_prefix,
        )

    async def get_context_info(self):
        """获取当前会话的上下文使用情况。"""
        return await self._agent.get_context_info()

    def clear_history(self) -> None:
        """清空会话历史和 token 使用记录。"""
        if self._closed:
            raise ChatSessionClosedError("Cannot clear history on a closed session")

        self._agent.clear_history()

        self._turn_number += 1
        reset_event = {
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "session_id": self.session_id,
            "turn_number": self._turn_number,
            "op": "conversation_reset",
            "conversation": {
                "items": []
            },
        }
        events_jsonl_append(self._context_jsonl, reset_event)
        usage_reset_event = {
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "session_id": self.session_id,
            "turn_number": self._turn_number,
            "op": "usage_reset",
            "usage": {
                "entries": []
            },
        }
        events_jsonl_append(self._context_jsonl, usage_reset_event)

    def set_level(self, level: LLMLevel) -> LLMSwitchedEvent:
        """Set the LLM level for this session."""
        if self._closed:
            raise ChatSessionClosedError("Cannot set level on a closed session")

        previous_level = self._agent.level
        previous_model = self._agent.llm.model if self._agent.llm else None

        self._agent.level = level

        if self._agent.llm_levels and level in self._agent.llm_levels:
            self._agent.llm = self._agent.llm_levels[level]

        new_model = None
        if self._agent.llm_levels and level in self._agent.llm_levels:
            new_model = self._agent.llm_levels[level].model

        event = LLMSwitchedEvent(
            previous_level=previous_level,
            new_level=level,
            previous_model=previous_model,
            new_model=new_model,
        )

        logger.info(f"Session {self.session_id}: {event}")
        return event

    def set_thinking_budget(self, budget_tokens: int | None) -> None:
        """Enable or disable extended thinking on the current LLM.

        Supports Anthropic, Google, and OpenAI providers. Each provider
        interprets the budget parameter differently:
        - Anthropic: Direct token budget for thinking
        - Google: Direct token budget for thinking
        - OpenAI: Converted to reasoning effort (None="low", >0="medium")

        Args:
            budget_tokens: Token budget for thinking, or None to disable.

        Raises:
            ChatSessionClosedError: If session is closed.
            TypeError: If the current LLM does not support thinking.
        """
        if self._closed:
            raise ChatSessionClosedError("Cannot set thinking on a closed session")

        llm = self._agent.llm
        if not hasattr(llm, "set_thinking_budget"):
            raise TypeError(f"LLM {type(llm).__name__} does not support thinking")

        llm.set_thinking_budget(budget_tokens)

    async def send(self, message: ChatMessage) -> None:
        if self._closed:
            raise ChatSessionClosedError("ChatSession is closed")
        if self._message_source is not None:
            raise ChatSessionError("send() is not allowed when message_source is provided")
        await self._queue.put(message)

    async def close(self) -> None:
        if self._closed:
            return
        if self._agent._hooks_session_started and not self._agent._hooks_session_ended:
            await self._agent.run_hook_event("SessionEnd")
            self._agent._hooks_session_ended = True
        # Runtime may be reused across ChatSession instances.
        self._agent._hooks_session_started = False
        self._agent._hooks_session_ended = False
        self._closed = True
        await self._queue.put(None)

    async def __aenter__(self) -> "ChatSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def query_stream(self, message: ChatMessage) -> AsyncIterator[AgentEvent]:
        if self._closed:
            raise ChatSessionClosedError("ChatSession is closed")

        if not self._init_event_emitted:
            self._init_event_emitted = True
            yield SessionInitEvent(session_id=self.session_id)

        self._turn_number += 1
        before = ConversationState.capture(self._agent._context.conversation.items)
        before_usage_count = len(self._agent._token_cost.usage_history)
        previous_controller = self._agent._run_controller
        self.run_controller.clear()
        self._agent._run_controller = self.run_controller
        try:
            async for event in self._agent.query_stream(message):  # type: ignore[arg-type]
                yield event
        finally:
            self._agent._run_controller = previous_controller
            self.run_controller.clear()
            evt = build_conversation_event(
                session_id=self.session_id,
                turn_number=self._turn_number,
                before=before,
                after_items=self._agent._context.conversation.items,
                offload_root=self._offload_root,
            )
            events_jsonl_append(self._context_jsonl, evt)
            usage_evt = build_usage_event(
                session_id=self.session_id,
                turn_number=self._turn_number,
                added_entries=slice_usage_delta(
                    self._agent._token_cost.usage_history,
                    before_count=before_usage_count,
                ),
            )
            if usage_evt is not None:
                events_jsonl_append(self._context_jsonl, usage_evt)

    async def events(self) -> AsyncIterator[AgentEvent]:
        if self._events_started:
            raise ChatSessionError("events() can only be consumed once")
        self._events_started = True

        if not self._init_event_emitted:
            self._init_event_emitted = True
            yield SessionInitEvent(session_id=self.session_id)

        async for message in self._message_iter():
            self._turn_number += 1
            before = ConversationState.capture(self._agent._context.conversation.items)
            before_usage_count = len(self._agent._token_cost.usage_history)
            previous_controller = self._agent._run_controller
            self.run_controller.clear()
            self._agent._run_controller = self.run_controller
            try:
                async for event in self._agent.query_stream(message):  # type: ignore[arg-type]
                    yield event
                    if isinstance(event, StopEvent):
                        break
            finally:
                self._agent._run_controller = previous_controller
                self.run_controller.clear()
                evt = build_conversation_event(
                    session_id=self.session_id,
                    turn_number=self._turn_number,
                    before=before,
                    after_items=self._agent._context.conversation.items,
                    offload_root=self._offload_root,
                )
                events_jsonl_append(self._context_jsonl, evt)
                usage_evt = build_usage_event(
                    session_id=self.session_id,
                    turn_number=self._turn_number,
                    added_entries=slice_usage_delta(
                        self._agent._token_cost.usage_history,
                        before_count=before_usage_count,
                    ),
                )
                if usage_evt is not None:
                    events_jsonl_append(self._context_jsonl, usage_evt)

    async def _message_iter(self) -> AsyncIterator[ChatMessage]:
        if self._message_source is not None:
            if hasattr(self._message_source, "__aiter__"):
                async for msg in self._message_source:  # type: ignore[misc]
                    if self._closed:
                        return
                    yield msg
            else:
                for msg in self._message_source:
                    if self._closed:
                        return
                    yield msg
            return

        while True:
            msg = await self._queue.get()
            if msg is None:
                return
            yield msg
