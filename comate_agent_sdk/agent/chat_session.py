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
    build_header_snapshot_event,
    build_usage_event,
    default_session_root,
    events_jsonl_append,
    replay_header_snapshot,
    replay_session_events,
    rewrite_session_id_in_jsonl,
    slice_usage_delta,
)
from comate_agent_sdk.llm.messages import ContentPartImageParam, ContentPartTextParam
from comate_agent_sdk.tokens.total_usage import TotalUsageSnapshot
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
        resolved_session_id = (
            session_id or (runtime.options.session_id if runtime is not None else None)
        )
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
            if runtime.options.session_id != self.session_id:
                raise ChatSessionError(
                    f"runtime.session_id={runtime.options.session_id} does not match session_id={self.session_id}"
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

        if runtime is None:
            self._persist_initial_header_snapshot_if_needed()

    def _persist_initial_header_snapshot_if_needed(self) -> None:
        if self._context_jsonl.exists():
            try:
                existing = self._context_jsonl.read_text(encoding="utf-8").strip()
            except Exception:
                existing = ""
            if existing:
                return

        snapshot = self._agent._context.export_header_snapshot()
        evt = build_header_snapshot_event(
            session_id=self.session_id,
            snapshot=snapshot,
            turn_number=0,
        )
        events_jsonl_append(self._context_jsonl, evt)

    @classmethod
    def resume(
        cls,
        template: AgentTemplate,
        *,
        session_id: str,
        storage_root: Path | None = None,
        message_source: MessageSource | None = None,
    ) -> "ChatSession":
        session_root = storage_root or default_session_root(session_id)
        context_jsonl = session_root / "context.jsonl"
        header_snapshot = replay_header_snapshot(path=context_jsonl)
        if header_snapshot is None:
            raise ChatSessionError(
                f"Session {session_id} is missing required header_snapshot in {context_jsonl}"
            )
        try:
            runtime = template.create_runtime_from_snapshot(
                header_snapshot=header_snapshot,
                session_id=session_id,
                offload_root_path=str(session_root / "offload"),
            )
        except Exception as e:
            raise ChatSessionError(
                f"Failed to restore header snapshot for session {session_id}: {e}"
            ) from e
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
            session._agent._context.replace_conversation(items)
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
                session._agent._context.reminder_engine.restore_todos(
                    todos=todos,
                    turn_number_at_update=turn_number_at_update,
                    current_turn=session._agent._context.turn_number,
                )
                logger.info(f"Restored {len(todos)} TODO items from todos.json")
            except Exception as e:
                logger.warning(f"Failed to restore TODO state for session_id={session_id}: {e}", exc_info=True)

        try:
            session._agent._context.rehydrate_reminder_state_from_conversation(
                suppress_task_nudge_on_next_turn=True,
            )
        except Exception as e:
            logger.warning(
                f"Failed to rehydrate reminder state for session_id={session_id}: {e}",
                exc_info=True,
            )

        # Header 从持久化快照恢复后，禁止自动改写 static header。
        session._agent._lock_header_from_snapshot = True

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

    def get_mode(self) -> str:
        if self._closed:
            raise ChatSessionClosedError("Cannot get mode on a closed session")
        return self._agent.get_mode()

    def set_mode(self, mode: str) -> str:
        if self._closed:
            raise ChatSessionClosedError("Cannot set mode on a closed session")
        self._agent.set_mode(mode)
        return self._agent.get_mode()

    def cycle_mode(self) -> str:
        if self._closed:
            raise ChatSessionClosedError("Cannot cycle mode on a closed session")
        return self._agent.cycle_mode()

    def has_pending_plan_approval(self) -> bool:
        if self._closed:
            raise ChatSessionClosedError("Cannot read pending plan approval on a closed session")
        return self._agent.has_pending_plan_approval()

    def get_pending_plan_approval(self) -> dict[str, str] | None:
        if self._closed:
            raise ChatSessionClosedError("Cannot read pending plan approval on a closed session")
        return self._agent.pending_plan_approval()

    def approve_plan(self) -> str:
        if self._closed:
            raise ChatSessionClosedError("Cannot approve plan on a closed session")
        return self._agent.approve_plan()

    def reject_plan(self) -> None:
        if self._closed:
            raise ChatSessionClosedError("Cannot reject plan on a closed session")
        self._agent.reject_plan()

    def _update_total_usage(self, new_entries: list) -> None:
        """把本 turn 的 usage entries 追加写入 total_usage.json。"""
        if not new_entries:
            return
        try:
            total_usage_path = self._storage_root / "total_usage.json"
            if total_usage_path.exists():
                snapshot = TotalUsageSnapshot.load(total_usage_path)
            else:
                snapshot = TotalUsageSnapshot.new(session_id=self.session_id)
            snapshot.add_turn_entries(new_entries)
            snapshot.save(total_usage_path)
        except Exception as e:
            logger.warning(f"Failed to update total_usage.json: {e}", exc_info=True)

    async def get_total_usage(self) -> TotalUsageSnapshot:
        """获取会话累计 token 使用快照（total_usage.json）。"""
        total_usage_path = self._storage_root / "total_usage.json"
        if total_usage_path.exists():
            return TotalUsageSnapshot.load(total_usage_path)
        return TotalUsageSnapshot.new(session_id=self.session_id)

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

    def restore_conversation_to_turn(self, *, target_turn: int) -> None:
        """仅回滚会话对话到指定 turn（不回滚 usage）。"""
        if self._closed:
            raise ChatSessionClosedError("Cannot restore conversation on a closed session")

        try:
            normalized_turn = int(target_turn)
        except Exception as exc:
            raise ChatSessionError(f"Invalid target_turn={target_turn}") from exc

        if normalized_turn < 0:
            raise ChatSessionError(f"target_turn must be >= 0, got {normalized_turn}")
        if normalized_turn > self._turn_number:
            raise ChatSessionError(
                f"target_turn={normalized_turn} exceeds current turn={self._turn_number}"
            )

        _, items, _ = replay_session_events(
            path=self._context_jsonl,
            offload_root=self._offload_root,
            max_turn=normalized_turn,
        )

        before = ConversationState.capture(self._agent._context.get_conversation_items_snapshot())
        self._agent._context.replace_conversation(list(items))

        self._turn_number += 1
        evt = build_conversation_event(
            session_id=self.session_id,
            turn_number=self._turn_number,
            before=before,
            after_items=self._agent._context.get_conversation_items_snapshot(),
            offload_root=self._offload_root,
        )
        events_jsonl_append(self._context_jsonl, evt)
        self._agent._context.set_turn_number(self._turn_number)
        self._agent._context.rehydrate_reminder_state_from_conversation(
            suppress_task_nudge_on_next_turn=True,
        )

    def set_level(self, level: LLMLevel) -> LLMSwitchedEvent:
        """Set the LLM level for this session."""
        if self._closed:
            raise ChatSessionClosedError("Cannot set level on a closed session")

        previous_level = self._agent.level
        previous_model = self._agent.llm.model if self._agent.llm else None

        self._agent.level = level

        if self._agent.options.llm_levels and level in self._agent.options.llm_levels:
            self._agent.llm = self._agent.options.llm_levels[level]

        new_model = None
        if self._agent.options.llm_levels and level in self._agent.options.llm_levels:
            new_model = self._agent.options.llm_levels[level].model

        event = LLMSwitchedEvent(
            previous_level=previous_level,
            new_level=level,
            previous_model=previous_model,
            new_model=new_model,
        )

        logger.info(f"Session {self.session_id}: {event}")
        return event

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
        before = ConversationState.capture(self._agent._context.get_conversation_items_snapshot())
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
                after_items=self._agent._context.get_conversation_items_snapshot(),
                offload_root=self._offload_root,
            )
            events_jsonl_append(self._context_jsonl, evt)
            new_entries = slice_usage_delta(
                self._agent._token_cost.usage_history,
                before_count=before_usage_count,
            )
            tracker = getattr(self._agent, "_context_usage_tracker", None)
            context_usage = tracker.context_usage if tracker is not None else None
            usage_evt = build_usage_event(
                session_id=self.session_id,
                turn_number=self._turn_number,
                added_entries=new_entries,
                context_usage=context_usage,
            )
            if usage_evt is not None:
                events_jsonl_append(self._context_jsonl, usage_evt)
            self._update_total_usage(new_entries)

    async def events(self) -> AsyncIterator[AgentEvent]:
        if self._events_started:
            raise ChatSessionError("events() can only be consumed once")
        self._events_started = True

        if not self._init_event_emitted:
            self._init_event_emitted = True
            yield SessionInitEvent(session_id=self.session_id)

        async for message in self._message_iter():
            self._turn_number += 1
            before = ConversationState.capture(self._agent._context.get_conversation_items_snapshot())
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
                    after_items=self._agent._context.get_conversation_items_snapshot(),
                    offload_root=self._offload_root,
                )
                events_jsonl_append(self._context_jsonl, evt)
                new_entries = slice_usage_delta(
                    self._agent._token_cost.usage_history,
                    before_count=before_usage_count,
                )
                tracker = getattr(self._agent, "_context_usage_tracker", None)
                context_usage = tracker.context_usage if tracker is not None else None
                usage_evt = build_usage_event(
                    session_id=self.session_id,
                    turn_number=self._turn_number,
                    added_entries=new_entries,
                    context_usage=context_usage,
                )
                if usage_evt is not None:
                    events_jsonl_append(self._context_jsonl, usage_evt)
                self._update_total_usage(new_entries)

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
