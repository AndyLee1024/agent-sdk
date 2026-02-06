from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from comate_agent_sdk.agent.events import AgentEvent, SessionInitEvent, StopEvent
from comate_agent_sdk.agent.service import Agent
from comate_agent_sdk.context.items import ContextItem, ItemType
from comate_agent_sdk.tokens.views import UsageSummary
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

logger = logging.getLogger("comate_agent_sdk.agent.chat_session")

ChatMessage = str | list[ContentPartTextParam | ContentPartImageParam]
MessageSource = AsyncIterator[ChatMessage] | Iterator[ChatMessage] | Iterable[ChatMessage]

PERSISTENCE_SCHEMA_VERSION = "2.0"


class ChatSessionError(Exception):
    pass


class ChatSessionClosedError(ChatSessionError):
    pass


def _default_session_root(session_id: str) -> Path:
    return Path.home() / ".agent" / "sessions" / session_id


def _as_relative_path(path_str: str, *, base: Path) -> str:
    try:
        p = Path(path_str)
    except Exception:
        return path_str

    if not p.is_absolute():
        return path_str

    try:
        return str(p.relative_to(base))
    except Exception:
        return path_str


def _serialize_message_for_persistence(
    message: BaseMessage,
    *,
    offload_root: Path,
    item_offload_path: str | None,
) -> dict:
    data = message.model_dump(mode="json")

    # ToolMessage: 避免将大输出写进 jsonl；同时把 offload_path 规范为相对路径，支持 fork/迁移
    if isinstance(message, ToolMessage):
        if data.get("offload_path"):
            data["offload_path"] = _as_relative_path(str(data["offload_path"]), base=offload_root)
        if item_offload_path:
            data["offload_path"] = item_offload_path

        # destroyed/offloaded 时不写 content（已卸载到文件/已丢弃）
        if bool(data.get("destroyed")) or bool(data.get("offloaded")):
            data["content"] = ""

    return data


def _deserialize_message(data: dict) -> BaseMessage:
    role = data.get("role")
    if role == "user":
        return UserMessage.model_validate(data)
    if role == "assistant":
        return AssistantMessage.model_validate(data)
    if role == "tool":
        return ToolMessage.model_validate(data)
    if role == "system":
        return SystemMessage.model_validate(data)
    if role == "developer":
        return DeveloperMessage.model_validate(data)
    raise ChatSessionError(f"Unsupported message role: {role}")


def _conversation_item_to_dict(*, item: ContextItem, offload_root: Path) -> dict:
    message_data = (
        _serialize_message_for_persistence(
            item.message,
            offload_root=offload_root,
            item_offload_path=item.offload_path,
        )
        if item.message is not None
        else None
    )

    return {
        "id": item.id,
        "item_type": item.item_type.value,
        "message": message_data,
        "content_text": item.content_text,
        "token_count": item.token_count,
        "priority": item.priority,
        "ephemeral": item.ephemeral,
        "destroyed": item.destroyed,
        "tool_name": item.tool_name,
        "created_at": item.created_at,
        "metadata": item.metadata,
        "cache_hint": item.cache_hint,
        "offload_path": item.offload_path,
        "offloaded": item.offloaded,
    }


def _conversation_item_dict_to_item(*, data: dict, offload_root: Path) -> ContextItem:
    message_data = data.get("message")
    message = _deserialize_message(message_data) if message_data else None

    # ToolMessage 的 offload_path 在 jsonl 中保存为相对路径；恢复到内存时转为绝对路径（便于 serializer 展示）
    if isinstance(message, ToolMessage) and message.offloaded and message.offload_path:
        msg_path = str(message.offload_path)
        try:
            if not Path(msg_path).is_absolute():
                message.offload_path = str(offload_root / msg_path)
        except Exception:
            pass

        if data.get("destroyed") is True:
            message.destroyed = True
        if data.get("offloaded") is True:
            message.offloaded = True

    return ContextItem(
        id=data.get("id") or uuid.uuid4().hex[:8],
        item_type=ItemType(data["item_type"]),
        message=message,
        content_text=data.get("content_text", ""),
        token_count=int(data.get("token_count", 0)),
        priority=int(data.get("priority", 50)),
        ephemeral=bool(data.get("ephemeral", False)),
        destroyed=bool(data.get("destroyed", False)),
        tool_name=data.get("tool_name"),
        created_at=float(data.get("created_at", 0.0)),
        metadata=data.get("metadata", {}) or {},
        cache_hint=bool(data.get("cache_hint", False)),
        offload_path=data.get("offload_path"),
        offloaded=bool(data.get("offloaded", False)),
    )


@dataclass(frozen=True)
class _ConversationState:
    """用于生成增量持久化的最小状态快照。"""

    ids: tuple[str, ...]
    by_id: dict[str, tuple[bool, bool, str | None, bool, str | None, bool]]
    # (item.destroyed, item.offloaded, item.offload_path, tool_msg.destroyed, tool_msg.offload_path, tool_msg.offloaded)

    @classmethod
    def capture(cls, items: list[ContextItem]) -> "_ConversationState":
        ids: list[str] = []
        by_id: dict[str, tuple[bool, bool, str | None, bool, str | None, bool]] = {}

        for item in items:
            ids.append(item.id)
            tool_destroyed = False
            tool_offload_path: str | None = None
            tool_offloaded = False
            if isinstance(item.message, ToolMessage):
                tool_destroyed = bool(item.message.destroyed)
                tool_offload_path = item.message.offload_path
                tool_offloaded = bool(item.message.offloaded)
            by_id[item.id] = (
                bool(item.destroyed),
                bool(item.offloaded),
                item.offload_path,
                tool_destroyed,
                tool_offload_path,
                tool_offloaded,
            )

        return cls(ids=tuple(ids), by_id=by_id)


def _build_conversation_event(
    *,
    session_id: str,
    turn_number: int,
    before: _ConversationState,
    after_items: list[ContextItem],
    offload_root: Path,
) -> dict:
    pre_set = set(before.ids)
    post_ids = [item.id for item in after_items]
    post_set = set(post_ids)

    removed_ids = [item_id for item_id in before.ids if item_id not in post_set]
    added_items = [item for item in after_items if item.id not in pre_set]

    expected_post_ids = [item_id for item_id in before.ids if item_id in post_set]
    expected_post_ids.extend([item.id for item in added_items])

    if post_ids != expected_post_ids:
        return {
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "session_id": session_id,
            "turn_number": turn_number,
            "op": "conversation_reset",
            "conversation": {
                "items": [
                    _conversation_item_to_dict(item=item, offload_root=offload_root)
                    for item in after_items
                ]
            },
        }

    updates: list[dict] = []
    for item in after_items:
        if item.id not in pre_set:
            continue

        prev = before.by_id.get(item.id)
        if prev is None:
            continue

        (
            prev_item_destroyed,
            prev_item_offloaded,
            prev_item_offload_path,
            prev_tool_destroyed,
            prev_tool_offload_path,
            prev_tool_offloaded,
        ) = prev

        cur_tool_destroyed = False
        cur_tool_offload_path: str | None = None
        cur_tool_offloaded = False
        if isinstance(item.message, ToolMessage):
            cur_tool_destroyed = bool(item.message.destroyed)
            cur_tool_offload_path = item.message.offload_path
            cur_tool_offloaded = bool(item.message.offloaded)

        cur = (
            bool(item.destroyed),
            bool(item.offloaded),
            item.offload_path,
            cur_tool_destroyed,
            cur_tool_offload_path,
            cur_tool_offloaded,
        )

        if cur == prev:
            continue

        upd: dict = {"id": item.id}
        if bool(item.destroyed) != prev_item_destroyed:
            upd["destroyed"] = bool(item.destroyed)
        if bool(item.offloaded) != prev_item_offloaded:
            upd["offloaded"] = bool(item.offloaded)
        if item.offload_path != prev_item_offload_path:
            upd["offload_path"] = item.offload_path

        if isinstance(item.message, ToolMessage):
            msg_upd: dict = {}
            if bool(item.message.destroyed) != prev_tool_destroyed:
                msg_upd["destroyed"] = bool(item.message.destroyed)
            if bool(item.message.offloaded) != prev_tool_offloaded:
                msg_upd["offloaded"] = bool(item.message.offloaded)
            if item.message.offload_path != prev_tool_offload_path:
                if item.message.offload_path:
                    msg_upd["offload_path"] = _as_relative_path(
                        str(item.message.offload_path),
                        base=offload_root,
                    )
                else:
                    msg_upd["offload_path"] = None
            if msg_upd:
                upd["message"] = msg_upd

        updates.append(upd)

    return {
        "schema_version": PERSISTENCE_SCHEMA_VERSION,
        "session_id": session_id,
        "turn_number": turn_number,
        "op": "conversation_delta",
        "conversation": {
            "adds": [
                _conversation_item_to_dict(item=item, offload_root=offload_root)
                for item in added_items
            ],
            "updates": updates,
            "removes": list(removed_ids),
        },
    }


def _replay_conversation_events(*, path: Path, offload_root: Path) -> tuple[int, list[ContextItem]]:
    if not path.exists():
        return 0, []

    items_by_id: dict[str, ContextItem] = {}
    order: list[str] = []
    last_turn = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            if data.get("schema_version") != PERSISTENCE_SCHEMA_VERSION:
                continue

            turn_number = int(data.get("turn_number", 0))
            if turn_number > last_turn:
                last_turn = turn_number

            op = data.get("op")
            convo = data.get("conversation") or {}

            if op == "conversation_reset":
                items_by_id = {}
                order = []
                for item_data in convo.get("items", []) or []:
                    item = _conversation_item_dict_to_item(data=item_data, offload_root=offload_root)
                    items_by_id[item.id] = item
                    order.append(item.id)
                continue

            if op != "conversation_delta":
                continue

            removes = convo.get("removes", []) or []
            if removes:
                remove_set = set(str(x) for x in removes)
                for rid in remove_set:
                    items_by_id.pop(rid, None)
                order = [item_id for item_id in order if item_id not in remove_set]

            for item_data in convo.get("adds", []) or []:
                item = _conversation_item_dict_to_item(data=item_data, offload_root=offload_root)
                items_by_id[item.id] = item
                order.append(item.id)

            for upd in convo.get("updates", []) or []:
                item_id = upd.get("id")
                if not item_id:
                    continue
                item = items_by_id.get(item_id)
                if item is None:
                    continue

                if "destroyed" in upd:
                    item.destroyed = bool(upd["destroyed"])
                if "offloaded" in upd:
                    item.offloaded = bool(upd["offloaded"])
                if "offload_path" in upd:
                    item.offload_path = upd["offload_path"]

                msg_upd = upd.get("message") or {}
                if isinstance(item.message, ToolMessage) and msg_upd:
                    if "destroyed" in msg_upd:
                        item.message.destroyed = bool(msg_upd["destroyed"])
                    if "offloaded" in msg_upd:
                        item.message.offloaded = bool(msg_upd["offloaded"])
                    if "offload_path" in msg_upd:
                        rel = msg_upd["offload_path"]
                        if rel:
                            try:
                                if not Path(str(rel)).is_absolute():
                                    item.message.offload_path = str(offload_root / str(rel))
                                else:
                                    item.message.offload_path = str(rel)
                            except Exception:
                                item.message.offload_path = str(rel)
                        else:
                            item.message.offload_path = None

    items: list[ContextItem] = []
    for item_id in order:
        item = items_by_id.get(item_id)
        if item is not None:
            items.append(item)

    return last_turn, items


def _events_jsonl_append(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _rewrite_session_id_in_jsonl(path: Path, *, session_id: str) -> None:
    if not path.exists():
        return

    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            data["session_id"] = session_id
            lines.append(json.dumps(data, ensure_ascii=False))

    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


class ChatSession:
    def __init__(
        self,
        agent: Agent,
        *,
        session_id: str | None = None,
        storage_root: Path | None = None,
        message_source: MessageSource | None = None,
    ):
        self._template_agent = agent
        self.session_id = session_id or str(uuid.uuid4())
        self._storage_root = storage_root or _default_session_root(self.session_id)
        self._offload_root = self._storage_root / "offload"
        self._context_jsonl = self._storage_root / "context.jsonl"

        session_options = replace(
            self._template_agent.options,
            session_id=self.session_id,
            offload_root_path=str(self._offload_root),
        )
        self._agent = replace(
            self._template_agent,
            options=session_options,
        )

        self._closed = False
        self._turn_number = 0
        self._init_event_emitted = False

        self._events_started = False
        self._queue: asyncio.Queue[ChatMessage | None] = asyncio.Queue()
        self._message_source = message_source

    @classmethod
    def resume(
        cls,
        agent: Agent,
        *,
        session_id: str,
        storage_root: Path | None = None,
        message_source: MessageSource | None = None,
    ) -> "ChatSession":
        session = cls(
            agent,
            session_id=session_id,
            storage_root=storage_root,
            message_source=message_source,
        )

        turn_number, items = _replay_conversation_events(
            path=session._context_jsonl,
            offload_root=session._offload_root,
        )
        session._turn_number = int(turn_number)
        if items:
            session._agent._context.conversation.items = items
        else:
            logger.info(f"No persisted conversation found for session_id={session_id}, starting empty")

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
        dst = storage_root or _default_session_root(new_session_id)

        if self._closed:
            raise ChatSessionClosedError("Cannot fork a closed session")

        if dst.exists():
            raise ChatSessionError(f"Fork destination already exists: {dst}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

        _rewrite_session_id_in_jsonl(dst / "context.jsonl", session_id=new_session_id)

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
            self._template_agent,
            session_id=new_session_id,
            storage_root=dst,
            message_source=message_source,
        )

    async def get_usage(self) -> UsageSummary:
        """获取会话的 token 使用统计。

        Returns:
            UsageSummary: Token 使用统计，包含总 tokens、成本、按模型/档位的统计

        Note:
            - 即使 session 已关闭，仍可调用此方法获取最终统计
            - 统计包括主 Agent 和所有 Subagent 的 token 使用
        """
        return await self._agent.get_usage()

    async def get_context_info(self):
        """获取当前会话的上下文使用情况

        Returns:
            ContextInfo: 包含上下文使用统计、分类明细、模型信息等

        Note:
            - 即使 session 已关闭，仍可调用此方法
        """
        return await self._agent.get_context_info()

    def clear_history(self) -> None:
        """清空会话历史和 token 使用记录。

        此方法会：
        1. 清空内存中的对话历史
        2. 重置 token 使用统计
        3. 重建 context header（system prompt、subagent、skill、memory）
        4. 在持久化 JSONL 文件中写入 conversation_reset 事件

        Raises:
            ChatSessionClosedError: 如果 session 已关闭

        Warning:
            - 此操作不可逆，清空后无法恢复历史
            - 如需保留历史，请先使用 fork_session() 创建副本
            - 不要在 query_stream() 执行过程中调用此方法
        """
        if self._closed:
            raise ChatSessionClosedError("Cannot clear history on a closed session")

        # 清空内存中的历史和 token 记录
        self._agent.clear_history()

        # 向 JSONL 写入 conversation_reset 事件
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
        _events_jsonl_append(self._context_jsonl, reset_event)

    async def send(self, message: ChatMessage) -> None:
        if self._closed:
            raise ChatSessionClosedError("ChatSession is closed")
        if self._message_source is not None:
            raise ChatSessionError("send() is not allowed when message_source is provided")
        await self._queue.put(message)

    async def close(self) -> None:
        if self._closed:
            return
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
        before = _ConversationState.capture(self._agent._context.conversation.items)
        try:
            async for event in self._agent.query_stream(message):  # type: ignore[arg-type]
                yield event
        finally:
            evt = _build_conversation_event(
                session_id=self.session_id,
                turn_number=self._turn_number,
                before=before,
                after_items=self._agent._context.conversation.items,
                offload_root=self._offload_root,
            )
            _events_jsonl_append(self._context_jsonl, evt)

    async def events(self) -> AsyncIterator[AgentEvent]:
        if self._events_started:
            raise ChatSessionError("events() can only be consumed once")
        self._events_started = True

        if not self._init_event_emitted:
            self._init_event_emitted = True
            yield SessionInitEvent(session_id=self.session_id)

        async for message in self._message_iter():
            self._turn_number += 1
            before = _ConversationState.capture(self._agent._context.conversation.items)
            try:
                async for event in self._agent.query_stream(message):  # type: ignore[arg-type]
                    yield event
                    if isinstance(event, StopEvent):
                        break
            finally:
                evt = _build_conversation_event(
                    session_id=self.session_id,
                    turn_number=self._turn_number,
                    before=before,
                    after_items=self._agent._context.conversation.items,
                    offload_root=self._offload_root,
                )
                _events_jsonl_append(self._context_jsonl, evt)

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
