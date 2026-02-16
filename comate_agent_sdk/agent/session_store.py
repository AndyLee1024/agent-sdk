"""Session persistence: serialization, delta events, and JSONL replay.

Extracted from chat_session.py to keep ChatSession focused on orchestration.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from comate_agent_sdk.context.items import ContextItem, ItemType
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from comate_agent_sdk.tokens.views import TokenUsageEntry

logger = logging.getLogger("comate_agent_sdk.agent.session_store")

PERSISTENCE_SCHEMA_VERSION = "2.0"


class SessionStoreError(Exception):
    pass


def default_session_root(session_id: str) -> Path:
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


def serialize_message_for_persistence(
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


def deserialize_message(data: dict) -> BaseMessage:
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
    raise SessionStoreError(f"Unsupported message role: {role}")


def conversation_item_to_dict(*, item: ContextItem, offload_root: Path) -> dict:
    message_data = (
        serialize_message_for_persistence(
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
        "is_tool_error": item.is_tool_error,
        "created_turn": item.created_turn,
    }


def conversation_item_dict_to_item(*, data: dict, offload_root: Path) -> ContextItem:
    message_data = data.get("message")
    message = deserialize_message(message_data) if message_data else None

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
        is_tool_error=bool(data.get("is_tool_error", False)),
        created_turn=int(data.get("created_turn", 0)),
    )


@dataclass(frozen=True)
class ConversationState:
    """用于生成增量持久化的最小状态快照。"""

    ids: tuple[str, ...]
    by_id: dict[str, tuple[bool, bool, str | None, bool, str | None, bool]]
    # (item.destroyed, item.offloaded, item.offload_path, tool_msg.destroyed, tool_msg.offload_path, tool_msg.offloaded)

    @classmethod
    def capture(cls, items: list[ContextItem]) -> "ConversationState":
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


def build_conversation_event(
    *,
    session_id: str,
    turn_number: int,
    before: ConversationState,
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
                    conversation_item_to_dict(item=item, offload_root=offload_root)
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
                conversation_item_to_dict(item=item, offload_root=offload_root)
                for item in added_items
            ],
            "updates": updates,
            "removes": list(removed_ids),
        },
    }


def token_usage_entry_to_dict(*, entry: TokenUsageEntry) -> dict:
    return entry.model_dump(mode="json")


def token_usage_entry_dict_to_entry(*, data: dict) -> TokenUsageEntry:
    return TokenUsageEntry.model_validate(data)


def build_usage_event(
    *,
    session_id: str,
    turn_number: int,
    added_entries: list[TokenUsageEntry],
) -> dict | None:
    if not added_entries:
        return None

    return {
        "schema_version": PERSISTENCE_SCHEMA_VERSION,
        "session_id": session_id,
        "turn_number": turn_number,
        "op": "usage_delta",
        "usage": {
            "adds": [
                token_usage_entry_to_dict(entry=entry)
                for entry in added_entries
            ]
        },
    }


def slice_usage_delta(
    usage_history: list[TokenUsageEntry],
    *,
    before_count: int,
) -> list[TokenUsageEntry]:
    if before_count < 0 or before_count > len(usage_history):
        return list(usage_history)
    return list(usage_history[before_count:])


def replay_session_events(
    *,
    path: Path,
    offload_root: Path,
) -> tuple[int, list[ContextItem], list[TokenUsageEntry]]:
    if not path.exists():
        return 0, [], []

    items_by_id: dict[str, ContextItem] = {}
    order: list[str] = []
    usage_entries: list[TokenUsageEntry] = []
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
            if op == "usage_reset":
                usage_entries = []
                continue
            if op == "usage_delta":
                usage = data.get("usage") or {}
                for entry_data in usage.get("adds", []) or []:
                    try:
                        usage_entries.append(token_usage_entry_dict_to_entry(data=entry_data))
                    except Exception as e:
                        logger.warning(f"Failed to replay usage entry: {e}", exc_info=True)
                continue

            convo = data.get("conversation") or {}

            if op == "conversation_reset":
                items_by_id = {}
                order = []
                for item_data in convo.get("items", []) or []:
                    item = conversation_item_dict_to_item(data=item_data, offload_root=offload_root)
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
                item = conversation_item_dict_to_item(data=item_data, offload_root=offload_root)
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

    return last_turn, items, usage_entries


def replay_conversation_events(*, path: Path, offload_root: Path) -> tuple[int, list[ContextItem]]:
    turn_number, items, _ = replay_session_events(path=path, offload_root=offload_root)
    return turn_number, items


def events_jsonl_append(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def rewrite_session_id_in_jsonl(path: Path, *, session_id: str) -> None:
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
