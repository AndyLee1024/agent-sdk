"""Header snapshot 导入导出。"""

from __future__ import annotations

from typing import Any

from comate_agent_sdk.context.items import ContextItem, ItemType
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


def _snapshot_item_to_dict(item: ContextItem) -> dict[str, Any]:
    message_data = item.message.model_dump(mode="json") if item.message is not None else None
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


def _snapshot_message_from_dict(data: dict[str, Any] | None) -> BaseMessage | None:
    if data is None:
        return None
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
    raise ValueError(f"Unsupported message role in header snapshot: {role}")


def _snapshot_item_from_dict(data: dict[str, Any]) -> ContextItem:
    message_data = data.get("message")
    message = _snapshot_message_from_dict(message_data if isinstance(message_data, dict) else None)
    kwargs: dict[str, Any] = {}
    item_id = data.get("id")
    if isinstance(item_id, str) and item_id.strip():
        kwargs["id"] = item_id
    return ContextItem(
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
        **kwargs,
    )


def export_header_snapshot(
    *,
    header_items: list[ContextItem],
    memory_item: ContextItem | None,
    header_item_order: dict[ItemType, int],
) -> dict[str, Any]:
    serialized_header_items = [
        _snapshot_item_to_dict(item)
        for item in header_items
        if item.item_type in header_item_order
    ]
    serialized_memory_item = _snapshot_item_to_dict(memory_item) if memory_item else None
    return {
        "schema_version": 1,
        "header_items": serialized_header_items,
        "memory_item": serialized_memory_item,
    }


def import_header_snapshot(
    *,
    snapshot: dict[str, Any],
    token_counter: Any,
    header_item_order: dict[ItemType, int],
) -> tuple[list[ContextItem], ContextItem | None]:
    if not isinstance(snapshot, dict):
        raise ValueError("header snapshot must be a dict")

    raw_items = snapshot.get("header_items")
    if not isinstance(raw_items, list):
        raise ValueError("header snapshot missing header_items list")

    allowed_header_types = set(header_item_order.keys())
    restored_header: list[ContextItem] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("header snapshot contains invalid header item")
        item = _snapshot_item_from_dict(raw_item)
        if item.item_type not in allowed_header_types:
            raise ValueError(f"unsupported header item_type in snapshot: {item.item_type.value}")
        item.message = None
        item.ephemeral = False
        item.tool_name = None
        item.token_count = token_counter.count(item.content_text or "")
        restored_header.append(item)

    restored_header.sort(key=lambda it: header_item_order.get(it.item_type, 999))

    raw_memory = snapshot.get("memory_item")
    if raw_memory is None:
        return restored_header, None
    if not isinstance(raw_memory, dict):
        raise ValueError("header snapshot memory_item must be a dict or null")

    restored_memory_item = _snapshot_item_from_dict(raw_memory)
    if restored_memory_item.item_type != ItemType.MEMORY:
        raise ValueError("header snapshot memory_item type must be memory")
    if restored_memory_item.message is None:
        raise ValueError("header snapshot memory_item.message is required")
    if not isinstance(restored_memory_item.message, UserMessage):
        raise ValueError("header snapshot memory_item.message must be UserMessage")
    if not bool(getattr(restored_memory_item.message, "is_meta", False)):
        raise ValueError("header snapshot memory_item.message must be meta user message")

    if not restored_memory_item.content_text:
        restored_memory_item.content_text = restored_memory_item.message.text
    restored_memory_item.token_count = token_counter.count(restored_memory_item.content_text)
    return restored_header, restored_memory_item
