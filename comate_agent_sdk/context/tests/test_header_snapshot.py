from __future__ import annotations

import pytest

from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.items import ItemType


def test_header_snapshot_roundtrip_restores_header_and_memory() -> None:
    source = ContextIR()
    source.set_system_prompt("SNAPSHOT_PROMPT", cache=False)
    source.set_tool_strategy("SNAPSHOT_TOOL_STRATEGY")
    source.set_system_env("<system_env>SNAPSHOT_ENV</system_env>")
    source.set_memory("SNAPSHOT_MEMORY", cache=False)

    snapshot = source.export_header_snapshot()

    restored = ContextIR()
    restored.import_header_snapshot(snapshot)

    prompt_item = restored.header.find_one_by_type(ItemType.SYSTEM_PROMPT)
    tool_item = restored.header.find_one_by_type(ItemType.TOOL_STRATEGY)
    env_item = restored.header.find_one_by_type(ItemType.SYSTEM_ENV)
    assert prompt_item is not None
    assert tool_item is not None
    assert env_item is not None
    assert prompt_item.content_text == "SNAPSHOT_PROMPT"
    assert tool_item.content_text == "SNAPSHOT_TOOL_STRATEGY"
    assert env_item.content_text == "<system_env>SNAPSHOT_ENV</system_env>"
    assert restored.memory_item is not None
    assert "SNAPSHOT_MEMORY" in restored.memory_item.content_text


def test_header_snapshot_rejects_non_header_item() -> None:
    ctx = ContextIR()
    snapshot = {
        "schema_version": 1,
        "header_items": [
            {
                "item_type": "user_message",
                "message": None,
                "content_text": "bad",
                "token_count": 1,
                "priority": 50,
                "ephemeral": False,
                "destroyed": False,
                "tool_name": None,
                "created_at": 0.0,
                "metadata": {},
                "cache_hint": False,
                "offload_path": None,
                "offloaded": False,
                "is_tool_error": False,
                "created_turn": 0,
            }
        ],
        "memory_item": None,
    }
    with pytest.raises(ValueError):
        ctx.import_header_snapshot(snapshot)
