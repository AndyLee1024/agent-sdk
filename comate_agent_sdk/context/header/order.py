"""Header item 顺序定义。"""

from __future__ import annotations

from comate_agent_sdk.context.items import ItemType

STATIC_HEADER_ITEM_ORDER: dict[ItemType, int] = {
    ItemType.SYSTEM_PROMPT: 0,
    ItemType.AGENT_LOOP: 1,
    ItemType.TOOL_STRATEGY: 2,
    ItemType.SUBAGENT_STRATEGY: 3,
    ItemType.SKILL_STRATEGY: 4,
}

SESSION_STATE_ITEM_ORDER: dict[ItemType, int] = {
    ItemType.SYSTEM_ENV: 0,
    ItemType.GIT_ENV: 1,
    ItemType.MCP_TOOL: 2,
    ItemType.OUTPUT_STYLE: 3,
}

STATIC_HEADER_ITEM_TYPES_IN_ORDER: tuple[ItemType, ...] = tuple(
    item_type for item_type, _ in sorted(STATIC_HEADER_ITEM_ORDER.items(), key=lambda kv: kv[1])
)

SESSION_STATE_ITEM_TYPES_IN_ORDER: tuple[ItemType, ...] = tuple(
    item_type for item_type, _ in sorted(SESSION_STATE_ITEM_ORDER.items(), key=lambda kv: kv[1])
)

# Backward-compatible aliases.
HEADER_ITEM_ORDER: dict[ItemType, int] = STATIC_HEADER_ITEM_ORDER
HEADER_ITEM_TYPES_IN_ORDER: tuple[ItemType, ...] = STATIC_HEADER_ITEM_TYPES_IN_ORDER
