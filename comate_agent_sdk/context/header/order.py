"""Header item 顺序定义。"""

from __future__ import annotations

from comate_agent_sdk.context.items import ItemType

HEADER_ITEM_ORDER: dict[ItemType, int] = {
    ItemType.SYSTEM_PROMPT: 0,
    ItemType.AGENT_LOOP: 1,
    ItemType.TOOL_STRATEGY: 2,
    ItemType.MCP_TOOL: 3,
    ItemType.SUBAGENT_STRATEGY: 4,
    ItemType.SKILL_STRATEGY: 5,
    ItemType.SYSTEM_ENV: 6,
    ItemType.GIT_ENV: 7,
}

HEADER_ITEM_TYPES_IN_ORDER: tuple[ItemType, ...] = tuple(
    item_type for item_type, _ in sorted(HEADER_ITEM_ORDER.items(), key=lambda kv: kv[1])
)
