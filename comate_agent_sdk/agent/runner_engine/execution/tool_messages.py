from __future__ import annotations

import json

from comate_agent_sdk.llm.messages import ToolCall, ToolMessage


def make_cancelled_tool_message(
    tool_call: ToolCall,
    *,
    reason: str = "user_interrupt",
) -> ToolMessage:
    return ToolMessage(
        tool_call_id=tool_call.id,
        tool_name=str(tool_call.function.name or "").strip() or "Tool",
        content=json.dumps(
            {
                "status": "cancelled",
                "reason": reason,
            },
            ensure_ascii=False,
        ),
        is_error=True,
    )
