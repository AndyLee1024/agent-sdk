from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from comate_agent_sdk.llm.messages import ToolCall, ToolMessage

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


@dataclass
class AskUserPolicyResult:
    tool_calls: list[ToolCall]
    repaired: bool
    askuser_repair_attempted: bool


def is_ask_user_question(call: ToolCall) -> bool:
    return str(call.function.name or "").strip() == "AskUserQuestion"


def policy_violation_message(
    *,
    call: ToolCall,
    code: str,
    message: str,
    required_fix: str,
) -> ToolMessage:
    content = json.dumps(
        {
            "error": {
                "code": code,
                "message": message,
                "required_fix": required_fix,
            }
        },
        ensure_ascii=False,
    )
    return ToolMessage(
        tool_call_id=call.id,
        tool_name=str(call.function.name or "").strip() or "Tool",
        content=content,
        is_error=True,
    )


async def enforce_ask_user_exclusive_policy(
    agent: "AgentRuntime",
    *,
    assistant_message,
    tool_calls: list[ToolCall],
    askuser_repair_attempted: bool,
) -> AskUserPolicyResult:
    has_ask = any(is_ask_user_question(call) for call in tool_calls)
    has_other = any(not is_ask_user_question(call) for call in tool_calls)
    if not (has_ask and has_other):
        return AskUserPolicyResult(
            tool_calls=tool_calls,
            repaired=False,
            askuser_repair_attempted=askuser_repair_attempted,
        )

    if not askuser_repair_attempted:
        expected_ids = [call.id for call in tool_calls if str(call.id).strip()]
        agent._context.begin_tool_barrier(expected_ids)

        results_by_id: dict[str, ToolMessage] = {}
        for call in tool_calls:
            if is_ask_user_question(call):
                results_by_id[call.id] = policy_violation_message(
                    call=call,
                    code="ASKUSER_EXCLUSIVE",
                    message="AskUserQuestion must be called alone in a single assistant tool_calls response.",
                    required_fix="Retry now with ONLY AskUserQuestion. Run other tools after the user answer in the next turn.",
                )
            else:
                results_by_id[call.id] = policy_violation_message(
                    call=call,
                    code="ASKUSER_BLOCKED_BY_ASK",
                    message="Blocked because AskUserQuestion was present in the same tool_calls response.",
                    required_fix="Retry without mixing tools with AskUserQuestion.",
                )

        agent._context.add_messages_atomic(
            [assistant_message] + [results_by_id[call.id] for call in tool_calls]
        )
        return AskUserPolicyResult(
            tool_calls=tool_calls,
            repaired=True,
            askuser_repair_attempted=True,
        )

    ask_call = next((call for call in tool_calls if is_ask_user_question(call)), None)
    final_calls = [ask_call] if ask_call is not None else tool_calls
    return AskUserPolicyResult(
        tool_calls=final_calls,
        repaired=False,
        askuser_repair_attempted=True,
    )
