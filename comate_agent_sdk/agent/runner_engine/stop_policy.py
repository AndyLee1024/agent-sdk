from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime

logger = logging.getLogger("comate_agent_sdk.agent")


async def should_continue_after_stop_block(
    agent: "AgentRuntime",
    stop_reason: str,
    *,
    drain_hidden_immediately: bool,
) -> bool:
    outcome = await agent.run_hook_event(
        "Stop",
        stop_reason=stop_reason,
        stop_hook_active=agent._stop_hook_block_count > 0,
    )
    if outcome is None or not outcome.should_block_stop:
        agent.reset_stop_hook_state()
        return False

    should_continue = agent.mark_stop_blocked_once()
    block_reason = outcome.reason or f"Stop blocked by hook: {stop_reason}"
    agent.add_hidden_user_message(block_reason)

    if drain_hidden_immediately:
        agent.drain_hidden_user_messages()

    if not should_continue:
        logger.warning(
            f"Stop hook blocked {agent._stop_hook_block_count} times, forcing stop (reason={stop_reason})"
        )
        agent.reset_stop_hook_state()
        return False

    return True
