from __future__ import annotations

from comate_agent_sdk.llm.messages import AssistantMessage, BaseMessage, ToolMessage


def validate_tool_call_sequence(messages: list[BaseMessage], *, provider: str) -> None:
    """Validate tool-call protocol invariants before provider serialization.

    Invariant:
    - If assistant message contains tool_calls, subsequent messages must include
      all corresponding ToolMessage(s) before any non-tool message appears.
    - ToolMessage.tool_call_id must match one of required tool_call ids.
    """
    total = len(messages)
    for idx, message in enumerate(messages):
        if not isinstance(message, AssistantMessage):
            continue
        if not message.tool_calls:
            continue

        required_ids = {tc.id for tc in message.tool_calls if tc.id}
        if not required_ids:
            continue

        seen_ids: set[str] = set()
        cursor = idx + 1
        while cursor < total:
            next_message = messages[cursor]
            if not isinstance(next_message, ToolMessage):
                if seen_ids >= required_ids:
                    break
                missing_ids = sorted(required_ids - seen_ids)
                raise ValueError(
                    _build_invariant_error(
                        provider=provider,
                        reason=(
                            "tool_calls must be followed by consecutive tool messages "
                            f"until all results arrive; encountered role={next_message.role!r}"
                        ),
                        index=idx,
                        required_ids=sorted(required_ids),
                        seen_ids=sorted(seen_ids),
                        missing_ids=missing_ids,
                        messages=messages,
                    )
                )

            tool_call_id = next_message.tool_call_id
            if tool_call_id not in required_ids:
                raise ValueError(
                    _build_invariant_error(
                        provider=provider,
                        reason=f"unexpected tool_call_id={tool_call_id!r} in tool result block",
                        index=idx,
                        required_ids=sorted(required_ids),
                        seen_ids=sorted(seen_ids),
                        missing_ids=sorted(required_ids - seen_ids),
                        messages=messages,
                    )
                )
            seen_ids.add(tool_call_id)
            cursor += 1

        if not required_ids.issubset(seen_ids):
            missing_ids = sorted(required_ids - seen_ids)
            raise ValueError(
                _build_invariant_error(
                    provider=provider,
                    reason="missing tool result messages for assistant tool_calls",
                    index=idx,
                    required_ids=sorted(required_ids),
                    seen_ids=sorted(seen_ids),
                    missing_ids=missing_ids,
                    messages=messages,
                )
            )


def _build_invariant_error(
    *,
    provider: str,
    reason: str,
    index: int,
    required_ids: list[str],
    seen_ids: list[str],
    missing_ids: list[str],
    messages: list[BaseMessage],
) -> str:
    excerpt = _format_message_excerpt(messages, center=index, radius=4)
    return (
        f"[{provider}] Invalid tool-call message sequence: {reason}. "
        f"assistant_index={index}, required_ids={required_ids}, seen_ids={seen_ids}, missing_ids={missing_ids}. "
        f"Recent messages: {excerpt}"
    )


def _format_message_excerpt(messages: list[BaseMessage], *, center: int, radius: int) -> str:
    start = max(0, center - radius)
    end = min(len(messages), center + radius + 1)
    parts: list[str] = []
    for idx in range(start, end):
        message = messages[idx]
        if isinstance(message, AssistantMessage) and message.tool_calls:
            tool_ids = [tc.id for tc in message.tool_calls]
            parts.append(f"{idx}:assistant(tool_calls={tool_ids})")
            continue
        if isinstance(message, ToolMessage):
            parts.append(f"{idx}:tool(tool_call_id={message.tool_call_id!r})")
            continue
        parts.append(f"{idx}:{message.role}")
    return " | ".join(parts)

