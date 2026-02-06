import asyncio
import logging
import os
from pathlib import Path

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import ComateAgentOptions
from comate_agent_sdk.agent.events import (
    SessionInitEvent,
    StepCompleteEvent,
    StepStartEvent,
    StopEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from comate_agent_sdk.llm import ChatOpenAI


logger = logging.getLogger("comate_agent_sdk.examples.chat_session_fork_scripted")


def _get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENAI_API_KEY. Run: export OPENAI_API_KEY='...'"
        )
    return api_key


def _session_root(session_id: str) -> Path:
    return Path.home() / ".agent" / "sessions" / session_id


def _log_event(event) -> None:
    match event:
        case SessionInitEvent(session_id=sid):
            logger.info(f"session_id={sid}")
            logger.info(f"storage_root={_session_root(sid)}")
        case ThinkingEvent(content=text):
            preview = text[:200] + "..." if len(text) > 200 else text
            logger.info(f"thinking={preview}")
        case TextEvent(content=text):
            logger.info(f"text={text}")
        case StepStartEvent(step_number=n, title=title):
            logger.info(f"step_start n={n} title={title}")
        case ToolCallEvent(tool=name, args=args):
            logger.info(f"tool_call tool={name} args={args}")
        case ToolResultEvent(tool=name, is_error=is_error, result=result):
            preview = result[:500] + "..." if len(result) > 500 else result
            status = "error" if is_error else "ok"
            logger.info(f"tool_result tool={name} status={status} result={preview}")
        case StepCompleteEvent(status=status, duration_ms=ms):
            logger.info(f"step_complete status={status} duration_ms={ms:.0f}")
        case StopEvent(reason=reason):
            logger.info(f"stop reason={reason}")
        case _:
            logger.debug(f"event={event!r}")


async def _run_events(session) -> None:
    async for event in session.events():
        _log_event(event)


async def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=_get_openai_api_key(),
    )
    agent = Agent(llm=llm, options=ComateAgentOptions(tools=[]))

    seed_messages = iter(
        [
            "My name is Alice.",
            "What's my name?",
        ]
    )
    parent = agent.chat(message_source=seed_messages)
    logger.info("=== parent (seed) ===")
    await _run_events(parent)

    forked = parent.fork_session(
        message_source=iter(
            [
                "We are now in the fork. What is my name?",
            ]
        )
    )
    logger.info("=== forked ===")
    await _run_events(forked)

    parent_cont = agent.chat(
        session_id=parent.session_id,
        message_source=iter(
            [
                "We are still in the parent. Summarize what you know about me in one sentence.",
            ]
        ),
    )
    logger.info("=== parent (continue) ===")
    await _run_events(parent_cont)

    logger.info(f"parent session_id={parent.session_id} storage_root={_session_root(parent.session_id)}")
    logger.info(f"forked session_id={forked.session_id} storage_root={_session_root(forked.session_id)}")
    logger.info("Run: uv run python comate_agent_sdk/examples/chat_session_fork_scripted.py")


if __name__ == "__main__":
    asyncio.run(main())
