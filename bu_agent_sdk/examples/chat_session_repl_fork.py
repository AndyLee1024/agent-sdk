import asyncio
import logging
import os
from pathlib import Path

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import ComateAgentOptions
from bu_agent_sdk.agent.events import (
    SessionInitEvent,
    StepCompleteEvent,
    StepStartEvent,
    StopEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from bu_agent_sdk.llm import ChatOpenAI


logger = logging.getLogger("bu_agent_sdk.examples.chat_session_repl_fork")


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


def _help_text() -> str:
    return (
        "Commands:\n"
        "  /help             Show this help\n"
        "  /session          Show current session id\n"
        "  /fork             Fork current session (new UUID)\n"
        "  /use <uuid>       Resume a session by UUID\n"
        "  /exit             Exit\n"
        "\n"
        "Run:\n"
        "  uv run python bu_agent_sdk/examples/chat_session_repl_fork.py\n"
    )


async def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    llm = ChatOpenAI(
        base_url='http://192.168.100.1:4141/v1',
        model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        api_key=_get_openai_api_key(),
    )
    agent = Agent(llm=llm, options=ComateAgentOptions(tools=[]))

    session = agent.chat()
    logger.info(_help_text())

    while True:
        line = await asyncio.to_thread(input, "> ")
        text = line.strip()
        if not text:
            continue

        if text == "/help":
            logger.info(_help_text())
            continue
        if text == "/exit":
            break
        if text == "/session":
            logger.info(f"session_id={session.session_id}")
            logger.info(f"storage_root={_session_root(session.session_id)}")
            continue
        if text == "/fork":
            session = agent.chat(fork_session=session.session_id)
            logger.info(f"forked session_id={session.session_id}")
            logger.info(f"storage_root={_session_root(session.session_id)}")
            continue
        if text.startswith("/use "):
            _, sid = text.split(" ", 1)
            sid = sid.strip()
            session = agent.chat(session_id=sid)
            logger.info(f"resumed session_id={session.session_id}")
            logger.info(f"storage_root={_session_root(session.session_id)}")
            continue
        if text.startswith("/"):
            logger.warning(f"Unknown command: {text}")
            continue

        async for event in session.query_stream(text):
            _log_event(event)


if __name__ == "__main__":
    asyncio.run(main())
