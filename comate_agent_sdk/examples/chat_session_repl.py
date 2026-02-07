"""Simple chat session REPL example.

Creates a new session when started and allows interactive chat.

Run:
    uv run python comate_agent_sdk/examples/chat_session_repl.py
"""
import asyncio
import logging
import os
import readline  # noqa: F401 - 启用更好的终端行编辑功能
from pathlib import Path

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig
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


logger = logging.getLogger("comate_agent_sdk.examples.chat_session_repl")


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
            print(f"session_id={sid}")
            print(f"storage_root={_session_root(sid)}")
        case ThinkingEvent(content=text):
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"thinking={preview}")
        case TextEvent(content=text):
            print(text)
        case StepStartEvent(step_number=n, title=title):
            print(f"step_start n={n} title={title}")
        case ToolCallEvent(tool=name, args=args):
            print(f"tool_call tool={name} args={args}")
        case ToolResultEvent(tool=name, is_error=is_error, result=result):
            preview = result[:500] + "..." if len(result) > 500 else result
            status = "error" if is_error else "ok"
            print(f"tool_result tool={name} status={status} result={preview}")
        case StepCompleteEvent(status=status, duration_ms=ms):
            print(f"step_complete status={status} duration_ms={ms:.0f}")
        case StopEvent(reason=reason):
            pass  # TextEvent 已打印，不重复
        case _:
            print(f"event={event!r}")


def _help_text() -> str:
    return (
        "Commands:\n"
        "  /help             Show this help\n"
        "  /session          Show current session id\n"
        "  /usage            Show token usage statistics\n"
        "  /context          Show context usage breakdown\n"
        "  /clear            Clear conversation history and token stats\n"
        "  /resume <sid>     Resume a specific session\n"
        "  /fork [sid]       Fork current session (or a specific session)\n"
        "  /exit             Exit\n"
        "\n"
        "Type any message to chat with the agent.\n"
    )


async def _show_usage(session) -> None:
    """Display token usage statistics."""
    # 使用新的公开 API
    usage = await session.get_usage()
    print("\n📊 Token Usage:")
    print(f"  总 tokens: {usage.total_tokens}")
    print(f"  总成本: ${usage.total_cost:.4f}")
    print(f"  Prompt tokens: {usage.total_prompt_tokens}")
    print(f"  Completion tokens: {usage.total_completion_tokens}")
    print(f"  调用次数: {usage.entry_count}")

    if usage.by_model:
        print("\n  按模型统计:")
        for model, stats in usage.by_model.items():
            print(f"    {model}: {stats.total_tokens} tokens, ${stats.cost:.4f}")

    if usage.by_level:
        print("\n  按档位统计:")
        for level, stats in usage.by_level.items():
            print(f"    {level}: {stats.total_tokens} tokens, ${stats.cost:.4f}")
    print()


async def _show_context(session) -> None:
    """Display context usage breakdown."""
    from comate_agent_sdk.context.formatter import format_context_view

    info = await session.get_context_info()
    output = format_context_view(info)
    print()
    print(output)
    print()


async def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Create a new agent (tools=None 会自动加载 SDK 内置 system tools)
    # include_cost=True 启用成本追踪
    agent = Agent(config=AgentConfig(include_cost=True))
    
    # Create a new session (will be initialized on first message)
    session = agent.chat()
    
    print("=== Chat Session REPL ===")
    print("A new session will be created when you send your first message.")
    print(_help_text())

    while True:
        try:
            line = await asyncio.to_thread(input, "> ")
        except EOFError:
            break
            
        text = line.strip()
        if not text:
            continue

        if text == "/help":
            print(_help_text())
            continue
        if text == "/exit":
            break
        if text == "/session":
            print(f"session_id={session.session_id}")
            print(f"storage_root={_session_root(session.session_id)}")
            continue
        if text == "/usage":
            await _show_usage(session)
            continue
        if text == "/context":
            await _show_context(session)
            continue
        if text == "/clear":
            session.clear_history()
            print("✅ Conversation history and token stats cleared.")
            continue
        if text.startswith("/resume"):
            parts = text.split()
            if len(parts) != 2:
                print("Usage: /resume <session_id>")
                continue
            new_sid = parts[1]
            try:
                # Close current session
                await session.close()
                # Resume new session
                session = agent.chat(session_id=new_sid)
                print(f"✅ Resumed session: {session.session_id}")
                print(f"storage_root={_session_root(session.session_id)}")
            except Exception as e:
                logger.error(f"Failed to resume session: {e}")
            continue

        if text.startswith("/fork"):
            parts = text.split()
            target_sid = None
            if len(parts) > 1:
                target_sid = parts[1]
            
            try:
                # Close current session if we are forking it (optional, but cleaner if we are switching)
                # But fork_session can be called on existing session object.
                # If target_sid is provided, we use agent.chat(fork_session=...)
                
                old_session = session
                if target_sid:
                    # Fork a specific session (not necessarily current)
                    session = agent.chat(fork_session=target_sid)
                else:
                    # Fork current session
                    session = old_session.fork_session()
                
                # Close the old session wrapper if it's different (it is)
                await old_session.close()
                
                print(f"✅ Forked to new session: {session.session_id}")
                print(f"storage_root={_session_root(session.session_id)}")
            except Exception as e:
                logger.error(f"Failed to fork session: {e}")
            continue

        if text.startswith("/"):
            logger.warning(f"Unknown command: {text}")
            continue

        # Send message and stream events
        async for event in session.query_stream(text):
            _log_event(event)


if __name__ == "__main__":
    asyncio.run(main())
