import logging
import asyncio

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 只关注 MCP 和 agent 相关日志
logging.getLogger("comate_agent_sdk.mcp.manager").setLevel(logging.DEBUG)
logging.getLogger("comate_agent_sdk.agent").setLevel(logging.DEBUG)

# 禁用其他噪音日志
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("mcp.client").setLevel(logging.WARNING)
logging.getLogger("langfuse").setLevel(logging.ERROR)

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import ComateAgentOptions, ToolResultEvent, ToolCallEvent, ChatSession, TextEvent, StopEvent

async def main():
    print("\n" + "="*80)
    print("Creating agent with MCP config...")
    print("="*80 + "\n")

    agent = Agent(
        options=ComateAgentOptions(
            mcp_servers={
                "exa_search": {
                    "type": "http",
                    "url": "https://mcp.exa.ai/mcp?exaApiKey=2ac4b289-8f68-473b-8cfd-3f8cb11595b7"
                }
            }
        ),
    )

    print("\n" + "="*80)
    print("Creating ChatSession...")
    print("="*80 + "\n")

    session = ChatSession(agent)

    print("\n" + "="*80)
    print("Starting query_stream...")
    print("="*80 + "\n")

    async for event in session.query_stream("使用 exa搜索 claude agent sdk"):
        match event:
            case ToolResultEvent(tool=tool, result=result, is_error=is_error):
                print(f"\n[Tool Result: {tool} is_error: {is_error}]")
                if is_error:
                    print(f"Error: {result}")
            case ToolCallEvent(tool=tool_name, args=arguments):
                print(f"\n[Tool Call: {tool_name}]")
            case TextEvent(content=text):
                print(text, end="", flush=True)
            case StopEvent(reason=reason):
                print(f"\n[Session Ended: {reason}]")
                break

if __name__ == "__main__":
    asyncio.run(main())
