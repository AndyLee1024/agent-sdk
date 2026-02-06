from comate_agent_sdk import Agent
from comate_agent_sdk.agent import ComateAgentOptions, ToolResultEvent, PreCompactEvent, ThinkingEvent,SessionInitEvent, ChatSession, TextEvent, StopEvent, ToolCallEvent
from comate_agent_sdk.llm import ChatOpenAI
from comate_agent_sdk.tools import tool
import asyncio

@tool("Add two numbers 涉及到加法运算 必须使用这个工具")
async def add(a: int, b: int) -> int:
    return a + b

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

async def main():
    session = ChatSession(agent)
    async for event in session.query_stream("使用 exa搜索 claude agent sdk文档，然后找到 对mcp的说明，最后总结一下 mcp 是什么。保存到 test.md"):
        match event:
            case SessionInitEvent(session_id=se):
                print(f"[Session Started: {se}]")
            case ThinkingEvent(content=thinking):
                print(f"\n[Thinking: {thinking}]")
            case PreCompactEvent(current_tokens=t, threshold=th, trigger=trig):                         
              print(f"压缩前: {t} tokens (阈值: {th})")    
            case ToolResultEvent(tool=tool, result=result, tool_call_id=tcid, is_error=is_error):
                print(f"\n[Tool Result: {tool} is_error: {is_error} tool_call_id: {tcid} with result {result}]")
            case ToolCallEvent(tool=tool_name, args=arguments, tool_call_id=tcid):
                print(f"\n[Tool Call: {tool_name} tool_call_id: {tcid} with args {arguments}]")
            case TextEvent(content=text):
                print(f"Message: {text}", end="", flush=True)
            case StopEvent(reason=reason):
                print(f"\n[Session Ended: {reason}]")
                break

if __name__ == "__main__":
    asyncio.run(main())
