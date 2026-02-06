import logging
import asyncio
from comate_agent_sdk import Agent
from comate_agent_sdk.agent import ComateAgentOptions

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 特别关注 MCP 相关的日志
logging.getLogger("comate_agent_sdk.mcp.manager").setLevel(logging.DEBUG)
logging.getLogger("comate_agent_sdk.agent").setLevel(logging.DEBUG)

async def main():
    print("=" * 80)
    print("开始测试 MCP 工具加载...")
    print("=" * 80)

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

    print("\n" + "=" * 80)
    print("Agent 已创建，现在尝试查询...")
    print("=" * 80 + "\n")

    try:
        result = await agent.query("test query to trigger MCP loading")
        print(f"\n查询结果: {result}")
    except Exception as e:
        print(f"\n查询失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
