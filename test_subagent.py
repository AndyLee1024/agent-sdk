#!/usr/bin/env python
"""
快速测试 Subagent 功能 - 展示终极简化版
"""

import asyncio

from bu_agent_sdk import Agent, AgentDefinition
from bu_agent_sdk.llm.anthropic.chat import ChatAnthropic
from bu_agent_sdk.tools import tool


async def main():
    print("=== Subagent 功能测试（终极简化版）===\n")

    # 1. 定义工具（自动注册到全局 registry）
    @tool("计算两个数的和")
    async def add(a: int, b: int) -> str:
        result = a + b
        print(f"  [add] {a} + {b} = {result}")
        return f"{a} + {b} = {result}"

    @tool("计算两个数的乘积")
    async def multiply(a: int, b: int) -> str:
        result = a * b
        print(f"  [multiply] {a} × {b} = {result}")
        return f"{a} × {b} = {result}"

    print(f"✓ 定义了 2 个工具（自动注册到全局 registry）")

    # 2. 定义 Subagent
    calculator = AgentDefinition(
        name="calculator",
        description="数学计算专家。当需要进行算术运算时使用。",
        prompt="""你是一个数学计算专家。

使用可用的工具来执行计算：
- add: 计算两个数的和
- multiply: 计算两个数的乘积

始终返回计算结果。""",
        tools=["add", "multiply"],
        model="haiku",
        timeout=30,
    )

    print(f"✓ 定义了 Subagent: {calculator.name}\n")

    # 3. 创建主 Agent（终极简化！）
    try:
        # 🎉 不需要传 tools，不需要传 tool_registry！
        agent = Agent(
            llm=ChatAnthropic(model="claude-sonnet-4-20250514"),
            agents=[calculator],
            system_prompt="你是一个助手，可以使用 calculator Subagent 来帮助用户进行计算。",
        )
        print(f"✓ 创建主 Agent 成功（零配置！）")
        print(f"  - 工具数量: {len(agent.tools)}")
        print(f"  - 工具列表: {[t.name for t in agent.tools]}")
        print(f"  - Tool Registry: {'是' if agent.tool_registry else '否'}")
    except Exception as e:
        print(f"✗ 创建主 Agent 失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. 测试查询
    print("\n=== 测试查询 ===")
    try:
        query = "请使用 calculator 帮我计算 12 加 34，然后再乘以 2"
        print(f"查询: {query}\n")
        result = await agent.query(query)
        print(f"\n结果: {result}")
    except Exception as e:
        print(f"✗ 查询失败: {e}")
        import traceback

        traceback.print_exc()

    # 5. 查看使用情况
    try:
        usage = await agent.get_usage()
        print(f"\n=== Token 使用情况 ===")
        print(f"总 tokens: {usage.total_tokens}")
        if usage.total_cost > 0:
            print(f"总成本: ${usage.total_cost:.6f}")
    except Exception as e:
        print(f"获取使用情况失败: {e}")

    print("\n✓ 测试完成")


if __name__ == "__main__":
    asyncio.run(main())
