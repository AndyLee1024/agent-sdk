"""
快速测试：验证 Tool Strategy 功能是否正常工作

运行方式：
    uv run python comate_agent_sdk/examples/test_tool_strategy.py
"""

import logging

from comate_agent_sdk.agent.service import Agent
from comate_agent_sdk.agent import ComateAgentOptions


logger = logging.getLogger(__name__)


def test_tool_strategy_generation():
    """测试 tool_strategy 是否正确生成"""
    logger.info("=== 测试 1: 验证 tool_strategy 生成 ===")

    # 创建 Agent（使用默认系统工具）
    agent = Agent(
        options=ComateAgentOptions(
            system_prompt="Test agent for tool strategy",
        ),
    )

    # 验证 ContextIR 中是否有 TOOL_STRATEGY
    from comate_agent_sdk.context.items import ItemType

    tool_strategy_item = agent._context.header.find_one_by_type(ItemType.TOOL_STRATEGY)
    assert tool_strategy_item is not None, "TOOL_STRATEGY 未找到"
    logger.info(
        f"✓ TOOL_STRATEGY 已成功写入 ContextIR header（tokens={tool_strategy_item.token_count}）"
    )

    # 验证内容包含关键元素
    content = tool_strategy_item.content_text
    assert "<system_tools_definition>" in content, "缺少 <system_tools_definition> 标签"
    assert "<tool_overview>" in content, "缺少 <tool_overview> 标签"
    assert "Bash" in content, "缺少 Bash 工具"
    assert "Read" in content, "缺少 Read 工具"
    logger.info("✓ 内容验证通过")


def test_custom_tools_no_usage_rules():
    """测试自定义工具（无 usage_rules）不出现在 tool_strategy 中"""
    logger.info("=== 测试 2: 自定义工具不应出现在 tool_strategy ===")

    from comate_agent_sdk.tools.decorator import tool

    @tool("Custom tool without usage rules")
    async def custom_tool(query: str) -> str:
        return f"Result: {query}"

    agent = Agent(
        options=ComateAgentOptions(
            system_prompt="Agent with custom tool",
            tools=[custom_tool],
        ),
    )

    from comate_agent_sdk.context.items import ItemType

    tool_strategy_item = agent._context.header.find_one_by_type(ItemType.TOOL_STRATEGY)

    assert tool_strategy_item is None, "自定义工具无 usage_rules 时，TOOL_STRATEGY 应为空"
    logger.info("✓ 自定义工具无 usage_rules，TOOL_STRATEGY 为空（符合预期）")


def test_system_message_order():
    """测试 system message 拼接顺序"""
    logger.info("=== 测试 3: 验证 system message 拼接顺序 ===")

    agent = Agent(
        options=ComateAgentOptions(
            system_prompt="TEST_SYSTEM_PROMPT",
        ),
    )

    # 手动添加 memory 用于测试顺序
    agent._context.set_memory("TEST_MEMORY")

    # Lower IR 并检查顺序
    messages = agent._context.lower()
    assert messages, "没有消息"

    system_msg = messages[0]
    content = system_msg.content

    # 验证顺序：SYSTEM_PROMPT → MEMORY → TOOL_STRATEGY → SUBAGENT → SKILL
    positions = {
        "SYSTEM_PROMPT": content.find("TEST_SYSTEM_PROMPT"),
        "MEMORY": content.find("TEST_MEMORY"),
        "TOOL_STRATEGY": content.find("<system_tools_definition>"),
    }

    assert (
        positions["SYSTEM_PROMPT"] < positions["MEMORY"] < positions["TOOL_STRATEGY"]
    ), f"顺序错误: {positions}"
    logger.info("✓ 顺序正确：SYSTEM_PROMPT → MEMORY → TOOL_STRATEGY")


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("开始测试 Tool Strategy 功能...")

    tests = [
        test_tool_strategy_generation,
        test_custom_tools_no_usage_rules,
        test_system_message_order,
    ]

    results = []
    for test in tests:
        try:
            test()
            results.append(True)
        except Exception as e:
            logger.exception(f"✗ 测试失败: {e}")
            results.append(False)

    logger.info("=" * 60)
    logger.info(f"测试结果: {sum(1 for r in results if r)}/{len(results)} 通过")
    logger.info("=" * 60)

    if all(results):
        logger.info("✓ 所有测试通过！")
        return 0
    else:
        logger.error("✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
