"""
快速测试：验证 Tool Strategy 功能是否正常工作

运行方式：
    uv run python comate_agent_sdk/examples/test_tool_strategy.py
"""

import logging

from comate_agent_sdk.agent.service import Agent
from comate_agent_sdk.agent import AgentConfig


logger = logging.getLogger(__name__)


def test_tool_strategy_generation():
    """测试 tool_strategy 是否正确生成"""
    logger.info("=== 测试 1: 验证 tool_strategy 生成 ===")

    # 创建 Agent（使用默认系统工具）
    template = Agent(
        config=AgentConfig(
            system_prompt="Test agent for tool strategy",
        ),
    )
    agent = template.create_runtime()

    # 验证 ContextIR 中是否有 TOOL_STRATEGY
    from comate_agent_sdk.context.items import ItemType

    tool_strategy_item = agent._context.header.find_one_by_type(ItemType.TOOL_STRATEGY)
    assert tool_strategy_item is not None, "TOOL_STRATEGY 未找到"
    logger.info(
        f"✓ TOOL_STRATEGY 已成功写入 ContextIR header（tokens={tool_strategy_item.token_count}）"
    )

    # 验证内容包含关键元素
    content = tool_strategy_item.content_text
    assert "<tools>" in content, "缺少 <tools> 标签"
    assert "- **Bash**:" in content, "缺少 Bash 工具概览"
    assert "- **Read**:" in content, "缺少 Read 工具概览"
    logger.info("✓ 内容验证通过")


def test_custom_tools_no_usage_rules():
    """测试自定义工具（无 usage_rules）不出现在 tool_strategy 中"""
    logger.info("=== 测试 2: 自定义工具不应出现在 tool_strategy ===")

    from comate_agent_sdk.tools.decorator import tool

    @tool("Custom tool without usage rules")
    async def custom_tool(query: str) -> str:
        return f"Result: {query}"

    template = Agent(
        config=AgentConfig(
            system_prompt="Agent with custom tool",
            tools=[custom_tool],
            agents=(),
        ),
    )
    agent = template.create_runtime()

    from comate_agent_sdk.context.items import ItemType

    tool_strategy_item = agent._context.header.find_one_by_type(ItemType.TOOL_STRATEGY)

    # 注意：即使只有自定义工具（无 usage_rules），如果有其他系统工具，TOOL_STRATEGY 仍会存在
    # 因此这个断言需要修改为：验证自定义工具不在 TOOL_STRATEGY 中
    if tool_strategy_item:
        content = tool_strategy_item.content_text
        assert "custom_tool" not in content.lower(), "自定义工具不应出现在 TOOL_STRATEGY 中"
        logger.info("✓ 自定义工具无 usage_rules，不出现在 TOOL_STRATEGY（符合预期）")
    else:
        logger.info("✓ TOOL_STRATEGY 为空（符合预期）")


def test_system_message_order():
    """测试 Memory 作为独立 UserMessage 的注入顺序"""
    logger.info("=== 测试 3: 验证 Memory 注入顺序（UserMessage） ===")

    template = Agent(
        config=AgentConfig(
            system_prompt="TEST_SYSTEM_PROMPT",
        ),
    )
    agent = template.create_runtime()

    # 手动添加 memory 用于测试顺序
    agent._context.set_memory("TEST_MEMORY_CONTENT")

    # Lower IR 并检查顺序
    messages = agent._context.lower()
    assert messages, "没有消息"
    assert len(messages) >= 2, f"消息数量不足：{len(messages)}"

    # 验证 messages[0] 是 SystemMessage
    from comate_agent_sdk.llm.messages import SystemMessage, UserMessage
    system_msg = messages[0]
    assert isinstance(system_msg, SystemMessage), f"messages[0] 应该是 SystemMessage，实际是 {type(system_msg)}"

    # 验证 SystemMessage 中不包含 Memory（Memory 已从 header 移除）
    system_content = system_msg.content
    assert "TEST_MEMORY_CONTENT" not in system_content, "SystemMessage 中不应包含 Memory 内容"
    assert "TEST_SYSTEM_PROMPT" in system_content, "SystemMessage 应包含 SYSTEM_PROMPT"
    # 验证 TOOL_STRATEGY 存在（使用实际格式 <tools>）
    assert "<tools>" in system_content, "SystemMessage 应包含 TOOL_STRATEGY (<tools> 标签)"

    # 验证 messages[1] 是 Memory UserMessage(is_meta=True)
    memory_msg = messages[1]
    assert isinstance(memory_msg, UserMessage), f"messages[1] 应该是 UserMessage，实际是 {type(memory_msg)}"
    assert getattr(memory_msg, "is_meta", False), "Memory UserMessage 应该有 is_meta=True"
    assert "TEST_MEMORY_CONTENT" in memory_msg.content, "Memory UserMessage 应包含 TEST_MEMORY_CONTENT"
    assert "<instructions>" in memory_msg.content, "Memory 应该用 <instructions> 标签包裹"
    assert "</instructions>" in memory_msg.content, "Memory 应该有闭合的 </instructions> 标签"

    logger.info("✓ 顺序正确：messages[0]=SystemMessage (不含 Memory), messages[1]=Memory UserMessage(is_meta=True)")


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
