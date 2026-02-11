"""
验证 Memory 从 SystemMessage 迁移到 UserMessage 的完整测试

运行方式：
    uv run python comate_agent_sdk/examples/test_memory_migration.py
"""

import logging
from comate_agent_sdk.agent.service import Agent
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import SystemMessage, UserMessage

logger = logging.getLogger(__name__)


def test_memory_as_usermessage():
    """测试 Memory 作为 UserMessage(is_meta=True) 注入"""
    logger.info("=== 测试 1: Memory 作为 UserMessage 注入 ===")

    template = Agent(config=AgentConfig(system_prompt="Test Agent"))
    agent = template.create_runtime()
    agent._context.set_memory("TEST_MEMORY_CONTENT")

    messages = agent._context.lower()

    # 验证消息结构
    assert len(messages) >= 2, f"消息数量不足: {len(messages)}"
    assert isinstance(messages[0], SystemMessage), "messages[0] 应该是 SystemMessage"
    assert isinstance(messages[1], UserMessage), "messages[1] 应该是 UserMessage"

    # 验证 Memory 在 UserMessage 中
    memory_msg = messages[1]
    assert getattr(memory_msg, "is_meta", False), "Memory UserMessage 应该有 is_meta=True"
    assert "<instructions>" in memory_msg.content, "Memory 应该用 <instructions> 标签包裹"
    assert "TEST_MEMORY_CONTENT" in memory_msg.content, "Memory 应包含设置的内容"

    # 验证 SystemMessage 不包含 Memory
    assert "TEST_MEMORY_CONTENT" not in messages[0].content, "SystemMessage 不应包含 Memory"

    logger.info("✓ Memory 正确注入为 UserMessage(is_meta=True)")


def test_memory_not_in_header():
    """测试 Memory 不在 header.items 中"""
    logger.info("=== 测试 2: Memory 不在 header.items 中 ===")

    template = Agent(config=AgentConfig(system_prompt="Test"))
    agent = template.create_runtime()
    agent._context.set_memory("TEST_MEMORY")

    # 验证 header 中没有 MEMORY 类型的 item
    memory_in_header = agent._context.header.find_one_by_type(ItemType.MEMORY)
    assert memory_in_header is None, "Memory 不应该在 header.items 中"

    # 验证 memory_item 字段存在
    assert agent._context.memory_item is not None, "memory_item 字段应该存在"
    assert agent._context.memory_item.item_type == ItemType.MEMORY, "memory_item 类型应该是 MEMORY"

    logger.info("✓ Memory 存储在独立字段 _memory_item 中，不在 header.items")


def test_clear_history_rebuilds_memory():
    """测试 clear_history 后 Memory 能正确重建"""
    logger.info("=== 测试 3: clear_history 后 Memory 重建 ===")

    template = Agent(config=AgentConfig(system_prompt="Test"))
    agent = template.create_runtime()

    # 设置 memory
    agent._context.set_memory("ORIGINAL_MEMORY")

    # 添加一些对话
    from comate_agent_sdk.llm.messages import UserMessage as UserMsg, AssistantMessage
    agent._context.add_message(UserMsg(content="Hello"))
    agent._context.add_message(AssistantMessage(content="Hi there"))

    # 验证 lowering 后有 memory
    messages_before = agent._context.lower()
    assert any("ORIGINAL_MEMORY" in getattr(m, "content", "") for m in messages_before), "清除前应该有 memory"

    # 清除上下文
    agent._context.clear()

    # 验证 memory_item 被清除
    assert agent._context.memory_item is None, "clear() 后 memory_item 应该为 None"

    # 重新设置（模拟 clear_history 的重建流程）
    agent._context.set_system_prompt("Test")
    agent._context.set_memory("REBUILT_MEMORY")

    # 验证 memory 重建成功
    messages_after = agent._context.lower()
    assert len(messages_after) >= 2, "重建后应该有至少 2 条消息"
    assert isinstance(messages_after[1], UserMessage), "messages[1] 应该是 Memory UserMessage"
    assert "REBUILT_MEMORY" in messages_after[1].content, "重建后的 memory 应该包含新内容"

    logger.info("✓ clear() 和重建流程正确")


def test_memory_token_accounting():
    """测试 Memory 的 token 统计"""
    logger.info("=== 测试 4: Memory token 统计 ===")

    template = Agent(config=AgentConfig(system_prompt="Test"))
    agent = template.create_runtime()

    # 注意：Agent 可能在初始化时已经加载了 AGENTS.md 作为 memory
    # 所以我们清除它，然后重新设置
    agent._context._memory_item = None

    # 记录清除后的 token 数
    tokens_without_memory = agent._context.total_tokens

    # 添加 memory
    memory_content = "This is a test memory content for token counting"
    agent._context.set_memory(memory_content)

    # 验证 total_tokens 增加
    assert agent._context.total_tokens > tokens_without_memory, "添加 memory 后 total_tokens 应该增加"

    # 验证 memory_item 有正确的 token_count
    assert agent._context.memory_item is not None, "memory_item 应该存在"
    assert agent._context.memory_item.token_count > 0, "memory_item 应该有 token_count"

    # 验证 get_budget_status 包含 memory tokens
    budget_status = agent._context.get_budget_status()
    assert ItemType.MEMORY in budget_status.tokens_by_type, "budget_status 应该包含 MEMORY 类型"
    assert budget_status.tokens_by_type[ItemType.MEMORY] > 0, "MEMORY 类型应该有 token 数"

    logger.info(f"✓ Memory token 统计正确 (memory tokens: {agent._context.memory_item.token_count})")


def test_memory_idempotent_update():
    """测试 Memory 的幂等更新"""
    logger.info("=== 测试 5: Memory 幂等更新 ===")

    template = Agent(config=AgentConfig(system_prompt="Test"))
    agent = template.create_runtime()

    # 第一次设置
    agent._context.set_memory("FIRST_MEMORY")
    first_id = agent._context.memory_item.id
    first_tokens = agent._context.memory_item.token_count

    # 第二次设置（应该覆盖）
    agent._context.set_memory("SECOND_MEMORY_LONGER_CONTENT")
    second_id = agent._context.memory_item.id
    second_tokens = agent._context.memory_item.token_count

    # 验证是同一个对象（幂等）
    assert first_id == second_id, "幂等更新应该复用同一个 ContextItem"

    # 验证内容已更新
    messages = agent._context.lower()
    memory_msg = messages[1]
    assert "SECOND_MEMORY_LONGER_CONTENT" in memory_msg.content, "内容应该更新为新内容"
    assert "FIRST_MEMORY" not in memory_msg.content, "旧内容应该被替换"

    # 验证 token 数更新
    assert second_tokens > first_tokens, "更长的内容应该有更多 tokens"

    logger.info("✓ Memory 幂等更新正确")


def test_memory_cache_hint():
    """测试 Memory 的 cache hint"""
    logger.info("=== 测试 6: Memory cache hint ===")

    template = Agent(config=AgentConfig(system_prompt="Test"))
    agent = template.create_runtime()

    # 设置 memory with cache=True（默认）
    agent._context.set_memory("CACHED_MEMORY", cache=True)

    messages = agent._context.lower()
    memory_msg = messages[1]

    # 验证 UserMessage 有 cache 属性
    assert hasattr(memory_msg, "cache"), "Memory UserMessage 应该有 cache 属性"
    assert memory_msg.cache is True, "Memory UserMessage 的 cache 应该为 True"

    logger.info("✓ Memory cache hint 正确")


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("开始测试 Memory 迁移...")

    tests = [
        test_memory_as_usermessage,
        test_memory_not_in_header,
        test_clear_history_rebuilds_memory,
        test_memory_token_accounting,
        test_memory_idempotent_update,
        test_memory_cache_hint,
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
        logger.info("✓ 所有测试通过！Memory 迁移成功！")
        return 0
    else:
        logger.error("✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
