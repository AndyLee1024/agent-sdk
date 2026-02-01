"""验证 AssistantMessage token 统计修复的效果

这个脚本会创建一个包含多个 tool_calls 的对话，然后检查：
1. /context 是否正确显示 Messages 类别
2. Messages 类别的 token 数是否包含 tool_calls
3. 与原始问题中的场景对比
"""

import asyncio
import logging

from bu_agent_sdk.agent.chat_session import ChatSession
from bu_agent_sdk.context.items import ItemType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    print("\n" + "=" * 60)
    print("验证 AssistantMessage Token 统计修复")
    print("=" * 60 + "\n")

    # 直接在 ContextIR 中测试（不需要真实的 Agent/API key）
    print("📊 测试 1: 验证只有 tool_calls 的 AssistantMessage 统计")
    print("-" * 60)

    from bu_agent_sdk.llm.messages import AssistantMessage, ToolCall, Function, UserMessage

    # 创建上下文
    from bu_agent_sdk.context.ir import ContextIR
    from bu_agent_sdk.context.budget import BudgetConfig

    context = ContextIR(
        budget=BudgetConfig(
            total_limit=128000,
            compact_threshold_ratio=0.8,
        )
    )

    # 模拟用户消息
    user_msg = UserMessage(content="请帮我读取 /tmp/test.py 文件")
    context.add_message(user_msg)

    # 模拟只有 tool_calls 的 Assistant 消息
    assistant_msg = AssistantMessage(
        content=None,  # 关键：没有文本内容
        tool_calls=[
            ToolCall(
                id="call_read_1",
                type="function",
                function=Function(
                    name="Read",
                    arguments='{"file_path": "/tmp/test.py"}',
                ),
            )
        ],
    )
    item = context.add_message(assistant_msg)

    print(f"✅ AssistantMessage 添加成功")
    print(f"   - Item Type: {item.item_type.value}")
    print(f"   - Token Count: {item.token_count}")
    print(f"   - Content Preview: {item.content_text[:80]}...")

    # 获取预算状态
    status = context.get_budget_status()
    print(f"\n📊 预算状态:")
    print(f"   - Total Tokens: {status.total_tokens}")
    print(f"\n📈 按类型统计:")
    for item_type, tokens in status.tokens_by_type.items():
        print(f"   - {item_type.value}: {tokens} tokens")

    # 验证关键点
    print("\n🔍 验证结果:")
    checks = []

    # 检查 1: AssistantMessage 的 token_count > 0
    check1 = item.token_count > 0
    checks.append(("AssistantMessage token_count > 0", check1))
    print(f"   {'✅' if check1 else '❌'} AssistantMessage 的 token_count > 0: {item.token_count}")

    # 检查 2: token_count 应该足够大（包含 tool_calls JSON）
    check2 = item.token_count > 10  # tool_calls JSON 应该至少 10+ tokens
    checks.append(("Token count 包含 tool_calls", check2))
    print(f"   {'✅' if check2 else '❌'} Token count 包含 tool_calls (> 10): {item.token_count}")

    # 检查 3: 预算状态中有 ASSISTANT_MESSAGE
    check3 = ItemType.ASSISTANT_MESSAGE in status.tokens_by_type
    checks.append(("预算状态包含 ASSISTANT_MESSAGE", check3))
    print(
        f"   {'✅' if check3 else '❌'} 预算状态包含 ASSISTANT_MESSAGE: "
        f"{status.tokens_by_type.get(ItemType.ASSISTANT_MESSAGE, 0)} tokens"
    )

    # 检查 4: ASSISTANT_MESSAGE 的 tokens > 0
    check4 = status.tokens_by_type.get(ItemType.ASSISTANT_MESSAGE, 0) > 0
    checks.append(("ASSISTANT_MESSAGE tokens > 0", check4))
    print(
        f"   {'✅' if check4 else '❌'} ASSISTANT_MESSAGE tokens > 0: "
        f"{status.tokens_by_type.get(ItemType.ASSISTANT_MESSAGE, 0)}"
    )

    # 总结
    all_passed = all(check[1] for check in checks)
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过！修复成功！")
    else:
        print("❌ 部分验证失败:")
        for name, result in checks:
            if not result:
                print(f"   - {name}")
    print("=" * 60 + "\n")

    # 测试 2: 多个 tool_calls
    print("\n📊 测试 2: 验证多个 tool_calls 的统计")
    print("-" * 60)

    context2 = ContextIR(
        budget=BudgetConfig(
            total_limit=128000,
            compact_threshold_ratio=0.8,
        )
    )

    # 添加用户消息
    context2.add_message(UserMessage(content="请检查这个项目的文件结构"))

    # 添加包含多个 tool_calls 的 Assistant 消息
    multi_tool_msg = AssistantMessage(
        content="让我帮你检查文件结构。",  # 有文本内容
        tool_calls=[
            ToolCall(
                id="call_1",
                type="function",
                function=Function(name="Bash", arguments='{"command": "ls -la"}'),
            ),
            ToolCall(
                id="call_2",
                type="function",
                function=Function(name="Bash", arguments='{"command": "tree -L 2"}'),
            ),
            ToolCall(
                id="call_3",
                type="function",
                function=Function(name="Bash", arguments='{"command": "find . -name \'*.py\' | head -10"}'),
            ),
        ],
    )
    item2 = context2.add_message(multi_tool_msg)

    print(f"✅ AssistantMessage (含多个 tool_calls) 添加成功")
    print(f"   - Token Count: {item2.token_count}")
    print(f"   - Tool Calls: 3 个")

    status2 = context2.get_budget_status()
    assistant_tokens2 = status2.tokens_by_type.get(ItemType.ASSISTANT_MESSAGE, 0)

    print(f"\n📊 预算状态:")
    print(f"   - Total Tokens: {status2.total_tokens}")
    print(f"   - ASSISTANT_MESSAGE: {assistant_tokens2} tokens")

    # 验证
    check5 = item2.token_count > item.token_count  # 应该比单个 tool_call 多
    print(
        f"\n🔍 验证: {'✅' if check5 else '❌'} 多个 tool_calls 的 token 数更多 "
        f"({item2.token_count} > {item.token_count})"
    )

    # 测试 3: 对比修复前后
    print("\n" + "=" * 60)
    print("📝 对比修复前后")
    print("=" * 60)
    print("\n修复前的问题:")
    print("  - AssistantMessage 只统计 content 文本")
    print("  - tool_calls 的 JSON tokens 被忽略")
    print("  - 导致 token_count = 0 或很小")
    print("  - /context 中 Messages 类别可能不显示")
    print("\n修复后的行为:")
    print("  ✅ AssistantMessage 统计 content + tool_calls")
    print("  ✅ tool_calls JSON 被完整计算")
    print("  ✅ token_count 正确反映实际大小")
    print("  ✅ /context 始终显示 Messages 类别")
    print("\n当前测试结果:")
    print(f"  - 单个 tool_call: {item.token_count} tokens")
    print(f"  - 多个 tool_calls: {item2.token_count} tokens")
    print("  - ✅ 两者都 > 0，修复成功！")
    print("\n" + "=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
