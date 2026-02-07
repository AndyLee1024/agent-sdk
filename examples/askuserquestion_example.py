"""演示 AskUserQuestion 工具的使用

这个示例展示了 Agent 如何在执行过程中向用户询问问题并处理答案。
"""
import asyncio
import logging
from pathlib import Path

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.events import UserQuestionEvent, StopEvent, TextEvent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


async def main():
    print("=" * 70)
    print("AskUserQuestion 工具演示")
    print("=" * 70)

    print("\n📝 场景: Agent 需要了解用户对项目设置的偏好")
    print("💡 注意: AskUserQuestion 是 system tool,所有 Agent 自动可用\n")

    # 模拟 Agent 执行
    message = """
    我想为我的 Web 项目设置测试框架和代码风格工具。
    请先询问我的偏好,然后帮我配置。
    """

    print(f"用户输入: {message}\n")
    print("🤖 Agent 开始执行...\n")

    # Agent 会调用 AskUserQuestion 工具
    print("期望流程:")
    print("1. Agent 调用 AskUserQuestion 询问偏好")
    print("2. 触发 UserQuestionEvent,包含问题列表")
    print("3. 触发 StopEvent(reason='waiting_for_input'),暂停执行")
    print("4. UI 展示问题,收集用户答案")
    print("5. 通过新的 UserMessage 发送答案")
    print("6. Agent 继续执行,根据答案完成配置\n")

    # 模拟事件流
    print("📢 模拟事件序列:\n")

    # 模拟 UserQuestionEvent
    mock_questions = [
        {
            "question": "Which testing framework would you like to use?",
            "header": "Framework",
            "options": [
                {
                    "label": "Jest (Recommended)",
                    "description": "Most popular, zero-config for most setups, great for React"
                },
                {
                    "label": "Vitest",
                    "description": "Faster, better Vite integration, similar API to Jest"
                },
                {
                    "label": "Mocha + Chai",
                    "description": "Flexible, modular, more setup required"
                }
            ],
            "multiSelect": False
        },
        {
            "question": "Which code style tools do you want?",
            "header": "Style tools",
            "options": [
                {
                    "label": "ESLint",
                    "description": "JavaScript linting to catch errors and enforce style"
                },
                {
                    "label": "Prettier",
                    "description": "Opinionated code formatter for consistent style"
                },
                {
                    "label": "TypeScript",
                    "description": "Static type checking for better code quality"
                }
            ],
            "multiSelect": True
        }
    ]

    # 显示 UserQuestionEvent
    event = UserQuestionEvent(
        questions=mock_questions,
        tool_call_id="toolu_123"
    )
    print(f"✅ {event}")

    print("\n📋 问题详情:")
    for i, q in enumerate(mock_questions, 1):
        print(f"\n  问题 {i}: {q['question']}")
        print(f"  Header: {q['header']}")
        print(f"  多选: {'是' if q['multiSelect'] else '否'}")
        print(f"  选项:")
        for j, opt in enumerate(q['options'], 1):
            print(f"    {j}. {opt['label']}")
            print(f"       {opt['description']}")

    # 模拟 StopEvent
    stop_event = StopEvent(reason="waiting_for_input")
    print(f"\n{stop_event}")
    print("⏸️  Agent 执行暂停,等待用户输入...\n")

    # 模拟用户回答
    print("=" * 70)
    print("用户回答")
    print("=" * 70)
    print("\n用户选择:")
    print("  问题 1: Jest (Recommended)")
    print("  问题 2: ESLint, Prettier\n")

    # 模拟答案通过 UserMessage 发送
    mock_answer = """
    我的选择:
    - 测试框架: Jest (Recommended)
    - 代码风格工具: ESLint, Prettier

    请根据这些选择帮我配置项目。
    """

    print(f"💬 UserMessage: {mock_answer}")

    # Agent 继续执行
    print("\n🤖 Agent 恢复执行...")
    print("   读取用户答案...")
    print("   根据选择配置项目...")
    print("   ✅ 完成!\n")

    # 模拟最终响应
    final_text = """
    好的,我已经根据你的选择配置好了项目:

    1. **测试框架 - Jest**
       - 安装 jest 和相关依赖
       - 创建 jest.config.js 配置文件
       - 添加测试脚本到 package.json

    2. **代码风格工具**
       - ESLint: 配置 .eslintrc.js,添加推荐规则
       - Prettier: 创建 .prettierrc 配置文件
       - 配置 ESLint 和 Prettier 协同工作

    你可以运行 `npm test` 来执行测试,运行 `npm run lint` 来检查代码风格。
    """

    print("=" * 70)
    print("Agent 最终响应")
    print("=" * 70)
    print(final_text)

    print("\n" + "=" * 70)
    print("✅ 演示完成!")
    print("=" * 70)
    print("\n关键要点:")
    print("1. AskUserQuestion 返回 status='waiting_for_input'")
    print("2. runner_stream 检测到后 yield UserQuestionEvent")
    print("3. 然后 yield StopEvent(reason='waiting_for_input') 暂停执行")
    print("4. 用户答案通过标准的 UserMessage 发送")
    print("5. Agent 在新一轮执行中继续处理")


if __name__ == "__main__":
    asyncio.run(main())
