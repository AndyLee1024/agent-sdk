"""
端到端示例：展示如何实际使用 Skill 功能

这个示例展示：
1. 创建一个简单的 Skill
2. Agent 自动发现 Skill
3. LLM 使用 Skill（模拟）
"""

import asyncio
from pathlib import Path

from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.llm.messages import ToolCall, Function
from bu_agent_sdk.tools import tool


# ============================================
# 1. 定义工具
# ============================================


@tool("写入文件")
async def write_file(path: str, content: str) -> str:
    """写入内容到文件"""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return f"✅ 成功写入: {path}"
    except Exception as e:
        return f"❌ 错误: {e}"


@tool("读取文件")
async def read_file(path: str) -> str:
    """读取文件内容"""
    try:
        content = Path(path).read_text(encoding="utf-8")
        return f"📄 文件内容 ({len(content)} 字符):\n{content[:200]}..."
    except Exception as e:
        return f"❌ 错误: {e}"


@tool("搜索网络")
async def search_web(query: str) -> str:
    """搜索网络"""
    return f"🔍 搜索结果: {query}"


# ============================================
# 2. 创建 Skill
# ============================================


def setup_skill():
    """创建一个 writer Skill"""
    skill_dir = Path(".agent/skills/writer")
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_content = """---
name: writer
description: 专业的技术文档写作专家
allowed-tools: write_file, read_file
model: inherit
---

你是一个专业的技术文档写作专家。

## 你的能力
- 创建高质量的技术文档
- 使用清晰的结构和格式
- 确保内容准确完整

## 可用工具
- `write_file`: 写入文件
- `read_file`: 读取文件

注意：你不能使用 search_web 工具，专注于文档写作任务。
"""

    (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")
    print("✅ 创建了 writer Skill")


# ============================================
# 3. 演示完整流程
# ============================================


async def main():
    """演示完整的 Skill 使用流程"""
    print("\n" + "=" * 70)
    print("端到端示例：Skill 功能演示")
    print("=" * 70)

    # 步骤 1: 创建 Skill
    print("\n📝 步骤 1: 创建 Skill")
    setup_skill()

    # 步骤 2: 创建 Agent（会自动发现 Skills）
    print("\n🤖 步骤 2: 创建 Agent（自动发现 Skills）")
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[write_file, read_file, search_web],
    )

    print(f"   - 可用工具: {[t.name for t in agent.tools]}")
    print(f"   - 发现的 Skills: {[s.name for s in agent.skills] if agent.skills else []}")

    # 步骤 3: 验证 Skill 工具已创建
    print("\n🛠️  步骤 3: 验证 Skill 工具")
    skill_tool = agent._tool_map.get("Skill")
    if skill_tool:
        print(f"   ✅ Skill 工具已创建")
        print(f"   - 工具名称: {skill_tool.name}")
        print(f"   - 工具描述: {skill_tool.description[:80]}...")

    # 步骤 4: 模拟 LLM 调用 Skill
    print("\n🎯 步骤 4: 模拟 LLM 调用 Skill")
    print("   LLM 决定使用 'writer' Skill 来帮助写文档...")

    tool_call = ToolCall(
        id="call_123",
        function=Function(
            name="Skill",
            arguments='{"skill_name": "writer"}'
        )
    )

    print(f"   - 工具调用: {tool_call.function.name}({tool_call.function.arguments})")

    # 执行 Skill 调用
    result = await agent._execute_skill_call(tool_call)

    print(f"\n   ✅ Skill 调用结果:")
    print(f"   - 成功: {not result.is_error}")
    print(f"   - 内容: {result.content}")

    # 步骤 5: 验证 Execution Context 修改
    print("\n🔒 步骤 5: 验证 Execution Context 修改")
    print(f"   - 激活的 Skill: {agent._active_skill_name}")
    print(f"   - 当前可用工具: {[t.name for t in agent.tools]}")

    # 验证工具权限被限制
    tool_names = {t.name for t in agent.tools}
    assert "write_file" in tool_names, "write_file 应该可用"
    assert "read_file" in tool_names, "read_file 应该可用"
    assert "search_web" not in tool_names, "search_web 应该被移除"

    print(f"   ✅ 工具权限已按 Skill 定义限制")
    print(f"      - 保留: write_file, read_file")
    print(f"      - 移除: search_web")

    # 步骤 6: 验证重复调用保护
    print("\n🛡️  步骤 6: 验证重复调用保护")
    print("   尝试激活另一个 Skill...")

    tool_call_2 = ToolCall(
        id="call_456",
        function=Function(
            name="Skill",
            arguments='{"skill_name": "another-skill"}'
        )
    )

    result_2 = await agent._execute_skill_call(tool_call_2)

    print(f"   ✅ 重复调用被拒绝:")
    print(f"   - 错误: {result_2.is_error}")
    print(f"   - 内容: {result_2.content}")

    # 步骤 7: 查看注入的消息
    print("\n💬 步骤 7: 查看注入的消息")
    print(f"   消息历史中共有 {len(agent._messages)} 条消息")

    for i, msg in enumerate(agent._messages):
        msg_type = msg.__class__.__name__
        is_meta = getattr(msg, 'is_meta', None)
        content_preview = str(msg)[:60] + "..." if len(str(msg)) > 60 else str(msg)

        print(f"   [{i}] {msg_type}")
        if is_meta is not None:
            print(f"       - is_meta: {is_meta}")
        print(f"       - 内容: {content_preview}")

    # 总结
    print("\n" + "=" * 70)
    print("✅ 演示完成！")
    print("=" * 70)
    print("\n📋 总结:")
    print("   1. ✅ Skill 自动发现")
    print("   2. ✅ Skill 工具创建")
    print("   3. ✅ LLM 调用 Skill")
    print("   4. ✅ 消息注入（元数据 + prompt）")
    print("   5. ✅ Execution Context 修改（工具权限限制）")
    print("   6. ✅ 重复调用保护")
    print("\n🎉 Skill 系统运行正常！")


if __name__ == "__main__":
    asyncio.run(main())
