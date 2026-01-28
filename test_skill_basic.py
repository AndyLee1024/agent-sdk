"""测试 Skill 功能的集成测试"""

import asyncio
from pathlib import Path

from bu_agent_sdk import Agent, SkillDefinition
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import tool


@tool("读取文件内容")
async def read_file(path: str) -> str:
    """读取文件并返回内容

    Args:
        path: 文件路径
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


@tool("写入文件")
async def write_file(path: str, content: str) -> str:
    """写入内容到文件

    Args:
        path: 文件路径
        content: 要写入的内容
    """
    try:
        Path(path).write_text(content, encoding="utf-8")
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool("搜索网络")
async def search_web(query: str) -> str:
    """搜索网络

    Args:
        query: 搜索查询
    """
    return f"Mock search results for: {query}"


async def test_skill_basic():
    """测试基础 Skill 功能"""
    print("\n=== 测试 1: 基础 Skill 加载 ===")

    # 创建测试 Skill
    test_skill_dir = Path(".agent/skills/test-skill")
    test_skill_dir.mkdir(parents=True, exist_ok=True)

    skill_content = """---
name: test-skill
description: A test skill for file operations
allowed-tools: read_file, write_file
model: inherit
---

你是一个文件操作专家。你只能使用 read_file 和 write_file 工具。

当用户请求文件操作时：
1. 使用 read_file 读取文件
2. 使用 write_file 写入文件

你不能使用任何其他工具。
"""

    (test_skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")

    # 创建 Agent
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[read_file, write_file, search_web],
    )

    # 验证 Skill 被发现
    assert agent.skills is not None, "Skills should be auto-discovered"
    assert len(agent.skills) > 0, "Should discover at least one skill"

    skill_names = [s.name for s in agent.skills]
    print(f"✅ 发现的 Skills: {skill_names}")

    # 验证 Skill 工具被创建
    skill_tool = agent._tool_map.get("Skill")
    assert skill_tool is not None, "Skill tool should be created"
    print(f"✅ Skill 工具已创建: {skill_tool.name}")

    # 验证工具列表
    tool_names = [t.name for t in agent.tools]
    print(f"✅ 可用工具: {tool_names}")
    assert "Skill" in tool_names, "Skill tool should be in tools list"


async def test_skill_manual_definition():
    """测试手动定义 Skill"""
    print("\n=== 测试 2: 手动定义 Skill ===")

    # 手动创建 SkillDefinition
    manual_skill = SkillDefinition(
        name="manual-skill",
        description="Manually defined skill",
        prompt="你是一个手动定义的 Skill。只能使用 search_web 工具。",
        allowed_tools=["search_web"],
        model="inherit",
    )

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[read_file, write_file, search_web],
        skills=[manual_skill],
    )

    # 验证手动 Skill 被加载
    skill_names = [s.name for s in agent.skills]
    print(f"✅ 加载的 Skills: {skill_names}")
    assert "manual-skill" in skill_names, "Manual skill should be loaded"


async def test_skill_from_markdown():
    """测试从 Markdown 解析 Skill"""
    print("\n=== 测试 3: 从 Markdown 解析 Skill ===")

    content = """---
name: parser-test
description: Test parsing from markdown
allowed-tools: read_file, write_file
model: gpt-4o
---

这是测试 Skill 的 prompt 内容。

## 使用说明
1. 步骤一
2. 步骤二
"""

    skill = SkillDefinition.from_markdown(
        content=content,
        dir_name="fallback-name",
        base_dir=Path("/test/base")
    )

    print(f"✅ 解析成功:")
    print(f"   - name: {skill.name}")
    print(f"   - description: {skill.description}")
    print(f"   - allowed_tools: {skill.allowed_tools}")
    print(f"   - model: {skill.model}")
    print(f"   - base_dir: {skill.base_dir}")

    # 测试 {baseDir} 替换
    skill.prompt = "Base directory: {baseDir}"
    prompt = skill.get_prompt()
    assert "/test/base" in prompt, "Should replace {baseDir}"
    print(f"✅ {'{baseDir}'} 替换成功: {prompt}")


async def test_disable_model_invocation():
    """测试 disable_model_invocation 标志"""
    print("\n=== 测试 4: disable_model_invocation ===")

    # 创建一个 disable_model_invocation=True 的 Skill
    disabled_skill = SkillDefinition(
        name="disabled-skill",
        description="This skill is disabled",
        prompt="Should not appear in tool description",
        disable_model_invocation=True,
    )

    normal_skill = SkillDefinition(
        name="normal-skill",
        description="This skill is enabled",
        prompt="Should appear in tool description",
        disable_model_invocation=False,
    )

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=[read_file],
        skills=[disabled_skill, normal_skill],
    )

    # 获取 Skill 工具的描述
    skill_tool = agent._tool_map.get("Skill")
    assert skill_tool is not None

    # 验证 disabled skill 不在描述中
    description = skill_tool.description
    assert "normal-skill" in description, "Normal skill should be in description"
    assert "disabled-skill" not in description, "Disabled skill should NOT be in description"
    print(f"✅ disable_model_invocation 过滤成功")


async def main():
    """运行所有测试"""
    try:
        await test_skill_basic()
        await test_skill_manual_definition()
        await test_skill_from_markdown()
        await test_disable_model_invocation()

        print("\n" + "=" * 50)
        print("✅ 所有测试通过！")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
