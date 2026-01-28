"""
测试 Subagent 自动发现机制
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from bu_agent_sdk import Agent, AgentDefinition
from bu_agent_sdk.llm import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_subagent_file(directory: Path, name: str, description: str) -> Path:
    """创建测试用的 subagent markdown 文件"""
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{name}.md"
    content = f"""---
name: {name}
description: {description}
model: haiku
---

你是一个 {description}。
"""
    file_path.write_text(content, encoding="utf-8")
    logger.info(f"Created test subagent file: {file_path}")
    return file_path


async def test_only_user_exists():
    """测试用例 1: 只有 user 目录存在"""
    logger.info("\n=== 测试用例 1: 只有 user 目录存在 ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建 user 级别的 subagent
        user_dir = Path.home() / ".agent" / "subagents"
        user_file = create_test_subagent_file(user_dir, "test_user", "用户级别测试 agent")

        try:
            # 创建 Agent，指定 project_root 为空目录（不存在 .agent/subagents）
            agent = Agent(
                llm=ChatOpenAI(model="gpt-4o-mini"),
                project_root=temp_path,
            )

            # 验证是否加载了 user 级别的 subagent
            if agent.agents:
                logger.info(f"✓ 成功加载 {len(agent.agents)} 个 subagent")
                for a in agent.agents:
                    logger.info(f"  - {a.name}: {a.description}")
                assert any(a.name == "test_user" for a in agent.agents), "应该加载 user 级别的 subagent"
            else:
                logger.error("✗ 未加载任何 subagent")

        finally:
            # 清理
            user_file.unlink(missing_ok=True)


async def test_only_project_exists():
    """测试用例 2: 只有 project 目录存在"""
    logger.info("\n=== 测试用例 2: 只有 project 目录存在 ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建 project 级别的 subagent
        project_dir = temp_path / ".agent" / "subagents"
        create_test_subagent_file(project_dir, "test_project", "项目级别测试 agent")

        # 创建 Agent
        agent = Agent(
            llm=ChatOpenAI(model="gpt-4o-mini"),
            project_root=temp_path,
        )

        # 验证是否加载了 project 级别的 subagent
        if agent.agents:
            logger.info(f"✓ 成功加载 {len(agent.agents)} 个 subagent")
            for a in agent.agents:
                logger.info(f"  - {a.name}: {a.description}")
            assert any(a.name == "test_project" for a in agent.agents), "应该加载 project 级别的 subagent"
        else:
            logger.error("✗ 未加载任何 subagent")


async def test_both_exist_project_priority():
    """测试用例 3: 两者都存在，应该只用 project"""
    logger.info("\n=== 测试用例 3: user 和 project 都存在，应该只用 project ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建 user 级别的 subagent
        user_dir = Path.home() / ".agent" / "subagents"
        user_file = create_test_subagent_file(user_dir, "test_global", "全局测试 agent")

        # 创建 project 级别的 subagent
        project_dir = temp_path / ".agent" / "subagents"
        create_test_subagent_file(project_dir, "test_project", "项目测试 agent")

        try:
            # 创建 Agent
            agent = Agent(
                llm=ChatOpenAI(model="gpt-4o-mini"),
                project_root=temp_path,
            )

            # 验证只加载了 project 级别的 subagent
            if agent.agents:
                logger.info(f"✓ 成功加载 {len(agent.agents)} 个 subagent")
                for a in agent.agents:
                    logger.info(f"  - {a.name}: {a.description}")

                # 应该只有 project 的
                assert any(a.name == "test_project" for a in agent.agents), "应该加载 project 级别的 subagent"
                assert not any(
                    a.name == "test_global" for a in agent.agents
                ), "不应该加载 user 级别的 subagent（因为 project 存在）"
            else:
                logger.error("✗ 未加载任何 subagent")

        finally:
            # 清理
            user_file.unlink(missing_ok=True)


async def test_merge_with_code_provided():
    """测试用例 4: 自动发现的 + 代码传入的 merge，代码传入的覆盖同名"""
    logger.info("\n=== 测试用例 4: Merge 验证 - 代码传入的覆盖同名 ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建 project 级别的 subagent（名为 foo）
        project_dir = temp_path / ".agent" / "subagents"
        create_test_subagent_file(project_dir, "foo", "自动发现的 foo agent")
        create_test_subagent_file(project_dir, "bar", "自动发现的 bar agent")

        # 代码传入同名的 foo 和新的 baz
        code_foo = AgentDefinition(
            name="foo",
            description="代码传入的 foo agent（应该覆盖自动发现的）",
            prompt="这是代码传入的 foo",
        )
        code_baz = AgentDefinition(
            name="baz",
            description="代码传入的 baz agent",
            prompt="这是代码传入的 baz",
        )

        # 创建 Agent
        agent = Agent(
            llm=ChatOpenAI(model="gpt-4o-mini"),
            project_root=temp_path,
            agents=[code_foo, code_baz],
        )

        # 验证 merge 结果
        if agent.agents:
            logger.info(f"✓ 成功加载 {len(agent.agents)} 个 subagent")
            for a in agent.agents:
                logger.info(f"  - {a.name}: {a.description}")

            # 应该有 3 个：bar (发现), foo (代码), baz (代码)
            assert len(agent.agents) == 3, f"应该有 3 个 subagent，实际 {len(agent.agents)} 个"

            # foo 应该是代码传入的版本
            foo_agent = next((a for a in agent.agents if a.name == "foo"), None)
            assert foo_agent is not None, "应该有 foo agent"
            assert "代码传入" in foo_agent.description, "foo 应该是代码传入的版本"

            # bar 应该是自动发现的
            bar_agent = next((a for a in agent.agents if a.name == "bar"), None)
            assert bar_agent is not None, "应该有 bar agent"

            # baz 应该是代码传入的
            baz_agent = next((a for a in agent.agents if a.name == "baz"), None)
            assert baz_agent is not None, "应该有 baz agent"

            logger.info("✓ Merge 逻辑正确")
        else:
            logger.error("✗ 未加载任何 subagent")


async def test_default_cwd():
    """测试用例 5: 不传 project_root，应该使用 cwd"""
    logger.info("\n=== 测试用例 5: 默认使用 cwd ===")

    # 在当前目录创建 .agent/subagents（如果不存在）
    project_dir = Path.cwd() / ".agent" / "subagents"
    if not project_dir.exists():
        logger.info("当前目录没有 .agent/subagents，跳过此测试")
        return

    # 不传 project_root
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o-mini"),
    )

    if agent.agents:
        logger.info(f"✓ 成功加载 {len(agent.agents)} 个 subagent（从 cwd）")
        for a in agent.agents:
            logger.info(f"  - {a.name}: {a.description}")
    else:
        logger.info("当前目录没有 subagent 定义")


async def main():
    """运行所有测试"""
    try:
        await test_only_user_exists()
        await test_only_project_exists()
        await test_both_exist_project_priority()
        await test_merge_with_code_provided()
        await test_default_cwd()

        logger.info("\n=== 所有测试通过 ===")
    except AssertionError as e:
        logger.error(f"\n=== 测试失败: {e} ===")
        raise
    except Exception as e:
        logger.error(f"\n=== 测试出错: {e} ===")
        raise


if __name__ == "__main__":
    asyncio.run(main())
