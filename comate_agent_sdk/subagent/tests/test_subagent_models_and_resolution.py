import unittest

from comate_agent_sdk.subagent.models import AgentDefinition
from comate_agent_sdk.subagent.task_tool import resolve_model
from comate_agent_sdk.llm.anthropic.chat import ChatAnthropic


class TestSubagentModelsAndResolution(unittest.TestCase):
    def test_agent_definition_parses_level_and_inherit_model(self) -> None:
        md = """---
name: researcher
description: Research things
level: low
model: inherit
tools: []
---

You are a researcher.
"""
        agent_def = AgentDefinition.from_markdown(md)
        self.assertEqual(agent_def.level, "LOW")
        self.assertIsNone(agent_def.model)

    def test_resolve_model_precedence_and_pool_mapping(self) -> None:
        parent = object()
        low = object()
        mid = parent
        high = object()
        llm_levels = {"LOW": low, "MID": mid, "HIGH": high}

        self.assertIs(
            resolve_model(model=None, level=None, parent_llm=parent, llm_levels=llm_levels),
            parent,
        )
        self.assertIs(
            resolve_model(model=None, level="LOW", parent_llm=parent, llm_levels=llm_levels),
            low,
        )
        self.assertIs(
            resolve_model(model="opus", level=None, parent_llm=parent, llm_levels=llm_levels),
            high,
        )
        self.assertIs(
            resolve_model(model="haiku", level="HIGH", parent_llm=parent, llm_levels=llm_levels),
            low,
        )

    def test_resolve_model_rejects_invalid_model(self) -> None:
        """测试：不支持的model值被忽略，回退到继承parent"""
        parent = ChatAnthropic(model="claude-sonnet-4-5")

        # 无效的model应该被忽略，返回parent
        result = resolve_model(
            model="gpt-4o",  # 无效值
            level=None,
            parent_llm=parent,
            llm_levels=None,
        )

        assert result is parent, "无效的model应该被忽略，返回parent_llm"

    def test_resolve_model_invalid_model_with_level_fallback(self) -> None:
        """测试：无效model被忽略后，回退到level"""
        parent = ChatAnthropic(model="claude-sonnet-4-5")
        low_llm = ChatAnthropic(model="claude-haiku-4-5")
        levels = {"LOW": low_llm, "MID": parent, "HIGH": parent}

        # 无效model + level: 应该回退到level
        result = resolve_model(
            model="gpt-4o",  # 无效值
            level="LOW",
            parent_llm=parent,
            llm_levels=levels,
        )

        assert result is low_llm, "无效model应该被忽略，使用level"

    def test_agent_definition_rejects_invalid_model(self) -> None:
        """测试：AgentDefinition解析时拒绝无效model"""
        content = """---
name: test-agent
description: Test
model: gpt-4o
---
Test prompt
"""
        agent_def = AgentDefinition.from_markdown(content)

        # 无效的model应该被置为None
        assert agent_def.model is None, "无效的model应该被置为None"


if __name__ == "__main__":
    unittest.main()

