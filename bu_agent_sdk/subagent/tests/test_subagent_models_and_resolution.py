import unittest

from bu_agent_sdk.subagent.models import AgentDefinition
from bu_agent_sdk.subagent.task_tool import resolve_model


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


if __name__ == "__main__":
    unittest.main()

