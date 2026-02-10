import unittest

from comate_agent_sdk.agent.tool_strategy import generate_tool_strategy
from comate_agent_sdk.tools.decorator import tool


class TestToolStrategyDefinitionMapping(unittest.TestCase):
    def test_usage_rules_goes_to_tool_definition_description(self) -> None:
        @tool(
            "Short overview for system prompt",
            name="DemoTool",
            usage_rules="Detailed usage policy for model tool definition.",
        )
        async def demo_tool(query: str) -> str:
            return query

        self.assertEqual(
            demo_tool.definition.description,
            "Detailed usage policy for model tool definition.",
        )

        prompt = generate_tool_strategy([demo_tool])
        self.assertIn("- **DemoTool**: Short overview for system prompt", prompt)
        self.assertNotIn("Detailed usage policy for model tool definition.", prompt)

    def test_tool_definition_description_falls_back_to_description(self) -> None:
        @tool("Fallback short description", name="NoRulesTool")
        async def no_rules_tool(query: str) -> str:
            return query

        self.assertEqual(
            no_rules_tool.definition.description,
            "Fallback short description",
        )

    def test_subagent_hides_task_and_ask_user_question_in_tool_strategy(self) -> None:
        @tool("Task overview", name="Task", usage_rules="Task policy")
        async def task_tool(query: str) -> str:
            return query

        @tool("Ask overview", name="AskUserQuestion", usage_rules="Ask policy")
        async def ask_tool(query: str) -> str:
            return query

        @tool("Read overview", name="Read", usage_rules="Read policy")
        async def read_tool(query: str) -> str:
            return query

        prompt = generate_tool_strategy(
            [task_tool, ask_tool, read_tool],
            is_subagent=True,
        )
        self.assertIn("- **Read**: Read overview", prompt)
        self.assertNotIn("- **Task**:", prompt)
        self.assertNotIn("- **AskUserQuestion**:", prompt)


if __name__ == "__main__":
    unittest.main()
