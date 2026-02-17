import unittest
from types import SimpleNamespace

from comate_agent_sdk.llm.anthropic.chat import ChatAnthropic


class TestAnthropicUsageMapping(unittest.TestCase):
    def test_get_usage_includes_cache_read_and_creation_in_prompt_and_total(self) -> None:
        llm = ChatAnthropic(model="claude-sonnet-4-5", api_key="test-key")
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=120,
                cache_read_input_tokens=40,
                cache_creation_input_tokens=10,
                output_tokens=20,
            )
        )

        usage = llm.get_usage(response)  # type: ignore[arg-type]

        assert usage is not None
        self.assertEqual(usage.prompt_tokens, 170)
        self.assertEqual(usage.prompt_cached_tokens, 40)
        self.assertEqual(usage.prompt_cache_creation_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 190)


if __name__ == "__main__":
    unittest.main()
