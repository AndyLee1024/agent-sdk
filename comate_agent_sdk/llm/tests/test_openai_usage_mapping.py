import unittest
from types import SimpleNamespace

from comate_agent_sdk.llm.openai.chat import ChatOpenAI


class TestOpenAIUsageMapping(unittest.TestCase):
    def test_get_usage_does_not_double_count_reasoning_tokens(self) -> None:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=12,
                prompt_tokens_details=SimpleNamespace(cached_tokens=3),
                completion_tokens=10,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
                total_tokens=22,
            )
        )

        usage = llm._get_usage(response)  # type: ignore[arg-type]

        assert usage is not None
        self.assertEqual(usage.prompt_tokens, 12)
        self.assertEqual(usage.prompt_cached_tokens, 3)
        self.assertEqual(usage.completion_tokens, 10)
        self.assertEqual(usage.total_tokens, 22)


if __name__ == "__main__":
    unittest.main()
