import unittest

from comate_agent_sdk.agent.compaction.models import TokenUsage
from comate_agent_sdk.llm.views import ChatInvokeUsage


class TestCompactionUsageNoCacheDoubleCount(unittest.TestCase):
    def test_total_tokens_prefers_reported_total(self) -> None:
        usage = ChatInvokeUsage(
            prompt_tokens=120,
            prompt_cached_tokens=40,
            prompt_cache_creation_tokens=10,
            prompt_image_tokens=None,
            completion_tokens=20,
            total_tokens=140,
        )

        token_usage = TokenUsage.from_usage(usage)
        self.assertEqual(token_usage.total_tokens, 140)

    def test_fallback_total_tokens_does_not_readd_cache_read(self) -> None:
        token_usage = TokenUsage(
            input_tokens=120,
            output_tokens=20,
            cache_creation_tokens=10,
            cache_read_tokens=40,
            reported_total_tokens=0,
        )

        self.assertEqual(token_usage.total_tokens, 150)


if __name__ == "__main__":
    unittest.main()
