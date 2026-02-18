import unittest
from types import SimpleNamespace

from comate_agent_sdk.llm.deepseek.chat import ChatDeepSeek


class TestDeepSeekUsageMapping(unittest.TestCase):
    def _make_llm(self, model: str = "deepseek-chat") -> ChatDeepSeek:
        return ChatDeepSeek(
            model=model,
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )

    def test_get_usage_public_delegates_to_private(self) -> None:
        llm = self._make_llm()
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
                prompt_cache_hit_tokens=None,
                prompt_cache_miss_tokens=None,
            )
        )

        usage = llm.get_usage(response)

        assert usage is not None
        self.assertEqual(usage.prompt_tokens, 20)
        self.assertEqual(usage.completion_tokens, 10)
        self.assertEqual(usage.total_tokens, 30)

    def test_get_context_window_known_model(self) -> None:
        llm = self._make_llm("deepseek-chat")
        result = llm.get_context_window()
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)  # type: ignore[operator]

    def test_get_context_window_unknown_model_returns_none(self) -> None:
        llm = self._make_llm("deepseek-unknown-xyz")
        result = llm.get_context_window()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
