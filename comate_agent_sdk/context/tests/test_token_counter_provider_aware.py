import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from comate_agent_sdk.context.budget import TokenCounter
from comate_agent_sdk.llm.messages import UserMessage


class _FakeEncoder:
    def __init__(self, count: int):
        self._count = count

    def encode(self, text: str):
        return list(range(self._count))


class TestTokenCounterProviderAware(unittest.TestCase):
    def test_count_for_model_openai_uses_model_encoder(self) -> None:
        counter = TokenCounter()
        fake_encoder = _FakeEncoder(7)

        with patch.object(counter, "_get_openai_model_encoder", return_value=fake_encoder) as mocked:
            result = counter.count_for_model(
                "hello world",
                provider="openai",
                model="gpt-4o",
            )

        self.assertEqual(result, 7)
        mocked.assert_called_once_with("gpt-4o")

    def test_count_messages_for_model_openai_calls_count_for_model(self) -> None:
        counter = TokenCounter()
        messages = [UserMessage(content="hello")]
        llm = SimpleNamespace(provider="openai", model="gpt-4o")

        with patch.object(counter, "count_for_model", return_value=42) as mocked_count:
            result = asyncio.run(
                counter.count_messages_for_model(
                    messages,
                    llm=llm,
                    timeout_ms=300,
                )
            )

        self.assertEqual(result, 42)
        self.assertEqual(mocked_count.call_count, 1)
        kwargs = mocked_count.call_args.kwargs
        self.assertEqual(kwargs["provider"], "openai")
        self.assertEqual(kwargs["model"], "gpt-4o")

    def test_count_messages_for_model_anthropic_falls_back_on_failure(self) -> None:
        counter = TokenCounter()
        messages = [UserMessage(content="fallback me")]
        llm = SimpleNamespace(provider="anthropic", model="claude-sonnet-4-5")

        with patch.object(
            counter,
            "_count_anthropic_messages",
            new=AsyncMock(return_value=None),
        ) as mocked_provider_count, patch.object(
            counter,
            "_fallback_count_messages",
            return_value=77,
        ) as mocked_fallback:
            result = asyncio.run(
                counter.count_messages_for_model(
                    messages,
                    llm=llm,
                    timeout_ms=300,
                )
            )

        self.assertEqual(result, 77)
        self.assertEqual(mocked_provider_count.await_count, 1)
        mocked_fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
