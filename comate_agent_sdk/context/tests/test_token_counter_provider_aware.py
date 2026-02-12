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


class _FakeAnthropicLLM:
    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5",
        input_tokens: int = 123,
        side_effect: Exception | None = None,
    ) -> None:
        self.provider = "anthropic"
        self.model = model
        if side_effect is not None:
            self.count_tokens_mock = AsyncMock(side_effect=side_effect)
        else:
            self.count_tokens_mock = AsyncMock(
                return_value=SimpleNamespace(input_tokens=input_tokens)
            )
        self._client = SimpleNamespace(
            messages=SimpleNamespace(count_tokens=self.count_tokens_mock)
        )

    def get_client(self):
        return self._client


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

    def test_count_messages_for_model_anthropic_enforces_min_timeout_floor(self) -> None:
        counter = TokenCounter()
        messages = [UserMessage(content="hello")]
        llm = _FakeAnthropicLLM(input_tokens=55)
        wait_for_timeouts: list[float] = []

        async def _wait_for_passthrough(awaitable, timeout):
            wait_for_timeouts.append(float(timeout))
            return await awaitable

        with patch(
            "comate_agent_sdk.context.budget.asyncio.wait_for",
            new=_wait_for_passthrough,
        ):
            result = asyncio.run(
                counter.count_messages_for_model(
                    messages,
                    llm=llm,
                    timeout_ms=300,
                )
            )

        self.assertEqual(result, 55)
        kwargs = llm.count_tokens_mock.call_args.kwargs
        self.assertAlmostEqual(kwargs["timeout"], 1.0, places=6)
        self.assertEqual(len(wait_for_timeouts), 1)
        self.assertAlmostEqual(wait_for_timeouts[0], 1.05, places=6)

    def test_count_messages_for_model_anthropic_keeps_larger_timeout(self) -> None:
        counter = TokenCounter()
        messages = [UserMessage(content="hello")]
        llm = _FakeAnthropicLLM(input_tokens=66)
        wait_for_timeouts: list[float] = []

        async def _wait_for_passthrough(awaitable, timeout):
            wait_for_timeouts.append(float(timeout))
            return await awaitable

        with patch(
            "comate_agent_sdk.context.budget.asyncio.wait_for",
            new=_wait_for_passthrough,
        ):
            result = asyncio.run(
                counter.count_messages_for_model(
                    messages,
                    llm=llm,
                    timeout_ms=1500,
                )
            )

        self.assertEqual(result, 66)
        kwargs = llm.count_tokens_mock.call_args.kwargs
        self.assertAlmostEqual(kwargs["timeout"], 1.5, places=6)
        self.assertEqual(len(wait_for_timeouts), 1)
        self.assertAlmostEqual(wait_for_timeouts[0], 1.55, places=6)

    def test_count_messages_for_model_anthropic_warning_contains_diagnostics(self) -> None:
        counter = TokenCounter()
        messages = [UserMessage(content="fallback me")]
        llm = _FakeAnthropicLLM(
            model="claude-sonnet-4-5",
            side_effect=RuntimeError("network timeout"),
        )

        with patch.object(
            counter,
            "_fallback_count_messages",
            return_value=77,
        ) as mocked_fallback, patch(
            "comate_agent_sdk.context.budget.logger.warning"
        ) as mocked_warning:
            result = asyncio.run(
                counter.count_messages_for_model(
                    messages,
                    llm=llm,
                    timeout_ms=300,
                )
            )

        self.assertEqual(result, 77)
        mocked_fallback.assert_called_once()
        mocked_warning.assert_called_once()
        warning_text = mocked_warning.call_args.args[0]
        self.assertIn("provider=anthropic", warning_text)
        self.assertIn("model=claude-sonnet-4-5", warning_text)
        self.assertIn("timeout_ms=1000", warning_text)
        self.assertIn("exc_type=RuntimeError", warning_text)


if __name__ == "__main__":
    unittest.main()
