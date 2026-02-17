import asyncio
import unittest
from types import SimpleNamespace

from comate_agent_sdk.agent.compaction.service import CompactionService


class _FakeLLM:
    def __init__(self, *, model: str, context_window: int | None) -> None:
        self.model = model
        self._context_window = context_window

    def get_context_window(self) -> int | None:
        return self._context_window


class _FakeTokenCost:
    def __init__(self, *, max_input_tokens: int | None, max_tokens: int | None = None) -> None:
        self._max_input_tokens = max_input_tokens
        self._max_tokens = max_tokens

    async def get_model_pricing(self, model: str):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            max_input_tokens=self._max_input_tokens,
            max_tokens=self._max_tokens,
        )


class TestCompactionContextLimitResolution(unittest.TestCase):
    def test_provider_context_window_takes_priority(self) -> None:
        service = CompactionService(token_cost=None, llm=None)
        llm = _FakeLLM(model="MiniMax-M2.5", context_window=111111)

        limit = asyncio.run(service.get_model_context_limit("MiniMax-M2.5", llm=llm))

        self.assertEqual(limit, 111111)

    def test_custom_pricing_is_used_when_provider_not_available(self) -> None:
        service = CompactionService(token_cost=None, llm=None)

        limit = asyncio.run(service.get_model_context_limit("MiniMax-M2.5"))

        self.assertEqual(limit, 204800)

    def test_token_pricing_fallback_when_custom_pricing_missing(self) -> None:
        service = CompactionService(
            token_cost=_FakeTokenCost(max_input_tokens=222000),
            llm=None,
        )

        limit = asyncio.run(service.get_model_context_limit("unknown-model"))

        self.assertEqual(limit, 222000)


if __name__ == "__main__":
    unittest.main()
