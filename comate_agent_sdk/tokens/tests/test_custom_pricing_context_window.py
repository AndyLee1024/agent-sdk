import unittest

from comate_agent_sdk.tokens.custom_pricing import (
    resolve_max_input_tokens_from_custom_pricing,
)


class TestCustomPricingContextWindow(unittest.TestCase):
    def test_resolve_exact_model(self) -> None:
        resolved = resolve_max_input_tokens_from_custom_pricing("MiniMax-M2.5")
        self.assertEqual(resolved, 204800)

    def test_resolve_provider_prefixed_model(self) -> None:
        resolved = resolve_max_input_tokens_from_custom_pricing("minimax/MiniMax-M2.5")
        self.assertEqual(resolved, 204800)

    def test_resolve_case_insensitive_key(self) -> None:
        resolved = resolve_max_input_tokens_from_custom_pricing("minimaxai/minimax-m2.5")
        self.assertEqual(resolved, 204800)

    def test_resolve_unknown_model(self) -> None:
        resolved = resolve_max_input_tokens_from_custom_pricing("unknown-model")
        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()
