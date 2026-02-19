import asyncio
import os
import unittest

from comate_agent_sdk.llm.views import ChatInvokeUsage
from comate_agent_sdk.tokens import TokenCost


class TestTokenCost(unittest.TestCase):
    def test_include_cost_param_used_when_env_unset(self) -> None:
        old = os.environ.pop("comate_agent_sdk_CALCULATE_COST", None)
        try:
            self.assertFalse(TokenCost(include_cost=False).include_cost)
            self.assertTrue(TokenCost(include_cost=True).include_cost)
        finally:
            if old is None:
                os.environ.pop("comate_agent_sdk_CALCULATE_COST", None)
            else:
                os.environ["comate_agent_sdk_CALCULATE_COST"] = old

    def test_include_cost_env_overrides_param(self) -> None:
        old = os.environ.get("comate_agent_sdk_CALCULATE_COST")
        try:
            os.environ["comate_agent_sdk_CALCULATE_COST"] = "false"
            self.assertFalse(TokenCost(include_cost=True).include_cost)

            os.environ["comate_agent_sdk_CALCULATE_COST"] = "true"
            self.assertTrue(TokenCost(include_cost=False).include_cost)
        finally:
            if old is None:
                os.environ.pop("comate_agent_sdk_CALCULATE_COST", None)
            else:
                os.environ["comate_agent_sdk_CALCULATE_COST"] = old

    def test_usage_summary_by_level(self) -> None:
        token_cost = TokenCost(include_cost=False)
        token_cost.add_usage(
            "m1",
            ChatInvokeUsage(
                prompt_tokens=10,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=5,
                reasoning_tokens=2,
                total_tokens=15,
            ),
            level="LOW",
            source="agent",
        )
        token_cost.add_usage(
            "m2",
            ChatInvokeUsage(
                prompt_tokens=7,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=3,
                reasoning_tokens=1,
                total_tokens=10,
            ),
            level="MID",
            source="webfetch",
        )
        token_cost.add_usage(
            "m3",
            ChatInvokeUsage(
                prompt_tokens=2,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=1,
                reasoning_tokens=0,
                total_tokens=3,
            ),
            level="MID",
            source="subagent:researcher",
        )

        summary = asyncio.run(token_cost.get_usage_summary())
        self.assertIn("LOW", summary.by_level)
        self.assertIn("MID", summary.by_level)
        self.assertEqual(summary.by_level["LOW"].invocations, 1)
        self.assertEqual(summary.by_level["MID"].invocations, 2)
        self.assertEqual(summary.by_level["MID"].total_tokens, 13)
        self.assertEqual(summary.total_reasoning_tokens, 3)

    def test_total_cost_does_not_double_count_cached_prompt(self) -> None:
        old = os.environ.pop("comate_agent_sdk_CALCULATE_COST", None)
        try:
            token_cost = TokenCost(include_cost=True)
        finally:
            if old is None:
                os.environ.pop("comate_agent_sdk_CALCULATE_COST", None)
            else:
                os.environ["comate_agent_sdk_CALCULATE_COST"] = old

        token_cost._initialized = True
        token_cost._pricing_data = {
            "dummy": {
                "input_cost_per_token": 1.0,
                "output_cost_per_token": 2.0,
                "cache_read_input_token_cost": 0.5,
                "cache_creation_input_token_cost": 0.25,
            }
        }

        token_cost.add_usage(
            "dummy",
            ChatInvokeUsage(
                prompt_tokens=100,
                prompt_cached_tokens=40,
                prompt_cache_creation_tokens=10,
                prompt_image_tokens=None,
                completion_tokens=50,
                total_tokens=150,
            ),
            level="MID",
            source="agent",
        )

        summary = asyncio.run(token_cost.get_usage_summary())
        self.assertAlmostEqual(summary.total_prompt_cost, 82.5, places=6)
        self.assertAlmostEqual(summary.total_prompt_cached_cost, 20.0, places=6)
        self.assertAlmostEqual(summary.total_completion_cost, 100.0, places=6)
        self.assertAlmostEqual(summary.total_cost, 182.5, places=6)
        self.assertAlmostEqual(summary.by_model["dummy"].cost, 182.5, places=6)
        self.assertAlmostEqual(summary.by_level["MID"].cost, 182.5, places=6)

    def test_usage_summary_source_prefix(self) -> None:
        token_cost = TokenCost(include_cost=False)
        token_cost.add_usage(
            "m-main",
            ChatInvokeUsage(
                prompt_tokens=20,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=10,
                total_tokens=30,
            ),
            level="MID",
            source="agent",
        )
        token_cost.add_usage(
            "m-sub",
            ChatInvokeUsage(
                prompt_tokens=12,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=8,
                total_tokens=20,
            ),
            level="MID",
            source="subagent:Explorer",
        )
        token_cost.add_usage(
            "m-sub-low",
            ChatInvokeUsage(
                prompt_tokens=7,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=5,
                total_tokens=12,
            ),
            level="LOW",
            source="subagent:Explorer:webfetch",
        )

        summary = asyncio.run(
            token_cost.get_usage_summary(source_prefix="subagent:Explorer")
        )
        self.assertEqual(summary.entry_count, 2)
        self.assertEqual(summary.total_prompt_tokens, 19)
        self.assertEqual(summary.total_completion_tokens, 13)
        self.assertEqual(summary.total_tokens, 32)

    def test_usage_summary_total_tokens_uses_reported_usage_total(self) -> None:
        token_cost = TokenCost(include_cost=False)
        token_cost.add_usage(
            "m-reported",
            ChatInvokeUsage(
                prompt_tokens=10,
                prompt_cached_tokens=2,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=4,
                total_tokens=99,
            ),
            level="MID",
            source="agent",
        )

        summary = asyncio.run(token_cost.get_usage_summary())
        self.assertEqual(summary.total_prompt_tokens, 10)
        self.assertEqual(summary.total_completion_tokens, 4)
        self.assertEqual(summary.total_tokens, 99)
        self.assertEqual(summary.by_model["m-reported"].total_tokens, 99)
        self.assertEqual(summary.by_level["MID"].total_tokens, 99)

    def test_usage_summary_by_source(self) -> None:
        token_cost = TokenCost(include_cost=False)
        token_cost.add_usage(
            "m1",
            ChatInvokeUsage(
                prompt_tokens=8,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=2,
                total_tokens=10,
            ),
            source="agent",
        )
        token_cost.add_usage(
            "m2",
            ChatInvokeUsage(
                prompt_tokens=6,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=3,
                total_tokens=9,
            ),
            source="subagent:Explorer",
        )
        token_cost.add_usage(
            "m3",
            ChatInvokeUsage(
                prompt_tokens=5,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=1,
                total_tokens=6,
            ),
            source="subagent:Explorer",
        )

        summary = asyncio.run(token_cost.get_usage_summary())
        self.assertEqual(summary.by_source["agent"].total_tokens, 10)
        self.assertEqual(summary.by_source["agent"].invocations, 1)
        self.assertEqual(summary.by_source["subagent:Explorer"].total_tokens, 15)
        self.assertEqual(summary.by_source["subagent:Explorer"].invocations, 2)

    def test_usage_observer_receives_new_entry(self) -> None:
        token_cost = TokenCost(include_cost=False)
        observed_sources: list[str] = []

        observer_id = token_cost.subscribe_usage(
            lambda entry: observed_sources.append(entry.source or "")
        )
        token_cost.add_usage(
            "m-observe",
            ChatInvokeUsage(
                prompt_tokens=5,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=2,
                total_tokens=7,
            ),
            level="MID",
            source="subagent:Explorer:tc_1",
        )
        token_cost.unsubscribe_usage(observer_id)

        self.assertEqual(observed_sources, ["subagent:Explorer:tc_1"])

    def test_usage_observer_can_unsubscribe(self) -> None:
        token_cost = TokenCost(include_cost=False)
        callback_count = 0

        def _observer(_entry) -> None:
            nonlocal callback_count
            callback_count += 1

        observer_id = token_cost.subscribe_usage(_observer)
        token_cost.unsubscribe_usage(observer_id)
        token_cost.add_usage(
            "m-observe",
            ChatInvokeUsage(
                prompt_tokens=3,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=1,
                total_tokens=4,
            ),
            level="LOW",
            source="agent",
        )
        self.assertEqual(callback_count, 0)


if __name__ == "__main__":
    unittest.main()
