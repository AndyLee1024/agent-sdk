import os
import unittest

from bu_agent_sdk.agent.llm_levels import resolve_llm_levels


class TestLLMLevels(unittest.TestCase):
    def test_env_requires_provider_prefix(self) -> None:
        old = os.environ.get("BU_AGENT_SDK_LLM_LOW")
        try:
            os.environ["BU_AGENT_SDK_LLM_LOW"] = "gpt-5-mini"
            with self.assertRaises(ValueError):
                resolve_llm_levels(explicit=None)
        finally:
            if old is None:
                os.environ.pop("BU_AGENT_SDK_LLM_LOW", None)
            else:
                os.environ["BU_AGENT_SDK_LLM_LOW"] = old

    def test_env_provider_openai_creates_openai_model(self) -> None:
        old = os.environ.get("BU_AGENT_SDK_LLM_LOW")
        old_base = os.environ.get("BU_AGENT_SDK_LLM_LOW_BASE_URL")
        try:
            os.environ["BU_AGENT_SDK_LLM_LOW"] = "openai:gpt-5-mini"
            os.environ["BU_AGENT_SDK_LLM_LOW_BASE_URL"] = "https://gw-low.example.com/v1"
            levels = resolve_llm_levels(explicit=None)
            self.assertIn("LOW", levels)
            self.assertEqual(levels["LOW"].provider, "openai")
            # base_url should be injected when env is used
            self.assertEqual(getattr(levels["LOW"], "base_url", None), "https://gw-low.example.com/v1")
        finally:
            if old is None:
                os.environ.pop("BU_AGENT_SDK_LLM_LOW", None)
            else:
                os.environ["BU_AGENT_SDK_LLM_LOW"] = old
            if old_base is None:
                os.environ.pop("BU_AGENT_SDK_LLM_LOW_BASE_URL", None)
            else:
                os.environ["BU_AGENT_SDK_LLM_LOW_BASE_URL"] = old_base

    def test_env_provider_anthropic_injects_base_url(self) -> None:
        old = os.environ.get("BU_AGENT_SDK_LLM_MID")
        old_base = os.environ.get("BU_AGENT_SDK_LLM_MID_BASE_URL")
        try:
            os.environ["BU_AGENT_SDK_LLM_MID"] = "anthropic:claude-sonnet-4-5"
            os.environ["BU_AGENT_SDK_LLM_MID_BASE_URL"] = "https://gw-mid.example.com"
            levels = resolve_llm_levels(explicit=None)
            self.assertIn("MID", levels)
            self.assertEqual(levels["MID"].provider, "anthropic")
            self.assertEqual(getattr(levels["MID"], "base_url", None), "https://gw-mid.example.com")
        finally:
            if old is None:
                os.environ.pop("BU_AGENT_SDK_LLM_MID", None)
            else:
                os.environ["BU_AGENT_SDK_LLM_MID"] = old
            if old_base is None:
                os.environ.pop("BU_AGENT_SDK_LLM_MID_BASE_URL", None)
            else:
                os.environ["BU_AGENT_SDK_LLM_MID_BASE_URL"] = old_base

    def test_env_provider_google_injects_http_base_url(self) -> None:
        old = os.environ.get("BU_AGENT_SDK_LLM_HIGH")
        old_base = os.environ.get("BU_AGENT_SDK_LLM_HIGH_BASE_URL")
        try:
            os.environ["BU_AGENT_SDK_LLM_HIGH"] = "google:gemini-2.5-pro"
            os.environ["BU_AGENT_SDK_LLM_HIGH_BASE_URL"] = "https://gw-high.example.com"
            levels = resolve_llm_levels(explicit=None)
            self.assertIn("HIGH", levels)
            self.assertEqual(levels["HIGH"].provider, "google")
            self.assertEqual(
                getattr(levels["HIGH"], "http_options", None),
                {"base_url": "https://gw-high.example.com"},
            )
        finally:
            if old is None:
                os.environ.pop("BU_AGENT_SDK_LLM_HIGH", None)
            else:
                os.environ["BU_AGENT_SDK_LLM_HIGH"] = old
            if old_base is None:
                os.environ.pop("BU_AGENT_SDK_LLM_HIGH_BASE_URL", None)
            else:
                os.environ["BU_AGENT_SDK_LLM_HIGH_BASE_URL"] = old_base


if __name__ == "__main__":
    unittest.main()
