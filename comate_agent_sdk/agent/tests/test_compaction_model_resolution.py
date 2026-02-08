from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from comate_agent_sdk.agent.compaction.models import CompactionConfig
from comate_agent_sdk.agent.init import _resolve_compaction_llm


@dataclass
class _DataclassLLM:
    model: str
    provider: str = "anthropic"
    api_key: str | None = None
    base_url: str | None = None
    _client: object | None = None

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(content="ok", usage=None)


class _NonDataclassLLM:
    def __init__(self, model: str) -> None:
        self.model = model
        self.provider = "anthropic"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(content="ok", usage=None)


class TestCompactionModelResolution(unittest.TestCase):
    def test_default_uses_mid_level_model(self) -> None:
        runtime = SimpleNamespace(
            llm=_DataclassLLM(model="main-model"),
            llm_levels={
                "LOW": _DataclassLLM(model="low-model"),
                "MID": _DataclassLLM(model="mid-model"),
                "HIGH": _DataclassLLM(model="high-model"),
            },
        )

        resolved = _resolve_compaction_llm(runtime, CompactionConfig())
        self.assertIs(resolved, runtime.llm_levels["MID"])

    def test_model_config_overrides_mid_level(self) -> None:
        runtime = SimpleNamespace(
            llm=_DataclassLLM(model="main-model"),
            llm_levels={
                "MID": _DataclassLLM(
                    model="mid-model",
                    provider="openai",
                    api_key="key-1",
                    base_url="https://api.example.com",
                    _client=object(),
                )
            },
        )

        resolved = _resolve_compaction_llm(
            runtime,
            CompactionConfig(model="claude-haiku-4-5"),
        )

        self.assertIsNot(resolved, runtime.llm_levels["MID"])
        self.assertEqual(resolved.model, "claude-haiku-4-5")
        self.assertEqual(resolved.provider, "openai")
        self.assertEqual(resolved.api_key, "key-1")
        self.assertEqual(resolved.base_url, "https://api.example.com")
        self.assertIsNone(resolved._client)

    def test_model_override_fallbacks_to_main_llm_on_clone_error(self) -> None:
        runtime = SimpleNamespace(
            llm=_DataclassLLM(model="main-model"),
            llm_levels={"MID": _NonDataclassLLM(model="mid-model")},
        )

        resolved = _resolve_compaction_llm(
            runtime,
            CompactionConfig(model="claude-haiku-4-5"),
        )
        self.assertIs(resolved, runtime.llm)


if __name__ == "__main__":
    unittest.main()
