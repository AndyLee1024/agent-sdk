from __future__ import annotations


def test_service_shim_exports():
    from comate_agent_sdk.agent.service import Agent, SystemPromptConfig, SystemPromptType

    assert Agent is not None
    assert SystemPromptConfig is not None
    assert SystemPromptType is not None

