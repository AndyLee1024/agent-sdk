import unittest

from comate_agent_sdk.agent.options import AgentConfig, build_runtime_options


class TestOptionsPrecheckConfig(unittest.TestCase):
    def test_build_runtime_options_copies_precheck_fields(self) -> None:
        cfg = AgentConfig(
            precheck_buffer_ratio=0.15,
            token_count_timeout_ms=450,
            emit_compaction_meta_events=True,
        )

        runtime = build_runtime_options(
            config=cfg,
            resolved_agents=None,
            resolved_skills=None,
            resolved_memory=None,
            resolved_llm_levels=None,
            session_id=None,
            offload_root_path=None,
        )

        self.assertEqual(runtime.precheck_buffer_ratio, 0.15)
        self.assertEqual(runtime.token_count_timeout_ms, 450)
        self.assertTrue(runtime.emit_compaction_meta_events)


if __name__ == "__main__":
    unittest.main()
