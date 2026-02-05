import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bu_agent_sdk.agent.settings import (
    USER_AGENTS_MD_PATH,
    USER_SETTINGS_PATH,
    SettingsConfig,
    discover_agents_md,
    discover_user_agents_md,
    load_settings_file,
    resolve_settings,
)


class TestLoadSettingsFile(unittest.TestCase):
    """测试 load_settings_file 单文件解析逻辑"""

    def test_valid_llm_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "settings.json"
            p.write_text('{"llm_levels": {"LOW": "openai:gpt-4o-mini", "HIGH": "anthropic:claude-opus-4-5"}}')
            result = load_settings_file(p)

        self.assertIsNotNone(result)
        self.assertEqual(result.llm_levels, {"LOW": "openai:gpt-4o-mini", "HIGH": "anthropic:claude-opus-4-5"})
        self.assertIsNone(result.llm_levels_base_url)

    def test_valid_with_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "settings.json"
            p.write_text(
                '{"llm_levels": {"LOW": "openai:gpt-4o-mini"}, '
                '"llm_levels_base_url": {"LOW": "https://api.example.com/v1", "HIGH": null}}'
            )
            result = load_settings_file(p)

        self.assertIsNotNone(result)
        self.assertEqual(result.llm_levels_base_url, {"LOW": "https://api.example.com/v1", "HIGH": None})

    def test_missing_file_returns_none(self) -> None:
        result = load_settings_file(Path("/nonexistent/path/settings.json"))
        self.assertIsNone(result)

    def test_invalid_json_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "settings.json"
            p.write_text("not valid json {{{")
            result = load_settings_file(p)
        self.assertIsNone(result)

    def test_empty_object_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "settings.json"
            p.write_text("{}")
            result = load_settings_file(p)
        self.assertIsNone(result)

    def test_non_object_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "settings.json"
            p.write_text('["not", "an", "object"]')
            result = load_settings_file(p)
        self.assertIsNone(result)

    def test_non_string_level_value_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "settings.json"
            p.write_text('{"llm_levels": {"LOW": 123, "HIGH": "anthropic:claude-opus-4-5"}}')
            result = load_settings_file(p)

        self.assertIsNotNone(result)
        self.assertNotIn("LOW", result.llm_levels)
        self.assertIn("HIGH", result.llm_levels)

    def test_only_base_url_no_levels(self) -> None:
        """仅有 base_url 无 llm_levels 也应返回有效配置"""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "settings.json"
            p.write_text('{"llm_levels_base_url": {"LOW": "https://api.example.com"}}')
            result = load_settings_file(p)

        self.assertIsNotNone(result)
        self.assertIsNone(result.llm_levels)
        self.assertEqual(result.llm_levels_base_url, {"LOW": "https://api.example.com"})


class TestResolveSettings(unittest.TestCase):
    """测试 resolve_settings 加载策略和合并逻辑（mock load_settings_file）"""

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_none_sources_returns_none(self, mock_load) -> None:
        result = resolve_settings(sources=None, project_root=Path("/tmp"))
        self.assertIsNone(result)
        mock_load.assert_not_called()

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_empty_sources_returns_none(self, mock_load) -> None:
        result = resolve_settings(sources=(), project_root=Path("/tmp"))
        self.assertIsNone(result)
        mock_load.assert_not_called()

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_user_only_loads_user_path(self, mock_load) -> None:
        user_cfg = SettingsConfig(llm_levels={"LOW": "openai:gpt-4o-mini"})
        mock_load.return_value = user_cfg

        result = resolve_settings(sources=("user",), project_root=Path("/tmp/proj"))

        self.assertIsNotNone(result)
        self.assertEqual(result.llm_levels, {"LOW": "openai:gpt-4o-mini"})
        mock_load.assert_called_once_with(USER_SETTINGS_PATH)

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_project_only_loads_project_path(self, mock_load) -> None:
        project_cfg = SettingsConfig(llm_levels={"HIGH": "anthropic:claude-opus-4-5"})
        mock_load.return_value = project_cfg

        result = resolve_settings(sources=("project",), project_root=Path("/tmp/proj"))

        self.assertIsNotNone(result)
        self.assertEqual(result.llm_levels, {"HIGH": "anthropic:claude-opus-4-5"})
        expected_path = Path("/tmp/proj").expanduser().resolve() / ".agent" / "settings.json"
        mock_load.assert_called_once_with(expected_path)

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_project_overrides_user_llm_levels(self, mock_load) -> None:
        """project 有 llm_levels 时，user 的 llm_levels 被完全忽略"""
        user_cfg = SettingsConfig(llm_levels={"LOW": "openai:gpt-4o-mini", "MID": "openai:gpt-4o"})
        project_cfg = SettingsConfig(llm_levels={"LOW": "anthropic:claude-haiku-4-5"})

        def _load(path):
            return user_cfg if path == USER_SETTINGS_PATH else project_cfg

        mock_load.side_effect = _load

        result = resolve_settings(sources=("user", "project"), project_root=Path("/tmp/proj"))

        # project 完全覆盖 → user 的 MID 消失
        self.assertEqual(result.llm_levels, {"LOW": "anthropic:claude-haiku-4-5"})

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_project_none_falls_back_to_user(self, mock_load) -> None:
        """project settings 不存在时，回退到 user"""
        user_cfg = SettingsConfig(llm_levels={"LOW": "openai:gpt-4o-mini"})

        def _load(path):
            return user_cfg if path == USER_SETTINGS_PATH else None

        mock_load.side_effect = _load

        result = resolve_settings(sources=("user", "project"), project_root=Path("/tmp/proj"))
        self.assertEqual(result.llm_levels, {"LOW": "openai:gpt-4o-mini"})

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_user_none_uses_project(self, mock_load) -> None:
        """user settings 不存在时，直接使用 project"""
        project_cfg = SettingsConfig(llm_levels={"HIGH": "anthropic:claude-opus-4-5"})

        def _load(path):
            return None if path == USER_SETTINGS_PATH else project_cfg

        mock_load.side_effect = _load

        result = resolve_settings(sources=("user", "project"), project_root=Path("/tmp/proj"))
        self.assertEqual(result.llm_levels, {"HIGH": "anthropic:claude-opus-4-5"})

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_both_none_returns_none(self, mock_load) -> None:
        mock_load.return_value = None
        result = resolve_settings(sources=("user", "project"), project_root=Path("/tmp/proj"))
        self.assertIsNone(result)

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_project_root_none_uses_cwd(self, mock_load) -> None:
        mock_load.return_value = None
        resolve_settings(sources=("project",), project_root=None)

        expected_path = Path.cwd().expanduser().resolve() / ".agent" / "settings.json"
        mock_load.assert_called_once_with(expected_path)

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_base_url_override_follows_same_rule(self, mock_load) -> None:
        """project 有 base_url 时完全覆盖 user 的 base_url"""
        user_cfg = SettingsConfig(
            llm_levels={"LOW": "openai:gpt-4o-mini"},
            llm_levels_base_url={"LOW": "https://user.example.com"},
        )
        project_cfg = SettingsConfig(
            llm_levels={"LOW": "openai:gpt-4o-mini"},
            llm_levels_base_url={"LOW": "https://project.example.com"},
        )

        def _load(path):
            return user_cfg if path == USER_SETTINGS_PATH else project_cfg

        mock_load.side_effect = _load

        result = resolve_settings(sources=("user", "project"), project_root=Path("/tmp/proj"))
        self.assertEqual(result.llm_levels_base_url, {"LOW": "https://project.example.com"})

    @patch("bu_agent_sdk.agent.settings.load_settings_file")
    def test_project_has_levels_but_no_base_url_falls_back_user_base_url(self, mock_load) -> None:
        """project 有 llm_levels 但无 base_url，user 有 base_url → base_url 回退到 user"""
        user_cfg = SettingsConfig(
            llm_levels={"LOW": "openai:gpt-4o-mini"},
            llm_levels_base_url={"LOW": "https://user.example.com"},
        )
        project_cfg = SettingsConfig(
            llm_levels={"LOW": "anthropic:claude-haiku-4-5"},
            llm_levels_base_url=None,  # project 无 base_url
        )

        def _load(path):
            return user_cfg if path == USER_SETTINGS_PATH else project_cfg

        mock_load.side_effect = _load

        result = resolve_settings(sources=("user", "project"), project_root=Path("/tmp/proj"))
        # llm_levels 被 project 覆盖，base_url 回退到 user（project 的 base_url 为 None）
        self.assertEqual(result.llm_levels, {"LOW": "anthropic:claude-haiku-4-5"})
        self.assertEqual(result.llm_levels_base_url, {"LOW": "https://user.example.com"})


class TestDiscoverAgentsMd(unittest.TestCase):
    """测试 AGENTS.md 文件发现逻辑"""

    def test_both_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("# Root")
            (root / ".agent").mkdir()
            (root / ".agent" / "AGENTS.md").write_text("# Agent dir")

            result = discover_agents_md(root)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], root / "AGENTS.md")
        self.assertEqual(result[1], root / ".agent" / "AGENTS.md")

    def test_only_root_agents_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("# Root")

            result = discover_agents_md(root)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], root / "AGENTS.md")

    def test_only_agent_dir_agents_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".agent").mkdir()
            (root / ".agent" / "AGENTS.md").write_text("# Agent dir")

            result = discover_agents_md(root)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], root / ".agent" / "AGENTS.md")

    def test_no_agents_md_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = discover_agents_md(Path(tmp))
        self.assertEqual(result, [])

    def test_none_project_root_does_not_crash(self) -> None:
        """project_root=None 时使用 cwd，不报错"""
        result = discover_agents_md(None)
        self.assertIsInstance(result, list)

    def test_agents_md_is_directory_not_included(self) -> None:
        """AGENTS.md 是目录而非文件时不收入"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").mkdir()  # 故意创建为目录

            result = discover_agents_md(root)

        self.assertEqual(result, [])


class TestDiscoverUserAgentsMd(unittest.TestCase):
    """测试 user 级 AGENTS.md 文件发现逻辑"""

    def test_user_agents_md_exists(self) -> None:
        """~/.agent/AGENTS.md 存在时返回单元素列表"""
        with tempfile.TemporaryDirectory() as tmp:
            fake_home = Path(tmp)
            fake_agent_dir = fake_home / ".agent"
            fake_agent_dir.mkdir()
            fake_agents_md = fake_agent_dir / "AGENTS.md"
            fake_agents_md.write_text("# User agents")

            with patch("bu_agent_sdk.agent.settings.USER_AGENTS_MD_PATH", fake_agents_md):
                result = discover_user_agents_md()

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], fake_agents_md)

    def test_user_agents_md_not_exists(self) -> None:
        """~/.agent/AGENTS.md 不存在时返回空列表"""
        with tempfile.TemporaryDirectory() as tmp:
            fake_path = Path(tmp) / "nonexistent" / "AGENTS.md"

            with patch("bu_agent_sdk.agent.settings.USER_AGENTS_MD_PATH", fake_path):
                result = discover_user_agents_md()

            self.assertEqual(result, [])

    def test_user_agents_md_is_directory(self) -> None:
        """~/.agent/AGENTS.md 是目录而非文件时返回空列表"""
        with tempfile.TemporaryDirectory() as tmp:
            fake_path = Path(tmp) / "AGENTS.md"
            fake_path.mkdir()  # 创建为目录

            with patch("bu_agent_sdk.agent.settings.USER_AGENTS_MD_PATH", fake_path):
                result = discover_user_agents_md()

            self.assertEqual(result, [])


class TestSettingsWithLLMLevels(unittest.TestCase):
    """验证 settings 与 resolve_llm_levels 的优先级集成"""

    def test_settings_creates_correct_openai_model(self) -> None:
        from bu_agent_sdk.agent.llm_levels import resolve_llm_levels

        settings = SettingsConfig(
            llm_levels={"LOW": "openai:gpt-4o-mini"},
            llm_levels_base_url={"LOW": "https://api.example.com/v1"},
        )
        levels = resolve_llm_levels(explicit=None, settings=settings)

        self.assertEqual(levels["LOW"].provider, "openai")
        self.assertEqual(getattr(levels["LOW"], "base_url", None), "https://api.example.com/v1")

    def test_settings_creates_correct_anthropic_model(self) -> None:
        from bu_agent_sdk.agent.llm_levels import resolve_llm_levels

        settings = SettingsConfig(
            llm_levels={"MID": "anthropic:claude-sonnet-4-5"},
            llm_levels_base_url={"MID": "https://gw.example.com"},
        )
        levels = resolve_llm_levels(explicit=None, settings=settings)

        self.assertEqual(levels["MID"].provider, "anthropic")
        self.assertEqual(getattr(levels["MID"], "base_url", None), "https://gw.example.com")

    def test_explicit_overrides_settings(self) -> None:
        from bu_agent_sdk.agent.llm_levels import resolve_llm_levels
        from bu_agent_sdk.llm.anthropic.chat import ChatAnthropic

        explicit_model = ChatAnthropic(model="claude-haiku-4-5")
        settings = SettingsConfig(llm_levels={"LOW": "openai:gpt-4o-mini"})

        levels = resolve_llm_levels(explicit={"LOW": explicit_model}, settings=settings)

        # explicit 优先，LOW 应该是传入的实例
        self.assertIs(levels["LOW"], explicit_model)

    def test_settings_none_falls_back_to_env(self) -> None:
        """settings=None 时回退到 env 优先级"""
        import os

        from bu_agent_sdk.agent.llm_levels import resolve_llm_levels

        old = os.environ.get("BU_AGENT_SDK_LLM_LOW")
        try:
            os.environ["BU_AGENT_SDK_LLM_LOW"] = "openai:gpt-4o-mini"
            levels = resolve_llm_levels(explicit=None, settings=None)
            self.assertEqual(levels["LOW"].provider, "openai")
        finally:
            if old is None:
                os.environ.pop("BU_AGENT_SDK_LLM_LOW", None)
            else:
                os.environ["BU_AGENT_SDK_LLM_LOW"] = old

    def test_settings_with_empty_llm_levels_falls_back_to_env(self) -> None:
        """settings 存在但 llm_levels 为空 dict 时，回退到 env"""
        import os

        from bu_agent_sdk.agent.llm_levels import resolve_llm_levels

        settings = SettingsConfig(llm_levels={})  # 空 dict，falsy
        old = os.environ.get("BU_AGENT_SDK_LLM_LOW")
        try:
            os.environ["BU_AGENT_SDK_LLM_LOW"] = "openai:gpt-4o-mini"
            levels = resolve_llm_levels(explicit=None, settings=settings)
            self.assertEqual(levels["LOW"].provider, "openai")
        finally:
            if old is None:
                os.environ.pop("BU_AGENT_SDK_LLM_LOW", None)
            else:
                os.environ["BU_AGENT_SDK_LLM_LOW"] = old

    def test_settings_invalid_provider_raises(self) -> None:
        from bu_agent_sdk.agent.llm_levels import resolve_llm_levels

        settings = SettingsConfig(llm_levels={"LOW": "unknown_provider:some-model"})
        with self.assertRaises(ValueError):
            resolve_llm_levels(explicit=None, settings=settings)

    def test_settings_bad_format_raises(self) -> None:
        from bu_agent_sdk.agent.llm_levels import resolve_llm_levels

        settings = SettingsConfig(llm_levels={"LOW": "no-colon-here"})
        with self.assertRaises(ValueError):
            resolve_llm_levels(explicit=None, settings=settings)


if __name__ == "__main__":
    unittest.main()
