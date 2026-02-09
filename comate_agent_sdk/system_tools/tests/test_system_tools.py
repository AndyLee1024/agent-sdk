import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from comate_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from comate_agent_sdk.system_tools import tools as system_tools_module
from comate_agent_sdk.system_tools.tools import (
    Bash,
    Edit,
    Glob,
    Grep,
    LS,
    MultiEdit,
    Read,
    TodoWrite,
    WebFetch,
    Write,
)
from comate_agent_sdk.tools import get_default_registry
from comate_agent_sdk.tools.system_context import bind_system_tool_context
from comate_agent_sdk.tokens import TokenCost


class TestSystemTools(unittest.TestCase):
    def setUp(self) -> None:
        system_tools_module._WEBFETCH_CACHE.clear()

    def test_default_registry_contains_core_system_tools(self) -> None:
        names = set(get_default_registry().names())
        self.assertTrue({"Read", "Grep", "Bash", "Glob"}.issubset(names))

    def test_read_offset_line_and_limit_lines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "a.txt"
            p.write_text("l1\nl2\nl3\n", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out0 = self._run(Read, file_path=str(p))
                self.assertTrue(out0["ok"])
                data0 = out0["data"]
                self.assertIn("\tl1", data0["content"])
                self.assertEqual(data0["total_lines"], 3)
                self.assertEqual(data0["lines_returned"], 3)

                out1 = self._run(Read, file_path=str(p), offset_line=1, limit_lines=1)
                self.assertTrue(out1["ok"])
                data1 = out1["data"]
                self.assertIn("\tl2", data1["content"])
                self.assertNotIn("\tl1", data1["content"])
                self.assertEqual(data1["lines_returned"], 1)

    def test_read_legacy_offset_limit_aliases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "a.txt"
            p.write_text("l1\nl2\nl3\n", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                with self.assertRaises(Exception):
                    self._run(Read, file_path=str(p), offset=1, limit=1)

    def test_read_accepts_relative_path_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nested = root / "skill" / "SKILL.md"
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.write_text("# skill\n", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out = self._run(Read, file_path="skill/SKILL.md")
                self.assertTrue(out["ok"])
                self.assertEqual(out["data"]["total_lines"], 1)
                self.assertIn("\t# skill", out["data"]["content"])

    def test_write_only_allows_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / ".agent_workspace"
            p_ok = workspace / "sub" / "b.txt"
            p_bad = root / "outside.txt"

            with bind_system_tool_context(project_root=root):
                ok_out = self._run(Write, file_path=str(p_ok), content="hello")
                self.assertTrue(ok_out["ok"])
                self.assertTrue(p_ok.exists())
                self.assertEqual(p_ok.read_text(encoding="utf-8"), "hello")

                bad_out = self._run(Write, file_path=str(p_bad), content="nope")
                self.assertFalse(bad_out["ok"])
                self.assertIn(bad_out["error"]["code"], {"PATH_ESCAPE", "PERMISSION_DENIED"})

    def test_edit_requires_unique_old_string_unless_replace_all(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / ".agent_workspace"
            p = workspace / "c.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x\nx\n", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out = self._run(Edit, file_path=str(p), old_string="x", new_string="y")
                self.assertFalse(out["ok"])
                self.assertEqual(out["error"]["code"], "CONFLICT")
                self.assertEqual(p.read_text(encoding="utf-8"), "x\nx\n")

                out2 = self._run(
                    Edit,
                    file_path=str(p),
                    old_string="x",
                    new_string="y",
                    replace_all=True,
                )
                self.assertTrue(out2["ok"])
                self.assertEqual(out2["data"]["replacements"], 2)
                self.assertEqual(p.read_text(encoding="utf-8"), "y\ny\n")

    def test_edit_can_target_existing_project_file_by_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "proj.txt"
            p.write_text("hello\n", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out = self._run(Edit, file_path="proj.txt", old_string="hello", new_string="hi")
                self.assertTrue(out["ok"])
                self.assertEqual(out["data"]["replacements"], 1)
                self.assertEqual(p.read_text(encoding="utf-8"), "hi\n")

    def test_glob_returns_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "d").mkdir()
            (root / "d" / "f1.txt").write_text("a", encoding="utf-8")
            (root / "d" / "f2.txt").write_text("b", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out = self._run(Glob, pattern="**/*.txt")
                self.assertTrue(out["ok"])
                self.assertEqual(out["data"]["count"], 2)
                self.assertTrue(all(not Path(m).is_absolute() for m in out["data"]["matches"]))

    def test_grep_alias_flags_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "e.txt"
            p.write_text("a\nfoo\nb\n", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out = self._run(
                    Grep,
                    **{
                        "pattern": "foo",
                        "path": str(root),
                        "output_mode": "content",
                        "-A": 1,
                        "-n": False,
                    },
                )
                self.assertTrue(out["ok"])
                self.assertEqual(out["data"]["total_matches"], 1)
                m = out["data"]["matches"][0]
                self.assertIsNone(m["line_number"])
                self.assertEqual(m["after_context"], ["b"])

    def test_bash_runs_and_returns_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "x.txt").write_text("ok", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out = self._run(Bash, args=["cat", "x.txt"])
                self.assertTrue(out["ok"])
                self.assertEqual(out["data"]["exit_code"], 0)
                self.assertIn("ok", out["data"]["stdout"])

    def test_multiedit_creates_new_file_with_first_empty_old_string(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / ".agent_workspace"
            p = workspace / "new.txt"

            with bind_system_tool_context(project_root=root):
                out = self._run(
                    MultiEdit,
                    file_path=str(p),
                    edits=[
                        {"old_string": "", "new_string": "hello\n", "replace_all": False},
                        {"old_string": "hello", "new_string": "hi", "replace_all": False},
                    ],
                )
                self.assertTrue(out["ok"])
                self.assertEqual(out["data"]["total_replacements"], 1)
                self.assertTrue(p.exists())
                self.assertEqual(p.read_text(encoding="utf-8"), "hi\n")

    def test_ls_returns_entries_and_supports_ignore(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("b", encoding="utf-8")
            (root / "d").mkdir()

            with bind_system_tool_context(project_root=root):
                out = self._run(LS, path=str(root), ignore=["b.txt"])
                self.assertTrue(out["ok"])
                names = [e["name"] for e in out["data"]["entries"]]
                self.assertIn("a.txt", names)
                self.assertIn("d", names)
                self.assertNotIn("b.txt", names)

    def test_todowrite_persists_to_session_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session_root = root / "sessions" / "s1"

            with bind_system_tool_context(project_root=root, session_id="s1", session_root=session_root):
                out = self._run(
                    TodoWrite,
                    todos=[
                        {"content": "t1", "status": "pending", "id": "1"},
                        {"content": "t2", "status": "completed", "id": "2"},
                    ],
                )
                self.assertTrue(out["ok"])
                self.assertTrue((session_root / "todos.json").exists())
                data = json.loads((session_root / "todos.json").read_text(encoding="utf-8"))
                self.assertEqual(data.get("schema_version"), 2)
                self.assertEqual(len(data["todos"]), 2)

    def test_webfetch_uses_low_llm_and_records_usage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            class DummyLLM:
                model = "dummy-low"

                @property
                def provider(self) -> str:
                    return "dummy"

                @property
                def name(self) -> str:
                    return self.model

                async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
                    return ChatInvokeCompletion(
                        content="answer",
                        usage=ChatInvokeUsage(
                            prompt_tokens=10,
                            prompt_cached_tokens=None,
                            prompt_cache_creation_tokens=None,
                            prompt_image_tokens=None,
                            completion_tokens=5,
                            total_tokens=15,
                        ),
                    )

            token_cost = TokenCost(include_cost=False)
            llm_levels = {"LOW": DummyLLM()}

            def fake_get(url, timeout, impersonate):
                return SimpleNamespace(status_code=200, url=url, text="<h1>Hello</h1>")

            with bind_system_tool_context(
                project_root=root,
                session_id="s1",
                session_root=root / "sessions" / "s1",
                token_cost=token_cost,
                llm_levels=llm_levels,
            ):
                with mock.patch.dict("sys.modules", {"curl_cffi": SimpleNamespace(requests=SimpleNamespace(get=fake_get))}):
                    with mock.patch.dict("sys.modules", {"markdownify": SimpleNamespace(markdownify=lambda html: "# Hello")}):
                        out = self._run_raw(WebFetch, url="https://example.com", prompt="Summarize")
                        self.assertTrue(out["ok"])
                        self.assertEqual(out["data"]["summary_text"], "answer")
                        self.assertIn("artifact", out["data"])
                        self.assertEqual(len(token_cost.usage_history), 1)
                        self.assertEqual(token_cost.usage_history[0].model, "dummy-low")
                        self.assertEqual(token_cost.usage_history[0].level, "LOW")
                        self.assertEqual(token_cost.usage_history[0].source, "webfetch")

    def test_webfetch_non_200_does_not_call_llm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            class ExplodingLLM:
                model = "dummy-low"

                @property
                def provider(self) -> str:
                    return "dummy"

                @property
                def name(self) -> str:
                    return self.model

                async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
                    raise AssertionError("should not be called")

            llm_levels = {"LOW": ExplodingLLM()}

            def fake_get(url, timeout, impersonate):
                return SimpleNamespace(status_code=404, url=url, text="not found")

            with bind_system_tool_context(
                project_root=root,
                session_id="s1",
                session_root=root / "sessions" / "s1",
                llm_levels=llm_levels,
            ):
                with mock.patch.dict("sys.modules", {"curl_cffi": SimpleNamespace(requests=SimpleNamespace(get=fake_get))}):
                    out = self._run_raw(WebFetch, url="https://example-404.com", prompt="Summarize")
                    self.assertFalse(out["ok"])
                    self.assertEqual(out["error"]["code"], "NOT_FOUND")

    def test_webfetch_records_subagent_prefixed_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            class DummyLLM:
                model = "dummy-low"

                @property
                def provider(self) -> str:
                    return "dummy"

                @property
                def name(self) -> str:
                    return self.model

                async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
                    return ChatInvokeCompletion(
                        content="answer",
                        usage=ChatInvokeUsage(
                            prompt_tokens=10,
                            prompt_cached_tokens=None,
                            prompt_cache_creation_tokens=None,
                            prompt_image_tokens=None,
                            completion_tokens=5,
                            total_tokens=15,
                        ),
                    )

            token_cost = TokenCost(include_cost=False)
            llm_levels = {"LOW": DummyLLM()}

            def fake_get(url, timeout, impersonate):
                return SimpleNamespace(status_code=200, url=url, text="<h1>Hello</h1>")

            with bind_system_tool_context(
                project_root=root,
                session_id="s1",
                session_root=root / "sessions" / "s1",
                token_cost=token_cost,
                llm_levels=llm_levels,
                subagent_name="researcher",
            ):
                with mock.patch.dict("sys.modules", {"curl_cffi": SimpleNamespace(requests=SimpleNamespace(get=fake_get))}):
                    with mock.patch.dict("sys.modules", {"markdownify": SimpleNamespace(markdownify=lambda html: "# Hello")}):
                        out = self._run_raw(WebFetch, url="https://example.com", prompt="Summarize")
                        self.assertTrue(out["ok"])
                        self.assertEqual(len(token_cost.usage_history), 1)
                        self.assertEqual(token_cost.usage_history[0].source, "subagent:researcher:webfetch")

    @staticmethod
    def _run(tool_obj, /, **kwargs):
        raw = asyncio.run(tool_obj.execute(**kwargs))
        if isinstance(raw, str) and raw.strip().startswith(("{", "[")):
            return json.loads(raw)
        return raw

    @staticmethod
    def _run_raw(tool_obj, /, **kwargs):
        return TestSystemTools._run(tool_obj, **kwargs)


if __name__ == "__main__":
    unittest.main()
