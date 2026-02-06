"""测试 AskUserQuestion 工具的基本功能"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from comate_agent_sdk.system_tools.tools import AskUserQuestion
from comate_agent_sdk.tools.system_context import bind_system_tool_context


class TestAskUserQuestion(unittest.TestCase):
    """AskUserQuestion 工具测试套件"""

    def test_askuserquestion_single_question(self):
        """测试单个问题的基本调用"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            with bind_system_tool_context(project_root=root):
                result = self._run(
                    AskUserQuestion,
                    questions=[
                        {
                            "question": "Which authentication method should we use?",
                            "header": "Auth method",
                            "options": [
                                {
                                    "label": "JWT tokens",
                                    "description": "Stateless, scalable, works well with REST APIs"
                                },
                                {
                                    "label": "Session cookies",
                                    "description": "Traditional server-side sessions"
                                },
                            ],
                            "multiSelect": False
                        }
                    ]
                )

                # 验证结果
                self.assertEqual(result["status"], "waiting_for_input")
                self.assertEqual(len(result["questions"]), 1)
                self.assertEqual(result["questions"][0]["question"], "Which authentication method should we use?")
                self.assertEqual(len(result["questions"][0]["options"]), 2)
                self.assertEqual(result["questions"][0]["multiSelect"], False)

    def test_askuserquestion_multiple_questions(self):
        """测试多个问题 + multiSelect"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            with bind_system_tool_context(project_root=root):
                result = self._run(
                    AskUserQuestion,
                    questions=[
                        {
                            "question": "Which testing framework would you like to use?",
                            "header": "Framework",
                            "options": [
                                {"label": "Jest", "description": "Most popular"},
                                {"label": "Vitest", "description": "Faster"},
                            ],
                            "multiSelect": False
                        },
                        {
                            "question": "Which types of tests do you want to set up?",
                            "header": "Test types",
                            "options": [
                                {"label": "Unit tests", "description": "Test individual functions"},
                                {"label": "Integration tests", "description": "Test components together"},
                                {"label": "E2E tests", "description": "Test full application flow"},
                            ],
                            "multiSelect": True
                        }
                    ]
                )

                self.assertEqual(result["status"], "waiting_for_input")
                self.assertEqual(len(result["questions"]), 2)
                self.assertEqual(result["questions"][1]["multiSelect"], True)
                self.assertEqual(len(result["questions"][1]["options"]), 3)

    def test_askuserquestion_validation_header_length(self):
        """测试 header 长度验证 (max 12 chars)"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            with bind_system_tool_context(project_root=root):
                # header 超过 12 字符应该失败
                with self.assertRaises(Exception) as ctx:
                    self._run(
                        AskUserQuestion,
                        questions=[
                            {
                                "question": "Test?",
                                "header": "This is way too long",  # 20 字符
                                "options": [
                                    {"label": "A", "description": "Option A"},
                                    {"label": "B", "description": "Option B"},
                                ],
                                "multiSelect": False
                            }
                        ]
                    )

    def test_askuserquestion_validation_options_count(self):
        """测试 options 数量验证 (2-4)"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            with bind_system_tool_context(project_root=root):
                # 只有 1 个 option 应该失败
                with self.assertRaises(Exception) as ctx:
                    self._run(
                        AskUserQuestion,
                        questions=[
                            {
                                "question": "Test?",
                                "header": "Test",
                                "options": [
                                    {"label": "Only one", "description": "Only one option"},
                                ],
                                "multiSelect": False
                            }
                        ]
                    )

    def test_askuserquestion_validation_questions_count(self):
        """测试 questions 数量验证 (1-4)"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            with bind_system_tool_context(project_root=root):
                # 空列表应该失败
                with self.assertRaises(Exception) as ctx:
                    self._run(
                        AskUserQuestion,
                        questions=[]
                    )

    @staticmethod
    def _run(tool_obj, /, **kwargs):
        """辅助方法: 运行 async 工具并解析结果"""
        raw = asyncio.run(tool_obj.execute(**kwargs))
        if isinstance(raw, str) and raw.strip().startswith(("{", "[")):
            return json.loads(raw)
        return raw


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
