import sys
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.question_view import AskUserQuestionUI


class TestQuestionView(unittest.TestCase):
    def test_single_select_then_submit(self) -> None:
        ui = AskUserQuestionUI()
        ok = ui.set_questions(
            [
                {
                    "question": "Pick one",
                    "header": "Choice",
                    "options": [
                        {"label": "Alpha", "description": "A"},
                        {"label": "Beta", "description": "B"},
                    ],
                    "multiSelect": False,
                }
            ]
        )

        self.assertTrue(ok)
        self.assertIsNone(ui.handle_enter())

        action = ui.handle_enter()
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "submit")
        self.assertIn("Alpha", action.message)

    def test_multi_select_space_then_submit(self) -> None:
        ui = AskUserQuestionUI()
        ok = ui.set_questions(
            [
                {
                    "question": "Pick tools",
                    "header": "Tools",
                    "options": [
                        {"label": "ESLint", "description": "lint"},
                        {"label": "Prettier", "description": "format"},
                        {"label": "TypeScript", "description": "types"},
                    ],
                    "multiSelect": True,
                }
            ]
        )

        self.assertTrue(ok)
        ui.toggle_current_selection()
        ui.move_option(1)
        ui.toggle_current_selection()
        ui.focus_submit()

        action = ui.handle_enter()
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "submit")
        self.assertIn("ESLint", action.message)
        self.assertIn("Prettier", action.message)

    def test_multi_select_custom_input_can_be_entered(self) -> None:
        ui = AskUserQuestionUI()
        ok = ui.set_questions(
            [
                {
                    "question": "Pick tools",
                    "header": "Tools",
                    "options": [
                        {"label": "ESLint", "description": "lint"},
                        {"label": "Prettier", "description": "format"},
                    ],
                    "multiSelect": True,
                }
            ]
        )
        self.assertTrue(ok)

        ui.toggle_current_selection()  # select ESLint
        ui.move_option(2)  # move to Type something
        self.assertIsNone(ui.handle_enter())  # activate custom input
        ui.set_custom_input("custom notes with space")
        self.assertIsNone(ui.handle_enter())  # go to preview
        action = ui.handle_enter()  # submit

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "submit")
        self.assertIn("ESLint", action.message)
        self.assertIn("custom notes with space", action.message)

    def test_custom_input_inline_submit(self) -> None:
        ui = AskUserQuestionUI()
        ok = ui.set_questions(
            [
                {
                    "question": "Type something",
                    "header": "Input",
                    "options": [
                        {"label": "A", "description": "x"},
                        {"label": "B", "description": "y"},
                    ],
                    "multiSelect": False,
                }
            ]
        )

        self.assertTrue(ok)
        ui.move_option(2)
        self.assertIsNone(ui.handle_enter())
        ui.set_custom_input("custom answer")
        self.assertIsNone(ui.handle_enter())

        action = ui.handle_enter()
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "submit")
        self.assertIn("custom answer", action.message)

    def test_preview_truncates_custom_input_but_submit_keeps_full_text(self) -> None:
        ui = AskUserQuestionUI()
        ok = ui.set_questions(
            [
                {
                    "question": "Type something",
                    "header": "Input",
                    "options": [
                        {"label": "A", "description": "x"},
                        {"label": "B", "description": "y"},
                    ],
                    "multiSelect": False,
                }
            ]
        )
        self.assertTrue(ok)

        long_text = "12345678901234567890"
        ui.move_option(2)
        self.assertIsNone(ui.handle_enter())  # activate custom input
        ui.set_custom_input(long_text)
        self.assertIsNone(ui.handle_enter())  # enter preview

        preview_text = "".join(text for _, text in ui._preview_fragments())
        self.assertIn("123456789012345...", preview_text)
        self.assertNotIn(long_text, preview_text)

        action = ui.handle_enter()  # submit
        self.assertIsNotNone(action)
        assert action is not None
        self.assertIn(long_text, action.message)

    def test_single_select_custom_overrides_selected_option(self) -> None:
        ui = AskUserQuestionUI()
        ok = ui.set_questions(
            [
                {
                    "question": "Pick one",
                    "header": "Mode",
                    "options": [
                        {"label": "Option A", "description": "a"},
                        {"label": "Option B", "description": "b"},
                    ],
                    "multiSelect": False,
                }
            ]
        )
        self.assertTrue(ok)

        self.assertIsNone(ui.handle_enter())  # select Option A
        ui.move_option(1)  # preview -> Cancel
        self.assertIsNone(ui.handle_enter())  # back to question mode
        ui.move_option(2)
        self.assertIsNone(ui.handle_enter())  # activate custom input
        ui.set_custom_input("custom only")
        self.assertIsNone(ui.handle_enter())  # move to preview
        action = ui.handle_enter()  # submit

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "submit")
        self.assertIn("custom only", action.message)
        self.assertNotIn("Option A", action.message)

    def test_preview_cancel_returns_to_question_mode(self) -> None:
        ui = AskUserQuestionUI()
        ui.set_questions(
            [
                {
                    "question": "Need choice",
                    "header": "Choice",
                    "options": [
                        {"label": "One", "description": "1"},
                        {"label": "Two", "description": "2"},
                    ],
                    "multiSelect": False,
                }
            ]
        )

        self.assertIsNone(ui.handle_enter())  # select One -> preview
        ui.move_option(1)  # move preview cursor to Cancel
        action = ui.handle_enter()
        self.assertIsNone(action)
        self.assertFalse(ui._state.is_preview_mode)

    def test_escape_returns_cancel_action(self) -> None:
        ui = AskUserQuestionUI()
        ui.set_questions(
            [
                {
                    "question": "Cancel?",
                    "header": "Cancel",
                    "options": [
                        {"label": "Yes", "description": "y"},
                        {"label": "No", "description": "n"},
                    ],
                    "multiSelect": False,
                }
            ]
        )

        action = ui.handle_escape()
        self.assertEqual(action.kind, "cancel")
        self.assertEqual(action.message, "user reject answer this question.")

    def test_chat_about_this_uses_fixed_message(self) -> None:
        ui = AskUserQuestionUI()
        ui.set_questions(
            [
                {
                    "question": "Need explanation?",
                    "header": "Explain",
                    "options": [
                        {"label": "Option A", "description": "a"},
                        {"label": "Option B", "description": "b"},
                    ],
                    "multiSelect": False,
                },
                {
                    "question": "Second question",
                    "header": "Second",
                    "options": [
                        {"label": "X", "description": "x"},
                        {"label": "Y", "description": "y"},
                    ],
                    "multiSelect": False,
                }
            ]
        )

        ui.move_option(3)
        action = ui.handle_enter()
        self.assertIsNone(action)
        self.assertEqual(ui._state.current_question_index, 1)
        self.assertFalse(ui._state.is_preview_mode)

    def test_chat_about_this_is_included_on_final_submit(self) -> None:
        ui = AskUserQuestionUI()
        ui.set_questions(
            [
                {
                    "question": "Need explanation?",
                    "header": "Explain",
                    "options": [
                        {"label": "Option A", "description": "a"},
                        {"label": "Option B", "description": "b"},
                    ],
                    "multiSelect": False,
                }
            ]
        )

        ui.move_option(3)
        self.assertIsNone(ui.handle_enter())  # choose preset answer and enter preview
        action = ui.handle_enter()  # submit from preview
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "submit")
        self.assertIn("Chat about this", action.message)

    def test_chat_about_this_basic_functionality(self) -> None:
        """测试 Chat about this 基本功能 - 验证用户选择 Chat about this 后，消息能正确提交"""
        ui = AskUserQuestionUI()
        ok = ui.set_questions(
            [
                {
                    "question": "你想如何处理这个问题？",
                    "header": "处理方式",
                    "options": [
                        {"label": "选项A", "description": "描述A"},
                        {"label": "选项B", "description": "描述B"},
                    ],
                    "multiSelect": False,
                }
            ]
        )

        self.assertTrue(ok)
        # 移动到 "Chat about this" 选项（索引3，因为0-1是选项，2是自定义输入，3是Chat about this）
        ui.move_option(3)
        # 第一次 Enter 应该选择 Chat about this 并进入预览模式
        action = ui.handle_enter()
        self.assertIsNone(action)  # 还没有提交，只是进入预览
        self.assertTrue(ui._state.is_preview_mode)

        # 第二次 Enter 应该提交
        action = ui.handle_enter()
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "submit")
        self.assertIn("Chat about this", action.message)

    def test_chat_about_this_message_format(self) -> None:
        """测试 Chat about this 消息格式 - 验证提交的消息格式是否符合预期"""
        ui = AskUserQuestionUI()
        ui.set_questions(
            [
                {
                    "question": "请选择你的偏好",
                    "header": "偏好设置",
                    "options": [
                        {"label": "红色", "description": "选择红色主题"},
                        {"label": "蓝色", "description": "选择蓝色主题"},
                    ],
                    "multiSelect": False,
                }
            ]
        )

        ui.move_option(3)  # 移动到 Chat about this
        ui.handle_enter()  # 进入预览
        action = ui.handle_enter()  # 提交

        self.assertIsNotNone(action)
        assert action is not None
        # 验证消息包含 "Chat about this" 文本
        self.assertEqual(action.message, "Chat about this")

    def test_chat_about_this_in_multi_question_scenario(self) -> None:
        """测试 Chat about this 在多问题场景下的行为"""
        ui = AskUserQuestionUI()
        ui.set_questions(
            [
                {
                    "question": "第一个问题",
                    "header": "问题1",
                    "options": [
                        {"label": "A1", "description": "a1"},
                        {"label": "A2", "description": "a2"},
                    ],
                    "multiSelect": False,
                },
                {
                    "question": "第二个问题",
                    "header": "问题2",
                    "options": [
                        {"label": "B1", "description": "b1"},
                        {"label": "B2", "description": "b2"},
                    ],
                    "multiSelect": False,
                },
            ]
        )

        # 在第一个问题选择 Chat about this
        ui.move_option(3)  # Chat about this
        self.assertIsNone(ui.handle_enter())  # 应该进入第二个问题，不是预览
        self.assertEqual(ui._state.current_question_index, 1)

        # 在第二个问题选择一个选项
        self.assertIsNone(ui.handle_enter())  # 选择 B1，进入预览
        self.assertTrue(ui._state.is_preview_mode)

        # 提交
        action = ui.handle_enter()
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "submit")
        # 验证消息包含第一个问题的 Chat about this 和第二个问题的 B1
        self.assertIn("Chat about this", action.message)
        self.assertIn("B1", action.message)

    def test_chat_about_this_interaction_with_other_options(self) -> None:
        """测试 Chat about this 与其他选项的交互"""
        ui = AskUserQuestionUI()
        ui.set_questions(
            [
                {
                    "question": "请选择一个选项",
                    "header": "选择",
                    "options": [
                        {"label": "选项X", "description": "描述X"},
                        {"label": "选项Y", "description": "描述Y"},
                    ],
                    "multiSelect": False,
                }
            ]
        )

        # 先选择一个普通选项
        self.assertIsNone(ui.handle_enter())  # 选择 选项X
        self.assertEqual(ui._state.current_question_index, 0)

        # 返回问题模式（从预览返回）
        ui.move_option(1)  # 移动到 Cancel
        ui.handle_enter()  # 取消，返回问题模式
        self.assertFalse(ui._state.is_preview_mode)

        # 现在改为选择 Chat about this
        ui.move_option(3)  # 移动到 Chat about this
        ui.handle_enter()  # 进入预览

        # 验证最终提交的消息是 Chat about this，不是选项X
        action = ui.handle_enter()
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "submit")
        self.assertIn("Chat about this", action.message)
        self.assertNotIn("选项X", action.message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
