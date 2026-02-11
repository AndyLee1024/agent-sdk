"""Tests for schema-aware tool argument coercion.

验证 _coerce_tool_arguments 及其辅助函数能正确处理
非原生 OpenAI 模型返回的字符串化参数问题。
"""

from __future__ import annotations

import json
import unittest

from comate_agent_sdk.agent.tool_exec import (
    _coerce_tool_arguments,
    _coerce_value,
    _extract_types,
)
from comate_agent_sdk.system_tools.tools import AskUserQuestionInput, BashInput, TodoWriteInput


# ── _extract_types ──────────────────────────────────────────────


class TestExtractTypes(unittest.TestCase):
    def test_single_type_string(self):
        assert _extract_types({"type": "integer"}) == {"integer"}

    def test_type_array(self):
        """OpenAI strict mode nullable: "type": ["object", "null"]"""
        assert _extract_types({"type": ["object", "null"]}) == {"object", "null"}

    def test_any_of(self):
        """Pydantic 原生 nullable: anyOf"""
        schema = {"anyOf": [{"type": "object"}, {"type": "null"}]}
        assert _extract_types(schema) == {"object", "null"}

    def test_empty_schema(self):
        assert _extract_types({}) == set()


# ── _coerce_value 基本类型转换 ──────────────────────────────────


class TestCoerceValuePrimitives(unittest.TestCase):
    def test_string_to_integer(self):
        assert _coerce_value("120000", {"type": "integer"}) == 120000
        assert isinstance(_coerce_value("120000", {"type": "integer"}), int)

    def test_string_to_number(self):
        result = _coerce_value("3.14", {"type": "number"})
        assert result == 3.14
        assert isinstance(result, float)

    def test_string_to_boolean_true(self):
        assert _coerce_value("true", {"type": "boolean"}) is True
        assert _coerce_value("1", {"type": "boolean"}) is True

    def test_string_to_boolean_false(self):
        assert _coerce_value("false", {"type": "boolean"}) is False
        assert _coerce_value("0", {"type": "boolean"}) is False

    def test_string_to_null(self):
        schema = {"type": ["string", "null"]}
        assert _coerce_value("null", schema) is None
        assert _coerce_value("none", schema) is None
        assert _coerce_value("", schema) is None

    def test_string_to_null_anyof(self):
        schema = {"anyOf": [{"type": "object"}, {"type": "null"}]}
        assert _coerce_value("null", schema) is None

    def test_invalid_integer_returns_original(self):
        """无法转换时原样返回"""
        assert _coerce_value("not_a_number", {"type": "integer"}) == "not_a_number"

    def test_invalid_boolean_returns_original(self):
        assert _coerce_value("maybe", {"type": "boolean"}) == "maybe"


# ── _coerce_value 复合类型 ──────────────────────────────────────


class TestCoerceValueComposite(unittest.TestCase):
    def test_string_to_array(self):
        result = _coerce_value('["git", "diff"]', {"type": "array", "items": {"type": "string"}})
        assert result == ["git", "diff"]
        assert isinstance(result, list)

    def test_string_to_object(self):
        result = _coerce_value('{"KEY": "VAL"}', {"type": "object"})
        assert result == {"KEY": "VAL"}
        assert isinstance(result, dict)

    def test_string_to_empty_object(self):
        result = _coerce_value("{}", {"type": "object"})
        assert result == {}

    def test_string_to_empty_array(self):
        result = _coerce_value("[]", {"type": "array", "items": {"type": "string"}})
        assert result == []

    def test_invalid_json_array_returns_original(self):
        assert _coerce_value("not json", {"type": "array"}) == "not json"

    def test_invalid_json_object_returns_original(self):
        assert _coerce_value("not json", {"type": "object"}) == "not json"

    def test_nested_array_of_objects(self):
        """array of objects，内部字段也需要递归 coerce"""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                },
            },
        }
        value = '[{"name": "a", "count": "3"}, {"name": "b", "count": "7"}]'
        result = _coerce_value(value, schema)
        assert result == [{"name": "a", "count": 3}, {"name": "b", "count": 7}]


# ── 已正确类型的值不被修改 ──────────────────────────────────────


class TestCoerceValuePassthrough(unittest.TestCase):
    def test_correct_int_unchanged(self):
        assert _coerce_value(120000, {"type": "integer"}) == 120000

    def test_correct_bool_unchanged(self):
        assert _coerce_value(True, {"type": "boolean"}) is True

    def test_correct_list_unchanged(self):
        val = ["git", "diff"]
        result = _coerce_value(val, {"type": "array", "items": {"type": "string"}})
        assert result == ["git", "diff"]

    def test_correct_dict_unchanged(self):
        val = {"KEY": "VAL"}
        result = _coerce_value(val, {"type": "object"})
        assert result == {"KEY": "VAL"}

    def test_correct_none_unchanged(self):
        result = _coerce_value(None, {"anyOf": [{"type": "object"}, {"type": "null"}]})
        assert result is None

    def test_string_stays_string_when_expected(self):
        assert _coerce_value("hello", {"type": "string"}) == "hello"

    def test_no_schema_type_returns_original(self):
        assert _coerce_value("anything", {}) == "anything"


# ── _coerce_tool_arguments 入口函数 ────────────────────────────


class TestCoerceToolArguments(unittest.TestCase):
    def test_bash_like_schema(self):
        """模拟 Bash 工具的 schema，所有参数都被字符串化"""
        schema = {
            "type": "object",
            "properties": {
                "args": {"type": "array", "items": {"type": "string"}},
                "timeout_ms": {"type": "integer"},
                "cwd": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "env": {"anyOf": [{"type": "object"}, {"type": "null"}]},
                "max_output_chars": {"type": "integer"},
            },
            "required": ["args"],
        }
        args = {
            "args": '["git", "diff"]',
            "timeout_ms": "120000",
            "cwd": "null",
            "env": "{}",
            "max_output_chars": "30000",
        }
        result = _coerce_tool_arguments(args, schema)
        assert result["args"] == ["git", "diff"]
        assert result["timeout_ms"] == 120000
        assert result["cwd"] is None
        assert result["env"] == {}
        assert result["max_output_chars"] == 30000

    def test_already_correct_args_unchanged(self):
        schema = {
            "type": "object",
            "properties": {
                "args": {"type": "array", "items": {"type": "string"}},
                "timeout_ms": {"type": "integer"},
            },
        }
        args = {"args": ["git", "status"], "timeout_ms": 120000}
        result = _coerce_tool_arguments(args, schema)
        assert result == args

    def test_empty_schema_properties(self):
        result = _coerce_tool_arguments({"foo": "bar"}, {"type": "object"})
        assert result == {"foo": "bar"}

    def test_unknown_keys_passthrough(self):
        """不在 schema properties 中的字段原样保留"""
        schema = {
            "type": "object",
            "properties": {"known": {"type": "integer"}},
        }
        args = {"known": "42", "unknown": "whatever"}
        result = _coerce_tool_arguments(args, schema)
        assert result["known"] == 42
        assert result["unknown"] == "whatever"


# ── BashInput defense-in-depth ──────────────────────────────────


class TestBashInputCoercion(unittest.TestCase):
    def test_all_stringified(self):
        """BashInput 全部参数字符串化的端到端场景"""
        data = {
            "args": '["git", "status"]',
            "timeout_ms": "120000",
            "cwd": "null",
            "env": "{}",
            "max_output_chars": "30000",
        }
        inp = BashInput(**data)
        assert inp.args == ["git", "status"]
        assert inp.timeout_ms == 120000
        assert inp.cwd is None
        assert inp.env == {}
        assert inp.max_output_chars == 30000

    def test_env_null_string(self):
        data = {"args": ["echo", "hi"], "env": "null"}
        inp = BashInput(**data)
        assert inp.env is None

    def test_env_none_string(self):
        data = {"args": ["echo", "hi"], "env": "none"}
        inp = BashInput(**data)
        assert inp.env is None

    def test_env_empty_string(self):
        data = {"args": ["echo", "hi"], "env": ""}
        inp = BashInput(**data)
        assert inp.env is None

    def test_env_valid_json_dict(self):
        data = {"args": ["echo", "hi"], "env": '{"PATH": "/usr/bin"}'}
        inp = BashInput(**data)
        assert inp.env == {"PATH": "/usr/bin"}

    def test_already_correct_types(self):
        """已正确类型时不受影响"""
        data = {
            "args": ["git", "status"],
            "timeout_ms": 120000,
            "env": {"K": "V"},
            "max_output_chars": 30000,
        }
        inp = BashInput(**data)
        assert inp.args == ["git", "status"]
        assert inp.timeout_ms == 120000
        assert inp.env == {"K": "V"}
        assert inp.max_output_chars == 30000


# ── 递归 coerce 对已正确的嵌套值 ───────────────────────────────


class TestCoerceNonStringRecursion(unittest.TestCase):
    def test_list_of_objects_with_stringified_fields(self):
        """已经是 list，但内部 object 有字符串化字段"""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "port": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                },
            },
        }
        value = [{"port": "8080", "enabled": "true"}]
        result = _coerce_value(value, schema)
        assert result == [{"port": 8080, "enabled": True}]

    def test_dict_with_stringified_nested_fields(self):
        """已经是 dict，但内部字段被字符串化"""
        schema = {
            "type": "object",
            "properties": {
                "retries": {"type": "integer"},
                "verbose": {"type": "boolean"},
            },
        }
        value = {"retries": "3", "verbose": "false"}
        result = _coerce_value(value, schema)
        assert result == {"retries": 3, "verbose": False}


# ── TodoWriteInput defense-in-depth ─────────────────────────────


class TestTodoWriteInputCoercion(unittest.TestCase):
    def test_stringified_todos(self):
        data = {
            "todos": '[{"id": "1", "content": "task one", "status": "pending", "priority": "high"}]'
        }
        inp = TodoWriteInput(**data)
        assert len(inp.todos) == 1
        assert inp.todos[0].id == "1"
        assert inp.todos[0].content == "task one"
        assert inp.todos[0].status == "pending"

    def test_already_correct_todos(self):
        data = {
            "todos": [{"id": "1", "content": "task one", "status": "pending", "priority": "high"}]
        }
        inp = TodoWriteInput(**data)
        assert len(inp.todos) == 1


# ── AskUserQuestionInput defense-in-depth ───────────────────────


class TestAskUserQuestionInputCoercion(unittest.TestCase):
    def test_stringified_questions(self):
        data = {
            "questions": json.dumps([{
                "question": "Which approach?",
                "header": "Approach",
                "options": [
                    {"label": "A", "description": "Option A"},
                    {"label": "B", "description": "Option B"},
                ],
                "multiSelect": False,
            }])
        }
        inp = AskUserQuestionInput(**data)
        assert len(inp.questions) == 1
        assert inp.questions[0].question == "Which approach?"

    def test_already_correct_questions(self):
        data = {
            "questions": [{
                "question": "Which?",
                "header": "Choice",
                "options": [
                    {"label": "A", "description": "Opt A"},
                    {"label": "B", "description": "Opt B"},
                ],
                "multiSelect": False,
            }]
        }
        inp = AskUserQuestionInput(**data)
        assert len(inp.questions) == 1


if __name__ == "__main__":
    unittest.main()
