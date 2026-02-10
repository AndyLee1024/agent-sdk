import unittest

from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import ToolMessage


class TestToolMessageMetadataInjection(unittest.TestCase):
    def test_add_message_injects_execution_meta_and_raw_envelope(self) -> None:
        ctx = ContextIR()
        message = ToolMessage(
            tool_call_id="tc1",
            tool_name="Read",
            content="# Read: src/main.py",
            is_error=False,
            raw_envelope={
                "ok": True,
                "data": {"content": "x"},
                "message": None,
                "error": None,
                "meta": {},
                "schema_version": 1,
            },
            execution_meta={
                "status": "ok",
                "truncation": {"formatter_truncated": True, "formatter_reason": "line_limit"},
            },
        )

        item = ctx.add_message(message)
        self.assertEqual(item.item_type, ItemType.TOOL_RESULT)
        self.assertIn("tool_execution_meta", item.metadata)
        self.assertIn("tool_raw_envelope", item.metadata)
        self.assertEqual(item.metadata["tool_execution_meta"]["status"], "ok")
        self.assertTrue(item.metadata["tool_execution_meta"]["truncation"]["formatter_truncated"])
        self.assertTrue(item.metadata["tool_raw_envelope"]["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
