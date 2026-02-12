from __future__ import annotations

import sys
import unittest
from pathlib import Path

from comate_agent_sdk.agent.events import TextEvent

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.rpc_protocol import (
    ErrorCodes,
    JSONRPCProtocolError,
    build_error_response,
    build_event_notification,
    build_success_response,
    parse_jsonrpc_message,
)


class TestRPCProtocol(unittest.TestCase):
    def test_parse_jsonrpc_message_accepts_valid_payload(self) -> None:
        parsed = parse_jsonrpc_message('{"jsonrpc":"2.0","id":1,"method":"initialize"}')
        self.assertEqual(parsed["method"], "initialize")

    def test_parse_jsonrpc_message_rejects_non_json(self) -> None:
        with self.assertRaises(JSONRPCProtocolError) as ctx:
            parse_jsonrpc_message("{bad-json")
        self.assertEqual(ctx.exception.code, ErrorCodes.PARSE_ERROR)

    def test_parse_jsonrpc_message_rejects_wrong_version(self) -> None:
        with self.assertRaises(JSONRPCProtocolError) as ctx:
            parse_jsonrpc_message('{"jsonrpc":"1.0","id":1,"method":"initialize"}')
        self.assertEqual(ctx.exception.code, ErrorCodes.INVALID_REQUEST)

    def test_build_event_notification_includes_event_type(self) -> None:
        payload = build_event_notification(TextEvent(content="hello"))
        self.assertEqual(payload["jsonrpc"], "2.0")
        self.assertEqual(payload["method"], "event")
        self.assertEqual(payload["params"]["event_type"], "TextEvent")
        self.assertEqual(payload["params"]["content"], "hello")

    def test_build_success_and_error_response(self) -> None:
        success = build_success_response(7, {"ok": True})
        error = build_error_response(
            request_id=7,
            code=ErrorCodes.INTERNAL_ERROR,
            message="boom",
        )
        self.assertEqual(success["id"], 7)
        self.assertIn("result", success)
        self.assertEqual(error["error"]["code"], ErrorCodes.INTERNAL_ERROR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
