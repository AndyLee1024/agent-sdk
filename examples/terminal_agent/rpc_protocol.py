from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


class ErrorCodes:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    INVALID_STATE = -32000


class JSONRPCProtocolError(Exception):
    def __init__(
        self,
        *,
        code: int,
        message: str,
        request_id: str | int | None = None,
        data: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id
        self.data = data


def parse_jsonrpc_message(raw_line: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise JSONRPCProtocolError(code=ErrorCodes.PARSE_ERROR, message=f"invalid json: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise JSONRPCProtocolError(
            code=ErrorCodes.INVALID_REQUEST,
            message="json-rpc payload must be an object",
        )

    if parsed.get("jsonrpc") != "2.0":
        raise JSONRPCProtocolError(
            code=ErrorCodes.INVALID_REQUEST,
            message="jsonrpc must be '2.0'",
            request_id=_coerce_request_id(parsed.get("id")),
        )

    return parsed


def build_success_response(request_id: str | int, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": _to_jsonable(result)}


def build_error_response(
    *,
    request_id: str | int | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": int(code), "message": str(message)},
    }
    if data is not None:
        payload["error"]["data"] = _to_jsonable(data)
    return payload


def build_event_notification(event: Any) -> dict[str, Any]:
    params = _to_jsonable(event)
    if isinstance(params, dict):
        params = {"event_type": type(event).__name__, **params}
    else:
        params = {"event_type": type(event).__name__, "value": params}
    return {"jsonrpc": "2.0", "method": "event", "params": params}


def _coerce_request_id(raw_value: Any) -> str | int | None:
    if isinstance(raw_value, str | int):
        return raw_value
    return None


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized[str(key)] = _to_jsonable(item)
        return normalized
    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]
    return str(value)
