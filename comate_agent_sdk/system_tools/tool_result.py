from __future__ import annotations

from typing import Any, Literal

ToolErrorCode = Literal[
    "INVALID_ARGUMENT",
    "NOT_FOUND",
    "IS_DIRECTORY",
    "PERMISSION_DENIED",
    "PATH_ESCAPE",
    "CONFLICT",
    "TIMEOUT",
    "TOOL_NOT_AVAILABLE",
    "RATE_LIMITED",
    "INTERNAL",
]


def ok(
    data: dict[str, Any] | None = None,
    *,
    message: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "data": data or {},
        "message": message,
        "error": None,
        "meta": meta or {},
        "schema_version": 1,
    }


def err(
    code: ToolErrorCode,
    message: str,
    *,
    field_errors: list[dict[str, Any]] | None = None,
    retryable: bool = False,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "data": {},
        "message": None,
        "error": {
            "code": code,
            "message": message,
            "field_errors": field_errors or [],
            "retryable": retryable,
        },
        "meta": meta or {},
        "schema_version": 1,
    }


def is_tool_result_envelope(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        "ok" in value
        and "data" in value
        and "message" in value
        and "error" in value
        and "meta" in value
        and "schema_version" in value
    )
