from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from comate_agent_sdk.agent import ChatSession

from terminal_agent.rpc_protocol import (
    ErrorCodes,
    JSONRPCProtocolError,
    build_error_response,
    build_event_notification,
    build_success_response,
    parse_jsonrpc_message,
)

logger = logging.getLogger(__name__)


class StdioRPCBridge:
    """JSON-RPC 2.0 bridge over stdin/stdout (NDJSON)."""

    def __init__(self, session: ChatSession) -> None:
        self._session = session
        self._write_lock = asyncio.Lock()
        self._closing = False
        self._active_prompt_task: asyncio.Task[dict[str, Any]] | None = None
        self._active_prompt_request_id: str | int | None = None

    async def run(self) -> None:
        while not self._closing:
            raw_line = await asyncio.to_thread(sys.stdin.readline)
            if raw_line == "":
                break

            line = raw_line.strip()
            if not line:
                continue

            await self._handle_incoming_line(line)

        await self._cancel_active_prompt()

    async def _handle_incoming_line(self, line: str) -> None:
        try:
            message = parse_jsonrpc_message(line)
        except JSONRPCProtocolError as exc:
            await self._send(
                build_error_response(
                    request_id=exc.request_id,
                    code=exc.code,
                    message=exc.message,
                    data=exc.data,
                )
            )
            return

        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}

        if method is None:
            await self._send(
                build_error_response(
                    request_id=request_id if isinstance(request_id, str | int) else None,
                    code=ErrorCodes.INVALID_REQUEST,
                    message="response payload is not supported on this endpoint",
                )
            )
            return

        if method == "initialize":
            await self._handle_initialize(request_id)
            return
        if method == "prompt":
            await self._handle_prompt(request_id, params)
            return
        if method == "cancel":
            await self._handle_cancel(request_id)
            return
        if method == "replay":
            await self._handle_replay(request_id)
            return

        await self._send(
            build_error_response(
                request_id=request_id if isinstance(request_id, str | int) else None,
                code=ErrorCodes.METHOD_NOT_FOUND,
                message=f"method not found: {method}",
            )
        )

    async def _handle_initialize(self, request_id: Any) -> None:
        if not isinstance(request_id, str | int):
            await self._send(
                build_error_response(
                    request_id=None,
                    code=ErrorCodes.INVALID_REQUEST,
                    message="initialize requires request id",
                )
            )
            return

        await self._send(
            build_success_response(
                request_id,
                {
                    "status": "ok",
                    "protocol_version": "1.0",
                    "session_id": self._session.session_id,
                },
            )
        )

    async def _handle_prompt(self, request_id: Any, params: Any) -> None:
        if not isinstance(request_id, str | int):
            await self._send(
                build_error_response(
                    request_id=None,
                    code=ErrorCodes.INVALID_REQUEST,
                    message="prompt requires request id",
                )
            )
            return

        if self._active_prompt_task is not None and not self._active_prompt_task.done():
            await self._send(
                build_error_response(
                    request_id=request_id,
                    code=ErrorCodes.INVALID_STATE,
                    message="prompt already running",
                )
            )
            return

        if not isinstance(params, dict):
            await self._send(
                build_error_response(
                    request_id=request_id,
                    code=ErrorCodes.INVALID_PARAMS,
                    message="params must be an object",
                )
            )
            return

        user_input = params.get("user_input")
        if not isinstance(user_input, str) or not user_input.strip():
            await self._send(
                build_error_response(
                    request_id=request_id,
                    code=ErrorCodes.INVALID_PARAMS,
                    message="params.user_input must be a non-empty string",
                )
            )
            return

        task = asyncio.create_task(
            self._run_prompt_stream(user_input),
            name=f"rpc-prompt-{request_id}",
        )
        self._active_prompt_task = task
        self._active_prompt_request_id = request_id
        asyncio.create_task(
            self._finalize_prompt_task(task=task, request_id=request_id),
            name=f"rpc-prompt-finalize-{request_id}",
        )

    async def _handle_cancel(self, request_id: Any) -> None:
        if not isinstance(request_id, str | int):
            await self._send(
                build_error_response(
                    request_id=None,
                    code=ErrorCodes.INVALID_REQUEST,
                    message="cancel requires request id",
                )
            )
            return

        cancelled = await self._cancel_active_prompt()
        await self._send(build_success_response(request_id, {"cancelled": cancelled}))

    async def _handle_replay(self, request_id: Any) -> None:
        if not isinstance(request_id, str | int):
            await self._send(
                build_error_response(
                    request_id=None,
                    code=ErrorCodes.INVALID_REQUEST,
                    message="replay requires request id",
                )
            )
            return

        await self._send(
            build_success_response(
                request_id,
                {
                    "supported": False,
                    "reason": "replay is not implemented for stdio bridge",
                },
            )
        )

    async def _run_prompt_stream(self, user_input: str) -> dict[str, Any]:
        waiting_for_input = False
        stop_reason = "completed"
        try:
            async for event in self._session.query_stream(user_input):
                await self._send(build_event_notification(event))
                reason = getattr(event, "reason", None)
                if reason == "waiting_for_input":
                    waiting_for_input = True
                    stop_reason = "waiting_for_input"
                if reason == "waiting_for_plan_approval":
                    stop_reason = "waiting_for_plan_approval"
        except asyncio.CancelledError:
            stop_reason = "cancelled"
            raise
        except Exception as exc:
            logger.exception("rpc prompt stream failed")
            raise RuntimeError(f"stream failed: {exc}") from exc

        if stop_reason == "waiting_for_plan_approval":
            status = "waiting_for_plan_approval"
        else:
            status = "waiting_for_input" if waiting_for_input else "completed"
        return {"status": status, "stop_reason": stop_reason}

    async def _finalize_prompt_task(
        self,
        *,
        task: asyncio.Task[dict[str, Any]],
        request_id: str | int,
    ) -> None:
        try:
            result = await task
            await self._send(build_success_response(request_id, result))
        except asyncio.CancelledError:
            await self._send(
                build_success_response(
                    request_id,
                    {"status": "cancelled", "stop_reason": "cancelled"},
                )
            )
        except Exception as exc:
            await self._send(
                build_error_response(
                    request_id=request_id,
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=str(exc),
                )
            )
        finally:
            if self._active_prompt_task is task:
                self._active_prompt_task = None
                self._active_prompt_request_id = None

    async def _cancel_active_prompt(self) -> bool:
        task = self._active_prompt_task
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    async def _send(self, payload: dict[str, Any]) -> None:
        encoded = f"{self._dump_json(payload)}\n"
        async with self._write_lock:
            await asyncio.to_thread(sys.stdout.write, encoded)
            await asyncio.to_thread(sys.stdout.flush)

    @staticmethod
    def _dump_json(payload: dict[str, Any]) -> str:
        import json

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
