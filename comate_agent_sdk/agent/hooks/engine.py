from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comate_agent_sdk.agent.hooks.models import (
    AggregatedHookOutcome,
    HookConfig,
    HookHandlerSpec,
    HookInput,
    HookMatcherGroup,
    HookResult,
    TOOL_HOOK_EVENTS,
    normalize_hook_event_name,
    resolve_project_root,
)

logger = logging.getLogger("comate_agent_sdk.agent.hooks")


@dataclass(slots=True)
class _HandlerExecution:
    result: HookResult | None = None
    blocked_by_exit_code: bool = False
    block_reason: str | None = None


class HookEngine:
    def __init__(
        self,
        *,
        config: HookConfig | None,
        project_root: Path | None,
        session_id: str,
        permission_mode: str = "default",
        transcript_path: str | None = None,
    ) -> None:
        self._config = config or HookConfig()
        self._project_root = resolve_project_root(project_root)
        self._session_id = session_id
        self._permission_mode = permission_mode
        self._transcript_path = transcript_path
        self._matcher_cache: dict[str, re.Pattern[str]] = {}
        self._runtime_python_groups: dict[str, list[tuple[int, HookMatcherGroup]]] = {}

    def register_python_hook(
        self,
        *,
        event_name: str,
        callback: Any,
        matcher: str = "*",
        order: int = 0,
        name: str | None = None,
    ) -> None:
        normalized = normalize_hook_event_name(event_name)
        if normalized is None:
            raise ValueError(f"Unknown hook event name: {event_name}")

        spec = HookHandlerSpec(
            type="python",
            callback=callback,
            source="runtime",
            name=name or getattr(callback, "__name__", "python_hook"),
        )
        group = HookMatcherGroup(matcher=matcher or "*", hooks=[spec], source="runtime")
        self._runtime_python_groups.setdefault(normalized, []).append((int(order), group))

    async def run_event(
        self,
        event_name: str,
        hook_input: HookInput,
    ) -> AggregatedHookOutcome:
        normalized = normalize_hook_event_name(event_name)
        if normalized is None:
            return AggregatedHookOutcome(event_name=event_name)

        groups = self._config.groups_for(normalized)
        runtime_groups = [
            group
            for _, group in sorted(
                self._runtime_python_groups.get(normalized, []),
                key=lambda item: item[0],
            )
        ]
        groups.extend(runtime_groups)

        if not groups:
            return AggregatedHookOutcome(event_name=normalized)

        additional_context_parts: list[str] = []
        ask_seen = False
        ask_reason: str | None = None
        allow_seen = False
        allow_reason: str | None = None
        updated_input: dict[str, Any] | None = None

        outcome = AggregatedHookOutcome(event_name=normalized)
        for group in groups:
            if not self._matches_group(group, normalized, hook_input.tool_name):
                continue

            for handler in group.hooks:
                execution = await self._execute_handler(handler, hook_input)
                if execution.blocked_by_exit_code:
                    block_reason = execution.block_reason or "blocked by command hook"
                    if normalized == "PreToolUse":
                        outcome.permission_decision = "deny"
                        outcome.reason = block_reason
                        outcome.updated_input = None
                        outcome.additional_context = _join_context(additional_context_parts)
                        return outcome
                    if normalized == "Stop":
                        outcome.decision = "block"
                        outcome.reason = block_reason
                        outcome.additional_context = _join_context(additional_context_parts)
                        return outcome
                    logger.warning(
                        f"hook exit code 2 ignored for event={normalized}, handler={handler.name or handler.source}"
                    )
                    continue

                result = execution.result
                if result is None:
                    continue

                if result.additional_context:
                    additional_context_parts.append(result.additional_context)

                if normalized == "PreToolUse":
                    if result.updated_input is not None:
                        updated_input = result.updated_input

                    decision = result.permission_decision
                    if decision == "deny":
                        outcome.permission_decision = "deny"
                        outcome.reason = result.reason or "blocked by hook"
                        outcome.updated_input = None
                        outcome.additional_context = _join_context(additional_context_parts)
                        return outcome
                    if decision == "ask":
                        ask_seen = True
                        if ask_reason is None and result.reason:
                            ask_reason = result.reason
                    if decision == "allow":
                        allow_seen = True
                        if allow_reason is None and result.reason:
                            allow_reason = result.reason
                    continue

                if normalized == "Stop" and result.decision == "block":
                    outcome.decision = "block"
                    outcome.reason = result.reason or "blocked by stop hook"
                    outcome.additional_context = _join_context(additional_context_parts)
                    return outcome

        if normalized == "PreToolUse":
            if ask_seen:
                outcome.permission_decision = "ask"
                outcome.reason = ask_reason
            elif allow_seen:
                outcome.permission_decision = "allow"
                outcome.reason = allow_reason
            outcome.updated_input = updated_input if outcome.permission_decision != "deny" else None

        outcome.additional_context = _join_context(additional_context_parts)
        return outcome

    def _matches_group(self, group: HookMatcherGroup, event_name: str, tool_name: str | None) -> bool:
        if event_name not in TOOL_HOOK_EVENTS:
            return True
        matcher = (group.matcher or "*").strip()
        if matcher in {"", "*"}:
            return True
        if tool_name is None:
            return False
        pattern = self._matcher_cache.get(matcher)
        if pattern is None:
            try:
                pattern = re.compile(matcher)
            except re.error as exc:
                logger.warning(f"Invalid hook matcher regex={matcher!r}: {exc}")
                return False
            self._matcher_cache[matcher] = pattern
        return bool(pattern.search(tool_name))

    async def _execute_handler(self, handler: HookHandlerSpec, hook_input: HookInput) -> _HandlerExecution:
        if handler.type == "python":
            return await self._execute_python_handler(handler, hook_input)
        return await self._execute_command_handler(handler, hook_input)

    async def _execute_python_handler(self, handler: HookHandlerSpec, hook_input: HookInput) -> _HandlerExecution:
        callback = handler.callback
        if callback is None:
            return _HandlerExecution()

        timeout = max(1, int(handler.timeout or 10))

        async def _run_callback() -> Any:
            if inspect.iscoroutinefunction(callback):
                result = callback(hook_input)
                if inspect.isawaitable(result):
                    return await result
                return result

            result = await asyncio.to_thread(callback, hook_input)
            if inspect.isawaitable(result):
                return await result
            return result

        try:
            result = await asyncio.wait_for(_run_callback(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                f"python hook 超时 event={hook_input.hook_event_name}, "
                f"handler={handler.name or callback}, timeout={timeout}s"
            )
            return _HandlerExecution()
        except Exception as exc:
            logger.warning(f"python hook 执行失败 {handler.name or callback}: {exc}", exc_info=True)
            return _HandlerExecution()

        parsed = _coerce_hook_result(result)
        if parsed is None:
            return _HandlerExecution()
        return _HandlerExecution(result=parsed)

    async def _execute_command_handler(self, handler: HookHandlerSpec, hook_input: HookInput) -> _HandlerExecution:
        command = (handler.command or "").strip()
        if not command:
            return _HandlerExecution()

        env = os.environ.copy()
        env["COMATE_PROJECT_DIR"] = str(self._project_root)
        env["COMATE_SESSION_ID"] = self._session_id
        env["COMATE_HOOK_EVENT"] = hook_input.hook_event_name
        env["COMATE_PERMISSION_MODE"] = hook_input.permission_mode or self._permission_mode
        if self._transcript_path:
            env["COMATE_TRANSCRIPT_PATH"] = self._transcript_path

        payload = hook_input.to_dict()
        stdin_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=str(self._project_root),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                proc.communicate(stdin_data),
                timeout=max(1, int(handler.timeout or 10)),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning(
                f"command hook 超时 event={hook_input.hook_event_name}, command={command}, timeout={handler.timeout}s"
            )
            return _HandlerExecution()

        exit_code = int(proc.returncode or 0)
        stdout_text = stdout_data.decode("utf-8", errors="replace").strip()
        stderr_text = stderr_data.decode("utf-8", errors="replace").strip()

        if exit_code == 2:
            return _HandlerExecution(
                blocked_by_exit_code=True,
                block_reason=stderr_text or "blocked by hook command exit code 2",
            )

        if exit_code != 0:
            logger.warning(
                f"command hook 非阻断错误 event={hook_input.hook_event_name}, exit_code={exit_code}, stderr={stderr_text}"
            )
            return _HandlerExecution()

        if not stdout_text:
            return _HandlerExecution()

        try:
            raw = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"command hook stdout JSON 解析失败 event={hook_input.hook_event_name}: {exc}; stdout={stdout_text[:500]!r}"
            )
            return _HandlerExecution()

        if not isinstance(raw, dict):
            logger.warning(
                f"command hook stdout 必须是 JSON object event={hook_input.hook_event_name}, got={type(raw).__name__}"
            )
            return _HandlerExecution()

        return _HandlerExecution(result=HookResult.from_mapping(raw))


def _coerce_hook_result(raw: Any) -> HookResult | None:
    if raw is None:
        return None
    if isinstance(raw, HookResult):
        return raw
    if isinstance(raw, dict):
        return HookResult.from_mapping(raw)
    logger.warning(f"python hook 返回值不支持，已忽略: {type(raw).__name__}")
    return None


def _join_context(parts: list[str]) -> str | None:
    clean = [part.strip() for part in parts if part and part.strip()]
    if not clean:
        return None
    return "\n".join(clean)
