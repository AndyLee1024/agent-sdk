from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, cast

PermissionDecision = Literal["allow", "ask", "deny"]
StopDecision = Literal["block"]
HookHandlerType = Literal["command", "python"]


class HookEventName(str, Enum):
    SESSION_START = "SessionStart"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"
    STOP = "Stop"
    SESSION_END = "SessionEnd"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"


TOOL_HOOK_EVENTS = frozenset(
    {
        HookEventName.PRE_TOOL_USE.value,
        HookEventName.POST_TOOL_USE.value,
        HookEventName.POST_TOOL_USE_FAILURE.value,
    }
)

HOOK_EVENT_NAMES = frozenset(e.value for e in HookEventName)


PythonHookCallback = Callable[
    ["HookInput"],
    "HookResult | dict[str, Any] | None | Awaitable[HookResult | dict[str, Any] | None]",
]


@dataclass(slots=True)
class HookHandlerSpec:
    type: HookHandlerType
    command: str | None = None
    timeout: int = 10
    callback: PythonHookCallback | None = None
    name: str = ""
    source: str = ""


@dataclass(slots=True)
class HookMatcherGroup:
    matcher: str | None = "*"
    hooks: list[HookHandlerSpec] = field(default_factory=list)
    source: str = ""


@dataclass(slots=True)
class HookConfig:
    events: dict[str, list[HookMatcherGroup]] = field(default_factory=dict)

    def groups_for(self, event_name: str) -> list[HookMatcherGroup]:
        return list(self.events.get(event_name, []))

    def extend(self, other: "HookConfig") -> None:
        for event_name, groups in other.events.items():
            self.events.setdefault(event_name, []).extend(groups)


@dataclass(slots=True)
class HookInput:
    session_id: str
    cwd: str
    permission_mode: str
    hook_event_name: str

    prompt: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_call_id: str | None = None
    tool_response: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    stop_hook_active: bool | None = None
    subagent_name: str | None = None
    subagent_description: str | None = None
    subagent_status: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "permission_mode": self.permission_mode,
            "hook_event_name": self.hook_event_name,
        }
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.tool_name is not None:
            data["tool_name"] = self.tool_name
        if self.tool_input is not None:
            data["tool_input"] = self.tool_input
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_response is not None:
            data["tool_response"] = self.tool_response
        if self.error is not None:
            data["error"] = self.error
        if self.stop_reason is not None:
            data["stop_reason"] = self.stop_reason
        if self.stop_hook_active is not None:
            data["stop_hook_active"] = self.stop_hook_active
        if self.subagent_name is not None:
            data["subagent_name"] = self.subagent_name
        if self.subagent_description is not None:
            data["subagent_description"] = self.subagent_description
        if self.subagent_status is not None:
            data["subagent_status"] = self.subagent_status
        if self.extra:
            data.update(self.extra)
        return data


@dataclass(slots=True)
class HookResult:
    additional_context: str | None = None
    permission_decision: PermissionDecision | None = None
    updated_input: dict[str, Any] | None = None
    decision: StopDecision | None = None
    reason: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "HookResult":
        permission_decision = raw.get("permissionDecision", raw.get("permission_decision"))
        decision = raw.get("decision")
        return cls(
            additional_context=cast(str | None, raw.get("additionalContext", raw.get("additional_context"))),
            permission_decision=cast(PermissionDecision | None, permission_decision),
            updated_input=cast(dict[str, Any] | None, raw.get("updatedInput", raw.get("updated_input"))),
            decision=cast(StopDecision | None, decision),
            reason=cast(str | None, raw.get("reason")),
        )


@dataclass(slots=True)
class AggregatedHookOutcome:
    event_name: str
    additional_context: str | None = None
    permission_decision: PermissionDecision | None = None
    updated_input: dict[str, Any] | None = None
    decision: StopDecision | None = None
    reason: str | None = None

    @property
    def should_block_stop(self) -> bool:
        return self.decision == "block"


def normalize_hook_event_name(name: str) -> str | None:
    if name in HOOK_EVENT_NAMES:
        return name
    return None


def resolve_project_root(project_root: Path | None) -> Path:
    return (project_root or Path.cwd()).expanduser().resolve()
