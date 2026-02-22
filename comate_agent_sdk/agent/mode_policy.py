from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


_RULE_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$")
_PLAN_ROOT = (Path.home() / ".agent" / "plans").expanduser().resolve()


@dataclass(frozen=True, slots=True)
class PermissionRule:
    tool: str
    selector: str
    raw: str


def _parse_rule(raw: str) -> PermissionRule | None:
    text = str(raw or "").strip()
    if not text:
        return None
    match = _RULE_PATTERN.match(text)
    if match is None:
        return None
    tool = match.group(1).strip()
    selector = match.group(2).strip()
    if not tool:
        return None
    return PermissionRule(tool=tool, selector=selector, raw=text)


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except Exception:
        return path.expanduser().absolute()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _normalize_pattern(selector: str, *, project_root: Path) -> str:
    text = selector.strip()
    if not text:
        return "*"
    if text.startswith("~"):
        return str(_safe_resolve(Path(text)))
    if text.startswith("/"):
        return str(_safe_resolve(Path(text)))
    return str(_safe_resolve(project_root / text))


def _extract_primary_path(
    *,
    tool_name: str,
    args: dict,
    project_root: Path,
) -> Path | None:
    key_by_tool = {
        "Read": "file_path",
        "Write": "file_path",
        "Edit": "file_path",
        "MultiEdit": "file_path",
        "Grep": "path",
        "Glob": "path",
        "LS": "path",
    }
    key = key_by_tool.get(tool_name)
    if key is None:
        return None

    raw_value = args.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        return _safe_resolve(project_root)

    raw_path = raw_value.strip()
    if raw_path.startswith("~"):
        return _safe_resolve(Path(raw_path))
    if Path(raw_path).is_absolute():
        return _safe_resolve(Path(raw_path))
    return _safe_resolve(project_root / raw_path)


def _matches_rule(
    *,
    rule: PermissionRule,
    tool_name: str,
    args: dict,
    project_root: Path,
) -> bool:
    if rule.tool != "*" and rule.tool != tool_name:
        return False

    selector = rule.selector.strip()
    if not selector:
        return True

    if tool_name == "Bash":
        raw_args = args.get("args")
        if not isinstance(raw_args, list) or not raw_args:
            return False
        cmd = str(raw_args[0]).strip()
        tail = " ".join(str(v).strip() for v in raw_args[1:] if str(v).strip())
        value = f"{cmd}:{tail}" if tail else cmd
        if ":" in selector:
            return fnmatch.fnmatchcase(value, selector)
        return fnmatch.fnmatchcase(cmd, selector)

    path = _extract_primary_path(tool_name=tool_name, args=args, project_root=project_root)
    if path is None:
        return False
    pattern = _normalize_pattern(selector, project_root=project_root)
    return fnmatch.fnmatchcase(str(path), pattern)


def evaluate_tool_permission(
    agent: "AgentRuntime",
    *,
    tool_name: str,
    args: dict,
) -> tuple[bool, str | None]:
    options = getattr(agent, "options", None)
    project_root_raw = getattr(options, "project_root", None)
    project_root = (project_root_raw or Path.cwd()).expanduser().resolve()

    mode_snapshot = getattr(agent, "_active_mode_snapshot", None)
    if mode_snapshot is None:
        get_mode = getattr(agent, "get_mode", None)
        if callable(get_mode):
            try:
                mode_snapshot = get_mode()
            except Exception:
                mode_snapshot = "act"
        else:
            mode_snapshot = "act"
    mode = str(mode_snapshot).strip().lower()
    allow_rules = [
        rule
        for raw in (getattr(options, "permission_rules_allow", None) or [])
        if (rule := _parse_rule(raw)) is not None
    ]
    deny_rules = [
        rule
        for raw in (getattr(options, "permission_rules_deny", None) or [])
        if (rule := _parse_rule(raw)) is not None
    ]

    if mode == "plan":
        if tool_name in {"Bash", "Write"}:
            return (
                False,
                f"ToolUnavailableInCurrentMode: {tool_name} is disabled in Plan Mode.",
            )

        if tool_name in {"Edit", "MultiEdit"}:
            target = _extract_primary_path(
                tool_name=tool_name,
                args=args,
                project_root=project_root,
            )
            if target is None:
                return (
                    False,
                    f"ToolUnavailableInCurrentMode: {tool_name} requires file_path in Plan Mode.",
                )
            builtin_allowed = _is_under(target, _PLAN_ROOT)
            custom_allowed = any(
                _matches_rule(
                    rule=rule,
                    tool_name=tool_name,
                    args=args,
                    project_root=project_root,
                )
                for rule in allow_rules
            )
            if not (builtin_allowed or custom_allowed):
                return (
                    False,
                    (
                        "ToolUnavailableInCurrentMode: "
                        f"{tool_name} is only allowed under {str(_PLAN_ROOT)} in Plan Mode."
                    ),
                )

    for rule in deny_rules:
        if _matches_rule(
            rule=rule,
            tool_name=tool_name,
            args=args,
            project_root=project_root,
        ):
            return (
                False,
                f"Tool use denied by permissions rule: {rule.raw}",
            )

    return True, None
