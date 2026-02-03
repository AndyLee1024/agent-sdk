"""
Event types for agent streaming.

These events are yielded by `agent.query_stream()` to provide visibility
into the agent's execution.

Usage:
    async for event in agent.query_stream("do something"):
        match event:
            case ToolCallEvent(tool=name, args=args):
                print(f"Calling {name} with {args}")
            case ToolResultEvent(tool=name, result=result):
                print(f"{name} returned: {result}")
            case TextEvent(content=text):
                print(f"Assistant: {text}")
            case StepStartEvent(step_id=id, title=title):
                print(f"Step started: {title}")
            case StepCompleteEvent(step_id=id, status=status):
                print(f"Step {status}")
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SessionInitEvent:
	"""Emitted once when a chat/session is initialized.

	This event is intended to return a stable session_id (UUID string) to the caller
	for resume/fork workflows.
	"""

	session_id: str
	"""Session id (UUID string)."""

	def __str__(self) -> str:
		return f"Session initialized: {self.session_id}"


@dataclass
class TextEvent:
	"""Emitted when the assistant produces text content."""

	content: str
	"""The text content from the assistant."""

	def __str__(self) -> str:
		preview = self.content[:100] + '...' if len(self.content) > 100 else self.content
		return f'💬 {preview}'


@dataclass
class ThinkingEvent:
	"""Emitted when the model produces thinking/reasoning content."""

	content: str
	"""The thinking content."""

	def __str__(self) -> str:
		preview = self.content[:80] + '...' if len(self.content) > 80 else self.content
		return f'🧠 {preview}'


@dataclass
class ToolCallEvent:
	"""Emitted when the assistant calls a tool."""

	tool: str
	"""The name of the tool being called."""

	args: dict[str, Any]
	"""The arguments passed to the tool."""

	tool_call_id: str
	"""The unique ID of this tool call."""

	display_name: str = ''
	"""Human-readable description of the tool call (e.g., 'Browsing https://...')."""

	def __str__(self) -> str:
		if self.display_name:
			return f'🔧 {self.display_name}'
		args_str = json.dumps(self.args, default=str)
		if len(args_str) > 80:
			args_str = args_str[:77] + '...'
		return f'🔧 {self.tool}({args_str})'


@dataclass
class ToolResultEvent:
	"""Emitted when a tool returns a result."""

	tool: str
	"""The name of the tool that was called."""

	result: str
	"""The result returned by the tool."""

	tool_call_id: str
	"""The unique ID of the tool call this result corresponds to."""

	is_error: bool = False
	"""Whether the tool execution resulted in an error."""

	screenshot_base64: str | None = None
	"""Base64-encoded screenshot if this was a browser tool."""

	def __str__(self) -> str:
		prefix = '❌' if self.is_error else '✓'
		preview = self.result[:80] + '...' if len(self.result) > 80 else self.result
		screenshot_indicator = ' 📸' if self.screenshot_base64 else ''
		return f'   {prefix} {self.tool}: {preview}{screenshot_indicator}'


@dataclass
class StopEvent:
	"""Emitted when the agent stops streaming events.

	This event is a pure termination signal and does not contain the final text.
	Final assistant text (if any) is emitted via `TextEvent`.
	"""

	reason: Literal['completed', 'max_iterations']
	"""Why the stream stopped."""

	def __str__(self) -> str:
		return f'🛑 Stop: {self.reason}'


# === New events for BU-like UI ===


@dataclass
class MessageStartEvent:
	"""Emitted when a new message starts (user or assistant)."""

	message_id: str
	"""Unique ID for this message."""

	role: Literal['user', 'assistant']
	"""The role of the message sender."""

	def __str__(self) -> str:
		return f'📨 Message started ({self.role})'


@dataclass
class MessageCompleteEvent:
	"""Emitted when a message is complete."""

	message_id: str
	"""The ID of the completed message."""

	content: str
	"""The full message content."""

	def __str__(self) -> str:
		preview = self.content[:80] + '...' if len(self.content) > 80 else self.content
		return f'📩 Message complete: {preview}'


@dataclass
class StepStartEvent:
	"""Emitted when the agent starts a logical step (tool execution group)."""

	step_id: str
	"""Unique ID for this step (typically same as tool_call_id)."""

	title: str
	"""Human-readable title for this step (e.g., 'Navigate to website')."""

	step_number: int = 0
	"""Sequential step number within the current query."""

	def __str__(self) -> str:
		return f'▶️  Step {self.step_number}: {self.title}'


@dataclass
class StepCompleteEvent:
	"""Emitted when a step completes."""

	step_id: str
	"""The ID of the completed step."""

	status: Literal['completed', 'error']
	"""The final status of the step."""

	duration_ms: float = 0.0
	"""Duration of the step in milliseconds."""

	def __str__(self) -> str:
		icon = '✅' if self.status == 'completed' else '❌'
		return f'{icon} Step complete ({self.duration_ms:.0f}ms)'


@dataclass
class SubagentStartEvent:
	"""Emitted when a subagent starts executing (Task tool)."""

	tool_call_id: str
	"""The tool_call_id for the Task call that launched this subagent."""

	subagent_name: str
	"""Subagent name (subagent_type)."""

	description: str = ''
	"""Optional short description for UI display."""

	def __str__(self) -> str:
		desc = f' · {self.description}' if self.description else ''
		return f'🤖 Subagent start: {self.subagent_name}{desc}'


@dataclass
class SubagentStopEvent:
	"""Emitted when a subagent finishes executing (Task tool)."""

	tool_call_id: str
	"""The tool_call_id for the Task call that launched this subagent."""

	subagent_name: str
	"""Subagent name (subagent_type)."""

	status: Literal['completed', 'error', 'timeout']
	"""Final status for the subagent execution."""

	duration_ms: float = 0.0
	"""Duration in milliseconds."""

	error: str | None = None
	"""Error message if status is error/timeout."""

	def __str__(self) -> str:
		icon = '✅' if self.status == 'completed' else '❌'
		return f'{icon} Subagent stop: {self.subagent_name} ({self.duration_ms:.0f}ms)'


@dataclass
class HiddenUserMessageEvent:
	"""Emitted when the agent injects a hidden user message (ex: incomplete todos prompt).
	Hidden messages are saved to history and sent to the LLM but not displayed in the UI.
	"""

	content: str
	"""The content of the hidden user message."""

	def __str__(self) -> str:
		preview = self.content[:80] + '...' if len(self.content) > 80 else self.content
		return f'👻 Hidden: {preview}'


# Union type for all events
AgentEvent = (
	SessionInitEvent
	| TextEvent
	| ThinkingEvent
	| ToolCallEvent
	| ToolResultEvent
	| StopEvent
	| MessageStartEvent
	| MessageCompleteEvent
	| StepStartEvent
	| StepCompleteEvent
	| SubagentStartEvent
	| SubagentStopEvent
	| HiddenUserMessageEvent
)
