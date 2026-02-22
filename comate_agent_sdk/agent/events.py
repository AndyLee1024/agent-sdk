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
from datetime import datetime
from typing import Any, Literal

LLMLevel = Literal["LOW", "MID", "HIGH"]


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
	"""Schema-normalized arguments passed to the tool."""

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

	metadata: dict[str, Any] | None = None
	"""Optional metadata for tool-specific data (e.g., diff for Edit/MultiEdit)."""

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

	reason: Literal['completed', 'max_iterations', 'waiting_for_input', 'waiting_for_plan_approval', 'interrupted']
	"""Why the stream stopped."""

	def __str__(self) -> str:
		return f'🛑 Stop: {self.reason}'


@dataclass
class ModeChangedEvent:
	"""Emitted when runtime mode changes."""

	mode: Literal['act', 'plan']
	applies_next_message: bool = True

	def __str__(self) -> str:
		suffix = " (next message)" if self.applies_next_message else ""
		return f'🔁 Mode: {self.mode}{suffix}'


@dataclass
class PlanApprovalRequiredEvent:
	"""Emitted when plan artifact is ready and waiting for user approval."""

	plan_path: str
	summary: str
	execution_prompt: str

	def __str__(self) -> str:
		return f'📋 Plan approval required: {self.plan_path}'


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

	status: Literal['completed', 'error', 'cancelled']
	"""The final status of the step."""

	duration_ms: float = 0.0
	"""Duration of the step in milliseconds."""

	def __str__(self) -> str:
		icon = '✅' if self.status == 'completed' else ('⏹️' if self.status == 'cancelled' else '❌')
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

	status: Literal['completed', 'error', 'timeout', 'cancelled']
	"""Final status for the subagent execution."""

	duration_ms: float = 0.0
	"""Duration in milliseconds."""

	error: str | None = None
	"""Error message if status is error/timeout."""

	def __str__(self) -> str:
		icon = '✅' if self.status == 'completed' else ('⏹️' if self.status == 'cancelled' else '❌')
		return f'{icon} Subagent stop: {self.subagent_name} ({self.duration_ms:.0f}ms)'


@dataclass
class UsageDeltaEvent:
	"""Emitted whenever token usage receives a new entry."""

	source: str
	"""Usage source (e.g., 'agent', 'subagent:Explorer:tc_1', 'subagent:Explorer:tc_1:webfetch')."""

	model: str
	"""Model identifier for this usage entry."""

	level: str | None = None
	"""Optional LLM level tag."""

	delta_prompt_tokens: int = 0
	"""Prompt token delta from this usage entry."""

	delta_prompt_cached_tokens: int = 0
	"""Prompt cached token delta from this usage entry."""

	delta_completion_tokens: int = 0
	"""Completion token delta from this usage entry."""

	delta_total_tokens: int = 0
	"""Total token delta from this usage entry."""

	def __str__(self) -> str:
		level = f' [{self.level}]' if self.level else ''
		return (
			f'📊 Usage{level} {self.source}: +{self.delta_total_tokens} tok '
			f'(p={self.delta_prompt_tokens}, c={self.delta_completion_tokens})'
		)


@dataclass
class SubagentProgressEvent:
	"""Emitted during subagent execution with live elapsed time and token usage."""

	tool_call_id: str
	"""The Task tool_call_id for correlation."""

	subagent_name: str
	"""Subagent name."""

	description: str = ''
	"""Subagent task description."""

	status: Literal['running', 'completed', 'error', 'timeout', 'cancelled'] = 'running'
	"""Current task status."""

	elapsed_ms: float = 0.0
	"""Elapsed time in milliseconds."""

	tokens: int = 0
	"""Subagent accumulated tokens for this task call."""

	def __str__(self) -> str:
		return (
			f'⏱️ Subagent progress: {self.subagent_name} '
			f'({self.status}, {self.elapsed_ms:.0f}ms, {self.tokens} tok)'
		)


@dataclass
class SubagentToolCallEvent:
	"""Emitted when a tool call occurs inside a subagent (TaskStream only)."""

	parent_tool_call_id: str
	"""The tool_call_id of the parent Task/TaskStream call."""

	subagent_name: str
	"""The name of the subagent making the tool call."""

	tool: str
	"""The name of the tool being called."""

	args: dict[str, Any]
	"""The arguments passed to the tool."""

	tool_call_id: str
	"""The tool_call_id within the subagent's context."""

	def __str__(self) -> str:
		return f'🔧 Subagent tool call: {self.subagent_name}.{self.tool}'


@dataclass
class SubagentToolResultEvent:
	"""Emitted when a tool result is received inside a subagent (TaskStream only)."""

	parent_tool_call_id: str
	"""The tool_call_id of the parent Task/TaskStream call."""

	subagent_name: str
	"""The name of the subagent that called the tool."""

	tool: str
	"""The name of the tool that was called."""

	tool_call_id: str
	"""The tool_call_id within the subagent's context."""

	is_error: bool = False
	"""Whether the tool call resulted in an error."""

	duration_ms: float = 0.0
	"""Duration of the tool call in milliseconds."""

	def __str__(self) -> str:
		icon = '✅' if not self.is_error else '❌'
		return f'{icon} Subagent tool result: {self.subagent_name}.{self.tool}'


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


@dataclass
class UserQuestionEvent:
	"""Emitted when agent requests user input via AskUserQuestion."""

	questions: list[dict[str, Any]]
	"""The questions to present to the user."""

	tool_call_id: str
	"""The tool call ID for correlation."""

	def __str__(self) -> str:
		count = len(self.questions)
		first_q = self.questions[0].get('question', '') if count > 0 else ''
		preview = first_q[:60] + '...' if len(first_q) > 60 else first_q
		return f'❓ Question: {preview} ({count} question{"s" if count > 1 else ""})'


@dataclass
class TodoUpdatedEvent:
	"""Emitted when the todo list is updated via TodoWrite tool.

	This event contains the full todo list state and is intended for UI rendering.
	"""

	todos: list[dict[str, Any]]
	"""The complete list of todo items with their current state."""

	def __str__(self) -> str:
		total = len(self.todos)
		completed = sum(1 for t in self.todos if t.get('status') == 'completed')
		return f'📋 Todo updated: {completed}/{total} completed'


@dataclass
class PreCompactEvent:
	"""Emitted before context compaction is performed."""

	current_tokens: int
	"""Current token count before compaction."""

	threshold: int
	"""The threshold that triggered compaction."""

	trigger: Literal['check', 'precheck']
	"""Which trigger initiated the compaction: 'check' for post-LLM, 'precheck' for post-tool."""

	def __str__(self) -> str:
		return f'📦 Pre-compact: {self.current_tokens} tokens (threshold: {self.threshold}, trigger: {self.trigger})'


@dataclass
class CompactionMetaEvent:
	"""调试用途的压缩元事件（默认关闭）。"""

	phase: Literal['selective_start', 'selective_done', 'summary_start', 'summary_done', 'rollback']
	"""压缩阶段。"""

	tokens_before: int
	"""阶段前 token 数。"""

	tokens_after: int
	"""阶段后 token 数。"""

	tool_blocks_kept: int = 0
	"""工具块保留数。"""

	tool_blocks_dropped: int = 0
	"""工具块删除数。"""

	tool_calls_truncated: int = 0
	"""tool_call.arguments 截断数。"""

	tool_results_truncated: int = 0
	"""tool_result.content 截断数。"""

	reason: str = ''
	"""阶段原因说明。"""

	def __str__(self) -> str:
		return (
			f'🧩 CompactionMeta phase={self.phase} '
			f'{self.tokens_before}->{self.tokens_after} '
			f'kept={self.tool_blocks_kept} dropped={self.tool_blocks_dropped} '
			f'tc={self.tool_calls_truncated} tr={self.tool_results_truncated} '
			f'reason={self.reason}'
		)


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
	| UsageDeltaEvent
	| SubagentProgressEvent
	| HiddenUserMessageEvent
	| UserQuestionEvent
	| TodoUpdatedEvent
	| PreCompactEvent
	| CompactionMetaEvent
)


@dataclass
class LLMSwitchedEvent:
	"""Emitted when the LLM level is switched during a session.

	This event is triggered when `session.set_level()` is called,
	indicating that the next query will use a different LLM.
	"""

	previous_level: LLMLevel | None
	"""The previous LLM level (None if not set)."""

	new_level: LLMLevel
	"""The new LLM level being switched to."""

	previous_model: str | None
	"""The name of the previous model (None if not set)."""

	new_model: str | None
	"""The name of the new model (None if level not configured)."""

	timestamp: datetime = field(default_factory=datetime.now)
	"""When the switch occurred."""

	def __str__(self) -> str:
		prev = self.previous_level or "unset"
		new = self.new_level
		return f'🔄 LLM switched: {prev} → {new}'


# Union type for all events (must be at the end after all event classes are defined)
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
	| UsageDeltaEvent
	| SubagentProgressEvent
	| SubagentToolCallEvent
	| SubagentToolResultEvent
	| HiddenUserMessageEvent
	| UserQuestionEvent
	| TodoUpdatedEvent
	| PreCompactEvent
	| CompactionMetaEvent
	| LLMSwitchedEvent
)
