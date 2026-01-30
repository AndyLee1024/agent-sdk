"""
Simple agentic loop implementation with native tool calling.

Usage:
    from bu_agent_sdk.llm import ChatOpenAI
    from bu_agent_sdk.tools import tool
    from bu_agent_sdk import Agent

    @tool("Search the web")
    async def search(query: str) -> str:
        return f"Results for {query}"

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        tools=[search],
    )

    response = await agent.query("Find information about Python")
    follow_up = await agent.query("Tell me more about that")

    # Compaction is enabled by default with dynamic thresholds based on model limits
    from bu_agent_sdk.agent.compaction import CompactionConfig

    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        tools=[search],
        # Custom threshold ratio (default is 0.80 = 80% of model's context window)
        compaction=CompactionConfig(threshold_ratio=0.70),
        # Or disable compaction entirely:
        # compaction=CompactionConfig(enabled=False),
    )

    # Access usage statistics:
    summary = await agent.usage
    print(f"Total tokens: {summary.total_tokens}")
    print(f"Total cost: ${summary.total_cost:.4f}")
"""


class TaskComplete(Exception):
    """Exception raised when a task is completed via the done tool.

    This provides explicit task completion signaling instead of relying on
    the absence of tool calls. The agent loop catches this exception and
    returns the completion message.

    Attributes:
        message: A description of why the task is complete and what was accomplished.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from bu_agent_sdk.agent.compaction import CompactionConfig, CompactionService
from bu_agent_sdk.context import ContextIR, SelectiveCompactionPolicy
from bu_agent_sdk.context.fs import ContextFileSystem
from bu_agent_sdk.context.offload import OffloadPolicy

logger = logging.getLogger("bu_agent_sdk.agent")
from bu_agent_sdk.agent.events import (
    AgentEvent,
    FinalResponseEvent,
    HiddenUserMessageEvent,
    StepCompleteEvent,
    StepStartEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from bu_agent_sdk.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from bu_agent_sdk.llm.exceptions import ModelProviderError, ModelRateLimitError
from bu_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from bu_agent_sdk.llm.views import ChatInvokeCompletion
from bu_agent_sdk.observability import Laminar, observe
from bu_agent_sdk.tokens import TokenCost, UsageSummary
from bu_agent_sdk.tools.decorator import Tool


@dataclass
class Agent:
    """
    Simple agentic loop that manages tool calling and message history.

    The agent will:
    1. Send the task to the LLM with available tools
    2. If the LLM returns tool calls, execute them and add results to history
    3. Repeat until the LLM returns a text response without tool calls
    4. Return the final response

    When compaction is enabled, the agent will automatically compress the
    conversation history when token usage exceeds the configured threshold.

    Attributes:
        llm: The language model to use for the agent.
        tools: List of Tool instances. If None, uses all tools from default registry.
        system_prompt: Optional system prompt to guide the agent.
        max_iterations: Maximum number of LLM calls before stopping.
        tool_choice: How the LLM should choose tools ('auto', 'required', 'none').
        compaction: Optional configuration for automatic context compaction.
        include_cost: Whether to calculate costs (requires fetching pricing data).
        dependency_overrides: Optional dict to override tool dependencies.
    """

    llm: BaseChatModel
    tools: list[Tool] | None = None
    system_prompt: str | None = None
    max_iterations: int = 200  # 200 steps max for now
    tool_choice: ToolChoice = "auto"
    compaction: CompactionConfig | None = None
    include_cost: bool = False
    dependency_overrides: dict | None = None
    ephemeral_storage_path: Path | None = None
    """Path to store destroyed ephemeral message content. If None, content is discarded."""
    ephemeral_keep_recent: int | None = None
    """Default keep_recent for ephemeral tools. Overrides tool's _ephemeral value."""

    # Context FileSystem 配置
    offload_enabled: bool = True
    """是否启用上下文卸载到文件系统"""
    offload_token_threshold: int = 2000
    """超过此 token 数的条目才会被卸载"""
    offload_root_path: str | None = None
    """卸载文件存储根目录。None 使用默认 ~/.agent/context/{session_id}"""

    require_done_tool: bool = False
    """If True, the agent will only finish when the 'done' tool is called, not when LLM returns no tool calls."""
    llm_max_retries: int = 5
    """Maximum retries for LLM errors at the agent level (matches browser-use default)."""
    llm_retry_base_delay: float = 1.0
    """Base delay in seconds for exponential backoff on LLM retries."""
    llm_retry_max_delay: float = 60.0
    """Maximum delay in seconds between LLM retry attempts."""
    llm_retryable_status_codes: set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )
    """HTTP status codes that trigger retries (matches browser-use)."""

    # Subagent support
    agents: list | None = None  # type: ignore  # list[AgentDefinition]
    """List of AgentDefinition for creating subagents."""
    tool_registry: object | None = None  # type: ignore  # ToolRegistry
    """Global tool registry for subagents to access tools by name."""
    project_root: Path | None = None
    """Project root directory for discovering subagents. Defaults to cwd."""
    _is_subagent: bool = field(default=False, repr=False)
    """Internal flag to prevent nested subagents."""

    # Skill support
    skills: list | None = None  # type: ignore  # list[SkillDefinition]
    """List of SkillDefinition for Skill support. Auto-discovered if None."""

    # Memory support
    memory: object | None = None  # type: ignore  # MemoryConfig
    """Memory configuration for loading static background knowledge."""

    # Internal state
    _context: ContextIR = field(default=None, repr=False)  # type: ignore  # 在 __post_init__ 中初始化
    _tool_map: dict[str, Tool] = field(default_factory=dict, repr=False)
    _compaction_service: CompactionService | None = field(default=None, repr=False)
    _token_cost: TokenCost = field(default=None, repr=False)  # type: ignore
    _context_fs: ContextFileSystem | None = field(default=None, repr=False)
    _session_id: str = field(default="", repr=False)

    def __post_init__(self):
        # ====== 自动推断 tools 和 tool_registry ======
        # 策略1: 都没传，使用全局默认
        if self.tools is None and self.tool_registry is None:
            from bu_agent_sdk.tools import get_default_registry

            _default = get_default_registry()
            self.tools = _default.all()
            self.tool_registry = _default
            logger.info(f"Using default registry with {len(self.tools)} tools")

        # 策略2: 只传了 tools，自动推断 registry
        elif self.tools is not None and self.tool_registry is None:
            if self.agents:
                # 需要 subagent，使用全局默认 registry
                from bu_agent_sdk.tools import get_default_registry

                self.tool_registry = get_default_registry()
                logger.debug("Using default registry for subagents")
            # else: 不需要 subagent，不需要 registry

        # 策略3: 只传了 registry，自动使用其所有工具
        elif self.tools is None and self.tool_registry is not None:
            self.tools = self.tool_registry.all()
            logger.debug(f"Using all {len(self.tools)} tools from provided registry")

        # 策略4: 都传了，使用用户指定的
        # 直接使用，不需要额外处理

        # 确保 tools 不是 None
        if self.tools is None:
            self.tools = []  # 空工具列表

        # ====== Subagent 自动发现和 Merge ======
        # 只在非 subagent 时执行自动发现（subagent 不支持嵌套）
        if not self._is_subagent:
            from bu_agent_sdk.subagent import discover_subagents

            discovered = discover_subagents(project_root=self.project_root)
            user_agents = self.agents or []

            if discovered or user_agents:
                # Merge：代码传入的覆盖自动发现的同名 subagent
                user_agent_names = {a.name for a in user_agents}
                merged = [a for a in discovered if a.name not in user_agent_names]
                merged.extend(user_agents)
                self.agents = merged if merged else None

                if discovered:
                    logger.info(f"Auto-discovered {len(discovered)} subagent(s)")

        # 检查嵌套 - Subagent 不能再定义 agents
        if self._is_subagent and self.agents:
            raise ValueError("Subagent 不能再定义 agents（不支持嵌套）")

        # Validate that all tools are Tool instances
        for t in self.tools:
            assert isinstance(t, Tool), (
                f"Expected Tool instance, got {type(t).__name__}. Did you forget to use the @tool decorator?"
            )

        # Build tool lookup map
        self._tool_map = {t.name: t for t in self.tools}

        # Initialize token cost service
        self._token_cost = TokenCost(include_cost=self.include_cost)

        # Generate session_id (timestamp + random)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        self._session_id = f"session_{timestamp}_{random_suffix}"

        # Initialize ContextFileSystem if offload enabled
        if self.offload_enabled:
            if self.offload_root_path:
                root_path = Path(self.offload_root_path).expanduser()
            else:
                root_path = Path.home() / ".agent" / "context" / self._session_id
            self._context_fs = ContextFileSystem(
                root_path=root_path,
                session_id=self._session_id,
            )
            logger.info(f"Context FileSystem enabled at {root_path}")

        # Initialize ContextIR
        self._context = ContextIR()

        # 保存原始 system_prompt（在 _setup_subagents/_setup_skills 修改之前）
        _original_system_prompt = self.system_prompt

        # Initialize compaction service (enabled by default)
        # Use provided config or create default (which has enabled=True)
        compaction_config = (
            self.compaction if self.compaction is not None else CompactionConfig()
        )
        self._compaction_service = CompactionService(
            config=compaction_config,
            llm=self.llm,
            token_cost=self._token_cost,
        )

        # Initialize subagent support
        # 注意：_setup_subagents 会同时写入 IR header 的 subagent_strategy
        # 和 拼接 self.system_prompt（向后兼容）
        if self.agents:
            self._setup_subagents()

        # Initialize skill support (所有 Agent 都支持，不区分主/子)
        # 注意：_setup_skills 会同时写入 IR header 的 skill_strategy
        # 和 拼接 self.system_prompt（向后兼容）
        self._setup_skills()

        # 将原始 system_prompt 写入 IR header（不含 subagent/skill 策略拼接）
        # subagent_strategy 和 skill_strategy 已由上面的 _setup 方法独立写入 IR
        if _original_system_prompt:
            self._context.set_system_prompt(_original_system_prompt, cache=True)

        # Initialize memory support
        if self.memory:
            self._setup_memory()

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for all registered tools."""
        return [t.definition for t in self.tools]

    @property
    def messages(self) -> list[BaseMessage]:
        """Get the current message history (read-only copy).

        Returns the lowered representation of the ContextIR,
        compatible with the old _messages format.
        """
        return self._context.lower()

    @property
    def token_cost(self) -> TokenCost:
        """Get the token cost service for direct access to usage tracking."""
        return self._token_cost

    async def get_usage(self) -> UsageSummary:
        """Get usage summary for the agent.

        Returns:
            UsageSummary with token counts and costs.
        """
        return await self._token_cost.get_usage_summary()

    def clear_history(self):
        """Clear the message history and token usage."""
        self._context.clear()
        self._token_cost.clear_history()
        # 如果有 system_prompt，重新写入 IR
        if self.system_prompt:
            self._context.set_system_prompt(self.system_prompt, cache=True)
        # 如果有 memory，重新加载
        if self.memory:
            self._setup_memory()

    def load_history(self, messages: list[BaseMessage]) -> None:
        """Load message history to continue a previous conversation.

        Use this to resume a conversation from previously saved state,
        e.g., when loading from a database on a new machine.

        Note: The system prompt will NOT be re-added on the next query()
        call since the context will be non-empty.

        Args:
                messages: List of BaseMessage instances to load.

        Example:
                # Load and parse messages from your DB
                messages = [parse_message(row) for row in db.query(...)]

                agent = BU(llm=llm, tools=tools, ...)
                agent.load_history(messages)

                # Continue with follow-up
                response = await agent.query("Continue the task...")
        """
        # 清空现有 conversation（保留 header）
        self._context.conversation.items.clear()
        self._token_cost.clear_history()

        # 逐条加载消息到 IR
        for msg in messages:
            self._context.add_message(msg)

    def _destroy_ephemeral_messages(self) -> None:
        """Destroy old ephemeral message content, keeping the last N per tool.

        Tools can specify how many outputs to keep via _ephemeral attribute:
        - _ephemeral = 3 means keep the last 3 outputs of this tool
        - _ephemeral = True is treated as _ephemeral = 1 (keep last 1)

        Older outputs beyond the limit have their content offloaded to filesystem
        via ContextFileSystem and replaced with a placeholder.

        This should be called after each LLM invocation.
        """
        # 构建每个工具的 keep_count 映射
        tool_keep_counts: dict[str, int] = {}
        for tool in (self.tools or []):
            if tool.ephemeral:
                # 如果设置了 ephemeral_keep_recent，覆盖工具的默认值
                if self.ephemeral_keep_recent is not None:
                    keep_count = self.ephemeral_keep_recent
                else:
                    keep_count = tool.ephemeral if isinstance(tool.ephemeral, int) else 1
                tool_keep_counts[tool.name] = keep_count

        # 遍历所有 ephemeral items，卸载需要被销毁的
        for tool_name, keep in tool_keep_counts.items():
            # 获取该工具的所有 ephemeral items
            same_tool_items = [
                item for item in self._context.conversation.items
                if item.item_type.value == "tool_result"
                and item.ephemeral and not item.destroyed
                and (item.tool_name or "") == tool_name
            ]

            # 卸载需要销毁的（不是最后 N 个的）
            if keep > 0 and len(same_tool_items) > keep:
                items_to_destroy = same_tool_items[:-keep]
                for item in items_to_destroy:
                    # 使用 ContextFS 卸载（如果启用）
                    if self._context_fs:
                        try:
                            path = self._context_fs.offload(item)
                            item.offload_path = path
                            item.offloaded = True
                            # 同步更新 message 对象
                            if isinstance(item.message, ToolMessage):
                                item.message.offloaded = True
                                item.message.offload_path = str(self._context_fs.root_path / path)
                        except Exception as e:
                            logger.warning(f"Failed to offload ephemeral item {item.id}: {e}")
                    # 标记为 destroyed（serializer 会用占位符）
                    item.destroyed = True
                    if isinstance(item.message, ToolMessage):
                        item.message.destroyed = True

        # 不再需要调用 ContextIR.destroy_ephemeral_items，因为已直接操作 items

    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        """Execute a single tool call and return the result as a ToolMessage."""
        tool_name = tool_call.function.name

        # Check if this is a Skill tool call (special handling)
        if tool_name == "Skill":
            return await self._execute_skill_call(tool_call)

        tool = self._tool_map.get(tool_name)

        if tool is None:
            return ToolMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                content=f"Error: Unknown tool '{tool_name}'",
                is_error=True,
            )

        # Create Laminar span for tool execution
        if Laminar is not None:
            span_context = Laminar.start_as_current_span(
                name=tool_name,
                input={
                    "tool": tool_name,
                    "arguments": tool_call.function.arguments,
                },
                span_type="TOOL",
            )
        else:
            span_context = nullcontext()

        # Handle TaskComplete outside the span context to avoid it being logged as an error
        task_complete_exception = None

        with span_context:
            try:
                # Parse arguments
                args = json.loads(tool_call.function.arguments)

                # Execute the tool (with dependency overrides if configured)
                result = await tool.execute(
                    _overrides=self.dependency_overrides, **args
                )

                # Check if the tool is marked as ephemeral (can be bool or int for keep count)
                is_ephemeral = bool(tool.ephemeral)  # Convert int to bool (2 -> True)

                tool_message = ToolMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=result,
                    is_error=False,
                    ephemeral=is_ephemeral,
                )

                # Set span output
                if Laminar is not None:
                    Laminar.set_span_output(
                        {
                            "result": result[:500]
                            if isinstance(result, str)
                            else str(result)[:500]
                        }
                    )

                return tool_message

            except json.JSONDecodeError as e:
                error_msg = f"Error parsing arguments: {e}"
                if Laminar is not None:
                    Laminar.set_span_output({"error": error_msg})
                return ToolMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=error_msg,
                    is_error=True,
                )
            except TaskComplete as e:
                # Capture TaskComplete to re-raise after span closes cleanly
                if Laminar is not None:
                    Laminar.set_span_output({"task_complete": True, "message": str(e)})
                task_complete_exception = e
            except Exception as e:
                error_msg = f"Error executing tool: {e}"
                if Laminar is not None:
                    Laminar.set_span_output({"error": error_msg})
                return ToolMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    content=error_msg,
                    is_error=True,
                )

        # Re-raise TaskComplete after span has closed cleanly
        if task_complete_exception is not None:
            raise task_complete_exception

        # This should be unreachable - all code paths either return or raise
        raise RuntimeError("Unexpected code path in _execute_tool_call")

    def _extract_screenshot(self, tool_message: ToolMessage) -> str | None:
        """Extract screenshot base64 from a tool message if present.

        Browser tools may return ContentPartImageParam with screenshots.
        This method extracts the base64 data from such messages.

        Args:
                tool_message: The tool message to extract screenshot from.

        Returns:
                Base64-encoded screenshot string, or None if no screenshot.
        """
        content = tool_message.content

        # If content is a string, no screenshot
        if isinstance(content, str):
            return None

        # If content is a list of content parts, look for images
        if isinstance(content, list):
            for part in content:
                # Check if it's an image content part
                if hasattr(part, "type") and part.type == "image_url":
                    image_url = getattr(part, "image_url", None)
                    if image_url:
                        url = getattr(image_url, "url", "") or image_url.get("url", "")
                        if url.startswith("data:image/png;base64,"):
                            return url.split(",", 1)[1]
                        elif url.startswith("data:image/jpeg;base64,"):
                            return url.split(",", 1)[1]
                # Handle dict format
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "")
                    if url.startswith("data:image/png;base64,"):
                        return url.split(",", 1)[1]
                    elif url.startswith("data:image/jpeg;base64,"):
                        return url.split(",", 1)[1]

        return None

    async def _invoke_llm(self) -> ChatInvokeCompletion:
        """Invoke the LLM with current messages and tools.

        Includes retry logic with exponential backoff for LLM errors
        """
        last_error: Exception | None = None

        for attempt in range(self.llm_max_retries):
            try:
                response = await self.llm.ainvoke(
                    messages=self._context.lower(),
                    tools=self.tool_definitions if self.tools else None,
                    tool_choice=self.tool_choice if self.tools else None,
                )

                # Track token usage
                if response.usage:
                    self._token_cost.add_usage(self.llm.model, response.usage)

                return response

            except ModelRateLimitError as e:
                # Rate limit errors are always retryable
                last_error = e
                if attempt < self.llm_max_retries - 1:
                    delay = min(
                        self.llm_retry_base_delay * (2**attempt),
                        self.llm_retry_max_delay,
                    )
                    jitter = random.uniform(
                        0, delay * 0.1
                    )  # 10% jitter (matches browser-use)
                    total_delay = delay + jitter
                    logger.warning(
                        f"⚠️ Got rate limit error, retrying in {total_delay:.1f}s... "
                        f"(attempt {attempt + 1}/{self.llm_max_retries})"
                    )
                    await asyncio.sleep(total_delay)
                    continue
                raise

            except ModelProviderError as e:
                last_error = e
                # Check if status code is retryable
                is_retryable = (
                    hasattr(e, "status_code")
                    and e.status_code in self.llm_retryable_status_codes
                )
                if is_retryable and attempt < self.llm_max_retries - 1:
                    delay = min(
                        self.llm_retry_base_delay * (2**attempt),
                        self.llm_retry_max_delay,
                    )
                    jitter = random.uniform(
                        0, delay * 0.1
                    )  # 10% jitter (matches browser-use)
                    total_delay = delay + jitter
                    logger.warning(
                        f"⚠️ Got {e.status_code} error, retrying in {total_delay:.1f}s... "
                        f"(attempt {attempt + 1}/{self.llm_max_retries})"
                    )
                    await asyncio.sleep(total_delay)
                    continue
                # Non-retryable or exhausted retries
                raise

            except Exception as e:
                # Handle timeout and connection errors (retryable)
                last_error = e
                error_message = str(e).lower()
                is_timeout = "timeout" in error_message or "cancelled" in error_message
                is_connection_error = (
                    "connection" in error_message or "connect" in error_message
                )

                if (
                    is_timeout or is_connection_error
                ) and attempt < self.llm_max_retries - 1:
                    delay = min(
                        self.llm_retry_base_delay * (2**attempt),
                        self.llm_retry_max_delay,
                    )
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    error_type = "timeout" if is_timeout else "connection error"
                    logger.warning(
                        f"⚠️ Got {error_type}, retrying in {total_delay:.1f}s... "
                        f"(attempt {attempt + 1}/{self.llm_max_retries})"
                    )
                    await asyncio.sleep(total_delay)
                    continue
                # Non-retryable error
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Retry loop completed without return or exception")

    async def _generate_max_iterations_summary(self) -> str:
        """Generate a summary of what was accomplished when max iterations is reached.

        Uses the LLM to summarize the conversation history and actions taken.
        """
        # Build a summary prompt
        summary_prompt = """The task has reached the maximum number of steps allowed.
Please provide a concise summary of:
1. What was accomplished so far
2. What actions were taken
3. What remains incomplete (if anything)
4. Any partial results or findings

Keep the summary brief but informative."""

        # Add the summary request as a user message temporarily
        temp_item = self._context.add_message(UserMessage(content=summary_prompt))

        try:
            # Invoke LLM without tools to get a summary response
            response = await self.llm.ainvoke(
                messages=self._context.lower(),
                tools=None,
                tool_choice=None,
            )
            summary = response.content or "Unable to generate summary."
        except Exception as e:
            logger.warning(f"Failed to generate max iterations summary: {e}")
            summary = f"Task stopped after {self.max_iterations} iterations. Unable to generate summary due to error."
        finally:
            # Remove the temporary summary prompt
            self._context.conversation.remove_by_id(temp_item.id)

        return f"[Max iterations reached]\n\n{summary}"

    async def _get_incomplete_todos_prompt(self) -> str | None:
        """Hook for subclasses to check for incomplete todos before finishing.

        This method is called when the LLM is about to stop (no more tool calls in CLI mode,
        or done tool called in autonomous mode).

        The prompt should ask the LLM to:
        1. Continue working on incomplete tasks
        2. Mark completed tasks as done
        3. Revise the todo list if tasks are no longer relevant
        """
        return None

    async def _check_and_compact(self, response: ChatInvokeCompletion) -> bool:
        """Check token usage and compact if threshold exceeded.

        Uses selective compaction (by type priority) first, falling back to
        full summary via CompactionService if needed.

        The threshold is calculated dynamically based on the model's context window.

        Args:
                response: The latest LLM response with usage information.

        Returns:
                True if compaction was performed, False otherwise.
        """
        if self._compaction_service is None:
            return False

        # Update token usage tracking
        self._compaction_service.update_usage(response.usage)

        # 检查是否需要压缩
        if not await self._compaction_service.should_compact(self.llm.model):
            return False

        # 获取压缩阈值
        threshold = await self._compaction_service.get_threshold_for_model(self.llm.model)

        # 使用 token usage 中的实际 total_tokens
        from bu_agent_sdk.agent.compaction.models import TokenUsage
        actual_tokens = TokenUsage.from_usage(response.usage).total_tokens

        # 创建 OffloadPolicy
        offload_policy = None
        if self.offload_enabled and self._context_fs:
            from bu_agent_sdk.context.offload import OffloadPolicy
            offload_policy = OffloadPolicy(
                enabled=True,
                token_threshold=self.offload_token_threshold,
            )

        # 创建选择性压缩策略
        policy = SelectiveCompactionPolicy(
            threshold=threshold,
            llm=self.llm,
            fallback_to_full_summary=True,
            fs=self._context_fs,
            offload_policy=offload_policy,
        )

        return await self._context.auto_compact(
            policy=policy,
            current_total_tokens=actual_tokens,
        )

    @observe(name="agent_query")
    async def query(self, message: str) -> str:
        """
        Send a message to the agent and get a response.

        Can be called multiple times for follow-up questions - message history
        is preserved between calls. System prompt is managed by ContextIR.

        When compaction is enabled, the agent will automatically compress the
        conversation history when token usage exceeds the configured threshold.
        After compaction, the conversation continues from the summary.

        Args:
            message: The user message.

        Returns:
            The agent's response text.
        """
        # Add the user message to context
        self._context.add_message(UserMessage(content=message))

        iterations = 0
        tool_calls_made = 0
        incomplete_todos_prompted = (
            False  # Track if we've already prompted about incomplete todos
        )
        done_tool_prompted = False  # Track if already prompted to call done tool

        while iterations < self.max_iterations:
            iterations += 1

            # Destroy ephemeral messages from previous iteration before LLM sees them again
            self._destroy_ephemeral_messages()

            # Invoke the LLM
            response = await self._invoke_llm()

            # Add assistant message to history
            assistant_msg = AssistantMessage(
                content=response.content,
                tool_calls=response.tool_calls if response.tool_calls else None,
            )
            self._context.add_message(assistant_msg)

            # If no tool calls, check if should finish
            if not response.has_tool_calls:
                if not self.require_done_tool:
                    # CLI mode: LLM stopped calling tools, check for incomplete todos before finishing
                    if not incomplete_todos_prompted:
                        incomplete_prompt = await self._get_incomplete_todos_prompt()
                        if incomplete_prompt:
                            incomplete_todos_prompted = True
                            self._context.add_message(
                                UserMessage(content=incomplete_prompt)
                            )
                            continue  # Give the LLM a chance to handle incomplete todos

                    # All done - return the response
                    await self._check_and_compact(response)
                    return response.content or ""
                # Autonomous mode: require done tool
                # If LLM returns no tool calls, prompt it to call done tool
                if not done_tool_prompted:
                    done_tool_prompted = True
                    done_prompt = (
                        "You have not called any tool. If you have completed the task, "
                        "please call the 'done' tool with a summary of what was accomplished. "
                        "If you still have work to do, please continue with the appropriate tools."
                    )
                    self._context.add_message(UserMessage(content=done_prompt))
                    continue
                else:
                    # Already prompted but still no tool calls - finish with current response
                    logger.warning(
                        "LLM did not call done tool after prompt, finishing anyway"
                    )
                    await self._check_and_compact(response)
                    return response.content or ""

            # Execute all tool calls
            for tool_call in response.tool_calls:
                tool_calls_made += 1
                try:
                    tool_result = await self._execute_tool_call(tool_call)
                    self._context.add_message(tool_result)

                    # 检查是否有待注入的 Skill items（必须在 ToolMessage 之后注入）
                    if self._context.has_pending_skill_items:
                        self._context.flush_pending_skill_items()
                except TaskComplete as e:
                    self._context.add_message(
                        ToolMessage(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.function.name,
                            content=f"Task completed: {e.message}",
                            is_error=False,
                        )
                    )
                    return e.message

            # Check for compaction after tool execution
            await self._check_and_compact(response)

        # Max iterations reached - generate summary of what was accomplished
        return await self._generate_max_iterations_summary()

    @observe(name="agent_query_stream")
    async def query_stream(
        self, message: str | list[ContentPartTextParam | ContentPartImageParam]
    ) -> AsyncIterator[AgentEvent]:
        """
        Send a message to the agent and stream events as they occur.

        Yields events for each step of the agent's execution, providing
        visibility into tool calls and intermediate results.

        Args:
            message: The user message. Can be a string or a list of content parts
                for multi-modal input (text + images).

        Yields:
            AgentEvent instances for each step:
            - TextEvent: When the assistant produces text
            - ThinkingEvent: When the model produces thinking content
            - ToolCallEvent: When a tool is being called
            - ToolResultEvent: When a tool returns a result
            - FinalResponseEvent: The final response (always last)

        Example:
            async for event in agent.query_stream("Schedule a meeting"):
                match event:
                    case ToolCallEvent(tool=name, args=args):
                        print(f"Calling {name}")
                    case ToolResultEvent(tool=name, result=result):
                        print(f"{name} returned: {result[:50]}")
                    case FinalResponseEvent(content=text):
                        print(f"Done: {text}")
        """
        # Add the user message to context (supports both string and multi-modal content)
        self._context.add_message(UserMessage(content=message))

        iterations = 0
        incomplete_todos_prompted = (
            False  # Track if already prompted about incomplete todos
        )
        done_tool_prompted = False  # Track if already prompted to call done tool

        while iterations < self.max_iterations:
            iterations += 1

            # Destroy ephemeral messages from previous iteration before LLM sees them again
            self._destroy_ephemeral_messages()

            # Invoke the LLM
            response = await self._invoke_llm()

            # Check for thinking content and yield it
            if response.thinking:
                yield ThinkingEvent(content=response.thinking)

            # Add assistant message to history
            assistant_msg = AssistantMessage(
                content=response.content,
                tool_calls=response.tool_calls if response.tool_calls else None,
            )
            self._context.add_message(assistant_msg)

            # If no tool calls, check if should finish
            if not response.has_tool_calls:
                if not self.require_done_tool:
                    # CLI mode: LLM stopped calling tools, check for incomplete todos before finishing
                    if not incomplete_todos_prompted:
                        incomplete_prompt = await self._get_incomplete_todos_prompt()
                        if incomplete_prompt:
                            incomplete_todos_prompted = True
                            self._context.add_message(
                                UserMessage(content=incomplete_prompt)
                            )
                            yield HiddenUserMessageEvent(content=incomplete_prompt)
                            continue  # Give the LLM a chance to handle incomplete todos

                    # All done - return the response
                    await self._check_and_compact(response)
                    if response.content:
                        yield TextEvent(content=response.content)
                    yield FinalResponseEvent(content=response.content or "")
                    return
                # Autonomous mode: require done tool
                # If LLM returns no tool calls, prompt it to call done tool
                if not done_tool_prompted:
                    done_tool_prompted = True
                    done_prompt = (
                        "You have not called any tool. If you have completed the task, "
                        "please call the 'done' tool with a summary of what was accomplished. "
                        "If you still have work to do, please continue with the appropriate tools."
                    )
                    self._context.add_message(UserMessage(content=done_prompt))
                    yield HiddenUserMessageEvent(content=done_prompt)
                    if response.content:
                        yield TextEvent(content=response.content)
                    continue
                else:
                    # Already prompted but still no tool calls - finish with current response
                    logger.warning(
                        "LLM did not call done tool after prompt, finishing anyway"
                    )
                    await self._check_and_compact(response)
                    if response.content:
                        yield TextEvent(content=response.content)
                    yield FinalResponseEvent(content=response.content or "")
                    return

            # Yield text content if present alongside tool calls
            if response.content:
                yield TextEvent(content=response.content)

            # Execute all tool calls, yielding events for each
            step_number = 0
            for tool_call in response.tool_calls:
                step_number += 1
                tool_name = tool_call.function.name

                # Yield the tool call event
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tool_call.function.arguments}

                # Emit step start event
                yield StepStartEvent(
                    step_id=tool_call.id,
                    title=tool_name,
                    step_number=step_number,
                )

                yield ToolCallEvent(
                    tool=tool_name,
                    args=args,
                    tool_call_id=tool_call.id,
                    display_name=tool_name,
                )

                # Execute the tool
                step_start_time = time.time()
                try:
                    tool_result = await self._execute_tool_call(tool_call)
                    self._context.add_message(tool_result)

                    # 检查是否有待注入的 Skill items（必须在 ToolMessage 之后注入）
                    if self._context.has_pending_skill_items:
                        self._context.flush_pending_skill_items()

                    # Extract screenshot if present (for browser tools)
                    screenshot_base64 = self._extract_screenshot(tool_result)

                    # Yield the tool result event
                    yield ToolResultEvent(
                        tool=tool_name,
                        result=tool_result.text,
                        tool_call_id=tool_call.id,
                        is_error=tool_result.is_error,
                        screenshot_base64=screenshot_base64,
                    )

                    # Emit step complete event
                    step_duration_ms = (time.time() - step_start_time) * 1000
                    yield StepCompleteEvent(
                        step_id=tool_call.id,
                        status="error" if tool_result.is_error else "completed",
                        duration_ms=step_duration_ms,
                    )
                except TaskComplete as e:
                    # done_autonomous already validates todos before raising TaskComplete,
                    # so can complete immediately
                    self._context.add_message(
                        ToolMessage(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.function.name,
                            content=f"Task completed: {e.message}",
                            is_error=False,
                        )
                    )
                    yield ToolResultEvent(
                        tool=tool_call.function.name,
                        result=f"Task completed: {e.message}",
                        tool_call_id=tool_call.id,
                        is_error=False,
                    )
                    yield FinalResponseEvent(content=e.message)
                    return

            # Check for compaction after tool execution
            await self._check_and_compact(response)

        # Max iterations reached - generate summary of what was accomplished
        summary = await self._generate_max_iterations_summary()
        yield FinalResponseEvent(content=summary)

    def _setup_subagents(self) -> None:
        """设置 Subagent 支持

        1. 生成 Subagent 策略提示并写入 ContextIR header
        2. 创建 Task 工具并添加到 tools 列表

        注意：同时保持 self.system_prompt 的拼接（向后兼容），
        但 IR 中的 subagent_strategy 是独立管理的。
        """
        from bu_agent_sdk.subagent.prompts import generate_subagent_prompt
        from bu_agent_sdk.subagent.task_tool import create_task_tool

        if not self.agents or not self.tool_registry:
            return

        # 生成 Subagent 策略提示
        subagent_prompt = generate_subagent_prompt(self.agents)

        # 写入 ContextIR header（独立段）
        self._context.set_subagent_strategy(subagent_prompt)

        # 同时保持 system_prompt 拼接（向后兼容 _rebuild_skill_tool 等）
        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{subagent_prompt}"
        else:
            self.system_prompt = subagent_prompt

        # 创建 Task 工具
        task_tool = create_task_tool(
            agents=self.agents,
            tool_registry=self.tool_registry,  # type: ignore
            parent_llm=self.llm,
            parent_dependency_overrides=self.dependency_overrides,  # type: ignore
        )

        # 添加到工具列表
        self.tools.append(task_tool)
        self._tool_map[task_tool.name] = task_tool

        logging.info(
            f"Initialized {len(self.agents)} subagents: {[a.name for a in self.agents]}"
        )

    def _setup_memory(self) -> None:
        """设置 Memory 静态背景知识

        从配置的文件中加载内容并写入 ContextIR header。
        Token 超限时只警告不截断。
        """
        from bu_agent_sdk.context.memory import MemoryConfig, load_memory_content

        if not isinstance(self.memory, MemoryConfig):
            logger.warning("memory 参数不是 MemoryConfig 实例，跳过初始化")
            return

        # 加载文件内容
        content = load_memory_content(self.memory)
        if not content:
            logger.warning("未能加载任何 memory 内容")
            return

        # Token 检查
        token_count = self._context.token_counter.count(content)
        if token_count > self.memory.max_tokens:
            logger.warning(
                f"Memory 内容超出 token 上限: {token_count} > {self.memory.max_tokens} "
                f"(不会截断，但可能影响上下文预算)"
            )

        # 写入 ContextIR
        self._context.set_memory(content, cache=self.memory.cache)
        logger.info(f"Memory 已加载: {token_count} tokens 来自 {len(self.memory.files)} 个文件")

    def _setup_skills(self) -> None:
        """设置 Skill 支持

        设计决策：
        - 每个 Agent 独立发现和加载 Skills
        - 主 Agent 和 Subagent 都能使用 Skill
        - 每个 Agent 可以激活多个不同的 Skills
        - 防止同一 Skill 被重复激活（避免无限递归）
        - 允许激活不同的 Skills（如先激活 Skill A，再激活 Skill B）

        架构模式（参考 Subagent）：
        - Skills 列表和使用规则 → ContextIR header（通过 set_skill_strategy）
        - Tool description → 只包含简洁的功能说明
        """
        from bu_agent_sdk.skill import create_skill_tool, discover_skills
        from bu_agent_sdk.skill.prompts import generate_skill_prompt

        # 自动发现 skills
        discovered = discover_skills(project_root=self.project_root)
        user_skills = self.skills or []

        if discovered or user_skills:
            # 合并（代码传入的覆盖自动发现的同名 skill）
            user_skill_names = {s.name for s in user_skills}
            merged = [s for s in discovered if s.name not in user_skill_names]
            merged.extend(user_skills)
            self.skills = merged if merged else None

            if discovered:
                logger.info(f"Auto-discovered {len(discovered)} skill(s)")

        if not self.skills:
            return

        # 生成 Skill 策略提示并写入 ContextIR header
        skill_prompt = generate_skill_prompt(self.skills)
        if skill_prompt:  # 只有当有 active skills 时才注入
            self._context.set_skill_strategy(skill_prompt)

            # 同时保持 system_prompt 拼接（向后兼容 _rebuild_skill_tool 等）
            if self.system_prompt:
                self.system_prompt = f"{self.system_prompt}\n\n{skill_prompt}"
            else:
                self.system_prompt = skill_prompt

        # 创建 Skill 工具（现在只包含简洁描述）
        skill_tool = create_skill_tool(self.skills)
        self.tools.append(skill_tool)
        self._tool_map[skill_tool.name] = skill_tool

        logger.info(f"Initialized {len(self.skills)} skill(s): {[s.name for s in self.skills]}")

    async def _execute_skill_call(self, tool_call: ToolCall) -> ToolMessage:
        """执行 Skill 调用（特殊处理）

        Skill 调用会：
        1. 返回 ToolMessage（必须先添加到 context 以满足 OpenAI API 要求）
        2. 通过 ContextIR.add_skill_injection() 存储待注入的 items
        3. 调用方会在添加 ToolMessage 后 flush pending skill items
        4. 应用 Skill 的 execution context 修改（持久化）
        5. 防止重复调用保护
        """
        args = json.loads(tool_call.function.arguments)
        skill_name = args.get("skill_name")

        # 查找 Skill
        skill_def = next((s for s in self.skills if s.name == skill_name), None) if self.skills else None
        if not skill_def:
            return ToolMessage(
                tool_call_id=tool_call.id,
                tool_name="Skill",
                content=f"Error: Unknown skill '{skill_name}'",
                is_error=True,
            )

        # 防重复调用：同一 Skill 不能重复激活（防止无限递归）
        if skill_name in self._context.active_skill_names:
            return ToolMessage(
                tool_call_id=tool_call.id,
                tool_name="Skill",
                content=f"Error: Skill '{skill_name}' is already active. Cannot activate the same skill twice to prevent infinite recursion.",
                is_error=True,
            )

        # 应用 Skill 的 execution context 修改（持久化）
        from bu_agent_sdk.skill.context import apply_skill_context

        apply_skill_context(self, skill_def)
        self._context.active_skill_names.add(skill_name)

        # 准备待注入的消息（将在 ToolMessage 添加到 context 后注入）
        metadata = f'<skill-message>The "{skill_name}" skill is loading</skill-message>\n<skill-name>{skill_name}</skill-name>'
        full_prompt = skill_def.get_prompt()

        # 通过 ContextIR 管理 pending skill items
        self._context.add_skill_injection(
            skill_name=skill_name,
            metadata_msg=UserMessage(content=metadata, is_meta=False),
            prompt_msg=UserMessage(content=full_prompt, is_meta=True),
        )

        # 返回 ToolMessage（调用方会先添加这个，再 flush pending items）
        return ToolMessage(
            tool_call_id=tool_call.id,
            tool_name="Skill",
            content=f"Skill '{skill_name}' loaded successfully and will remain active",
            is_error=False,
        )

    def _rebuild_skill_tool(self) -> None:
        """重建 Skill 工具（用于 Subagent Skills 筛选后更新工具描述）

        这个方法在 Subagent 创建后，如果筛选了 Skills，需要调用以重新生成 Skill 工具。

        通过 ContextIR 管理 skill_strategy，不再需要手动操作 SystemMessage。
        """
        from bu_agent_sdk.skill import create_skill_tool, generate_skill_prompt

        # 1. 移除旧的 Skill 工具
        self.tools = [t for t in self.tools if t.name != "Skill"]
        self._tool_map.pop("Skill", None)

        # 2. 移除 IR 中的旧 Skill 策略
        self._context.remove_skill_strategy()

        # 3. 同时更新 system_prompt（向后兼容）
        skill_prompt_marker = "\n\n## Skill 工具使用指南"
        if self.system_prompt and skill_prompt_marker in self.system_prompt:
            idx = self.system_prompt.find(skill_prompt_marker)
            if idx > 0:
                self.system_prompt = self.system_prompt[:idx]

        # 4. 如果还有 skills，重新生成
        if self.skills:
            skill_prompt = generate_skill_prompt(self.skills)
            if skill_prompt:
                # 写入 IR header
                self._context.set_skill_strategy(skill_prompt)

                # 同时更新 system_prompt（向后兼容）
                if self.system_prompt:
                    self.system_prompt = f"{self.system_prompt}\n\n{skill_prompt}"
                else:
                    self.system_prompt = skill_prompt

            # 创建新的 Skill 工具
            skill_tool = create_skill_tool(self.skills)
            self.tools.append(skill_tool)
            self._tool_map["Skill"] = skill_tool

            logger.debug(f"Rebuilt Skill tool with {len(self.skills)} skill(s)")
        else:
            logger.debug("Removed Skill tool (no skills remaining)")

        # ContextIR 的 lower() 会自动从 header 段构建 SystemMessage，
        # 不再需要手动同步 _messages[0]
