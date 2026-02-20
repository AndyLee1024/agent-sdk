from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from comate_agent_sdk.llm.exceptions import ModelProviderError, ModelRateLimitError
from comate_agent_sdk.llm.views import ChatInvokeCompletion

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import AgentRuntime


async def invoke_llm(agent: "AgentRuntime") -> ChatInvokeCompletion:
    """调用 LLM（包含 retry + exponential backoff）。"""
    last_error: Exception | None = None
    inject_due_reminders = getattr(agent._context, "inject_due_reminders", None)
    if callable(inject_due_reminders):
        inject_due_reminders()

    for attempt in range(agent.options.llm_max_retries):
        try:
            # MCP tools：首次调用前加载；resume 后 dirty 时刷新
            try:
                await agent.ensure_mcp_tools_loaded()
            except Exception as e:
                # MCP 加载失败不应阻断所有场景；
                # 但若用户显式请求了某些 mcp__ 工具名，则应直接抛出，避免 silent degrade。
                if getattr(agent, "_tools_allowlist_mode", False) and getattr(agent, "_mcp_pending_tool_names", []):
                    raise
                logger.warning(f"MCP tools 加载失败（已降级继续）：{e}")

            tool_definitions = agent.tool_definitions
            response = await agent.llm.ainvoke(
                messages=agent._context.lower(),
                tools=tool_definitions or None,
                tool_choice=agent.options.tool_choice if tool_definitions else None,
            )

            # Track token usage
            if response.usage:
                source = "agent"
                if agent._subagent_source_prefix:
                    source = agent._subagent_source_prefix
                agent._token_cost.add_usage(
                    agent.llm.model,
                    response.usage,
                    level=agent._effective_level,
                    source=source,
                )
                tracker = getattr(agent, "_context_usage_tracker", None)
                if tracker is not None:
                    try:
                        tracker.observe_response(
                            total_tokens=int(response.usage.total_tokens),
                            ir_total=agent._context.total_tokens,
                        )
                    except Exception as e:
                        logger.debug(f"更新 context usage tracker 失败: {e}", exc_info=True)

            return response

        except ModelRateLimitError as e:
            # Rate limit errors are always retryable
            last_error = e
            if attempt < agent.options.llm_max_retries - 1:
                delay = min(
                    agent.options.llm_retry_base_delay * (2**attempt),
                    agent.options.llm_retry_max_delay,
                )
                jitter = random.uniform(0, delay * 0.1)  # 10% jitter (matches browser-use)
                total_delay = delay + jitter
                logger.warning(
                    f"⚠️ Got rate limit error, retrying in {total_delay:.1f}s... "
                    f"(attempt {attempt + 1}/{agent.options.llm_max_retries})"
                )
                await asyncio.sleep(total_delay)
                continue
            raise

        except ModelProviderError as e:
            last_error = e
            # Check if status code is retryable
            is_retryable = (
                hasattr(e, "status_code")
                and e.status_code in agent.options.llm_retryable_status_codes
            )
            if is_retryable and attempt < agent.options.llm_max_retries - 1:
                delay = min(
                    agent.options.llm_retry_base_delay * (2**attempt),
                    agent.options.llm_retry_max_delay,
                )
                jitter = random.uniform(0, delay * 0.1)  # 10% jitter (matches browser-use)
                total_delay = delay + jitter
                logger.warning(
                    f"⚠️ Got {e.status_code} error, retrying in {total_delay:.1f}s... "
                    f"(attempt {attempt + 1}/{agent.options.llm_max_retries})"
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
            is_connection_error = "connection" in error_message or "connect" in error_message

            if (is_timeout or is_connection_error) and attempt < agent.options.llm_max_retries - 1:
                delay = min(
                    agent.options.llm_retry_base_delay * (2**attempt),
                    agent.options.llm_retry_max_delay,
                )
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter
                error_type = "timeout" if is_timeout else "connection error"
                logger.warning(
                    f"⚠️ Got {error_type}, retrying in {total_delay:.1f}s... "
                    f"(attempt {attempt + 1}/{agent.options.llm_max_retries})"
                )
                await asyncio.sleep(total_delay)
                continue
            # Non-retryable error
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry loop completed without return or exception")
