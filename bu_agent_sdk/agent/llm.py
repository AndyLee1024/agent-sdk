from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from bu_agent_sdk.llm.exceptions import ModelProviderError, ModelRateLimitError
from bu_agent_sdk.llm.views import ChatInvokeCompletion

logger = logging.getLogger("bu_agent_sdk.agent")

if TYPE_CHECKING:
    from bu_agent_sdk.agent.core import Agent


async def invoke_llm(agent: "Agent") -> ChatInvokeCompletion:
    """调用 LLM（包含 retry + exponential backoff）。"""
    last_error: Exception | None = None

    for attempt in range(agent.llm_max_retries):
        try:
            response = await agent.llm.ainvoke(
                messages=agent._context.lower(),
                tools=agent.tool_definitions if agent.tools else None,
                tool_choice=agent.tool_choice if agent.tools else None,
            )

            # Track token usage
            if response.usage:
                source = "agent"
                if agent._is_subagent:
                    source = f"subagent:{agent.name}" if agent.name else "subagent"
                agent._token_cost.add_usage(
                    agent.llm.model,
                    response.usage,
                    level=agent._effective_level,
                    source=source,
                )

            return response

        except ModelRateLimitError as e:
            # Rate limit errors are always retryable
            last_error = e
            if attempt < agent.llm_max_retries - 1:
                delay = min(
                    agent.llm_retry_base_delay * (2**attempt),
                    agent.llm_retry_max_delay,
                )
                jitter = random.uniform(0, delay * 0.1)  # 10% jitter (matches browser-use)
                total_delay = delay + jitter
                logger.warning(
                    f"⚠️ Got rate limit error, retrying in {total_delay:.1f}s... "
                    f"(attempt {attempt + 1}/{agent.llm_max_retries})"
                )
                await asyncio.sleep(total_delay)
                continue
            raise

        except ModelProviderError as e:
            last_error = e
            # Check if status code is retryable
            is_retryable = hasattr(e, "status_code") and e.status_code in agent.llm_retryable_status_codes
            if is_retryable and attempt < agent.llm_max_retries - 1:
                delay = min(
                    agent.llm_retry_base_delay * (2**attempt),
                    agent.llm_retry_max_delay,
                )
                jitter = random.uniform(0, delay * 0.1)  # 10% jitter (matches browser-use)
                total_delay = delay + jitter
                logger.warning(
                    f"⚠️ Got {e.status_code} error, retrying in {total_delay:.1f}s... "
                    f"(attempt {attempt + 1}/{agent.llm_max_retries})"
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

            if (is_timeout or is_connection_error) and attempt < agent.llm_max_retries - 1:
                delay = min(
                    agent.llm_retry_base_delay * (2**attempt),
                    agent.llm_retry_max_delay,
                )
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter
                error_type = "timeout" if is_timeout else "connection error"
                logger.warning(
                    f"⚠️ Got {error_type}, retrying in {total_delay:.1f}s... "
                    f"(attempt {attempt + 1}/{agent.llm_max_retries})"
                )
                await asyncio.sleep(total_delay)
                continue
            # Non-retryable error
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry loop completed without return or exception")

