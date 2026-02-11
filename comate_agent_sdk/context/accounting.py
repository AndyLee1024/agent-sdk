"""Token accounting for context-window estimation and provider usage calibration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from comate_agent_sdk.context.lower import LoweringPipeline

if TYPE_CHECKING:
    from comate_agent_sdk.context.ir import ContextIR
    from comate_agent_sdk.llm.base import BaseChatModel, ToolDefinition

logger = logging.getLogger("comate_agent_sdk.context.accounting")


@dataclass(frozen=True)
class NextStepEstimate:
    """Estimated token usage for the next model invocation."""

    model_key: str
    message_tokens: int
    tool_definition_tokens: int
    raw_total_tokens: int
    calibration_ratio: float
    calibrated_tokens: int
    safety_margin_ratio: float
    buffered_tokens: int


class ContextTokenAccounting:
    """Dual-track token accounting.

    - ReportedUsage: provider-returned `usage.total_tokens`
    - NextStepEstimate: local estimate with EMA calibration and safety margin
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.2,
        ratio_min: float = 0.5,
        ratio_max: float = 2.0,
        safety_margin_ratio: float = 0.12,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.safety_margin_ratio = float(max(0.0, safety_margin_ratio))
        self._ratio_by_model_key: dict[str, float] = {}
        self._last_estimate: NextStepEstimate | None = None
        self._last_reported_total_tokens: int = 0

    @staticmethod
    def build_model_key(llm: "BaseChatModel") -> str:
        provider = str(getattr(llm, "provider", "unknown") or "unknown").strip().lower()
        model = str(getattr(llm, "model", "unknown") or "unknown").strip()
        return f"{provider}:{model}"

    def _clamp_ratio(self, ratio: float) -> float:
        return min(max(float(ratio), self.ratio_min), self.ratio_max)

    def get_ratio(self, model_key: str) -> float:
        return self._ratio_by_model_key.get(model_key, 1.0)

    @property
    def last_estimate(self) -> NextStepEstimate | None:
        return self._last_estimate

    @property
    def last_reported_total_tokens(self) -> int:
        return self._last_reported_total_tokens

    def observe_reported_usage(
        self,
        *,
        llm: "BaseChatModel",
        reported_total_tokens: int,
        estimated_raw_tokens: int | None = None,
    ) -> None:
        """Update reported total and calibration ratio for model."""
        reported = int(max(0, reported_total_tokens))
        self._last_reported_total_tokens = reported

        if reported <= 0:
            return
        if estimated_raw_tokens is None or int(estimated_raw_tokens) <= 0:
            return

        model_key = self.build_model_key(llm)
        raw_estimate = int(estimated_raw_tokens)
        observed_ratio = self._clamp_ratio(reported / raw_estimate)
        prev_ratio = self.get_ratio(model_key)
        updated_ratio = self._clamp_ratio(
            (1.0 - self.ema_alpha) * prev_ratio + self.ema_alpha * observed_ratio
        )
        self._ratio_by_model_key[model_key] = updated_ratio
        logger.debug(
            f"更新 token 校准 ratio: model={model_key}, prev={prev_ratio:.4f}, "
            f"observed={observed_ratio:.4f}, updated={updated_ratio:.4f}, "
            f"reported={reported}, estimated={raw_estimate}"
        )

    async def estimate_next_step(
        self,
        *,
        context: "ContextIR",
        llm: "BaseChatModel",
        tool_definitions: list["ToolDefinition"] | None,
        timeout_ms: int,
    ) -> NextStepEstimate:
        """Estimate tokens for the next invocation with calibration and margin."""
        lowered_messages = LoweringPipeline.lower(context)
        message_tokens = await context.token_counter.count_messages_for_model(
            lowered_messages,
            llm=llm,
            timeout_ms=int(timeout_ms),
        )
        if message_tokens <= 0:
            message_tokens = context.total_tokens

        tool_definition_tokens = 0
        if tool_definitions:
            tool_defs_json = json.dumps(
                [d.model_dump() for d in tool_definitions],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            tool_definition_tokens = context.token_counter.count(tool_defs_json)

        raw_total = int(max(message_tokens + tool_definition_tokens, 0))
        model_key = self.build_model_key(llm)
        ratio = self.get_ratio(model_key)
        calibrated_tokens = int(max(raw_total * ratio, 0))
        buffered_tokens = int(max(calibrated_tokens * (1.0 + self.safety_margin_ratio), 0))

        estimate = NextStepEstimate(
            model_key=model_key,
            message_tokens=message_tokens,
            tool_definition_tokens=tool_definition_tokens,
            raw_total_tokens=raw_total,
            calibration_ratio=ratio,
            calibrated_tokens=calibrated_tokens,
            safety_margin_ratio=self.safety_margin_ratio,
            buffered_tokens=buffered_tokens,
        )
        self._last_estimate = estimate
        return estimate
