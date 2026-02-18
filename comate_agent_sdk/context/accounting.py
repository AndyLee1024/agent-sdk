"""Token accounting: tracks provider-reported baseline for incremental precheck estimation."""

from __future__ import annotations

import logging

logger = logging.getLogger("comate_agent_sdk.context.accounting")


class ContextTokenAccounting:
    """Baseline token accounting.

    Tracks the last provider-reported ``total_tokens`` and the ContextIR token
    count at the time of that report.  This lets ``precheck_and_compact`` do a
    cheap incremental estimate without re-counting the full context every turn::

        estimated = last_reported + max(0, current_ir_total - ir_total_at_last_report)

    After compaction the baseline is reset so the next precheck falls back to
    the raw IR total until the next ``invoke_llm`` rebuilds ground truth.
    """

    def __init__(
        self,
        *,
        safety_margin_ratio: float = 0.12,
    ) -> None:
        self.safety_margin_ratio = float(max(0.0, safety_margin_ratio))
        self._last_reported_total_tokens: int = 0
        self._ir_total_at_last_report: int = 0

    @property
    def last_reported_total_tokens(self) -> int:
        return self._last_reported_total_tokens

    @property
    def ir_total_at_last_report(self) -> int:
        return self._ir_total_at_last_report

    def observe_reported_usage(
        self,
        *,
        reported_total_tokens: int,
        ir_total: int = 0,
    ) -> None:
        """Record the provider-reported total and the IR snapshot at that moment."""
        self._last_reported_total_tokens = max(0, int(reported_total_tokens))
        self._ir_total_at_last_report = max(0, int(ir_total))
        logger.debug(
            f"token baseline updated: reported={self._last_reported_total_tokens}, "
            f"ir_snapshot={self._ir_total_at_last_report}"
        )

    def reset_baseline(self) -> None:
        """Reset baseline after compaction so next precheck uses IR total as fallback."""
        self._last_reported_total_tokens = 0
        self._ir_total_at_last_report = 0
        logger.debug("token baseline reset after compaction")
