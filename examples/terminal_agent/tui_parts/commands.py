from __future__ import annotations

import asyncio
import logging
from typing import Any

from comate_agent_sdk.agent.llm_levels import ALL_LEVELS

from terminal_agent.selection_menu import SelectionResult, build_model_level_options
from terminal_agent.slash_commands import SLASH_COMMAND_SPECS, parse_slash_command_call
from terminal_agent.tui_parts.ui_mode import UIMode

logger = logging.getLogger(__name__)


class CommandsMixin:
    async def _execute_command(self, command: str) -> None:
        parsed = parse_slash_command_call(command)
        normalized = command.strip()
        if parsed is None:
            self._renderer.append_system_message(
                f"Unknown command: {normalized}",
                is_error=True,
            )
            self._refresh_layers()
            return

        spec = self._slash_lookup.get(parsed.name)
        if spec is None:
            self._renderer.append_system_message(
                f"Unknown command: {normalized}",
                is_error=True,
            )
            self._refresh_layers()
            return

        handler = self._slash_handlers.get(spec.name)
        if handler is None:
            logger.error(f"slash handler missing for command: {spec.name}")
            self._renderer.append_system_message(
                f"Unknown command: {parsed.raw_input}",
                is_error=True,
            )
            self._refresh_layers()
            return

        result = handler(parsed.args)
        if asyncio.iscoroutine(result):
            await result
        self._refresh_layers()

    def _slash_help(self, _args: str) -> None:
        lines = []
        for spec in SLASH_COMMAND_SPECS:
            alias_text = ""
            if spec.aliases:
                alias_text = f" ({', '.join(f'/{alias}' for alias in spec.aliases)})"
            lines.append(f"/{spec.name}{alias_text} - {spec.description}")
        self._renderer.append_system_message("\n".join(lines))

    def _slash_session(self, _args: str) -> None:
        self._renderer.append_system_message(f"Session ID: {self._session.session_id}")

    async def _slash_usage(self, _args: str) -> None:
        await self._append_usage_snapshot()

    async def _slash_context(self, _args: str) -> None:
        await self._append_context_snapshot()

    def _slash_exit(self, _args: str) -> None:
        self._exit_app()

    def _slash_model(self, args: str) -> None:
        """Handle /model command - switch model level."""
        # If args provided, try to use it directly
        if args.strip():
            level = args.strip().upper()
            if level in ALL_LEVELS:
                self._switch_model_level(level)
                return
            # Invalid level, show error and open menu
            self._renderer.append_system_message(
                f"Invalid model level: {args.strip()}. Use LOW, MID, or HIGH.",
                is_error=True,
            )

        # Open selection menu
        self._enter_selection_mode()

    def _switch_model_level(self, level: str) -> None:
        """Switch to the specified model level."""
        try:
            llm_level = level  # type: ignore[assignment]
            event = self._session.set_level(llm_level)

            # Get model names for display
            prev_model = event.previous_model or "unknown"
            new_model = event.new_model or "unknown"

            self._renderer.append_system_message(
                f"Model switched: {event.previous_level} → {event.new_level}\n"
                f"  ({prev_model} → {new_model})"
            )
            logger.info(f"Model level switched: {event}")

            # Update status bar model name - 使用 event 中的新模型名
            self._status_bar.set_model_name(new_model)
            self._invalidate()
        except Exception as e:
            logger.exception("Failed to switch model level")
            self._renderer.append_system_message(
                f"Failed to switch model: {e}",
                is_error=True,
            )

    def _update_status_bar_model(self) -> None:
        """Update status bar with current model name from session."""
        try:
            agent = getattr(self._session, "_agent", None)
            llm = getattr(agent, "llm", None)
            model = getattr(llm, "model", "")
            if model:
                self._status_bar.set_model_name(str(model))
                self._invalidate()
        except Exception:
            logger.exception("Failed to update status bar model name")

    def _enter_selection_mode(self) -> None:
        """Enter model selection menu mode."""
        # Get current level and llm_levels
        # Default to MID if level is not set
        current_level = "MID"
        llm_levels = None
        try:
            agent_level = self._session._agent.level
            if agent_level:
                current_level = agent_level
            llm_levels = self._session._agent.llm_levels
        except Exception:
            pass

        # Setup selection menu
        def on_confirm(value: str) -> None:
            self._exit_selection_mode()
            self._switch_model_level(value)
            self._refresh_layers()

        def on_cancel() -> None:
            self._exit_selection_mode()
            self._renderer.append_system_message("Model switch cancelled.")
            self._refresh_layers()

        title, options = build_model_level_options(
            current_level=current_level,
            llm_levels=llm_levels,
        )
        ok = self._selection_ui.set_options(
            title=title,
            options=options,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
        )
        if not ok:
            self._renderer.append_system_message(
                "No model options available.",
                is_error=True,
            )
            self._refresh_layers()
            return
        self._selection_ui.refresh()

        # Enter selection mode
        self._ui_mode = UIMode.SELECTION
        self._sync_focus_for_mode()
        self._invalidate()

    def _exit_selection_mode(self) -> None:
        """Exit selection menu mode."""
        self._ui_mode = UIMode.NORMAL
        self._selection_ui.clear()
        self._sync_focus_for_mode()
        self._invalidate()

    def _handle_selection_result(self, result: SelectionResult | None) -> None:
        """Handle selection result."""
        if result is None:
            return

        if not result.confirmed:
            self._exit_selection_mode()
            self._renderer.append_system_message("Model switch cancelled.")
            return

        # The callback handles the actual switch
        self._exit_selection_mode()

    async def _append_usage_snapshot(self) -> None:
        usage = await self._session.get_usage()
        include_cost = bool(getattr(self._session._agent, "include_cost", False))
        prompt_new_tokens = max(
            usage.total_prompt_tokens - usage.total_prompt_cached_tokens,
            0,
        )

        lines = [
            "Token Usage",
            f"- total: {usage.total_tokens:,}",
            f"- entries: {usage.entry_count}",
            f"- prompt: {usage.total_prompt_tokens:,}",
            f"- prompt_cached: {usage.total_prompt_cached_tokens:,}",
            f"- prompt_new: {prompt_new_tokens:,}",
            f"- completion: {usage.total_completion_tokens:,}",
        ]

        if include_cost:
            lines.extend(
                [
                    f"- prompt_cost: ${usage.total_prompt_cost:.4f}",
                    f"- completion_cost: ${usage.total_completion_cost:.4f}",
                    f"- total_cost: ${usage.total_cost:.4f}",
                ]
            )
        self._renderer.append_system_message("\n".join(lines))

    async def _append_context_snapshot(self) -> None:
        info = await self._session.get_context_info()
        utilization = float(getattr(info, "utilization_percent", 0.0))
        context_limit = getattr(info, "context_limit_tokens", None)
        used_tokens = getattr(info, "used_tokens", None)

        lines = [
            "Context Usage",
            f"- utilization: {utilization:.1f}%",
            f"- left: {max(0.0, 100.0 - utilization):.1f}%",
        ]
        if used_tokens is not None:
            lines.append(f"- used_tokens: {used_tokens}")
        if context_limit is not None:
            lines.append(f"- context_limit_tokens: {context_limit}")

        self._renderer.append_system_message("\n".join(lines))
