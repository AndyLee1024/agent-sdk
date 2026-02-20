from __future__ import annotations

import asyncio
import logging
from typing import Any
from pathlib import Path

from prompt_toolkit.document import Document

from comate_agent_sdk.agent.llm_levels import ALL_LEVELS
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.messages import UserMessage

from terminal_agent.rewind_store import RewindCheckpoint, RewindRestorePlan, RewindStore
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

    async def _slash_context(self, args: str) -> None:
        normalized = args.strip()
        show_details = False
        if normalized:
            if normalized in {"--details", "-d"}:
                show_details = True
            else:
                self._renderer.append_system_message(
                    "Usage: /context [--details]",
                    is_error=True,
                )
                return
        await self._append_context_snapshot(show_details=show_details)

    async def _slash_rewind(self, args: str) -> None:
        if args.strip():
            self._renderer.append_system_message(
                "Usage: /rewind",
                is_error=True,
            )
            return
        if self._busy:
            self._renderer.append_system_message(
                "当前已有任务在运行，请稍后再执行 /rewind。",
                is_error=True,
            )
            return

        checkpoints = self._rewind_store.list_checkpoints()
        if not checkpoints:
            self._renderer.append_system_message(
                "No checkpoints yet. 先发起至少一轮用户消息再执行 /rewind。"
            )
            return
        self._show_rewind_checkpoint_menu(checkpoints)

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
            llm_levels = self._session._agent.options.llm_levels
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

    def _show_rewind_checkpoint_menu(self, checkpoints: list[RewindCheckpoint]) -> None:
        options: list[dict[str, str]] = []
        for cp in checkpoints:
            preview = cp.user_preview or "(empty)"
            label = f"#{cp.checkpoint_id} turn={cp.turn_number}: {preview}"
            desc = cp.created_at
            options.append(
                {
                    "value": str(cp.checkpoint_id),
                    "label": label,
                    "description": desc,
                }
            )

        def on_confirm(value: str) -> None:
            checkpoint = next(
                (cp for cp in checkpoints if str(cp.checkpoint_id) == value),
                None,
            )
            if checkpoint is None:
                self._renderer.append_system_message(
                    f"Checkpoint not found: {value}",
                    is_error=True,
                )
                return
            self._schedule_background(self._open_rewind_mode_menu_async(checkpoint))

        def on_cancel() -> None:
            self._renderer.append_system_message("Rewind cancelled.")

        ok = self._selection_ui.set_options(
            title="Select checkpoint",
            options=options,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
        )
        if not ok:
            self._renderer.append_system_message(
                "No checkpoint options available.",
                is_error=True,
            )
            return
        self._selection_ui.refresh()
        self._ui_mode = UIMode.SELECTION
        self._sync_focus_for_mode()
        self._invalidate()

    async def _open_rewind_mode_menu_async(self, checkpoint: RewindCheckpoint) -> None:
        await asyncio.sleep(0)
        try:
            plan = self._rewind_store.build_restore_plan(
                checkpoint_id=checkpoint.checkpoint_id,
            )
        except Exception as exc:
            self._renderer.append_system_message(
                f"Failed to prepare rewind preview: {exc}",
                is_error=True,
            )
            self._refresh_layers()
            return

        summary = self._format_restore_plan_summary(plan)
        mode_options = [
            {
                "value": "restore_both",
                "label": "Restore code and conversation",
                "description": (
                    "The conversation will be forked. "
                    f"The code will be restored {summary}."
                ),
            },
            {
                "value": "restore_conversation",
                "label": "Restore conversation",
                "description": (
                    "The conversation will be forked. "
                    "The code will be unchanged."
                ),
            },
            {
                "value": "restore_code",
                "label": "Restore code",
                "description": (
                    "The conversation will be unchanged. "
                    f"The code will be restored {summary}."
                ),
            },
            {
                "value": "never_mind",
                "label": "Never mind",
                "description": "The conversation and code will be unchanged.",
            },
        ]

        def on_confirm(mode_value: str) -> None:
            self._schedule_background(
                self._execute_rewind_mode(
                    checkpoint=checkpoint,
                    mode=mode_value,
                )
            )

        def on_cancel() -> None:
            self._renderer.append_system_message("Rewind cancelled.")

        ok = self._selection_ui.set_options(
            title=f"Checkpoint #{checkpoint.checkpoint_id} (turn={checkpoint.turn_number})",
            options=mode_options,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
        )
        if not ok:
            self._renderer.append_system_message(
                "No rewind mode options available.",
                is_error=True,
            )
            self._refresh_layers()
            return
        self._selection_ui.refresh()
        self._ui_mode = UIMode.SELECTION
        self._sync_focus_for_mode()
        self._invalidate()

    async def _execute_rewind_mode(
        self,
        *,
        checkpoint: RewindCheckpoint,
        mode: str,
    ) -> None:
        prefill_text: str | None = None
        rewind_succeeded = False
        if mode == "never_mind":
            self._renderer.append_system_message("Rewind cancelled.")
            self._refresh_layers()
            return
        if self._busy:
            self._renderer.append_system_message(
                "当前已有任务在运行，请稍后再执行 /rewind。",
                is_error=True,
            )
            self._refresh_layers()
            return

        self._set_busy(True)
        try:
            if mode == "restore_code":
                self._rewind_store.restore_code_before_checkpoint(
                    checkpoint_id=checkpoint.checkpoint_id
                )
                self._rewind_store.prune_after_checkpoint(
                    checkpoint_id=checkpoint.checkpoint_id
                )
                prefill_text = self._resolve_checkpoint_user_text(
                    checkpoint=checkpoint,
                    fallback=checkpoint.user_message,
                )
                await self._status_bar.refresh()
                rewind_succeeded = True

            elif mode == "restore_conversation":
                new_session = self._session.fork_session()
                fork_store = RewindStore(
                    session=new_session,
                    project_root=Path.cwd(),
                )
                try:
                    rewind_turn = self._rewind_turn_before_checkpoint(checkpoint.turn_number)
                    new_session.restore_conversation_to_turn(
                        target_turn=rewind_turn
                    )
                    fork_store.prune_after_checkpoint(
                        checkpoint_id=checkpoint.checkpoint_id
                    )
                    await self._replace_session(
                        new_session,
                        close_old=True,
                        replay_history=False,
                    )
                except Exception:
                    await new_session.close()
                    raise

                prefill_text = self._resolve_checkpoint_user_text(
                    checkpoint=checkpoint,
                    fallback=checkpoint.user_message,
                )
                rewind_succeeded = True

            elif mode == "restore_both":
                new_session = self._session.fork_session()
                fork_store = RewindStore(
                    session=new_session,
                    project_root=Path.cwd(),
                )
                try:
                    rewind_turn = self._rewind_turn_before_checkpoint(checkpoint.turn_number)
                    new_session.restore_conversation_to_turn(
                        target_turn=rewind_turn
                    )
                    fork_store.restore_code_before_checkpoint(
                        checkpoint_id=checkpoint.checkpoint_id
                    )
                    fork_store.prune_after_checkpoint(
                        checkpoint_id=checkpoint.checkpoint_id
                    )
                    await self._replace_session(
                        new_session,
                        close_old=True,
                        replay_history=False,
                    )
                except Exception:
                    await new_session.close()
                    raise
                prefill_text = self._resolve_checkpoint_user_text(
                    checkpoint=checkpoint,
                    fallback=checkpoint.user_message,
                )
                rewind_succeeded = True

            else:
                self._renderer.append_system_message(
                    f"Unknown rewind mode: {mode}",
                    is_error=True,
                )

            if rewind_succeeded:
                await self._replay_scrollback_after_rewind()
        except Exception as exc:
            logger.exception("rewind failed")
            self._renderer.append_system_message(
                f"Rewind failed: {exc}",
                is_error=True,
            )
        finally:
            self._set_busy(False)
            if prefill_text:
                self._prefill_user_input(prefill_text)
            self._refresh_layers()

    def _resolve_checkpoint_user_text(
        self,
        *,
        checkpoint: RewindCheckpoint,
        fallback: str,
    ) -> str:
        items = getattr(self._session._agent._context.conversation, "items", [])
        for item in reversed(items):
            if item.item_type != ItemType.USER_MESSAGE:
                continue
            if int(getattr(item, "created_turn", 0) or 0) != checkpoint.turn_number:
                continue
            message = getattr(item, "message", None)
            if isinstance(message, UserMessage) and not bool(getattr(message, "is_meta", False)):
                text = str(message.text or "").strip()
                if text:
                    return text
            content_text = str(getattr(item, "content_text", "")).strip()
            if content_text:
                return content_text
        return fallback.strip()

    def _prefill_user_input(self, text: str) -> None:
        normalized = str(text).strip()
        if not normalized:
            return
        self._clear_paste_state()
        self._last_input_len = len(normalized)
        self._last_input_text = normalized
        buffer = self._input_area.buffer
        buffer.cancel_completion()
        self._suppress_input_change_hook = True
        try:
            buffer.set_document(
                Document(normalized, cursor_position=len(normalized)),
                bypass_readonly=True,
            )
        finally:
            self._suppress_input_change_hook = False

    @staticmethod
    def _rewind_turn_before_checkpoint(turn_number: int) -> int:
        return max(0, int(turn_number) - 1)

    @staticmethod
    def _format_restore_plan_summary(plan: RewindRestorePlan) -> str:
        return (
            f"+{plan.total_added_lines} -{plan.total_removed_lines} "
            f"in {plan.writable_files_count} file(s)"
        )

    def _render_rewind_done_message(
        self,
        *,
        mode: str,
        checkpoint: RewindCheckpoint,
        plan: RewindRestorePlan,
        new_session_id: str | None,
        dropped_checkpoints: int,
    ) -> str:
        lines = [
            f"Rewind done: checkpoint #{checkpoint.checkpoint_id} (turn={checkpoint.turn_number})",
        ]
        if mode == "restore_both":
            lines.append(f"- conversation: forked to session {new_session_id}")
            lines.append(f"- code: restored {self._format_restore_plan_summary(plan)}")
        elif mode == "restore_conversation":
            lines.append(f"- conversation: forked to session {new_session_id}")
            lines.append("- code: unchanged")
        elif mode == "restore_code":
            lines.append("- conversation: unchanged")
            lines.append(f"- code: restored {self._format_restore_plan_summary(plan)}")

        if plan.skipped_binary_count > 0:
            lines.append(f"- skipped(binary): {plan.skipped_binary_count}")
        if plan.skipped_unknown_count > 0:
            lines.append(f"- skipped(unknown): {plan.skipped_unknown_count}")
        lines.append(f"- dropped_checkpoints_after_target: {dropped_checkpoints}")
        return "\n".join(lines)

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
        total_tokens = prompt_new_tokens + usage.total_completion_tokens

        lines = [
            "Token Usage",
            f"- total: {total_tokens:,}",
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

    async def _append_context_snapshot(self, *, show_details: bool = False) -> None:
        info = await self._session.get_context_info()
        context_limit = int(getattr(info, "context_limit", 0) or 0)
        next_step_estimated_tokens = int(getattr(info, "next_step_estimated_tokens", 0) or 0)
        last_step_reported_tokens = int(getattr(info, "last_step_reported_tokens", 0) or 0)
        ir_used_tokens = int(getattr(info, "used_tokens", 0) or 0)

        headroom_left_percent = 0.0
        next_step_used_percent = 0.0
        actual_used_percent = 0.0
        if context_limit > 0:
            next_step_used_percent = max(
                0.0,
                min((next_step_estimated_tokens / context_limit) * 100.0, 100.0),
            )
            headroom_left_percent = max(0.0, 100.0 - next_step_used_percent)
            if last_step_reported_tokens > 0:
                actual_used_percent = max(
                    0.0,
                    min((last_step_reported_tokens / context_limit) * 100.0, 100.0),
                )

        lines = ["Context Usage"]
        lines.append(
            f"- Headroom (est): {headroom_left_percent:.1f}% left "
            f"(est={next_step_estimated_tokens:,}, limit={context_limit:,}; includes buffer)"
        )
        if last_step_reported_tokens > 0:
            lines.append(
                f"- Last call (actual): {actual_used_percent:.1f}% used "
                f"(actual={last_step_reported_tokens:,})"
            )
        else:
            lines.append("- Last call (actual): unavailable")

        if show_details:
            lines.append("Context Details")
            lines.append(f"- next_step_estimated_tokens: {next_step_estimated_tokens:,}")
            lines.append(f"- last_step_reported_tokens: {last_step_reported_tokens:,}")
            lines.append(f"- ir_used_tokens: {ir_used_tokens:,}")
            lines.append(f"- delta_ir_vs_actual: {ir_used_tokens - last_step_reported_tokens:+,}")
            lines.extend(self._build_last_usage_breakdown_lines())

        self._renderer.append_system_message("\n".join(lines))

    def _build_last_usage_breakdown_lines(self) -> list[str]:
        """Build I/R/W/O breakdown lines from the latest usage entry."""
        token_cost = getattr(getattr(self._session, "_agent", None), "_token_cost", None)
        usage_history = getattr(token_cost, "usage_history", None)
        if not usage_history:
            return ["- breakdown(last call): unavailable"]

        latest = usage_history[-1]
        usage = getattr(latest, "usage", None)
        if usage is None:
            return ["- breakdown(last call): unavailable"]

        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cache_read_tokens = int(getattr(usage, "prompt_cached_tokens", 0) or 0)
        cache_creation_tokens = int(getattr(usage, "prompt_cache_creation_tokens", 0) or 0)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        input_tokens = max(prompt_tokens - cache_read_tokens - cache_creation_tokens, 0)

        lines = [
            (
                "- breakdown(last call): "
                f"I={input_tokens:,} (derived), "
                f"R={cache_read_tokens:,}, "
                f"W={cache_creation_tokens:,}, "
                f"O={output_tokens:,}"
            )
        ]

        prefix_total = cache_read_tokens + cache_creation_tokens
        if prefix_total > 0:
            cache_hit_ratio = (cache_read_tokens / prefix_total) * 100.0
            lines.append(f"- cache_hit_prefix: {cache_hit_ratio:.1f}% (R / (R + W))")
        else:
            lines.append("- cache_hit_prefix: n/a")

        return lines
