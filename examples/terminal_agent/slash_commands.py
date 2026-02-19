from __future__ import annotations

import re
from dataclasses import dataclass

from prompt_toolkit.completion import (
    Completer,
    Completion,
    FuzzyCompleter,
    WordCompleter,
)
from prompt_toolkit.document import Document


@dataclass(frozen=True, slots=True)
class SlashCommandSpec:
    name: str
    description: str
    aliases: tuple[str, ...] = ()

    def slash_name(self) -> str:
        if self.aliases:
            return f"/{self.name} ({', '.join(self.aliases)})"
        return f"/{self.name}"


@dataclass(frozen=True, slots=True)
class _SlashCommandCall:
    name: str
    args: str
    raw_input: str


def parse_slash_command_call(user_input: str) -> _SlashCommandCall | None:
    text = user_input.strip()
    if not text or not text.startswith("/"):
        return None

    match = re.match(r"^\/([a-zA-Z0-9_-]+(?::[a-zA-Z0-9_-]+)*)", text)
    if match is None:
        return None
    if len(text) > match.end() and not text[match.end()].isspace():
        return None

    return _SlashCommandCall(
        name=match.group(1),
        args=text[match.end() :].lstrip(),
        raw_input=text,
    )


SLASH_COMMAND_SPECS: tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec(
        name="help",
        description="Show available slash commands",
        aliases=("h",),
    ),
    SlashCommandSpec(
        name="model",
        description="Switch model level (LOW/MID/HIGH)",
        aliases=("m",),
    ),
    SlashCommandSpec(name="session", description="Show current session ID"),
    SlashCommandSpec(name="usage", description="Show token usage summary"),
    SlashCommandSpec(name="context", description="Show context usage summary"),
    SlashCommandSpec(name="rewind", description="Rewind to a checkpoint"),
    SlashCommandSpec(name="exit", description="Exit terminal agent", aliases=("quit",)),
)
SLASH_COMMANDS: tuple[str, ...] = tuple(f"/{cmd.name}" for cmd in SLASH_COMMAND_SPECS)


class SlashCommandCompleter(Completer):
    def __init__(self, commands: tuple[SlashCommandSpec, ...]) -> None:
        self._commands = commands
        self._command_lookup: dict[str, list[SlashCommandSpec]] = {}
        words: list[str] = []

        for cmd in sorted(self._commands, key=lambda item: item.name):
            if cmd.name not in self._command_lookup:
                self._command_lookup[cmd.name] = []
                words.append(cmd.name)
            self._command_lookup[cmd.name].append(cmd)
            for alias in cmd.aliases:
                if alias in self._command_lookup:
                    self._command_lookup[alias].append(cmd)
                else:
                    self._command_lookup[alias] = [cmd]
                    words.append(alias)

        self._word_pattern = re.compile(r"[^\s]+")
        self._fuzzy_pattern = r"^[^\s]*"
        self._word_completer = WordCompleter(
            words,
            WORD=False,
            pattern=self._word_pattern,
        )
        self._fuzzy = FuzzyCompleter(
            self._word_completer,
            WORD=False,
            pattern=self._fuzzy_pattern,
        )

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if document.text_after_cursor.strip():
            return

        last_space = text.rfind(" ")
        token = text[last_space + 1 :]
        prefix = text[: last_space + 1] if last_space != -1 else ""
        if prefix:
            return
        if not token.startswith("/"):
            return

        typed = token[1:]
        if typed:
            commands = self._command_lookup.get(typed, [])
            if commands and any(
                typed == cmd.name or typed in cmd.aliases for cmd in commands
            ):
                return

        typed_doc = Document(text=typed, cursor_position=len(typed))
        candidates = list(self._fuzzy.get_completions(typed_doc, complete_event))
        seen: set[str] = set()

        for candidate in candidates:
            commands = self._command_lookup.get(candidate.text)
            if not commands:
                continue
            for cmd in commands:
                if cmd.name in seen:
                    continue
                seen.add(cmd.name)
                yield Completion(
                    text=f"/{cmd.name}",
                    start_position=-len(token),
                    display=cmd.slash_name(),
                    display_meta=cmd.description,
                )
