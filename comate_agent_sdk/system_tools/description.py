"""
System Tools rules (System Prompt layer)

Guiding principle:
- Tool contract (name/short desc/schema/strict/examples) lives in the Tool API layer.
- System prompt rules are POLICIES: when/how to use tools safely and efficiently.
- Keep global policies small and stable; inject long playbooks only when needed.
"""

# -----------------------------
# Bash
# -----------------------------

BASH_USAGE_RULES = """Executes non-interactive shell commands with argv-style args and optional timeout/cwd/env.

Core policy:
1) Prefer specialized tools over Bash:
   - Search: Grep / Glob / Task
   - List dirs: LS
   - Read files: Read
   - Modify files: Edit / MultiEdit / Write

2) Hard bans inside Bash (do NOT use these as substitutes for tools):
   - grep, find
   - cat, head, tail
   - ls

3) Quoting & paths:
   - Always quote paths with spaces using double quotes.
   - Prefer absolute paths; avoid 'cd' unless the user explicitly asks.

4) Output & safety:
   - Design commands to be concise; large outputs may be truncated.
   - When issuing multiple commands, separate with ';' or '&&' (no newlines).
   - Standardized output includes a short summary; when truncated/error, follow "Recommended next step".

EXCEPTION: You MAY use ripgrep `rg` in Bash ONLY if ALL conditions are met:
A) Scope is already narrowed to specific files/dirs (via Glob/LS). Do NOT run `rg` blindly at repo root.
B) Output is bounded: MUST include --max-count (or an equivalent hard bound).
C) Stable, parseable format: MUST include --line-number --no-heading --color=never
D) `rg` is for locating matches only (file + line numbers). Use Read to view content.
E) If you expect many matches (broad terms like "TODO", "error"), do NOT use `rg`; use Grep tool instead.

Good:
  rg --line-number --no-heading --color=never --max-count 50 "foo|bar" src/
Bad:
  rg "TODO" .
"""


# -----------------------------
# Read / Write / Edit
# -----------------------------

READ_USAGE_RULES = """Reads a file from the local filesystem (text, images, PDFs, notebooks).

Policy:
- Assume file paths provided by the user are valid. If the file does not exist, an error is returned.
- file_path supports absolute and relative paths (relative resolves against workspace first, then project root).

Required parameters:
- ALWAYS provide offset_line and limit_lines (do NOT use offset/limit).
- offset_line >= 0
- limit_lines in [1, 5000] (default 500)

Best practices:
- Default read: offset_line=0, limit_lines=500
- For large/unfamiliar codebases: prefer Grep → Read (use Grep line numbers to choose offset_line).
- Batch reads when multiple files are likely relevant.

Output characteristics:
- Returned with line numbers (cat -n style), starting at 1.
- Lines longer than 2000 chars may be truncated.
- PDFs are processed page-by-page; images are presented visually.
- Response is standardized as title + line-range summary + body.
- When truncated or error, a "Recommended next step (token-efficient)" footer is included; prefer those actions.
"""


WRITE_USAGE_RULES = """Writes a file to the local filesystem (overwrites if exists).

Rules:
- If the target file already exists, you MUST Read it earlier in the conversation before Write.
- Prefer editing existing files; do not create new files unless explicitly required by the user.
- Do NOT create documentation (*.md/README) unless explicitly requested.
- Avoid emojis in files unless the user explicitly asks.
- Output includes concise write receipt (operation/bytes/hash/path).
- Use Read immediately after write when validation is needed.
"""


EDIT_USAGE_RULES = """Performs an exact string replacement in a file.

Preconditions:
- You MUST have used Read at least once in this conversation; otherwise Edit must fail.

Matching rules:
- old_string must match file content EXACTLY (including whitespace/indentation).
- Do NOT include any line-number prefix from Read output in old_string/new_string.
- If old_string is not unique, either enlarge context to make it unique or use replace_all.

Policy:
- Prefer editing existing files; do not create new files unless explicitly required.
- Avoid emojis unless the user explicitly asks.
- Output includes replacement summary and file hashes for verification.
"""


MULTIEDIT_USAGE_RULES = """Applies multiple exact string replacements to a single file atomically.

When to use:
- Prefer MultiEdit over Edit when you need multiple changes in one file.

Preconditions:
- Read the file first (same conversation), otherwise MultiEdit must fail.
- file_path MUST be an absolute path (starts with '/').

Edits behavior:
- Edits apply sequentially; each edit sees the result of the previous edit.
- Atomic: if any edit fails, none are applied.
- Each edit requires exact match rules identical to Edit.

Tips:
- Use replace_all for systematic renames.
- Ensure earlier edits do not invalidate later matches.
- Avoid emojis unless the user explicitly asks.
- Output includes aggregate replacement summary and before/after hashes.
"""


# -----------------------------
# Search / List
# -----------------------------

GLOB_USAGE_RULES = """Fast file pattern matching across any codebase size.

Usage:
- Provide glob patterns like "**/*.py" or "src/**/*.ts"
- Returns matching file paths (sorted by modification time)

Guidance:
- Use Glob when you know name/path patterns.
- For content search use Grep.
- For open-ended multi-round exploration, use Task (if available).
- Batch multiple globs together when useful.
- Large results are truncated; use footer suggestions (pagination/artifact Read) instead of full dumps.
"""


GREP_USAGE_RULES = """Content search tool built on ripgrep (preferred default for searching).

Default policy:
- Use Grep for searches. Do NOT use `grep` or `find` in Bash.
- Do NOT use `rg` in Bash by default.

Exception:
- `rg` in Bash is allowed ONLY under the strict conditions defined in BASH_USAGE_RULES.

Capabilities:
- Regex supported (ripgrep syntax)
- Filter scope with glob or type
- Output modes: files_with_matches (default), content, count
- Multiline: set multiline=true for cross-line patterns (e.g. struct blocks)

Search strategy (performance-critical):
- Prefer ONE Grep call with regex alternation (|) over many sequential calls.
- Include naming variants (PascalCase|snake_case|camelCase) and nearby synonyms when relevant.
- When output is truncated or lower-bound, follow footer suggestions (refine pattern/path/mode or Read artifact).
"""


LS_USAGE_RULES = """Lists files/directories under a path.

Rules:
- path MUST be an absolute path (starts with '/').
- Use ignore (glob patterns) to exclude noisy paths if needed.

Guidance:
- Prefer Glob/Grep when you know what you're looking for.
- Use LS for quick directory shape confirmation and parent-dir verification before creating files/dirs.
- For large directories, use truncated output + footer guidance; avoid repeatedly requesting huge listings.
"""


# -----------------------------
# Todo / Task list
# -----------------------------

TODO_USAGE_RULES = """Creates and updates a structured todo list for the current session.

When to use:
- Multi-step work (>= 3 distinct actions), non-trivial refactors, or multi-file changes.
- User provides multiple requirements or explicitly asks for a todo list.
- After receiving new constraints that affect the plan.
- When you need progress visibility or checkpointing.

When NOT to use:
- Single trivial step, purely informational answers, or a one-off command the user requested.

Operational rules:
- States: pending, in_progress, completed
- Keep ONLY ONE task in_progress at a time.
- Mark tasks completed immediately when truly done.
- If blocked, keep task in_progress and add a new task describing the blocker/resolution.
- Remove tasks that become irrelevant.

Task schema:
- id: string (unique)
- content: string (actionable)
- status: pending|in_progress|completed
- priority: high|medium|low (default medium)
"""


# -----------------------------
# Web Fetch
# -----------------------------

WEBFETCH_USAGE_RULES = """Fetches content from a URL, converts HTML to markdown, and processes it with an AI model.

Usage:
- Inputs: url + prompt (what to extract/summarize)
- Read-only; does not modify files.
- Results may be summarized if content is very large.

Rules:
- Prefer MCP-provided web fetch tools (named like "mcp__*") if available.
- URL must be valid; http is upgraded to https automatically.
- If redirected to a different host, re-run WebFetch with the redirected URL when instructed.

Caching:
- Includes a short-lived cache to speed up repeated fetches.
"""
 
