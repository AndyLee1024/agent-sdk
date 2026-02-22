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

BASH_USAGE_RULES = """Execute shell command. Returns stdout, stderr, exit_code (0 means success).

Critical: Use args as list: ['git', 'status'] NOT string 'git status'
For shell features (pipes, &&, etc): use ['sh', '-c', 'command here']

Set cwd to run command in specific directory (default: project root).
Set timeout_ms to prevent hanging commands (default: 30000ms).
"""


# -----------------------------
# Read / Write / Edit
# -----------------------------

READ_USAGE_RULES = """Read text file with line numbers. Use to understand file content before editing.

For large files: automatically returns has_more=true and next_offset_line for pagination.

format='line_numbers': shows line numbers (recommended for code)
format='raw': plain text without line numbers
"""


WRITE_USAGE_RULES = """Create new file or overwrite/append to existing file.

For existing files: Must Read first.
mode='overwrite' (default): replaces entire file
mode='append': adds content to end of file
"""


EDIT_USAGE_RULES = """Replace exact string in file. Must Read file first.

Critical: old_string must match EXACTLY (including whitespace/indentation).
If string appears multiple times: either add more context to old_string OR use replace_all=true.
"""


MULTIEDIT_USAGE_RULES = """Make multiple replacements in one file atomically (all succeed or all fail). Must Read first.

Each old_string must be unique in the file OR set replace_all=true.
To create new file: first edit with old_string='', new_string='full content'
"""


# -----------------------------
# Search / List
# -----------------------------

GLOB_USAGE_RULES = """Find files by name/path pattern. Use when you need to LOCATE files, not search their content.

When to use:
- Finding files by name: 'config.json', 'test_*.py'
- Listing all files of a type: '**/*.ts' (recursive), '*.md' (current dir)
- Exploring project structure

Don't use for: Searching file content (use Grep instead)
"""


GREP_USAGE_RULES = """Search file CONTENT using regex patterns. Use when you need to find WHERE code/text appears.

When to use:
- Finding function definitions, class names, API calls
- Locating error messages, TODO comments, specific logic
- Understanding where a concept is implemented

Key practice: Combine variations with '|' for better coverage
Example: 'cache|缓存|cached|caching' not just 'cache'

Don't use for: Finding files by name (use Glob instead)
"""


LS_USAGE_RULES = """List files and directories in a path (like 'ls' command). Returns name, type, size, mtime.

Use for: Exploring what's inside a specific directory
Don't use for: Finding files by pattern across subdirectories (use Glob instead)
"""


# -----------------------------
# Todo / Task list
# -----------------------------

TODO_USAGE_RULES = """Create or update task list for current session to track your work progress.

When to use:
- Multi-step work (>= 3 distinct actions), non-trivial refactors, or multi-file changes.
- User provides multiple requirements or explicitly asks for a todo list.
- After receiving new constraints that affect the plan.
- When you need progress visibility or checkpointing.

When NOT to use:
- Single trivial step, purely informational answers, or a one-off command the user requested.

Workflow: Set status to 'in_progress' when starting a task, 'completed' when done.
Update the list after completing each task to stay organized.
"""


# -----------------------------
# Web Fetch
# -----------------------------

WEBFETCH_USAGE_RULES = """Fetch webpage, convert to markdown, and summarize with small LLM based on your prompt.

Returns summary_text (LLM's response to your prompt) + artifact (full markdown).
Result is cached for 1 hour.

Use when: You need webpage content analyzed or specific info extracted.
"""


# -----------------------------
# AskUserQuestion
# -----------------------------

ASK_USER_QUESTION_USAGE_RULES = """Use this tool ONLY when executing a specific task and you need structured input from the user to proceed. This is for task execution workflow, NOT for casual conversation.

Use this tool when:
1. User has given you a task, and you need to gather specific requirements/preferences before implementation
2. You encounter ambiguous instructions during task execution that require clarification
3. You need the user to make implementation choices (e.g., framework selection, design options)
4. You need to offer multiple approaches and let user decide

DO NOT use this tool when:
- Having casual conversation or greetings (just respond naturally)
- The user hasn't given you a specific task yet
- You can reasonably infer the answer from context
- Asking meta questions about whether to start working (just start or ask naturally in your response)

Usage notes:
- Set multiSelect: true to allow multiple selections
- Put recommended options first and add "(Recommended)" to the label
- In plan mode: Use this ONLY to clarify requirements or choose approaches BEFORE finalizing the plan. Use ExitPlanMode for plan approval, NOT this tool.

Hard rule:
- AskUserQuestion MUST be called alone. If you call AskUserQuestion in a response, DO NOT call any other tools in that same response.
- The tool result means "questions have been asked / waiting for user input", NOT the user's answers. The user's answers will arrive as the next normal user message.
"""


# -----------------------------
# Exit Plan Mode
# -----------------------------

EXIT_PLAN_MODE_USAGE_RULES = """Use this tool when you are in Plan Mode and your plan artifact is ready for user approval.

Requirements:
- Provide complete markdown plan content in plan_markdown.
- Provide a concise summary for approval UI.
- Optionally provide execution_prompt for immediate act-mode execution after approval.

This tool does NOT execute the plan. It only writes plan artifact and requests approval.
"""
 
