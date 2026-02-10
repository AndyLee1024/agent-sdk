# Tool Result Mock Outputs (Standardized)

本文展示每个 system tool 在新格式下的 mock 返回，分为：

- `LLM-visible text`：注入给模型的标准化文本
- `execution_meta`：写入 `ContextItem.metadata.tool_execution_meta`
- `raw_envelope`：写入 `ContextItem.metadata.tool_raw_envelope`

## Read (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "content": "     1\tdef main():\n     2\t    return 42",
    "total_lines": 9321,
    "lines_returned": 2,
    "has_more": true,
    "next_offset_line": 2,
    "truncated": true,
    "artifact": {
      "relpath": ".agent_workspace/.artifacts/read/a1.txt"
    }
  },
  "message": null,
  "error": null,
  "meta": {
    "file_path": "app/main.py",
    "offset_line": 0,
    "limit_lines": 2,
    "file_bytes": 120394
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# Read: app/main.py

Lines 1-2 of 9321 (TRUNCATED: line_limit, output_spilled)

     1	def main():
     2	    return 42

Recommended next step (token-efficient):
* Read(file_path="app/main.py", offset_line=2, limit_lines=500)
* Grep(pattern="<keyword>", path="app/main.py", output_mode="content")
* Read(file_path=".agent_workspace/.artifacts/read/a1.txt", format="raw", offset_line=0, limit_lines=500)
```

### execution_meta
```json
{
  "tool_name": "Read",
  "tool_call_id": "tc_read",
  "status": "ok",
  "truncation": {
    "truncated": true,
    "reason": "line_limit,output_spilled",
    "shown_range": {
      "start_line": 1,
      "end_line": 2
    },
    "total_estimate": 9321
  },
  "retrieval_hints": [
    {
      "action": "Read",
      "priority": "high",
      "args": {
        "file_path": "app/main.py",
        "offset_line": 2,
        "limit_lines": 500
      }
    },
    {
      "action": "Grep",
      "priority": "medium",
      "args": {
        "pattern": "<keyword>",
        "path": "app/main.py",
        "output_mode": "content"
      }
    },
    {
      "action": "Read",
      "priority": "low",
      "args": {
        "file_path": ".agent_workspace/.artifacts/read/a1.txt",
        "format": "raw",
        "offset_line": 0,
        "limit_lines": 500
      }
    }
  ]
}
```

## Write (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "bytes_written": 128,
    "file_bytes": 1024,
    "created": false,
    "sha256": "after_sha",
    "relpath": "config/settings.json"
  },
  "message": null,
  "error": null,
  "meta": {
    "file_path": "config/settings.json",
    "operation": "overwrite",
    "sha256_before": "before_sha",
    "sha256_after": "after_sha"
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# Write: config/settings.json

Operation: overwrite (created=False)
Bytes written: 128
File bytes: 1024
SHA256 before: before_sha
SHA256 after: after_sha
Path: config/settings.json
```

### execution_meta
```json
{
  "tool_name": "Write",
  "tool_call_id": "tc_write",
  "status": "ok",
  "file_ops": {
    "file_path": "config/settings.json",
    "operation": "overwrite",
    "bytes_written": 128,
    "sha256_before": "before_sha",
    "sha256_after": "after_sha"
  },
  "retrieval_hints": [
    {
      "action": "Read",
      "priority": "high",
      "args": {
        "file_path": "config/settings.json",
        "offset_line": 0,
        "limit_lines": 200
      }
    }
  ]
}
```

## Edit (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "replacements": 1,
    "before_sha256": "b_sha",
    "after_sha256": "a_sha",
    "relpath": "src/service.py"
  },
  "message": null,
  "error": null,
  "meta": {
    "file_path": "src/service.py",
    "operation": "replace",
    "replace_all": false,
    "replacements": 1,
    "sha256_before": "b_sha",
    "sha256_after": "a_sha"
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# Edit: src/service.py

Replacements: 1
SHA256 before: b_sha
SHA256 after: a_sha
Path: src/service.py
```

### execution_meta
```json
{
  "tool_name": "Edit",
  "tool_call_id": "tc_edit",
  "status": "ok",
  "file_ops": {
    "file_path": "src/service.py",
    "operation": "replace",
    "bytes_written": 0,
    "sha256_before": "b_sha",
    "sha256_after": "a_sha"
  },
  "retrieval_hints": [
    {
      "action": "Read",
      "priority": "high",
      "args": {
        "file_path": "src/service.py",
        "offset_line": 0,
        "limit_lines": 200
      }
    }
  ]
}
```

## MultiEdit (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "total_replacements": 3,
    "created": false,
    "before_sha256": "old_sha",
    "after_sha256": "new_sha",
    "bytes": 4096,
    "relpath": "src/core.py"
  },
  "message": null,
  "error": null,
  "meta": {
    "file_path": "src/core.py",
    "operation": "multi_edit",
    "created": false,
    "edit_count": 2,
    "total_replacements": 3,
    "sha256_before": "old_sha",
    "sha256_after": "new_sha"
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# MultiEdit: src/core.py

Total replacements: 3
Created: False
File bytes: 4096
SHA256 before: old_sha
SHA256 after: new_sha
Path: src/core.py
```

### execution_meta
```json
{
  "tool_name": "MultiEdit",
  "tool_call_id": "tc_multiedit",
  "status": "ok",
  "file_ops": {
    "file_path": "src/core.py",
    "operation": "multi_edit",
    "bytes_written": 4096,
    "sha256_before": "old_sha",
    "sha256_after": "new_sha"
  },
  "retrieval_hints": [
    {
      "action": "Read",
      "priority": "high",
      "args": {
        "file_path": "src/core.py",
        "offset_line": 0,
        "limit_lines": 200
      }
    }
  ]
}
```

## Glob (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "matches": [
      "src/a.py",
      "src/b.py",
      "src/c.py"
    ],
    "count": 328,
    "search_path": "/repo/src",
    "truncated": true,
    "artifact": {
      "relpath": ".agent_workspace/.artifacts/glob/a2.json"
    }
  },
  "message": null,
  "error": null,
  "meta": {
    "pattern": "**/*.py",
    "path": "/repo/src",
    "search_path": "/repo/src",
    "head_limit": 100
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# Glob: **/*.py

Path: /repo/src
Matches shown: 3 of 328 (TRUNCATED)

- src/a.py
- src/b.py
- src/c.py

Recommended next step (token-efficient):
* Glob(pattern="**/*.py", path="/repo/src", head_limit=328)
* Read(file_path=".agent_workspace/.artifacts/glob/a2.json", format="raw", offset_line=0, limit_lines=500)
```

### execution_meta
```json
{
  "tool_name": "Glob",
  "tool_call_id": "tc_glob",
  "status": "ok",
  "truncation": {
    "truncated": true,
    "reason": "output_spilled",
    "total_estimate": 328
  },
  "retrieval_hints": [
    {
      "action": "Glob",
      "priority": "high",
      "args": {
        "pattern": "**/*.py",
        "path": "/repo/src",
        "head_limit": 328
      }
    },
    {
      "action": "Read",
      "priority": "low",
      "args": {
        "file_path": ".agent_workspace/.artifacts/glob/a2.json",
        "format": "raw",
        "offset_line": 0,
        "limit_lines": 500
      }
    }
  ]
}
```

## Grep (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "matches": [
      {
        "file": "src/a.py",
        "line_number": 88,
        "line": "class Foo:",
        "before_context": null,
        "after_context": null
      },
      {
        "file": "src/b.py",
        "line_number": 13,
        "line": "class FooBar:",
        "before_context": null,
        "after_context": null
      }
    ],
    "total_matches": 512,
    "truncated": true,
    "total_matches_is_lower_bound": true,
    "artifact": {
      "relpath": ".agent_workspace/.artifacts/grep_content/a3.txt"
    }
  },
  "message": null,
  "error": null,
  "meta": {
    "engine": "rg",
    "pattern": "class\\s+Foo",
    "path": "/repo/src",
    "search_path": "/repo/src",
    "output_mode": "content",
    "head_limit": 100
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# Grep (content): class\s+Foo

Path: /repo/src

Matches shown: 2 of 512 (LOWER BOUND) (TRUNCATED)

- src/a.py:88 | class Foo:
- src/b.py:13 | class FooBar:

Recommended next step (token-efficient):
* Read(file_path="src/a.py", offset_line=68, limit_lines=120)
* Grep(pattern="class\\s+Foo", path="/repo/src", output_mode="content", head_limit=300)
* Read(file_path=".agent_workspace/.artifacts/grep_content/a3.txt", format="raw", offset_line=0, limit_lines=500)
```

### execution_meta
```json
{
  "tool_name": "Grep",
  "tool_call_id": "tc_grep",
  "status": "ok",
  "truncation": {
    "truncated": true,
    "reason": "output_spilled",
    "total_estimate": 512
  },
  "retrieval_hints": [
    {
      "action": "Read",
      "priority": "high",
      "args": {
        "file_path": "src/a.py",
        "offset_line": 68,
        "limit_lines": 120
      }
    },
    {
      "action": "Grep",
      "priority": "medium",
      "args": {
        "pattern": "class\\s+Foo",
        "path": "/repo/src",
        "output_mode": "content",
        "head_limit": 300
      }
    },
    {
      "action": "Read",
      "priority": "low",
      "args": {
        "file_path": ".agent_workspace/.artifacts/grep_content/a3.txt",
        "format": "raw",
        "offset_line": 0,
        "limit_lines": 500
      }
    }
  ]
}
```

## LS (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "entries": [
      {
        "name": "a.py",
        "type": "file",
        "size": 132,
        "mtime": 1738970101
      },
      {
        "name": "pkg",
        "type": "dir",
        "size": 0,
        "mtime": 1738970102
      }
    ],
    "count": 540,
    "truncated": true,
    "path": "/repo/src",
    "artifact": {
      "relpath": ".agent_workspace/.artifacts/ls/a4.json"
    }
  },
  "message": null,
  "error": null,
  "meta": {
    "path": "/repo/src",
    "resolved_path": "/repo/src",
    "head_limit": 100,
    "include_hidden": false,
    "sort_by": "name"
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# LS: /repo/src

Entries shown: 2 of 540 (sorted by name) (TRUNCATED)

- [FILE] a.py (size=132)
- [DIR] pkg (size=0)

Recommended next step (token-efficient):
* LS(path="/repo/src", head_limit=540, sort_by="name")
* Read(file_path=".agent_workspace/.artifacts/ls/a4.json", format="raw", offset_line=0, limit_lines=500)
```

### execution_meta
```json
{
  "tool_name": "LS",
  "tool_call_id": "tc_ls",
  "status": "ok",
  "truncation": {
    "truncated": true,
    "reason": "output_spilled",
    "total_estimate": 540
  },
  "retrieval_hints": [
    {
      "action": "LS",
      "priority": "high",
      "args": {
        "path": "/repo/src",
        "head_limit": 540,
        "sort_by": "name"
      }
    },
    {
      "action": "Read",
      "priority": "low",
      "args": {
        "file_path": ".agent_workspace/.artifacts/ls/a4.json",
        "format": "raw",
        "offset_line": 0,
        "limit_lines": 500
      }
    }
  ]
}
```

## Bash (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "stdout": "line1\\nline2",
    "stderr": "",
    "exit_code": 0,
    "timed_out": false,
    "killed": false,
    "duration_ms": 123,
    "truncated": true,
    "artifact": {
      "relpath": ".agent_workspace/.artifacts/bash/a5.json"
    }
  },
  "message": null,
  "error": null,
  "meta": {
    "args": [
      "rg",
      "-n",
      "TODO",
      "src"
    ],
    "cwd": "/repo",
    "timeout_ms": 120000,
    "max_output_chars": 30000,
    "duration_ms": 123
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# Bash: rg -n TODO src

Exit code: 0
Timed out: False
Duration: 123.0ms

stdout:
line1\nline2

stderr:
(empty)

Recommended next step (token-efficient):
* Read(file_path=".agent_workspace/.artifacts/bash/a5.json", format="raw", offset_line=0, limit_lines=500)
* Bash(args=["<command>", "<with narrower output>"], max_output_chars=5000)
```

### execution_meta
```json
{
  "tool_name": "Bash",
  "tool_call_id": "tc_bash",
  "status": "ok",
  "truncation": {
    "truncated": true,
    "reason": "output_spilled",
    "total_estimate": 0
  },
  "retrieval_hints": [
    {
      "action": "Read",
      "priority": "low",
      "args": {
        "file_path": ".agent_workspace/.artifacts/bash/a5.json",
        "format": "raw",
        "offset_line": 0,
        "limit_lines": 500
      }
    },
    {
      "action": "Bash",
      "priority": "medium",
      "args": {
        "args": [
          "<command>",
          "<with narrower output>"
        ],
        "max_output_chars": 5000
      }
    }
  ],
  "duration_ms": 123.0
}
```

## TodoWrite (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "count": 4,
    "active_count": 2,
    "persisted": true,
    "todo_path": "todos.json"
  },
  "message": "Remember to keep using the TODO list...",
  "error": null,
  "meta": {
    "count": 4,
    "active_count": 2,
    "persisted": true,
    "todo_path": "todos.json"
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# TodoWrite

Total todos: 4
Active todos: 2
Persisted: True
Todo path: todos.json

Remember to keep using the TODO list...
```

### execution_meta
```json
{
  "tool_name": "TodoWrite",
  "tool_call_id": "tc_todowrite",
  "status": "ok"
}
```

## WebFetch (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "final_url": "https://example.com/page",
    "status": 200,
    "cached": false,
    "artifact": {
      "relpath": ".agent_workspace/.artifacts/webfetch/a6.md"
    },
    "truncated_for_llm": true,
    "summary_text": "This page explains...",
    "model_used": "gpt-4.1-mini"
  },
  "message": null,
  "error": null,
  "meta": {
    "url": "https://example.com",
    "prompt_length": 34,
    "final_url": "https://example.com/page",
    "status": 200,
    "cached": false,
    "truncated_for_llm": true,
    "model_used": "gpt-4.1-mini"
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# WebFetch: https://example.com/page

Status: 200
Cached: False
Model: gpt-4.1-mini

Summary:
This page explains...

Recommended next step (token-efficient):
* Read(file_path=".agent_workspace/.artifacts/webfetch/a6.md", format="raw", offset_line=0, limit_lines=500)
```

### execution_meta
```json
{
  "tool_name": "WebFetch",
  "tool_call_id": "tc_webfetch",
  "status": "ok",
  "truncation": {
    "truncated": true,
    "reason": "llm_input_limit",
    "total_estimate": 0
  },
  "retrieval_hints": [
    {
      "action": "Read",
      "priority": "high",
      "args": {
        "file_path": ".agent_workspace/.artifacts/webfetch/a6.md",
        "format": "raw",
        "offset_line": 0,
        "limit_lines": 500
      }
    }
  ]
}
```

## AskUserQuestion (Success)

### raw_envelope
```json
{
  "ok": true,
  "data": {
    "status": "waiting_for_input",
    "questions": [
      {
        "question": "Which auth method should we use?",
        "header": "Auth",
        "options": [
          {
            "label": "JWT",
            "description": "Stateless"
          },
          {
            "label": "Session",
            "description": "Server-side"
          }
        ],
        "multiSelect": false
      }
    ]
  },
  "message": "Prepared 1 question(s) for user",
  "error": null,
  "meta": {
    "status": "waiting_for_input",
    "question_count": 1
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# AskUserQuestion

Status: waiting_for_input
Question count: 1

1. [Auth] Which auth method should we use?
```

### execution_meta
```json
{
  "tool_name": "AskUserQuestion",
  "tool_call_id": "tc_askuserquestion",
  "status": "ok"
}
```

## Read (Error)

### raw_envelope
```json
{
  "ok": false,
  "data": {},
  "message": null,
  "error": {
    "code": "NOT_FOUND",
    "message": "File not found: missing.py",
    "field_errors": [],
    "retryable": false
  },
  "meta": {},
  "schema_version": 1
}
```

### LLM-visible text
```text
# Read Error

Code: NOT_FOUND
Message: File not found: missing.py
Retryable: False

Recommended next step (token-efficient):
* LS(path="<parent_directory>", head_limit=200)
```

### execution_meta
```json
{
  "tool_name": "Read",
  "tool_call_id": "tc_read_err",
  "status": "error",
  "retrieval_hints": [
    {
      "action": "LS",
      "priority": "medium",
      "args": {
        "path": "<parent_directory>",
        "head_limit": 200
      }
    }
  ],
  "error_code": "NOT_FOUND"
}
```

## Grep (Error)

### raw_envelope
```json
{
  "ok": false,
  "data": {},
  "message": null,
  "error": {
    "code": "INVALID_ARGUMENT",
    "message": "invalid regex pattern",
    "field_errors": [
      {
        "field": "pattern",
        "message": "unterminated character set"
      }
    ],
    "retryable": false
  },
  "meta": {},
  "schema_version": 1
}
```

### LLM-visible text
```text
# Grep Error

Code: INVALID_ARGUMENT
Message: invalid regex pattern
Retryable: False

Field errors:
- pattern: unterminated character set

Recommended next step (token-efficient):
* Grep(pattern="<correct_value>")
```

### execution_meta
```json
{
  "tool_name": "Grep",
  "tool_call_id": "tc_grep_err",
  "status": "error",
  "retrieval_hints": [
    {
      "action": "Grep",
      "priority": "high",
      "args": {
        "pattern": "<correct_value>"
      }
    }
  ],
  "error_code": "INVALID_ARGUMENT",
  "error_field": "pattern"
}
```

## Bash (Error)

### raw_envelope
```json
{
  "ok": false,
  "data": {},
  "message": null,
  "error": {
    "code": "TIMEOUT",
    "message": "Command timed out after 120000ms",
    "field_errors": [],
    "retryable": true
  },
  "meta": {
    "timed_out": true,
    "truncated": true,
    "artifact": {
      "relpath": ".agent_workspace/.artifacts/bash/timed_out.json"
    }
  },
  "schema_version": 1
}
```

### LLM-visible text
```text
# Bash Error

Code: TIMEOUT
Message: Command timed out after 120000ms
Retryable: True

Recommended next step (token-efficient):
* Bash(retry=true)
```

### execution_meta
```json
{
  "tool_name": "Bash",
  "tool_call_id": "tc_bash_err",
  "status": "error",
  "truncation": {
    "truncated": true,
    "reason": "output_spilled",
    "total_estimate": 0
  },
  "retrieval_hints": [
    {
      "action": "Bash",
      "priority": "medium",
      "args": {
        "retry": true
      }
    }
  ],
  "error_code": "TIMEOUT"
}
```

