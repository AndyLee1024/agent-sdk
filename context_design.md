# Conversation Context 管理设计方案

## 核心概念

- **Context = Chat History** = 一次对话的完整记录
- 一个 Agent 可以创建/切换多个 Context
- Fork = 复制 JSONL 文件，在副本上继续对话

---

## 数据结构

### JSONL Entry 类型

```python
from typing import Literal, Any
from pydantic import BaseModel
from datetime import datetime

class MessageEntry(BaseModel):
    id: str
    ts: datetime
    type: Literal["msg"] = "msg"
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Any
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_calls: list[dict] | None = None
    is_error: bool = False
    is_meta: bool = False
    compacted: bool = False

class EventEntry(BaseModel):
    id: str
    ts: datetime
    type: Literal["evt"] = "evt"
    event: str
    data: dict
    ref_msg_id: str | None = None

ContextEntry = MessageEntry | EventEntry
```

### JSONL 示例

```jsonl
{"id":"m001","ts":"2026-01-28T10:00:00Z","type":"msg","role":"system","content":"You are..."}
{"id":"m002","ts":"2026-01-28T10:00:01Z","type":"msg","role":"user","content":"帮我创建PPT"}
{"id":"m003","ts":"2026-01-28T10:00:02Z","type":"msg","role":"assistant","content":null,"tool_calls":[...]}
{"id":"e001","ts":"2026-01-28T10:00:02Z","type":"evt","event":"skill_activated","data":{"skill":"pptx"}}
{"id":"m004","ts":"2026-01-28T10:00:03Z","type":"msg","role":"tool","tool_call_id":"tc001","tool_name":"Skill","content":"loaded"}
{"id":"m005","ts":"2026-01-28T10:00:03Z","type":"msg","role":"user","content":"<skill-prompt>...","is_meta":true}
{"id":"e002","ts":"2026-01-28T10:01:00Z","type":"evt","event":"compaction","data":{"from":"m001","to":"m010","summary":"m011"}}
{"id":"m011","ts":"2026-01-28T10:01:00Z","type":"msg","role":"user","content":"[Summary] ..."}
{"id":"m001","ts":"2026-01-28T10:00:00Z","type":"msg","role":"system","content":"...","compacted":true}
```

---

## ConversationContext 类（简化版）

```python
from pathlib import Path
from dataclasses import dataclass, field
import shutil

@dataclass
class ConversationContext:
    path: Path | None = None
    _entries: list[ContextEntry] = field(default_factory=list)
    _active_skills: set[str] = field(default_factory=set)
    _msg_counter: int = 0
    _pending_writes: list[ContextEntry] = field(default_factory=list)
    # ... 见详细设计 ...
```

- 支持 append、get_llm_messages、activate_skill、record_event、checkpoint、mark_compacted、save/load、fork、fork_from_checkpoint 等方法

---

## Agent 集成

- Agent 可通过 new_conversation/load_conversation/fork_conversation 管理多个 Context
- Context 负责所有消息和 Skill 状态的持久化与恢复

---

## 使用示例

- 内存模式：`agent = Agent(llm=llm, tools=tools)`
- 持久化：`agent.new_conversation(Path("./chats/task-001.jsonl"))`
- Resume：`agent.load_conversation(Path("./chats/task-001.jsonl"))`
- Fork：`agent.fork_conversation(Path("./chats/task-001-retry.jsonl"))`
- Checkpoint/Fork from checkpoint 见详细设计

---

## 文件结构

```
bu_agent_sdk/
├── context/
│   ├── __init__.py
│   ├── models.py
│   ├── context.py
│   └── serializer.py
├── agent/
│   └── service.py
└── llm/
    └── messages.py
```

---

## 设计原则

- 一个 Agent 可以管理多个 Context（即多个 chat history）
- Context 以 JSONL 形式持久化，支持 resume、fork、checkpoint
- Compaction 只标记，不物理删除
- Fork 直接复制 JSONL 文件
- 不考虑并发和大文件问题
