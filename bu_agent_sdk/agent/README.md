# Agent

Simple agentic loop with native tool calling.

## Quick Start

```python
from bu_agent_sdk import Agent
from bu_agent_sdk.agent import ComateAgentOptions
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import tool

@tool("Add two numbers")
async def add(a: int, b: int) -> int:
    return a + b

agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    options=ComateAgentOptions(tools=[add]),
)

result = await agent.query("What is 2 + 2?")
```

## Streaming Events

See exactly what the agent is doing:

```python
from bu_agent_sdk.agent import StopEvent, TextEvent, ToolCallEvent, ToolResultEvent

async for event in agent.query_stream("do something"):
    match event:
        case ToolCallEvent(tool=name, args=args):
            print(f"Calling {name}: {args}")
        case ToolResultEvent(tool=name, result=result):
            print(f"{name} returned: {result}")
        case TextEvent(content=text):
            print(text)
        case StopEvent(reason=reason):
            print(f"Done: {reason}")
```

## Pydantic Models

Group related parameters:

```python
from pydantic import BaseModel, Field

class EmailParams(BaseModel):
    to: str = Field(description="Recipient")
    subject: str
    body: str

@tool("Send an email")
async def send_email(params: EmailParams) -> str:
    return f"Sent to {params.to}"
```

## Dependency Injection

Inject shared resources (DB, API clients):

```python
from typing import Annotated
from bu_agent_sdk.tools import Depends

def get_db() -> Database:
    return Database()

@tool("Query database")
async def query(sql: str, db: Annotated[Database, Depends(get_db)]) -> str:
    return await db.execute(sql)
```

## Multi-turn Conversations

History is preserved between calls:

```python
await agent.query("My name is Alice")
await agent.query("What's my name?")  # Remembers "Alice"
agent.clear_history()  # Reset
```

## Token Usage

```python
usage = await agent.get_usage()
print(f"Tokens: {usage.total_tokens}, Cost: ${usage.total_cost:.4f}")
```
