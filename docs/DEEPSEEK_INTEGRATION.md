# DeepSeek Integration Guide

## Overview

DeepSeek support has been added to the SDK with full thinking mode (reasoning_content) integration. DeepSeek uses an OpenAI-compatible API with a custom `reasoning_content` field for extended thinking capabilities.

## Quick Start

### Installation

No additional dependencies required - DeepSeek uses the standard OpenAI SDK.

### Basic Usage

```python
from comate_agent_sdk.llm import ChatDeepSeek, UserMessage

# Initialize DeepSeek client
llm = ChatDeepSeek(
    model="deepseek-reasoner",  # or "deepseek-chat"
    api_key="your-api-key",      # or set DEEPSEEK_API_KEY env var
    base_url="https://api.deepseek.com"  # default value
)

# Simple invocation
response = await llm.ainvoke(
    messages=[UserMessage(content="Explain quantum computing")]
)

# Access thinking content
if response.thinking:
    print(f"Model reasoning: {response.thinking}")
print(f"Final answer: {response.content}")
```

### Using the Model Factory

```python
import os
from comate_agent_sdk import llm

# Set environment variable
os.environ["DEEPSEEK_API_KEY"] = "your-api-key"

# Access via factory
model = llm.deepseek_chat       # deepseek-chat model
reasoner = llm.deepseek_reasoner  # deepseek-reasoner model (thinking mode)

# Or use get_llm_by_name
model = llm.get_llm_by_name("deepseek_chat")
```

## Features

### 1. Thinking Mode (reasoning_content)

DeepSeek's thinking mode provides extended reasoning capabilities through the `reasoning_content` field:

```python
llm = ChatDeepSeek(model="deepseek-reasoner")

response = await llm.ainvoke(
    messages=[UserMessage(content="Solve: 25 * 47")]
)

# Access thinking content
print(response.thinking)  # "Let me break this down... 25 * 47 = ..."
print(response.content)   # "The answer is 1175"
```

### 2. Tool Calling with Thinking

Thinking content is preserved across tool call loops:

```python
from comate_agent_sdk.llm import ToolDefinition

tools = [
    ToolDefinition(
        name="calculate",
        description="Perform arithmetic",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    )
]

response = await llm.ainvoke(
    messages=[UserMessage(content="What is 123 + 456?")],
    tools=tools
)

# Model will think, then call the tool
if response.has_tool_calls:
    print(f"Thinking: {response.thinking}")
    for tc in response.tool_calls:
        print(f"Tool: {tc.function.name}({tc.function.arguments})")
```

### 3. Auto-cleaning Historical Reasoning

By default, historical `reasoning_content` is automatically cleaned on new turns to save bandwidth:

```python
llm = ChatDeepSeek(
    model="deepseek-reasoner",
    auto_clean_reasoning=True  # default
)

# Turn 1
response1 = await llm.ainvoke([UserMessage(content="Question 1")])

# Turn 2 - historical reasoning_content will be stripped
messages = [
    UserMessage(content="Question 1"),
    AssistantMessage(content=response1.raw_content_blocks),
    UserMessage(content="Question 2")
]
response2 = await llm.ainvoke(messages)
```

To disable auto-cleaning:

```python
llm = ChatDeepSeek(
    model="deepseek-reasoner",
    auto_clean_reasoning=False
)
```

### 4. Context Cache Statistics

DeepSeek automatically uses **disk-based context caching** to reduce costs. The SDK correctly tracks cache usage:

```python
import os
os.environ["comate_agent_sdk_LLM_DEBUG"] = "1"  # Enable debug logging

response = await llm.ainvoke(messages)

# Check cache statistics
if response.usage:
    total_input = response.usage.prompt_tokens
    cache_hit = response.usage.prompt_cached_tokens or 0
    cache_miss = total_input - cache_hit

    print(f"Total input: {total_input:,} tokens")
    print(f"Cache hit: {cache_hit:,} tokens ({cache_hit/total_input*100:.1f}%)")
    print(f"Cache miss: {cache_miss:,} tokens")
```

**DeepSeek Cache Pricing**:
- Cache hit: ¥0.1 / 1M tokens (90% discount)
- Cache miss: ¥1.0 / 1M tokens (normal price)

**Debug Output Example**:
```
📊 deepseek-reasoner: 200 in + 800 cached (hit) + 150 out
   💾 Cache hit rate: 80.0%
```

**How Cache Works** (from DeepSeek docs):
- Caches are built for every request (64 tokens minimum)
- Only matching **prefixes** trigger cache hits
- Caches last for hours to days
- Best for: Few-shot learning, long documents, multi-turn conversations

## Configuration

### Environment Variables

```bash
# Required
export DEEPSEEK_API_KEY="your-api-key"

# Optional
export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # default value

# Enable debug logging to see token usage and cache statistics
export comate_agent_sdk_LLM_DEBUG=1
```

### Model Parameters

```python
llm = ChatDeepSeek(
    model="deepseek-reasoner",
    api_key="...",
    base_url="...",

    # OpenAI-compatible parameters
    temperature=0.7,
    max_completion_tokens=16384,
    top_p=1.0,
    seed=42,

    # DeepSeek-specific
    auto_clean_reasoning=True,
)
```

## Models

| Model | Description | Thinking Mode |
|-------|-------------|---------------|
| `deepseek-chat` | Standard chat model | No |
| `deepseek-reasoner` | Extended thinking model | Yes |

## Architecture Details

### Thinking Infrastructure

DeepSeek's `reasoning_content` is mapped to the SDK's unified thinking infrastructure:

1. **Response Parsing**: `reasoning_content` → `ChatInvokeCompletion.thinking`
2. **Structured Content**: Preserved in `raw_content_blocks` as `ContentPartThinkingParam`
3. **Serialization**: `ContentPartThinkingParam` → `reasoning_content` (for tool call loops)

### Serialization Flow

```
User Input
    ↓
SDK serializes messages
    ↓
thinking blocks → reasoning_content
    ↓
DeepSeek API
    ↓
Response with reasoning_content
    ↓
SDK parses → ChatInvokeCompletion
    ↓
Saved as raw_content_blocks
    ↓
Next turn: raw_content_blocks → reasoning_content
```

## Troubleshooting

### "Missing `thinking` block" Error

This error occurs when DeepSeek expects `reasoning_content` but it's missing in the request. Solutions:

1. **Use ChatDeepSeek instead of ChatOpenAI**: DeepSeek has a custom protocol that requires specific handling.

2. **Check auto_clean_reasoning**: Ensure historical reasoning is preserved in tool call loops (same turn).

3. **Verify message structure**: Make sure `AssistantMessage.content` uses `raw_content_blocks` from previous responses.

### Missing Thinking Content

If `response.thinking` is None when you expect it:

1. **Check model**: Only `deepseek-reasoner` supports thinking mode
2. **Check response**: Some queries may not trigger thinking
3. **Check API limits**: Verify your API key has access to thinking features

## Examples

### Example: Multi-turn Conversation

```python
from comate_agent_sdk.llm import ChatDeepSeek, UserMessage, AssistantMessage

llm = ChatDeepSeek(model="deepseek-reasoner")

# Turn 1
response1 = await llm.ainvoke([
    UserMessage(content="What is machine learning?")
])
print(f"Thinking: {response1.thinking}")
print(f"Answer: {response1.content}")

# Turn 2
messages = [
    UserMessage(content="What is machine learning?"),
    AssistantMessage(content=response1.raw_content_blocks),
    UserMessage(content="How does it differ from traditional programming?")
]
response2 = await llm.ainvoke(messages)
print(f"Answer: {response2.content}")
```

### Example: Tool Calling Loop

```python
from comate_agent_sdk.llm import ChatDeepSeek, UserMessage, AssistantMessage, ToolMessage
from comate_agent_sdk.llm.base import ToolDefinition

llm = ChatDeepSeek(model="deepseek-reasoner")

# Define tools
tools = [
    ToolDefinition(name="search", description="Search the web", parameters={...}),
    ToolDefinition(name="calculate", description="Perform math", parameters={...})
]

# Initial request
response = await llm.ainvoke(
    messages=[UserMessage(content="What is 25% of 1000?")],
    tools=tools
)

# Execute tools
if response.has_tool_calls:
    tool_results = []
    for tc in response.tool_calls:
        result = execute_tool(tc)  # Your tool execution logic
        tool_results.append(ToolMessage(
            tool_call_id=tc.id,
            content=result
        ))

    # Continue conversation with tool results
    messages = [
        UserMessage(content="What is 25% of 1000?"),
        AssistantMessage(
            content=response.raw_content_blocks,  # Preserves thinking!
            tool_calls=response.tool_calls
        ),
        *tool_results
    ]

    final_response = await llm.ainvoke(messages, tools=tools)
    print(final_response.content)
```

## Testing

Run the test suite:

```bash
# Offline tests (no API key required)
uv run python test_deepseek_offline.py

# Full integration tests (requires DEEPSEEK_API_KEY)
uv run python test_deepseek.py
```

## References

- [DeepSeek API Documentation](https://platform.deepseek.com/api-docs/)
- [DeepSeek Thinking Mode](https://platform.deepseek.com/api-docs/thinking-mode/)
- SDK Thinking Infrastructure: `comate_agent_sdk/llm/messages.py` (ContentPartThinkingParam)
