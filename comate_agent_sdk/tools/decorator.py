"""
Tool decorator for creating type-safe, auto-documented tools.

Usage:
    @tool("Search the web for information")
    async def search(query: str, max_results: int = 10) -> str:
        # Implementation here
        return "results"

    # With Pydantic models for complex parameters
    class SearchParams(BaseModel):
        query: str
        max_results: int = 10

    @tool("Search with complex params")
    async def search_complex(params: SearchParams) -> str:
        return "results"

    # With dependency injection (type-safe)
    from typing import Annotated

    @tool("Query database")
    async def query(sql: str, db: Annotated[Database, Depends(get_db)]) -> str:
        return await db.execute(sql)
"""

import inspect
import json
import types
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    Literal,
    ParamSpec,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from pydantic import BaseModel

from comate_agent_sdk.llm.base import ToolDefinition
from comate_agent_sdk.llm.messages import ContentPartImageParam, ContentPartTextParam
from comate_agent_sdk.llm.schema import SchemaOptimizer
from comate_agent_sdk.tools.depends import Depends

# Type alias for tool return content
ToolContent = str | list[ContentPartTextParam | ContentPartImageParam]

T = TypeVar("T")
P = ParamSpec("P")


def _python_type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Convert a Python type to JSON schema type."""
    origin = get_origin(python_type)

    # Handle Union types (including X | None and Optional[X])
    if origin is Union or origin is types.UnionType:
        args = get_args(python_type)
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            # Optional[X] or X | None - just return the non-None type
            return _python_type_to_json_schema(non_none_args[0])
        # Multiple non-None types - use anyOf
        return {"anyOf": [_python_type_to_json_schema(a) for a in non_none_args]}

    # Handle Literal types
    if origin is Literal:
        values = get_args(python_type)
        return {"type": "string", "enum": list(values)}

    # Handle basic types
    if python_type is str:
        return {"type": "string"}
    elif python_type is int:
        return {"type": "integer"}
    elif python_type is float:
        return {"type": "number"}
    elif python_type is bool:
        return {"type": "boolean"}
    elif python_type is type(None):
        return {"type": "null"}

    # Handle list types
    if origin is list:
        args = get_args(python_type)
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}

    # Handle dict types (both dict and dict[K, V])
    if origin is dict or python_type is dict:
        args = get_args(python_type)
        if len(args) >= 2:
            return {
                "type": "object",
                "additionalProperties": _python_type_to_json_schema(args[1]),
            }
        return {"type": "object"}

    # Handle Pydantic models
    if inspect.isclass(python_type) and issubclass(python_type, BaseModel):
        return python_type.model_json_schema()

    # Handle TypedDict
    if is_typeddict(python_type):
        hints = get_type_hints(python_type)
        required_keys = getattr(python_type, "__required_keys__", set())
        properties = {k: _python_type_to_json_schema(v) for k, v in hints.items()}
        return {
            "type": "object",
            "properties": properties,
            "required": list(required_keys),
            "additionalProperties": False,
        }

    # Default to string for unknown types
    return {"type": "string"}


def _get_param_description(func: Callable, param_name: str) -> str | None:
    """Extract parameter description from docstring if available."""
    docstring = func.__doc__
    if not docstring:
        return None

    # Simple docstring parsing for Args section
    lines = docstring.split("\n")
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if stripped.startswith(param_name + ":"):
                return stripped.split(":", 1)[1].strip()
            if stripped and not stripped.startswith(" ") and ":" in stripped:
                # Check if it's another param
                potential_param = stripped.split(":")[0].strip()
                if potential_param != param_name:
                    continue
            if stripped.lower().startswith(("returns:", "raises:", "example")):
                break
    return None


@dataclass
class Tool:
    """
    Wrapper class for a tool function with its metadata and definition.

    This class is created by the @tool decorator and provides:
    - Automatic JSON schema generation from function signature
    - Dependency injection resolution
    - Type-safe execution with result serialization
    - Ephemeral output management (keep last N outputs in context)
    """

    func: Callable[..., Awaitable[Any]]
    description: str
    name: str = field(default="")
    ephemeral: int | bool = False
    """How many outputs to keep in context. False=not ephemeral, True=keep 1, int=keep N."""
    usage_rules: str | None = field(default=None, repr=False)
    """Detailed usage rules for tool API layer. If provided, it overrides ToolDefinition.description."""
    _definition: ToolDefinition | None = field(default=None, repr=False)
    _dependencies: dict[str, Depends] = field(default_factory=dict, repr=False)
    _param_types: dict[str, type] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = self.func.__name__

        # Analyze function signature
        self._analyze_signature()

    def _analyze_signature(self):
        """Analyze function signature to extract parameters and dependencies."""
        sig = inspect.signature(self.func)

        # Get type hints, including Annotated types
        try:
            hints = get_type_hints(self.func, include_extras=True)
        except Exception:
            hints = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Check if default value is a Depends instance (FastAPI style: x = Depends(fn))
            if isinstance(param.default, Depends):
                self._dependencies[param_name] = param.default
                continue

            # Get the type hint for this parameter
            hint = hints.get(param_name)

            # Check for Annotated[Type, Depends(...)]
            if get_origin(hint) is Annotated:
                args = get_args(hint)
                actual_type = args[0] if args else str
                # Look for Depends in the metadata
                for metadata in args[1:]:
                    if isinstance(metadata, Depends):
                        self._dependencies[param_name] = metadata
                        break
                else:
                    # No Depends found, treat as regular parameter
                    self._param_types[param_name] = actual_type
                continue

            # Check if annotation is a Depends instance directly (legacy style)
            if isinstance(hint, Depends):
                self._dependencies[param_name] = hint
                continue

            # Regular parameter
            if hint is None:
                hint = str
            if get_origin(hint) is type(None):
                continue
            self._param_types[param_name] = hint

    def _build_tool_definition_description(self) -> str:
        """Resolve description sent to model tool definition.

        Priority:
        1) usage_rules (detailed and actionable policy)
        2) description (short summary fallback)
        """
        if self.usage_rules and self.usage_rules.strip():
            return self.usage_rules.strip()
        return self.description

    @property
    def definition(self) -> ToolDefinition:
        """Generate the ToolDefinition for this tool."""
        if self._definition is not None:
            return self._definition

        # Build JSON schema from parameters
        properties: dict[str, Any] = {}
        required: list[str] = []

        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)

        for param_name, param_type in self._param_types.items():
            # Check if it's a Pydantic model (single model parameter pattern)
            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                # Use optimized schema that ensures additionalProperties: false for OpenAI
                schema = SchemaOptimizer.create_optimized_json_schema(param_type)
                self._definition = ToolDefinition(
                    name=self.name,
                    description=self._build_tool_definition_description(),
                    parameters=schema,
                    strict=True,
                )
                return self._definition

            # Build property schema
            prop_schema = _python_type_to_json_schema(param_type)

            # Add description from docstring if available
            param_desc = _get_param_description(self.func, param_name)
            if param_desc:
                prop_schema["description"] = param_desc

            properties[param_name] = prop_schema

            # Check if required (no default value)
            param = sig.parameters[param_name]
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

        self._definition = ToolDefinition(
            name=self.name,
            description=self._build_tool_definition_description(),
            parameters=schema,
            strict=True,
        )
        return self._definition

    async def execute(
        self, _overrides: dict | None = None, **kwargs: Any
    ) -> ToolContent:
        """
        Execute the tool function with the given arguments.

        Handles:
        - Dependency injection resolution (with optional overrides)
        - Pydantic model instantiation
        - Result serialization (strings, dicts, or content part lists for images)

        Args:
            _overrides: Optional dependency overrides dict. Maps original
                       dependency functions to replacement functions.
            **kwargs: Tool arguments from the LLM.

        Returns:
            Either a string or a list of content parts (for images/multimodal).
        """
        # Resolve dependencies (with optional overrides)
        resolved_deps = {}
        for dep_name, depends in self._dependencies.items():
            resolved_deps[dep_name] = await depends.resolve(_overrides)

        # Merge dependencies with provided kwargs
        call_kwargs = {**kwargs, **resolved_deps}

        # Handle Pydantic model parameters
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)

        for param_name, param_type in self._param_types.items():
            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                # If we have a single Pydantic model param, instantiate it from kwargs
                if param_name in call_kwargs:
                    if isinstance(call_kwargs[param_name], dict):
                        call_kwargs[param_name] = param_type(**call_kwargs[param_name])
                else:
                    # Kwargs are the model fields directly.
                    # Keep all user-provided keys so model_config(extra="forbid")
                    # can reject unknown fields instead of silently dropping them.
                    model_kwargs = dict(kwargs)
                    call_kwargs = {
                        param_name: param_type(**model_kwargs),
                        **resolved_deps,
                    }
                    break

        # Execute the function
        result = await self.func(**call_kwargs)

        # Serialize result to string
        return self._serialize_result(result)

    def _serialize_result(self, result: Any) -> ToolContent:
        """Serialize the tool result to string or content parts.

        Returns content parts (list) directly for multimodal results,
        otherwise serializes to string.
        """
        if result is None:
            return ""
        if isinstance(result, str):
            return result

        # Check if result is already a list of content parts
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, (ContentPartTextParam, ContentPartImageParam)):
                return result

        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)


def tool(
    description: str,
    *,
    name: str | None = None,
    registry: "ToolRegistry | None" = None,  # type: ignore
    ephemeral: int | bool = False,
    usage_rules: str | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Tool]:
    """
    Decorator to create a tool from an async function.

    Args:
        description: Short overview used in system prompt tool overview.
        name: Optional custom name for the tool. Defaults to function name.
        registry: Optional ToolRegistry to register the tool to.
                  If None, the tool is NOT registered globally. The caller must
                  explicitly pass the returned Tool to an Agent (tools=[...]) or
                  register it into a registry.
        ephemeral: How many outputs to keep in context before older ones are removed.
                   False = not ephemeral (keep all), True = keep last 1, int = keep last N.
        usage_rules: Detailed usage rules for tool JSON schema/function description.
                     If provided, ToolDefinition.description uses usage_rules.

    Returns:
        A Tool instance wrapping the decorated function.

    Example:
        # Basic usage - returns a Tool (not globally registered)
        @tool("Search the web for information")
        async def search(query: str) -> str:
            return f"Results for: {query}"

        # Custom registry (explicit registration)
        from comate_agent_sdk.tools import ToolRegistry
        custom_registry = ToolRegistry()
        @tool("Special tool", registry=custom_registry)
        async def special() -> str:
            return "..."

        # With ephemeral outputs (keep last 2)
        @tool("Get browser state", ephemeral=2)
        async def get_state() -> str:
            return "..."
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Tool:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                f"Tool '{func.__name__}' must be an async function. Use 'async def {func.__name__}(...)' instead."
            )

        # 创建 Tool 实例
        t = Tool(
            func=func,
            description=description,
            name=name or func.__name__,
            ephemeral=ephemeral,
            usage_rules=usage_rules,
        )

        # ====== 显式注册逻辑 ======
        # 默认不做任何全局注册：工具的生命周期应当由 Agent/调用方管理。
        if registry is not None:
            registry.register(t)

        return t

    return decorator
