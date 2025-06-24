import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, get_type_hints, TypeVar, overload, Union
T = TypeVar("T", bound=Callable[..., Any])

logger = logging.getLogger("workflow-engine.tools")

# Flag to track if OpenAI's Agents SDK is available
OPENAI_AGENTS_AVAILABLE = False

# Try to import OpenAI's tools
try:
    from agents import Tool as OpenAITools

    OPENAI_AGENTS_AVAILABLE = True
    logger.info("OpenAI Agents SDK tools are available")
except ImportError:
    logger.info("OpenAI Agents SDK not available. Built-in tools will be simulated.")
    raise


class Tool(ABC):
    """Base class for all tools that can be used by agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the tool."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Get the parameters schema for the tool."""
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """default is `functional` for most cases"""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given parameters."""
        pass

    def to_schema(self) -> Dict[str, Any]:
        """Convert the tool to a schema representation."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class OpenAITool(Tool):
    """Tool implementation that wraps a OpenAI tool."""

    def __init__(
        self,
        openai_tool: OpenAITools,
        name: str,
        description: str,
    ):
        """Initialize an OpenAI tool wrapper.

        Args:
            openai_tool: The OpenAI tool instance to wrap
        """
        self._func = openai_tool
        self._name = name
        self._description = description

    @property
    def function(self) -> OpenAITools:
        """Get the wrapped function."""
        return self._func

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        raise NotImplementedError("Parameter should not be called for OpenAI tools.")

    @property
    def type(self) -> str:
        return "openai"

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the OpenAI tool with the given parameters."""
        raise NotImplementedError(
            "Execute is not implemented for OpenAI tools directly."
        )


class FunctionTool(Tool):
    """Tool implementation that wraps a function."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize a function tool.

        Args:
            func: The function to wrap
            name: Optional name for the tool, defaults to function name
            description: Optional description, defaults to function docstring
        """
        self._func = func
        self._name = name or func.__name__
        self._description = (
            description or inspect.getdoc(func) or f"Execute the {self._name} function"
        )
        self._parameters = self._build_parameters_schema()

    @property
    def function(self) -> Callable:
        """Get the wrapped function."""
        return self._func

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def type(self) -> str:
        return "functional"

    def _build_parameters_schema(self) -> Dict[str, Any]:
        """Build JSON schema for function parameters."""
        schema: dict[str, Any] = {"type": "object",
                  "properties": {},
                  "required": []
                  }

        # Get function signature
        sig = inspect.signature(self._func)
        type_hints = get_type_hints(self._func)

        for param_name, param in sig.parameters.items():
            # Skip self parameter for methods
            if param_name == "self":
                continue

            param_schema = {"type": "string"}  # Default type

            # Try to determine parameter type from type hints
            if param_name in type_hints:
                param_type = type_hints[param_name]
                param_schema = self._type_to_schema(param_type)

            # Check if the parameter has a default value
            if param.default is not inspect.Parameter.empty:
                param_schema["default"] = param.default
            else:
                schema["required"].append(param_name)

            # Add parameter description if available
            # Try to extract from docstring
            docstring = inspect.getdoc(self._func)
            if docstring:
                param_desc = self._extract_param_description(docstring, param_name)
                if param_desc:
                    param_schema["description"] = param_desc

            schema["properties"][param_name] = param_schema

        return schema

    def _type_to_schema(self, typ: Type) -> Dict[str, Any]:
        """Convert Python type to JSON schema."""
        if typ is str:
            return {"type": "string"}
        elif typ is int:
            return {"type": "integer"}
        elif typ is float:
            return {"type": "number"}
        elif typ is bool:
            return {"type": "boolean"}
        elif typ is list or getattr(typ, "__origin__", None) is list:
            item_type = {"type": "string"}  # Default item type
            if hasattr(typ, "__args__") and typ.__args__:
                item_type = self._type_to_schema(typ.__args__[0])
            return {"type": "array", "items": item_type}
        elif typ is dict or getattr(typ, "__origin__", None) is dict:
            return {"type": "object"}
        else:
            return {"type": "string"}  # Default to string for complex types

    def _extract_param_description(
        self, docstring: str, param_name: str
    ) -> Optional[str]:
        """Extract parameter description from function docstring."""
        lines = docstring.split("\n")
        param_lines = [
            line.strip() for line in lines if line.strip().startswith(f"{param_name}:")
        ]

        if param_lines:
            parts = param_lines[0].split(":", 1)
            if len(parts) > 1:
                return parts[1].strip()

        return None

    async def execute(self, **kwargs: dict[str, Any]) -> Any:
        """Execute the function with the given parameters."""
        try:
            result = self._func(**kwargs)

            # Handle both regular and async functions
            if inspect.iscoroutine(result):
                return await result
            return result
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return {"error": str(e)}


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """Register a tool with the registry.

        Args:
            tool: The tool to register
        """
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    @overload
    def register_function(self,
                          func: None = ...,
                          *,
                          name: Optional[str] = None,
                          description: Optional[str] = None) -> Callable[[T], T]: ...

    @overload
    def register_function(self,
                          func: T,
                          *,
                          name: Optional[str] = None,
                          description: Optional[str] = None) -> T: ...


    def register_function(self,
                          func: Optional[T] = None,
                          *,
                          name: Optional[str] = None,
                          description: Optional[str] = None) -> Union[Callable[[T], T], T]:
        """Decorator to register a function as a tool.

        Args:
            func: The function to register
            name: Optional name for the tool
            description: Optional description for the tool
        """

        def decorator(f: T) -> T:
            tool_ = FunctionTool(f, name=name, description=description)
            self.register_tool(tool_)
            return f

        if func is None:
            return decorator
        return decorator(func)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name.

        Args:
            name: The name of the tool to get

        Returns:
            The tool if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> Dict[str, Tool]:
        """Get all registered tools.

        Returns:
            Dictionary of tool names to tool objects
        """
        return self._tools.copy()

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get schema representations for all tools.

        Returns:
            List of tool schemas
        """
        return [t.to_schema() for t in self._tools.values()]

    async def execute_tool(self, name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool by name with the given parameters.

        Args:
            name: The name of the tool to execute
            parameters: The parameters to pass to the tool

        Returns:
            The result of the tool execution

        Raises:
            ValueError: If the tool is not found
        """
        tool_ = self.get_tool(name)
        if not tool_:
            raise ValueError(f"Tool not found: {name}")

        logger.info(f"Executing tool: {name} with parameters: {parameters}")
        result = await tool_.execute(**parameters)
        logger.info(f"Tool {name} execution result: {type(result).__name__}")

        return result


# Create a global tool registry instance
global_tool_registry = ToolRegistry()


# Convenience function to register a tool
def register_tool(tool: Tool) -> None:
    """Register a tool with the global registry.

    Args:
        tool: The tool to register
    """
    global_tool_registry.register_tool(tool)


# Convenience decorator to register a function as a tool
def tool(name: str,
         description: Optional[str] = None) -> Callable[[T], T]:
    """Decorator to register a function as a tool.

    Args:
        name: Optional name for the tool
        description: Optional description for the tool
    """
    return global_tool_registry.register_function(name=name, description=description)
