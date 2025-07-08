# workflow_engine/models.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypedDict, TypeVar, Union, Literal, NewType

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    create_model,
    model_validator,
    root_validator,
)
from typing_extensions import Self
import logging

# Strongly typed workflow source types
WorkflowSourceFile = NewType('WorkflowSourceFile', str)
WorkflowSourceYAML = NewType('WorkflowSourceYAML', str)
WorkflowSourceDict = Dict[str, Any]

# Union type for all workflow sources
WorkflowSource = Union[WorkflowSourceFile, WorkflowSourceYAML, WorkflowSourceDict]

# Type variable for models
T = TypeVar("T", bound=BaseModel)


@dataclass
class ModelSettings:
    """Settings to use when calling an LLM.
    
    This class holds optional model configuration parameters (e.g. temperature,
    top_p, penalties, truncation, etc.).
    
    Not all models/providers support all of these parameters, so please check the API documentation
    for the specific model and provider you are using.
    """
    
    temperature: Optional[float] = None
    """The temperature to use when calling the model."""
    
    top_p: Optional[float] = None
    """The top_p to use when calling the model."""
    
    frequency_penalty: Optional[float] = None
    """The frequency penalty to use when calling the model."""
    
    presence_penalty: Optional[float] = None
    """The presence penalty to use when calling the model."""
    
    tool_choice: Optional[Union[Literal["auto", "required", "none"], str]] = None
    """The tool choice to use when calling the model."""
    
    parallel_tool_calls: Optional[bool] = None
    """Whether to use parallel tool calls when calling the model.
    Defaults to False if not provided."""
    
    truncation: Optional[Literal["auto", "disabled"]] = None
    """The truncation strategy to use when calling the model."""
    
    max_tokens: Optional[int] = None
    """The maximum number of output tokens to generate."""
    
    store: Optional[bool] = None
    """Whether to store the generated model response for later retrieval.
    Defaults to True if not provided."""
    
    include_usage: Optional[bool] = None
    """Whether to include usage chunk.
    Defaults to True if not provided."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model settings to a dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class ModelRegistry:
    """Registry to store dynamically created Pydantic models."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model: Type[BaseModel]) -> None:
        """Register a model with the registry."""
        cls._models[name] = model

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        """Get a model from the registry."""
        return cls._models.get(name)

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a model exists in the registry."""
        return name in cls._models


class ModelGenerator(ABC):
    """Abstract base class for dynamic model generators."""

    @abstractmethod
    def create_model(self, name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Create a model from a schema definition.

        Args:
            name: Name of the model to create
            schema: Schema definition for the model

        Returns:
            Created model type
        """
        pass


class DynamicModelGenerator(ModelGenerator):
    """Generate Pydantic models from schema definitions."""

    @staticmethod
    def _get_field_type(type_str: str, item_type: Optional[str] = None) -> Any:
        """Convert type string to Python type."""
        basic_types = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "dict": dict,
            "list": list,
        }

        if type_str == "list" and item_type:
            # Handle list with specified item type
            item_python_type = DynamicModelGenerator._get_field_type(item_type)
            return List[item_python_type]
        elif type_str in basic_types:
            return basic_types[type_str]
        else:
            # Assume it's a reference to another model
            if ModelRegistry.has(type_str):
                return ModelRegistry.get(type_str)
            else:
                # Default to Any if model not found (will be resolved later)
                return Any

    def create_model(self, name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Create a Pydantic model from a schema definition."""
        fields = {}

        for field_name, field_def in schema.items():
            field_type = self._get_field_type(
                field_def.get("type", "str"), field_def.get("item_type")
            )

            field_args = {
                "description": field_def.get("description", ""),
            }

            # Add default if specified
            if "value" in field_def:
                field_args["default"] = field_def["value"]
            elif "required" in field_def and not field_def["required"]:
                field_args["default"] = None

            # Set field as Optional if not required
            if "required" in field_def and not field_def["required"]:
                field_type = Optional[field_type]

            # Add the field to our fields dict
            fields[field_name] = (field_type, Field(**field_args))

        # Create the model
        model = create_model(name, **fields, __base__=BaseModel)

        # Register the model for future reference
        ModelRegistry.register(name, model)

        return model


# Structured models for YAML schemas and provider configuration


class Tool(BaseModel):
    """Tool configuration for an agent."""

    name: str
    description: Optional[str] = None
    type: Optional[str] = "function"
    parameters: Optional[Dict[str, Any]] = None


class AgentConfig(BaseModel):
    """Agent configuration within a workflow task."""

    # Reference to an external agent definition
    ref: Optional[str] = None

    # Agent implementation details
    agent_type: Optional[str] = None

    # Basic identification
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None

    # Tool configuration
    tools: Optional[List[Tool]] = None

    # MCP server configuration
    mcp_servers: Optional[List[str]] = None

    # Input/output schema
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    # Prompt templates
    system_prompt: Optional[str] = None

    # Resource requirements
    resources: Optional[Dict[str, Any]] = None

    # Retry and error handling
    retry: Optional[Dict[str, Any]] = None

    @root_validator(skip_on_failure=True)
    def validate_agent_config(cls, values):
        """Ensure that either ref or agent_type is provided."""
        if not values.get("ref") and not values.get("agent_type"):
            raise ValueError("Either 'ref' or 'agent_type' must be provided")
        return values

    @computed_field
    @property
    def pydantic_output_schema(self) -> Union[type[BaseModel], None]:
        """
        Convert the agent's output_schema to a Pydantic model.

        Returns:
            A dynamically created Pydantic model class based on the output schema
        """
        if not self.output_schema:
            # Return none so it can be handled at call site
            return None

        spec = self.output_schema
        agent_name = self.id

        # DSL → Python types
        type_map: dict[str, type] = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": float,
            "bool": bool,
            "boolean": bool,
            "list": List,
            "array": List,
            "dict": dict,
            "object": dict,
        }

        pydantic_fields: dict[str, tuple[type, object]] = {}
        for field_name, info in spec.items():
            # pick the base type (or default to str)
            raw = type_map.get(info["type"], str)
            # if it’s a List, force-parameterize with item_type (default to str)
            if raw is List:
                item_py = type_map.get(info.get("item_type", "string"), str)
                base = List[item_py]  # <-- now List[str], List[int], etc.
            else:
                base = raw

            default = ... if info.get("required", True) else None
            pydantic_fields[field_name] = (base, default)

        model_name = f"{agent_name.replace(' ', '')}"
        return create_model(model_name, __base__=BaseModel, **pydantic_fields)


class WorkflowTask(BaseModel):
    """Definition of an individual task in a workflow stage."""

    name: str
    description: Optional[str] = None
    agent: AgentConfig
    prompt: str = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    condition: Optional[str] = None
    timeout: Optional[int] = None

    #     add an optional provider field of `BaseProviderConfig` type that'll be resolved at runtime
    provider: Optional["BaseProviderConfig"] = None
    
    # Store the initialized agent instance
    initialized_agent: Optional[Any] = None

    def process_inputs(
        self, response_store: "ResponseStore", workflow_inputs: "WorkflowInput"
    ) -> Dict[str, Any]:
        """
        Process task inputs to replace variables with values from response_store.

        Args:
            response_store: The response store containing previous task outputs
            workflow_inputs: The workflow inputs

        Returns:
            Processed task inputs with variables replaced by values
        """

        from agent_workflow.expressions.expressions import (
            ExpressionEvaluator,
            create_evaluation_context,
        )

        logger = logging.getLogger("workflow-engine.task")

        if not self.inputs:
            logger.warning("No inputs are registered for this task")
            return {}

        if not self.outputs:
            logger.warning("No outputs are registered for this task")
            pass

        # Use the utility function to create the evaluation context
        context = create_evaluation_context(response_store, workflow_inputs)

        # Process each input defined in task.inputs
        processed_inputs = {}
        for input_name, input_value in self.inputs.items():
            # Use ExpressionEvaluator to process the value
            processed_value = ExpressionEvaluator.evaluate_expression(
                input_value, context
            )
            processed_inputs[input_name] = processed_value
            logger.debug(
                f"Processed input '{input_name}': {input_value} -> {processed_value}"
            )

        return processed_inputs

    def process_output(self, raw_output: Any) -> Dict[str, Any]:
        """
        Process the raw agent output according to task.outputs or agent.output_schema.

        Args:
            raw_output: The raw output from the agent

        Returns:
            Processed output according to the schema
        """

        # Handle Pydantic model outputs from OpenAI SDK
        if isinstance(raw_output, BaseModel):
            # Convert Pydantic model to dict
            return raw_output.dict()

        # If raw_output is already a dict, use it directly
        if isinstance(raw_output, dict):
            return raw_output

        # Handle string outputs that might be JSON
        if isinstance(raw_output, str):
            # Fall back to simple string response
            return {"response": raw_output}

        # For any other type, convert to string
        return {"response": str(raw_output)}


class StageExecutionStrategy(str, Enum):
    """Execution types for workflow stages"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HANDOFF = "handoff"


class WorkflowStage(BaseModel):
    """Definition of a workflow stage containing tasks."""

    name: str
    description: Optional[str] = None
    execution_type: StageExecutionStrategy
    tasks: List[WorkflowTask]
    condition: Optional[str] = None
    timeout: Optional[int] = None
    #
    # @model_validator(mode="after")
    # def validate_tasks(self) -> Self:

    @model_validator(mode="before")
    @classmethod
    def validate_execution_type(cls, data):
        """Validate the execution_type field and convert string to enum if needed."""
        if isinstance(data, dict) and "execution_type" in data:
            exec_type = data["execution_type"]
            # If it's already an ExecutionType enum, keep it
            if isinstance(exec_type, StageExecutionStrategy):
                return data

            # If it's a string, try to convert it
            if isinstance(exec_type, str):
                try:
                    # Try to match the string to an enum value
                    for enum_val in StageExecutionStrategy:
                        if exec_type.lower() == enum_val.value:
                            data["execution_type"] = enum_val
                            return data

                    # If we get here, the string didn't match any enum value
                    valid_types = [e.value for e in StageExecutionStrategy]
                    raise ValueError(
                        f"Invalid execution_type: '{exec_type}'. Valid types are: {valid_types}"
                    )
                except Exception as e:
                    raise ValueError(f"Invalid execution_type: {e}")

            # If it's not a string or an enum, raise an error
            valid_types = [e.value for e in StageExecutionStrategy]
            raise ValueError(
                f"Invalid execution_type: '{exec_type}'. Valid types are: {valid_types}"
            )

        return data


class UserInputConfig(BaseModel):
    """Configuration for user input in workflow."""

    enabled: bool = False
    prompt_template: Optional[str] = (
        "Review the output from {agent_name}. You can modify it if needed:"
    )
    timeout_seconds: Optional[int] = 300  # Default 5 minutes timeout


class Workflow(BaseModel):
    """Top-level workflow definition."""

    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    stages: List[WorkflowStage]
    user_input: Optional[UserInputConfig] = None


class LLMAgent(BaseModel):
    """Definition of an LLM-based agent."""

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    agent_type: Optional[str] = "LLMAgent"
    tools: Optional[List[Tool]] = None
    mcp_servers: Optional[List[str]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    resources: Optional[Dict[str, Any]] = None
    retry: Optional[Dict[str, Any]] = None

    def pydantic_output_schema(self) -> type[BaseModel]:
        """
        Convert the agent's output_schema to a Pydantic model.

        Returns:
            A dynamically created Pydantic model class based on the output schema
        """
        if not self.output_schema:
            # Return a simple empty model if no schema is defined
            return create_model(f"{self.id}", __base__=BaseModel)

        spec = self.output_schema
        agent_name = self.id

        # DSL → Python types
        type_map: dict[str, type] = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": float,
            "bool": bool,
            "boolean": bool,
            "list": List,
            "array": List,
            "dict": dict,
            "object": dict,
        }

        pydantic_fields: dict[str, tuple[type, object]] = {}
        for field_name, info in spec.items():
            # pick the base type (or default to str)
            field_type = info.get("type", "string")
            raw = type_map.get(field_type, str)

            # if it's a List, force-parameterize with item_type (default to str)
            if raw is List:
                item_py = type_map.get(info.get("item_type", "string"), str)
                base = List[item_py]  # <-- now List[str], List[int], etc.
            else:
                base = raw

            default = ... if info.get("required", True) else None
            pydantic_fields[field_name] = (base, default)

        model_name = f"{agent_name.replace(' ', '')}"
        return create_model(model_name, __base__=BaseModel, **pydantic_fields)


# Workflow Input Model
class WorkflowInput(BaseModel):
    """Model for structured workflow inputs."""

    user_query: str
    provider_mapping: Optional[Dict[str, str]] = None
    provider_config: Optional[Union["ProviderConfiguration", Dict[str, Any]]] = None
    workflow: Dict[str, Any]
    # _execution: Optional[Dict[str, Any]] = None

    # Allow additional fields for flexibility
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling nested Pydantic models."""
        result = {}
        for key, value in self.dict(exclude_unset=True).items():
            if isinstance(value, BaseModel):
                result[key] = value.dict()
            else:
                result[key] = value
        return result


# Provider Configuration Models


class ProviderType(str, Enum):
    """Supported LLM provider types"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    GEMINI = "gemini"


class BaseProviderConfig(BaseModel):
    """Base class for provider configuration"""

    provider_type: ProviderType
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # https://github.com/google-gemini/deprecated-generative-ai-python/issues/515
    # JSON Schema is not available on Gemini while calling tools.
    # Change this to False if you see following error logged in console: Error code: 400 - [{'error': {'code': 400, 'message': "Function calling with a response mime type: 'application/json' is unsupported", 'status': 'INVALID_ARGUMENT'}}]
    enforce_structured_output: bool = True
    # Optional model settings for controlling model behavior during inference
    model_settings: Optional[ModelSettings] = None


class OpenAIProviderConfig(BaseProviderConfig):
    """OpenAI provider configuration"""

    provider_type: ProviderType = ProviderType.OPENAI
    organization: Optional[str] = None


class AnthropicProviderConfig(BaseProviderConfig):
    """Anthropic provider configuration"""

    provider_type: ProviderType = ProviderType.ANTHROPIC


class BedrockProviderConfig(BaseProviderConfig):
    """AWS Bedrock provider configuration"""

    provider_type: ProviderType = ProviderType.BEDROCK
    AWS_REGION: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str


class GeminiProviderConfig(BaseProviderConfig):
    """Google Gemini provider configuration"""

    provider_type: ProviderType = ProviderType.GEMINI
    access_token: str
    model: str
    project_id: str
    location: str
    base_url: str

    @model_validator(mode="after")
    def set_api_key(self) -> Self:
        # set the access token as api_key for compatibility
        if not self.api_key or self.api_key == "":
            self.api_key = self.access_token
        return self


class ProviderConfiguration(BaseModel):
    """Configuration for all providers in a workflow"""

    providers: Dict[
        str,
        BaseProviderConfig,
    ]

    @classmethod
    def from_dict(
        cls, provider_dict: Dict[str, Dict[str, Any]]
    ) -> "ProviderConfiguration":
        """Create a ProviderConfiguration from a dictionary of provider configs"""
        provider_configs = {}

        for provider_id, config in provider_dict.items():
            # Process model_settings if it exists
            model_settings_dict = config.get("model_settings")
            if model_settings_dict:
                # Remove from config to avoid passing it to parse_obj
                # We'll set it manually after parsing the config
                del config["model_settings"]
                model_settings = ModelSettings(**model_settings_dict)
            else:
                model_settings = None

            provider_type = config.get("provider_type")
            if provider_type == ProviderType.OPENAI:
                provider_config = OpenAIProviderConfig.parse_obj(config)
            elif provider_type == ProviderType.ANTHROPIC:
                provider_config = AnthropicProviderConfig.parse_obj(config)
            elif provider_type == ProviderType.BEDROCK:
                provider_config = BedrockProviderConfig.parse_obj(config)
            elif provider_type == ProviderType.GEMINI:
                provider_config = GeminiProviderConfig.parse_obj(config)
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")

            # Set model_settings after parsing
            if model_settings:
                provider_config.model_settings = model_settings
                
            provider_configs[provider_id] = provider_config

        return cls(providers=provider_configs)


@dataclass
class AgentOutput:
    """
    Standardized output for an individual agent execution.
    """

    agent: str  # Agent name/identifier
    output: Any  # The agent's output content
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata about the execution


class ResponseStore(BaseModel):
    """
    Structured storage for execution responses from all stages and tasks.

    Maintains a nested dictionary structure:
    {
        "stage_name": {
            "task_name": {
                ...
            },
            ...
        },
        ...
    }
    """

    responses: Dict[str, Dict[str, "TaskExecutionResult"]] = Field(default_factory=dict)

    def add(self, stage: str, task: str, data: "TaskExecutionResult") -> None:
        """
        Add response data for a specific stage and task.

        Args:
            stage: Stage name
            task: Task name
            data: Response data dictionary
        """
        if stage not in self.responses:
            self.responses[stage] = {}

        self.responses[stage][task] = data

    def get(self, stage: str, task: str, key: Optional[str] = None) -> Any:
        """
        Retrieve response data for a specific stage and task.

        Args:
            stage: Stage name
            task: Task name
            key: Optional specific key to retrieve. If None, returns all data for the task.

        Returns:
            The requested data, or None if not found

        Raises:
            ValueError: If the stage or task doesn't exist
        """
        if stage not in self.responses:
            raise ValueError(f"Stage '{stage}' not found")

        if task not in self.responses[stage]:
            raise ValueError(f"Task '{task}' not found in stage '{stage}'")

        if key is None:
            return self.responses[stage][task]

        if key not in self.responses[stage][task]:
            raise ValueError(
                f"Key '{key}' not found in task '{task}' of stage '{stage}'"
            )

        return self.responses[stage][task][key]

    def has_stage(self, stage: str) -> bool:
        """Check if a stage exists in the response store."""
        return stage in self.responses

    def has_task(self, stage: str, task: str) -> bool:
        """Check if a task exists in a specific stage."""
        return stage in self.responses and task in self.responses[stage]

    def has_key(self, stage: str, task: str, key: str) -> bool:
        """Check if a key exists for a specific task in a specific stage."""
        return (
            stage in self.responses
            and task in self.responses[stage]
            and key in self.responses[stage][task]
        )

    def get_stages(self) -> List[str]:
        """Get a list of all stages."""
        return list(self.responses.keys())

    def get_tasks(self, stage: str) -> List[str]:
        """Get a list of all tasks in a specific stage."""
        if stage not in self.responses:
            raise ValueError(f"Stage '{stage}' not found")
        return list(self.responses[stage].keys())

    def get_keys(self, stage: str, task: str) -> List[str]:
        """Get a list of all keys for a specific task in a specific stage."""
        if stage not in self.responses:
            raise ValueError(f"Stage '{stage}' not found")
        if task not in self.responses[stage]:
            raise ValueError(f"Task '{task}' not found in stage '{stage}'")
        return list(self.responses[stage][task].keys())

    def to_dict(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Convert to a nested dictionary."""
        return self.responses


@dataclass
class ExecutionResult:
    """
    Standardized result format for all execution engines.
    """

    agent_outputs: List[AgentOutput]  # List of all agent outputs in execution order
    final_result: Any  # The final output value
    all_agents: List[str]  # List of all agent names/identifiers
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata about the execution
    response_store: Optional[ResponseStore] = (
        None  # Structured storage for all stage and task outputs
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the execution result to a dictionary format."""
        result = {
            "agent_outputs": [
                {
                    "agent": output.agent,
                    "output": output.output,
                    **({"metadata": output.metadata} if output.metadata else {}),
                }
                for output in self.agent_outputs
            ],
            "final_result": self.final_result,
            "all_agents": self.all_agents,
        }

        if self.metadata:
            result["metadata"] = self.metadata

        if self.response_store:
            result["responses"] = self.response_store.to_dict()

        return result


@dataclass
class StageExecutionResult:
    """
    Standardized result format for a workflow stage execution.
    Contains the results of all tasks in the stage.
    """

    stage_name: str  # Name of the stage
    tasks_results: Dict[
        str, Dict[str, Any]
    ]  # Dictionary mapping task names to their results
    completed: bool = True  # Whether the stage execution completed successfully
    error: Optional[str] = None  # Error message if execution failed
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata about the execution

    def to_dict(self) -> Dict[str, Any]:
        """Convert the stage execution result to a dictionary format."""
        result = {
            "stage_name": self.stage_name,
            "tasks_results": self.tasks_results,
            "completed": self.completed,
        }
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def get_task_result(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get the result of a specific task."""
        return self.tasks_results.get(task_name)


@dataclass
class TaskExecutionResult:
    """
    Standardized result format for a workflow task execution.
    Contains the results of the tasks in the stage.
    """

    task_name: str  # Name of the task
    result: Dict[str, Any]  # Dictionary mapping of task results
    completed: bool = True  # Whether the task execution completed successfully
    error: Optional[str] = None  # Error message if execution failed
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata about the execution
    structured_output_enforced: bool = (
        True  # Whether the output is enforced to be structured according to the schema
    )


# support for MCP


class MCPServerType(str, Enum):
    """Type of MCP server."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


class MCPServerSpec:
    """Specification for an MCP server."""

    def __init__(
        self,
        params: Dict[str, Any],
        server_type: MCPServerType,
        name: str,
        cache_tools_list: bool = False,
        client_session_timeout: float = 5,
    ):
        """Initialize the MCP server specification.

        Args:
            params: The name of the MCP server
            server_type: The type of the MCP server (studio or see)
            cache_tools_list: Whether to cache the tool list
            name: Optional name for the server
            client_session_timeout: Timeout for the client session
        """
        self.params = params
        self.server_type = server_type
        self.cache_tools_list = cache_tools_list
        self.name = name
        self.client_session_timeout = client_session_timeout
