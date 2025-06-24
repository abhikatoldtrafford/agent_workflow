from abc import ABC, abstractmethod
from datetime import datetime
import uuid
from typing import TypedDict, Dict, List, Union, Any, TypeVar, ParamSpec, Optional, TYPE_CHECKING, Literal
from enum import Enum

# Helper functions for generating IDs
def gen_trace_id() -> str:
    """Generate a random trace ID."""
    return str(uuid.uuid4())

def gen_span_id() -> str:
    """Generate a random span ID."""
    return str(uuid.uuid4())

def gen_group_id() -> str:
    """Generate a random group ID."""
    return str(uuid.uuid4())

if TYPE_CHECKING:
    from agent_workflow.workflow_engine.models import AgentConfig

# -----------------------------
# Strongly-typed metadata types
# -----------------------------
class TraceStatus(Enum):
    FAILED = 0
    SUCCESS = 1
    IN_PROGRESS = 2

class RequestData(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    latency_ms: int
    total_tokens: int

class CommonMetadata(TypedDict, total=False):
    timestamp: datetime              # when the event occurred
    tags: Dict[str, str]  # arbitrary key/value tags
    request_data: RequestData
    status: TraceStatus

class TraceMetadata(CommonMetadata, total=False):
    agent_id: str                    # ID of the agent (if this trace is an agent run)
    workflow_id: str                 # ID of the workflow (if applicable)
    description: str  # human-readable note

class ToolSpan(CommonMetadata, total=False):
    tool_type: Union[Literal['functional'], Literal['openai']]
    # fill in more tool fields

class DebugSpan(CommonMetadata, total=False):
    log_level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]

class SpanMetadata(CommonMetadata, total=False):
    span: Union[ToolSpan, DebugSpan]

class RunContextMetadata(CommonMetadata, total=False):
    requests: int

class CallMetadata(CommonMetadata, total=False):
    prompt_type: str                 # e.g. "zero-shot", "chain-of-thought"
    response_tokens: int
    cost_usd: float

class ScoreMetadata(CommonMetadata, total=False):
    evaluator: str                   # e.g. "BLEU", "ROUGE", "custom"
    details: Dict[str, str]

class ToolMetadata(CommonMetadata, total=False):
    tool_type: str                   # e.g. "function", "openai", etc.
    duration_ms: float               # execution time in milliseconds
    success: bool                    # whether the tool execution succeeded
    error: str                       # error message if tool execution failed


P = ParamSpec("P")
R = TypeVar("R")

# ----------------------------------
# Core observability provider contract
# ----------------------------------
class LLMObservabilityProvider(ABC):
    """
    Abstract base for LLM observability:
     - Traces & trace groups
     - Spans
     - LLM call logging
     - Tool execution logging
     - Metrics / scores
     - Arbitrary events
    """

    # -- Trace groups --------------------------------
    @abstractmethod
    async def start_trace_group(
        self,
        metadata: TraceMetadata
    ) -> str:
        """Begin a named group of traces (e.g. all runs in a batch). Return group ID"""

    @abstractmethod
    async def end_trace_group(
        self,
        group_id: str,
        metadata: TraceMetadata
    ) -> None:
        """End/close a previously created trace group."""

    # -- Traces --------------------------------------
    @abstractmethod
    async def start_trace(
        self,
        name: str,
        prompt: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]],
        metadata: TraceMetadata,
        group_id: Optional[str],
    ) -> str:
        """Start a new trace; optionally assign it to a group."""

    @abstractmethod
    async def end_trace(
        self,
        prompt: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]],
        response: Any,
        trace_id: str,
        metadata: TraceMetadata,
    ) -> None:
        """End a trace and record any final metadata."""

    # -- Spans ---------------------------------------
    @abstractmethod
    async def start_span(
        self,
        name: str,
        parent_span_id: Optional[str],
        metadata: SpanMetadata
    ) -> str:
        """Start a new span in a trace; returns the span_id."""

    @abstractmethod
    async def end_span(
        self,
        span_id: str,
        metadata: SpanMetadata
    ) -> None:
        """End a span and attach any metadata (e.g. token counts)."""

    # -- LLM call logging ----------------------------
    @abstractmethod
    async def log_trace(
            self,
            name: str,
            model: str,
            prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
            response: Any,
            metadata: CallMetadata,
            trace_id: Optional[str]) -> None:
        """Log a full LLM trace, tied to a trace/span if desired.
           we can use this method to repeatedly log trace in between a start and end trace
        """

    @abstractmethod
    async def log_span(
            self,
            name: str,
            model: str,
            prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
            response: Any,
            metadata: CallMetadata,
            span_id: str) -> None:
        pass

    # -- Scores / metrics ----------------------------
    @abstractmethod
    async def record_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        metadata: ScoreMetadata,
        span_id: str
    ) -> None:
        """Attach a numeric evaluation (e.g. quality, latency) to a trace/span."""

    # -- Agent execution callback ----------------------------
    @abstractmethod
    async def on_agent_start(
        self,
        name: str,
        parent_span_id: Optional[str],
        system_prompt: str,
        context_metadata: Optional[RunContextMetadata] = None,
        agent_config: Optional["AgentConfig"] = None,
    ) -> None:
        """Agent starts callback
        
        Args:
            name: The agent name
            parent_span_id: parent span or trace id
            system_prompt: system prompt
            context_metadata: an optional `RunContextMetadata` object
            agent_config: Optional agent configuration object
        """

    @abstractmethod
    async def on_agent_end(
        self,
        name: str,
        output: object,
        parent_span_id: Optional[str],
        system_prompt: str,
        context_metadata: Optional[RunContextMetadata] = None,
        agent_config: Optional["AgentConfig"] = None,
    ) -> None:
        """Agent end callback
        
        Args:
            name: The agent name
            output: The agent output
            parent_span_id: parent span or trace id
            system_prompt: system prompt
            context_metadata: an optional `RunContextMetadata` object
            agent_config: Optional agent configuration object
        """

    @abstractmethod
    async def on_handoff(
            self,
            from_agent_name: str,
            to_agent_name: str,
            parent_span_id: Optional[str],
            from_agent_config: Optional['AgentConfig'] = None,
            to_agent_config: Optional['AgentConfig'] = None,
    ) -> None:
        """Agent handoff callback
        
        Args:
            from_agent_name: The name of the agent handing off control
            to_agent_name: The name of the agent receiving control
            parent_span_id: parent span or trace id
            from_agent_config: Optional configuration of the agent handing off control
            to_agent_config: Optional configuration of the agent receiving control
        """
# --------------------------------------------------------------------
# Base implementation for easier overloading
# --------------------------------------------------------------------

class BaseLLMObservabilityProvider(LLMObservabilityProvider):

    async def start_trace_group(self, metadata: TraceMetadata) -> str:
        return ""

    async def end_trace_group(self, group_id: str, metadata: TraceMetadata) -> None:
        pass

    async def start_trace(self, name: str, prompt: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]],
                          metadata: TraceMetadata, group_id: Optional[str]) -> str:
        return ""

    async def end_trace(self, prompt: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]], response: Any,
                        trace_id: str, metadata: TraceMetadata) -> None:
        pass

    async def start_span(self, name: str, parent_span_id: Optional[str], metadata: SpanMetadata) -> str:
        return ""

    async def end_span(self, span_id: str, metadata: SpanMetadata) -> None:
        pass

    async def log_trace(self, name: str, model: str, prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
                        response: Any, metadata: CallMetadata, trace_id: Optional[str]) -> None:
        pass

    async def log_span(self, name: str, model: str, prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
                       response: Any, metadata: CallMetadata, span_id: str) -> None:
        pass

    async def record_score(self, trace_id: str, name: str, value: float, metadata: ScoreMetadata, span_id: str) -> None:
        pass

    async def on_agent_start(self, name: str, parent_span_id: Optional[str], system_prompt: str,
                             context_metadata: Optional[RunContextMetadata] = None,
                             agent_config: Optional["AgentConfig"] = None) -> None:
        pass

    async def on_agent_end(self, name: str, output: object, parent_span_id: Optional[str], system_prompt: str,
                           context_metadata: Optional[RunContextMetadata] = None,
                           agent_config: Optional["AgentConfig"] = None) -> None:
        pass

    async def on_handoff(self, from_agent_name: str, to_agent_name: str, parent_span_id: Optional[str],
                         from_agent_config: Optional['AgentConfig'] = None,
                         to_agent_config: Optional['AgentConfig'] = None) -> None:
        pass


# ----------------------------------
# No-op implementation
# ----------------------------------
class NoOpLLMObservabilityProvider(BaseLLMObservabilityProvider):
    """
    A no-op implementation of the LLMObservabilityProvider interface.
    
    This implementation does nothing for all operations, but satisfies the interface
    requirements. It can be used as a default provider when no observability is needed.
    """

    @staticmethod
    def gen_group_id() -> str:
        """Generate a random group ID."""
        return str(uuid.uuid4())

    @staticmethod
    def gen_trace_id() -> str:
        """Generate a random trace ID."""
        return str(uuid.uuid4())

    @staticmethod
    def gen_span_id() -> str:
        """Generate a random span ID."""
        return str(uuid.uuid4())
        
    async def start_trace_group(
        self,
        metadata: TraceMetadata
    ) -> str:
        """No-op implementation of starting a trace group."""
        return gen_group_id()

    async def start_trace(
        self,
        name: str,
        prompt: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]],
        metadata: TraceMetadata,
        group_id: Optional[str]
    ) -> str:
        """No-op implementation of starting a trace."""
        return str(uuid.uuid4())
        
    async def start_span(
        self,
        name: str,
        parent_span_id: Optional[str],
        metadata: SpanMetadata
    ) -> str:
        """No-op implementation of starting a span."""
        return str(uuid.uuid4())