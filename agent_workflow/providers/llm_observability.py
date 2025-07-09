"""
LLM Observability Provider interfaces and implementations.

This module defines the contract for LLM observability providers and includes
implementations for various observability platforms.
"""
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, ParamSpec, TypeVar, Union, TYPE_CHECKING, Tuple, Literal
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from agent_workflow.workflow_engine.models import AgentConfig

logger = logging.getLogger("workflow-engine.providers.llm_observability")

# ----------------------------------
# Status enums
# ----------------------------------
class TraceStatus(Enum):
    """Status of a trace or span"""
    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"

# ----------------------------------
# Type definitions for metadata
# ----------------------------------
class CommonMetadata(TypedDict, total=False):
    """Common metadata fields across all observability events."""
    timestamp: datetime
    tags: Dict[str, str]

class RequestData(TypedDict, total=False):
    """Request data for LLM calls"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int

class TraceMetadata(CommonMetadata, total=False):
    description: str
    workflow_id: str
    workflow_name: str
    request_data: RequestData
    status: TraceStatus

class ToolSpan(TypedDict, total=False):
    tool_type: str
    tags: Dict[str, str]

class DebugSpan(TypedDict, total=False):
    log_level: str

class SpanMetadata(CommonMetadata, total=False):
    span_type: str
    span: Union[ToolSpan, DebugSpan]
    request_data: RequestData
    status: TraceStatus

class RunContextMetadata(TypedDict, total=False):
    """Metadata from execution context (e.g., from OpenAI hooks)"""
    timestamp: datetime
    request_data: RequestData
    requests: int
    tags: Dict[str, str]
    status: TraceStatus

class CallMetadata(CommonMetadata, total=False):
    prompt_type: str                 # e.g. "zero-shot", "chain-of-thought"
    response_tokens: int
    cost_usd: float
    request_data: RequestData

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
        return self.gen_group_id()

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


# ----------------------------------
# Langfuse implementation
# ----------------------------------
class LangfuseLLMObservabilityProvider(BaseLLMObservabilityProvider):
    """
    Langfuse implementation of LLMObservabilityProvider.
    Provides observability for LLM interactions using Langfuse.
    
    This implementation:
    - Gracefully handles missing Langfuse package
    - Prevents empty traces for non-executing agents
    - Provides comprehensive error handling
    - Maintains full compatibility with the abstract interface
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
        userid: Optional[str] = None,
        enabled: bool = True
    ) -> None:
        """
        Initialize with Langfuse credentials.
        
        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL (defaults to Langfuse Cloud)
            userid: Optional user ID for tracking
            enabled: Whether to enable Langfuse (can be disabled for testing)
        """
        self.enabled = enabled
        self.client = None
        self._langfuse_available = False
        
        if self.enabled:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                self._langfuse_available = True
                logger.info("Langfuse observability provider initialized successfully")
            except ImportError:
                logger.warning("Langfuse package not installed. Install with: pip install langfuse")
                self.enabled = False
            except Exception as e:
                logger.error(f"Error initializing Langfuse client: {e}")
                self.enabled = False
        
        self._sessionID: Optional[str] = None
        self._current_span: Any = None
        self._current_agent: Any = None
        self._current_trace: Any = None
        self._current_trace_id: Optional[str] = None
        self.userid = userid
        self._pending_traces: Dict[str, Dict[str, Any]] = {}
        
    # -- Generate ids --------------------------------
    def gen_group_id(self) -> str:
        """Generate the group id and return"""
        return str(uuid.uuid4())

    def gen_trace_id(self) -> str:
        """Generate the trace id and return"""
        return str(uuid.uuid4())

    def gen_span_id(self) -> str:
        """Generate the span id and return"""
        return str(uuid.uuid4())

    # -- Trace groups --------------------------------
    async def start_trace_group(
        self,
        metadata: TraceMetadata
    ) -> str:
        """Begin a named group of traces (e.g. all runs in a batch). Return group ID"""
        group_id = self.gen_group_id()
        self._sessionID = group_id
        return group_id

    async def end_trace_group(
        self,
        group_id: str,
        metadata: TraceMetadata
    ) -> None:
        """End/close a previously created trace group."""
        # No explicit action needed for trace groups in Langfuse
        self._sessionID = None
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.debug(f"Error flushing Langfuse client: {e}")

    @staticmethod
    def _parse_trace_status(trace_status: TraceStatus, tags: Dict[str, str]) -> Tuple[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"], str]:
        """Parse trace status to Langfuse level and status message."""
        if trace_status == TraceStatus.FAILED:
            return "ERROR", str(tags) if tags else "Failed"
        elif trace_status == TraceStatus.IN_PROGRESS:
            return "DEBUG", "In Progress"
        else:  # SUCCESS is the default
            return "DEFAULT", "Success"

    # -- Traces --------------------------------------
    async def start_trace(
        self,
        name: str,
        prompt: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]],
        metadata: TraceMetadata,
        group_id: Optional[str],
    ) -> str:
        """Start a new trace; optionally assign it to a group."""
        trace_id = self.gen_trace_id()
        
        if not self.enabled or not self.client:
            return trace_id
        
        # Store trace info but DON'T create the trace yet
        self._pending_traces[trace_id] = {
            "name": name,
            "prompt": prompt,
            "metadata": metadata,
            "group_id": group_id or self._sessionID,
            "start_time": metadata.get("timestamp", datetime.now())
        }
        
        # Don't create the actual trace yet
        self._current_trace_id = trace_id
        return trace_id

    async def end_trace(
        self,
        prompt: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]],
        response: Any,
        trace_id: str,
        metadata: TraceMetadata,
    ) -> None:
        """End a trace and record any final metadata."""
        if not self.enabled or not self.client:
            return
        
        # Get pending trace info
        pending_trace = self._pending_traces.get(trace_id)
        if not pending_trace:
            return
            
        try:
            # Check if this was a real execution
            request_data = metadata.get("request_data", {})
            input_tokens = request_data.get("input_tokens", 0) if request_data else 0
            output_tokens = request_data.get("output_tokens", 0) if request_data else 0
            total_tokens = request_data.get("total_tokens", 0) if request_data else 0
            
            # ONLY create trace if there was actual execution
            if input_tokens > 0 or output_tokens > 0:
                time = metadata.get("timestamp", datetime.now())
                level, status_message = self._parse_trace_status(
                    metadata.get("status", TraceStatus.SUCCESS), metadata.get("tags", {})
                )
                
                usage = {
                    "promptTokens": input_tokens,
                    "completionTokens": output_tokens,
                    "totalTokens": total_tokens
                }
                
                # Create the complete trace with start and end
                self.client.trace(
                    name=pending_trace["name"],
                    user_id=self.userid,
                    session_id=pending_trace["group_id"],
                    input=prompt or pending_trace["prompt"],
                    output=response,
                    id=trace_id,
                    metadata={**pending_trace["metadata"], **metadata},
                    usage=usage,
                    start_time=pending_trace["start_time"],
                    end_time=time,
                    level=level,
                    status_message=status_message,
                )
            
        except Exception as e:
            logger.debug(f"Error ending Langfuse trace: {e}")
        finally:
            # Clean up
            if trace_id in self._pending_traces:
                del self._pending_traces[trace_id]
            if self._current_trace_id == trace_id:
                self._current_trace_id = None

    # -- Spans ---------------------------------------
    async def start_span(
        self,
        name: str,
        parent_span_id: Optional[str],
        metadata: SpanMetadata,
    ) -> str:
        """Start a new span in a trace; returns the span_id."""
        span_id = self.gen_span_id()
        
        if not self.enabled or not self.client:
            return span_id
            
        try:
            # Extract metadata fields
            timestamp = metadata.get("timestamp", datetime.now())
            tags_dict = metadata.get("tags", {})
            tags = [f"{k}:{v}" for k, v in tags_dict.items()]
            
            # Extract request data if available
            request_data = metadata.get("request_data", {})
            input_tokens = request_data.get("input_tokens", 0) if request_data else 0
            output_tokens = request_data.get("output_tokens", 0) if request_data else 0
            total_tokens = request_data.get("total_tokens", 0) if request_data else 0
            latency_ms = request_data.get("latency_ms", 0) if request_data else 0
            
            # Create a metadata dict for Langfuse
            langfuse_metadata = {
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "trace_id": parent_span_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
            }
            
            # Add any additional metadata fields
            for key, value in metadata.items():
                if key not in ["timestamp", "tags", "request_data", "status"]:
                    langfuse_metadata[key] = value

            level, status_message = self._parse_trace_status(
                metadata.get("status", TraceStatus.SUCCESS), {}
            )

            param_dict = {
                "name": name,
                "trace_id": self._current_trace_id or span_id,
                "parent_observation_id": parent_span_id,
                "metadata": langfuse_metadata,
                "start_time": timestamp,
                "tags": tags,
                "level": level,
                "status_message": status_message
            }

            # Create the span in Langfuse
            if self._current_trace:
                self._current_span = self._current_trace.span(**param_dict)
            else:
                self._current_span = self.client.span(**param_dict)
                
        except Exception as e:
            logger.debug(f"Error starting Langfuse span: {e}")
        
        return span_id

    async def end_span(
        self,
        span_id: str,
        metadata: SpanMetadata
    ) -> None:
        """End a span and attach any metadata (e.g. token counts)."""
        if not self.enabled or not self.client or not self._current_span:
            return
            
        try:
            tags_dict = metadata.get("tags", {})
            tags = [f"{k}:{v}" for k, v in tags_dict.items()]
            
            # Extract request data if available
            request_data = metadata.get("request_data", {})
            input_tokens = request_data.get("input_tokens", 0) if request_data else 0
            output_tokens = request_data.get("output_tokens", 0) if request_data else 0
            total_tokens = request_data.get("total_tokens", 0) if request_data else 0
            latency_ms = request_data.get("latency_ms", 0) if request_data else 0
            end_time = metadata.get("timestamp", datetime.now())
            
            # Create a metadata dict for Langfuse
            langfuse_metadata = {
                "span_id": span_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
            }
            
            # Add any additional metadata fields
            for key, value in metadata.items():
                if key not in ["timestamp", "tags", "request_data", "status"]:
                    langfuse_metadata[key] = value

            level, status_message = self._parse_trace_status(
                metadata.get("status", TraceStatus.SUCCESS), {}
            )

            # Update and end the span
            self._current_span.end(
                metadata=langfuse_metadata,
                tags=tags,
                end_time=end_time,
                level=level,
                status_message=status_message,
            )
        except Exception as e:
            logger.debug(f"Error ending Langfuse span: {e}")
        finally:
            self._current_span = None

    # -- LLM call logging ----------------------------
    async def log_trace(
        self,
        name: str,
        model: str,
        prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        response: Any,
        metadata: CallMetadata,
        trace_id: Optional[str]
    ) -> None:
        """Log a full LLM interaction, tied to a trace/span if desired."""
        if not self.enabled or not self.client:
            return
            
        try:
            timestamp = metadata.get("timestamp", datetime.now())
            
            # Extract token usage from metadata
            request_data = metadata.get("request_data", {})
            usage = None
            if request_data:
                usage = {
                    "promptTokens": request_data.get("input_tokens", 0),
                    "completionTokens": request_data.get("output_tokens", 0),
                    "totalTokens": request_data.get("total_tokens", 0),
                }
            
            # Create tags
            tags_dict = metadata.get("tags", {})
            tags = [f"{k}:{v}" for k, v in tags_dict.items()]
            
            # Create a generation trace for the LLM call
            if self._current_trace and trace_id == self._current_trace_id:
                # Update existing trace with LLM call details
                self._current_trace.generation(
                    name=f"llm:{model}:{name}",
                    model=model,
                    input=prompt,
                    output=response,
                    usage=usage,
                    metadata=metadata,
                    timestamp=timestamp,
                    tags=tags
                )
            else:
                # Create standalone generation
                self.client.generation(
                    name=f"llm:{model}:{name}",
                    model=model,
                    input=prompt,
                    output=response,
                    usage=usage,
                    metadata=metadata,
                    timestamp=timestamp,
                    trace_id=trace_id,
                    tags=tags
                )
        except Exception as e:
            logger.debug(f"Error logging LLM trace to Langfuse: {e}")

    async def log_span(
        self,
        name: str,
        model: str,
        prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        response: Any,
        metadata: CallMetadata,
        span_id: str
    ) -> None:
        """Log a span - currently not implemented for Langfuse."""
        # Langfuse doesn't have a direct log_span equivalent
        # This could be implemented as a generation within a span if needed
        pass

    # -- Agent execution callbacks (with fix for empty traces) ----------------------------
    async def on_agent_start(
        self,
        name: str,
        parent_span_id: Optional[str],
        system_prompt: str,
        context_metadata: Optional[RunContextMetadata] = None,
        agent_config: Optional["AgentConfig"] = None,
    ) -> None:
        """Agent starts callback - only creates trace for executing agents"""
        if not self.enabled or not self.client:
            return
            
        # CRITICAL FIX: Check if this is a real execution
        has_execution_context = False
        if context_metadata and "request_data" in context_metadata:
            request_data = context_metadata.get("request_data", {})
            has_execution_context = (
                request_data.get("requests", 0) > 0 or
                request_data.get("input_tokens", 0) > 0
            )
        
        # Only proceed if this is an actual execution
        if not has_execution_context:
            self._current_agent = None
            return
        
        try:
            # Extract metadata
            timestamp = datetime.now()
            new_tags = {}
            if context_metadata:
                timestamp = context_metadata.get("timestamp", datetime.now())
                new_tags = context_metadata.get("tags", {})

            # Create tags
            tags_dict = {"event_type": "agent_start", "agent_name": name}
            if agent_config:
                tags_dict["agent_id"] = str(agent_config.id) if agent_config.id else "unknown"
                if hasattr(agent_config, "agent_type") and agent_config.agent_type:
                    tags_dict["agent_type"] = str(agent_config.agent_type)
            
            tags = [f"{k}:{v}" for k, v in tags_dict.items()]
            
            # Create metadata
            langfuse_metadata: Dict[str, Any] = {
                "timestamp": timestamp.isoformat(),
                "agent_name": name,
            }
            if agent_config:
                langfuse_metadata["agent_config"] = {
                    "id": str(agent_config.id) if agent_config.id else "unknown",
                    "type": str(agent_config.agent_type) if hasattr(agent_config, "agent_type") else "unknown",
                    "description": str(agent_config.description) if hasattr(agent_config, "description") else ""
                }

            level = "DEFAULT"
            status_message = "Success"
            if context_metadata:
                langfuse_metadata["context_metadata"] = context_metadata
                level, status_message = self._parse_trace_status(
                    context_metadata.get("status", TraceStatus.SUCCESS), {}
                )
            langfuse_metadata.update(new_tags)

            # Create generation trace
            if self._current_trace:
                self._current_agent = self._current_trace.generation(
                    name=f"agent:start:{name}",
                    input={"system_prompt": system_prompt},
                    metadata=langfuse_metadata,
                    timestamp=timestamp,
                    parent_observation_id=parent_span_id,
                    tags=tags,
                    level=level,
                    status_message=status_message
                )
            else:
                self._current_agent = self.client.generation(
                    name=f"agent:start:{name}",
                    input={"system_prompt": system_prompt},
                    metadata=langfuse_metadata,
                    timestamp=timestamp,
                    parent_observation_id=parent_span_id,
                    tags=tags,
                    level=level,
                    status_message=status_message,
                )
        except Exception as e:
            logger.debug(f"Error in on_agent_start: {e}")
            self._current_agent = None

    async def on_agent_end(
        self,
        name: str,
        output: object,
        parent_span_id: Optional[str],
        system_prompt: str,
        context_metadata: Optional[RunContextMetadata] = None,
        agent_config: Optional["AgentConfig"] = None,
    ) -> None:
        """Agent end callback - only updates trace if agent was started"""
        if not self.enabled or not self.client:
            return
            
        # CRITICAL FIX: If no agent was created in on_agent_start, skip
        if self._current_agent is None:
            return
        
        try:
            # Extract metadata
            timestamp = datetime.now()
            new_tags: Dict[str, str] = {}
            input_tokens, output_tokens, total_tokens = 0, 0, 0
            if context_metadata:
                timestamp = context_metadata.get("timestamp", datetime.now())
                new_tags = context_metadata.get("tags", {})
                request_data = context_metadata.get("request_data", {})
                input_tokens = request_data.get("input_tokens", 0)
                output_tokens = request_data.get("output_tokens", 0)
                total_tokens = request_data.get("total_tokens", 0)

            # Create metadata
            update_metadata: Dict[str, Any] = {
                "event_type": "agent_end",
                "agent_name": name,
                "output": str(output)[:1000] if output else None,  # Truncate long outputs
            }

            if agent_config:
                update_metadata["agent_config"] = {
                    "id": str(agent_config.id) if agent_config.id else "unknown",
                    "type": str(agent_config.agent_type) if hasattr(agent_config, "agent_type") else "unknown",
                    "description": str(agent_config.description) if hasattr(agent_config, "description") else "",
                }

            level = "DEFAULT"
            status_message = "Success"
            if context_metadata:
                update_metadata["context_metadata"] = context_metadata
                level, status_message = self._parse_trace_status(
                    context_metadata.get("status", TraceStatus.SUCCESS), {}
                )

            update_metadata.update(new_tags)

            usage = None
            if input_tokens or output_tokens:
                usage = {
                    "promptTokens": input_tokens,
                    "completionTokens": output_tokens,
                    "totalTokens": total_tokens,
                }

            # End the agent generation trace
            self._current_agent.end(
                name=f"agent:start:{name}",
                input={"system_prompt": system_prompt},
                output=output,
                metadata=update_metadata,
                usage=usage,
                tags=["event_type:agent_end", f"agent_name:{name}"],
                end_time=timestamp,
                level=level,
                status_message=status_message
            )
        except Exception as e:
            logger.debug(f"Error in on_agent_end: {e}")
        finally:
            self._current_agent = None

    async def on_handoff(
        self,
        from_agent_name: str,
        to_agent_name: str,
        parent_span_id: Optional[str],
        from_agent_config: Optional["AgentConfig"] = None,
        to_agent_config: Optional["AgentConfig"] = None,
    ) -> None:
        """Agent handoff callback"""
        if not self.enabled or not self.client:
            return
            
        try:
            span_id = self.gen_span_id()
            timestamp = datetime.now()
            
            # Create tags
            tags_dict = {
                "event_type": "agent_handoff", 
                "from_agent": from_agent_name,
                "to_agent": to_agent_name
            }
            tags = [f"{k}:{v}" for k, v in tags_dict.items()]
            
            # Create metadata
            langfuse_metadata: Dict[str, Any] = {
                "span_id": span_id,
                "timestamp": timestamp.isoformat(),
                "from_agent": from_agent_name,
                "to_agent": to_agent_name
            }
            
            if from_agent_config:
                langfuse_metadata["from_agent_config"] = {
                    "id": str(from_agent_config.id) if from_agent_config.id else "unknown",
                    "type": str(from_agent_config.agent_type) if hasattr(from_agent_config, "agent_type") else "unknown"
                }
                
            if to_agent_config:
                langfuse_metadata["to_agent_config"] = {
                    "id": str(to_agent_config.id) if to_agent_config.id else "unknown",
                    "type": str(to_agent_config.agent_type) if hasattr(to_agent_config, "agent_type") else "unknown"
                }

            # Log handoff as a span
            handoff_span = self.client.span(
                name=f"agent:handoff:{from_agent_name}-to-{to_agent_name}",
                session_id=self._sessionID,
                metadata=langfuse_metadata,
                timestamp=timestamp,
                parent_observation_id=parent_span_id,
                tags=tags,
            )
            handoff_span.end(end_time=timestamp)
        except Exception as e:
            logger.debug(f"Error in on_handoff: {e}")

    async def record_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        metadata: ScoreMetadata,
        span_id: Optional[str] = None
    ) -> None:
        """Attach a numeric evaluation to a trace/span."""
        if not self.enabled or not self.client:
            return
            
        try:
            evaluator = metadata.get("evaluator", "custom")
            details = metadata.get("details", {})
            
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=f"Evaluator: {evaluator}",
                metadata={**details, **metadata}
            )
        except Exception as e:
            logger.debug(f"Error recording score to Langfuse: {e}")