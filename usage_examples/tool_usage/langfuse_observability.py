"""
Langfuse implementation of LLMObservabilityProvider.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, Tuple, Literal

from langfuse.types import SpanLevel

if TYPE_CHECKING:
    from agent_workflow.workflow_engine.models import AgentConfig

from langfuse import Langfuse

from agent_workflow.providers import (
    BaseLLMObservabilityProvider,
    TraceMetadata,
    SpanMetadata,
    CallMetadata,
    ScoreMetadata,
    CommonMetadata,
    RunContextMetadata,
)
from agent_workflow.providers.llm_observability import TraceStatus


class LangfuseLLMObservabilityProvider(BaseLLMObservabilityProvider):
    """
    Langfuse implementation of LLMObservabilityProvider.
    Provides observability for LLM interactions using Langfuse.
    
    Implements:
     - Traces & trace groups
     - Spans
     - LLM call logging
     - Tool execution logging
     - Metrics / scores
     - Arbitrary events
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
        userid: Optional[str] = None
    ) -> None:
        """
        Initialize with Langfuse credentials.
        
        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL (defaults to Langfuse Cloud)
        """
        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        self._sessionID: Optional[str] = None
        self._current_span: Any = None
        self._current_agent: Any = None
        self._current_trace: Any = None
        self._current_trace_id: Optional[str] = None

        self.userid = userid
        
    # -- Generate ids --------------------------------
    def gen_group_id(self) -> str:
        """generate the group id and return"""
        return str(uuid.uuid4())

    def gen_trace_id(self) -> str:
        """generate the trace id and return"""
        return str(uuid.uuid4())

    def gen_span_id(self) -> str:
        """generate the span id and return"""
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
        pass

    @staticmethod
    def _parse_trace_status(trace_status: TraceStatus, tags: dict[str, str]) -> Tuple[Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] , str]:
        # Handle different trace statuses and map to appropriate log levels
        if trace_status == TraceStatus.FAILED:
            return "ERROR", str(tags)
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
        trace_id = self.gen_trace_id()
        time = metadata.get("timestamp", datetime.now())
        level, status_message = LangfuseLLMObservabilityProvider._parse_trace_status(
            metadata.get("status", TraceStatus.SUCCESS), {}
        )
        self._current_trace = self.client.trace(
            name=name,
            user_id=self.userid,
            session_id=group_id,
            input=prompt,
            id=trace_id,
            metadata=metadata,
            start_time=time,
            level=level,
            status_message=status_message,
        )
        return trace_id

    async def end_trace(
        self,
        prompt: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]],
        response: Any,
        trace_id: str,
        metadata: TraceMetadata,
    ) -> None:
        """End a trace and record any final metadata."""
        # Extract request data if available
        request_data = metadata.get("request_data", {})
        input_tokens = request_data.get("input_tokens", 0) if request_data else 0
        output_tokens = request_data.get("output_tokens", 0) if request_data else 0
        total_tokens = request_data.get("total_tokens", 0) if request_data else 0
        time = metadata.get("timestamp", datetime.now())
        tags = metadata.get("tags", {})
        trace_status = metadata.get("status", TraceStatus.SUCCESS)

        usage = {"promptTokens": input_tokens,
                 "completionTokens": output_tokens,
                 "totalTokens": total_tokens}


        if self._current_trace:

            level, status_message = LangfuseLLMObservabilityProvider._parse_trace_status(trace_status, tags)
            self._current_trace.end (
                input=prompt,
                response= response,
                id=trace_id,
                metadata=metadata,
                usgae= usage,
                end_time=time,
                level=level,
                status_message=status_message
            )
            self._current_trace = None


    # -- Spans ---------------------------------------
    async def start_span(
        self,
        name: str,
        parent_span_id: Optional[str],
        metadata: SpanMetadata,
    ) -> str:
        span_id = self.gen_span_id()

        # Extract metadata fields
        timestamp = metadata.get("timestamp", datetime.now())
        tags_dict = metadata.get("tags", {})
        # trace_status = metadata.get("status", TraceStatus.SUCCESS)
        # Convert dictionary tags to list format for Langfuse API
        tags = [f"{k}:{v}" for k, v in tags_dict.items()]
        
        # Extract request data if available
        request_data = metadata.get("request_data", {})
        input_tokens = request_data.get("input_tokens", 0) if request_data else 0
        output_tokens = request_data.get("output_tokens", 0) if request_data else 0
        total_tokens = request_data.get("total_tokens", 0) if request_data else 0
        latency_ms = request_data.get("latency_ms", 0) if request_data else 0
        time = request_data.get("timestamp", datetime.now())
        
        # Extract span type specific details
        span_data = metadata.get("span", {})
        if span_data:
            if span_data.get("tool_type"):
                # This is a tool span
                tags_dict["span_type"] = "tool"
                tags_dict["tool_type"] = str(span_data.get("tool_type", "functional"))
                if span_data.get("tags"):
                    tags_dict.update(span_data.get("tags", {}))
                # Regenerate tags list after updates
                tags = [f"{k}:{v}" for k, v in tags_dict.items()]
            elif span_data.get("log_level"):
                # This is a debug span
                tags_dict["span_type"] = "debug"
                tags_dict["log_level"] = str(span_data.get("log_level", "DEFAULT"))
                # Regenerate tags list after updates
                tags = [f"{k}:{v}" for k, v in tags_dict.items()]
        
        # Create a metadata dict for Langfuse
        langfuse_metadata = {
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "trace_id": parent_span_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            **metadata
        }

        level, status_message = LangfuseLLMObservabilityProvider._parse_trace_status(
            metadata.get("status", TraceStatus.SUCCESS), {}
        )

        param_dict = {
            "name": name,
            "trace_id": span_id,
            "parent_observation_id": parent_span_id,
            "metadata": langfuse_metadata,
            "timestamp": timestamp,
            "tags": tags,
            "start_time": time,
            "level": level,
            "status_message": status_message
        }

        # Create the span in Langfuse
        if self._current_trace:
            self._current_span = self._current_trace.span(
                **param_dict
            )
        else:
            # Create the span in Langfuse
            self._current_span = self.client.span(
                **param_dict
            )
        
        # Return the generated span ID
        return span_id

    async def end_span(
        self,
        span_id: str,
        metadata: SpanMetadata
    ) -> None:
        """End a span and attach any metadata (e.g. token counts)."""
        if not self._current_span:
            return
            
        tags_dict = metadata.get("tags", {})
        tags = [f"{k}:{v}" for k, v in tags_dict.items()]
        
        # Extract request data if available
        request_data = metadata.get("request_data", {})
        input_tokens = request_data.get("input_tokens", 0) if request_data else 0
        output_tokens = request_data.get("output_tokens", 0) if request_data else 0
        total_tokens = request_data.get("total_tokens", 0) if request_data else 0
        latency_ms = request_data.get("latency_ms", 0) if request_data else 0
        end_time = request_data.get("timestamp", datetime.now())
        
        # Create a metadata dict for Langfuse
        langfuse_metadata = {
            "span_id": span_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            **metadata
        }

        level, status_message = LangfuseLLMObservabilityProvider._parse_trace_status(
            metadata.get("status", TraceStatus.SUCCESS), {}
        )

        # Update and end the span
        if self._current_span:
            self._current_span.end(
                metadata=langfuse_metadata,
                tags=tags,
                end_time=end_time,
                level=level,
                status_message=status_message,
            )
            self._current_span = None


    # -- LLM call logging ----------------------------
    async def log_trace(
            self,
            name: str,
            model: str,
            prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
            response: Any,
            metadata: CallMetadata,
            trace_id: Optional[str]) -> None:
        """Log a full LLM interaction, tied to a trace/span if desired."""
        pass
        # timestamp = metadata.get("timestamp", datetime.now())
        # tags_dict: Dict[str, str] = metadata.get("tags", {})
        # # Convert dictionary tags to list format for Langfuse API
        # tags = [f"{k}:{v}" for k, v in tags_dict.items()]
        #
        # if self._current_trace is None:
        #     # If no trace is active, create a new one
        #     self._current_trace = self.client.trace(
        #         name=name,
        #         model=model,
        #         user_id=self.userid,
        #         session_id=self._sessionID,
        #         id=trace_id,  # Use provided trace_id
        #         input=prompt,
        #         output=response,
        #         metadata=metadata,
        #         timestamp=timestamp,
        #         tags=tags
        #     )
        # else:
        #     # If a trace is already active (from start_trace), update it
        #     # This connects the log_trace call with the previously started trace
        #     self._current_trace.update(
        #         name=name,
        #         model=model,
        #         id=trace_id,  # Use provided trace_id
        #         user_id=self.userid,
        #         input=prompt,
        #         output=response,
        #         metadata=metadata,
        #         timestamp=timestamp,
        #         tags=tags
        #     )


    async def log_span(
        self,
        name: str,
        model: str,
        prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        response: Any,
        metadata: CallMetadata,
        span_id: str,
    ) -> None:
        timestamp = metadata.get("timestamp", datetime.now())
        tags_dict: Dict[str, str] = metadata.get("tags", {})
        # Convert dictionary tags to list format for Langfuse API
        tags = [f"{k}:{v}" for k, v in tags_dict.items()]

        level, status_message = LangfuseLLMObservabilityProvider._parse_trace_status(
            metadata.get("status", TraceStatus.SUCCESS), {}
        )

        if self._current_span:
            self._current_span.update(
                name=name,
                model=model,
                parent_observation_id=self._sessionID,
                input=prompt,
                output=response,
                metadata=metadata,
                timestamp=timestamp,
                tags=tags,
                level=level,
                status_message=status_message
            )
        else:
            span = self.client.span(
                name=name,
                model=model,
                parent_observation_id=self._sessionID,
                input=prompt,
                output=response,
                metadata=metadata,
                timestamp=timestamp,
                tags=tags,
                level=level,
                status_message=status_message,
            )
            span.end(end_time=timestamp)

    # -- Scores / metrics ----------------------------
    async def record_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        metadata: ScoreMetadata,
        span_id: str
    ) -> None:
        """Attach a numeric evaluation (e.g. quality, latency) to a trace/span."""
        pass

    # -- General events / logs -----------------------
    async def log_event(
        self,
        message: str,
        level: str,
        trace_id: str,
        span_id: str,
        metadata: CommonMetadata
    ) -> None:
        """Emit a structured log or annotation within a trace/span."""
        pass

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
            parent_span_id: Optional ID of the parent span
            system_prompt: The system prompt used by the agent
            context_metadata: Optional metadata about the execution context
            agent_config: Optional agent configuration object
        """
        timestamp = datetime.now()
        new_tags = {}
        if context_metadata:
            timestamp = context_metadata.get("timestamp", datetime.now())
            # request_data = context_metadata['request_data']
            new_tags = context_metadata.get("tags", {})

        # Extract agent details for tags
        tags_dict = {"event_type": "agent_start", "agent_name": name}
        if agent_config:
            tags_dict["agent_id"] = str(agent_config.id) if agent_config.id else "unknown"
            tags_dict["agent_type"] = str(agent_config.agent_type) if hasattr(agent_config, "agent_type") and agent_config.agent_type else "unknown"
            tags_dict["agent_version"] = str(agent_config.version) if hasattr(agent_config, "agent_version") and agent_config.version else "unknown"
        
        # Convert dictionary tags to list format for Langfuse API
        tags = [f"{k}:{v}" for k, v in tags_dict.items()]
        
        # Create a metadata dict for Langfuse
        langfuse_metadata: Dict[str, Any] = {
            "timestamp": timestamp,
            "agent_name": name,
        }
        if agent_config:
            config = {
                "id": agent_config.id,
                "type": agent_config.agent_type if hasattr(agent_config, "agent_type") else "unknown",
                "description": agent_config.description if hasattr(agent_config, "description") else ""
            }

            langfuse_metadata["agent_config"] = config

        level = "DEFAULT"
        status_message = "SUCCESS"
        if context_metadata:
            langfuse_metadata["context_metadata"] = context_metadata
            level, status_message = (
                LangfuseLLMObservabilityProvider._parse_trace_status(
                    context_metadata.get("status", TraceStatus.SUCCESS), {}
                )
            )
        langfuse_metadata.update(new_tags)


        if self._current_agent:
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
            # Log agent start as a span
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
            parent_span_id: Parent span ID
            system_prompt: The system prompt used by the agent
            context_metadata: Optional metadata about the execution context
            agent_config: Optional agent configuration object
        """
        timestamp = datetime.now()
        new_tags: dict[str, str] = {}
        input_tokens, output_tokens, total_tokens = 0, 0, 0
        if context_metadata:
            timestamp = context_metadata.get("timestamp", datetime.now())
            new_tags = context_metadata.get("tags", {})
            request_data = context_metadata.get("request_data", {})
            input_tokens = request_data.get("input_tokens", 0)
            output_tokens = request_data.get("output_tokens", 0)
            total_tokens = request_data.get("total_tokens", 0)

        # Create a metadata dict for Langfuse
        update_metadata: Dict[str, Any] = {
            "event_type": "agent_end",
            "agent_name": name,
            "output": output,  # Truncate if too long
        }

        if agent_config:
            config = {
                "id": agent_config.id,
                "type": agent_config.agent_type if hasattr(agent_config, "agent_type") else "unknown",
                "description": agent_config.description if hasattr(agent_config, "description") else "",
            }
            update_metadata["agent_config"] = config

        level = "DEFAULT"
        status_message = "SUCCESS"
        if context_metadata:
            update_metadata["context_metadata"] = context_metadata
            level, status_message = (
                LangfuseLLMObservabilityProvider._parse_trace_status(
                    context_metadata.get("status", TraceStatus.SUCCESS), {}
                )
            )

        update_metadata.update(new_tags)

        usage = {
            "promptTokens": input_tokens,
            "completionTokens": output_tokens,
            "totalTokens": total_tokens,
        }

        if self._current_agent:
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
            self._current_agent = None


    async def on_handoff(
        self,
        from_agent_name: str,
        to_agent_name: str,
        parent_span_id: Optional[str],
        from_agent_config: Optional["AgentConfig"] = None,
        to_agent_config: Optional["AgentConfig"] = None,
    ) -> None:
        """Agent handoff callback
        
        Args:
            from_agent_name: The name of the agent handing off control
            to_agent_name: The name of the agent receiving control
            parent_span_id: Optional[str],
            from_agent_config: Optional configuration of the agent handing off control
            to_agent_config: Optional configuration of the agent receiving control
        """
        span_id = self.gen_span_id()
        timestamp = datetime.now()
        
        # Extract agent details for tags
        tags_dict = {
            "event_type": "agent_handoff", 
            "from_agent": from_agent_name,
            "to_agent": to_agent_name
        }
        
        # Convert dictionary tags to list format for Langfuse API
        tags = [f"{k}:{v}" for k, v in tags_dict.items()]
        
        # Create a metadata dict for Langfuse
        langfuse_metadata: Dict[str, Any] = {
            "span_id": span_id,
            "timestamp": timestamp,
            "from_agent": from_agent_name,
            "to_agent": to_agent_name
        }
        
        if from_agent_config:
            langfuse_metadata["from_agent_config"] = {
                "id": from_agent_config.id,
                "type": from_agent_config.agent_type if hasattr(from_agent_config, "agent_type") else "unknown"
            }
            
        if to_agent_config:
            langfuse_metadata["to_agent_config"] = {
                "id": to_agent_config.id,
                "type": to_agent_config.agent_type if hasattr(to_agent_config, "agent_type") else "unknown"
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


# Helper function to easily create a provider
def create_langfuse_provider(
    public_key: str, 
    secret_key: str, 
    host: str = "https://cloud.langfuse.com"
) -> LangfuseLLMObservabilityProvider:
    """
    Create and configure a Langfuse LLM observability provider.
    
    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse host URL (defaults to Langfuse Cloud)
        
    Returns:
        Configured LangfuseLLMObservabilityProvider instance
    """
    return LangfuseLLMObservabilityProvider(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )