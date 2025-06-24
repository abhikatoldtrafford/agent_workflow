"""
Utility class for LLM tracing that can be used by any ExecutionEngine implementation.

This module provides common patterns for tracing LLM interactions at different levels:
workflow, stage, and task. It includes consistent error handling and utility functions
that can be reused across different execution engines.
"""
import logging
import asyncio
from typing import Any, Dict, Optional, TypeVar, cast, Callable, List, Set, Tuple, Union
from datetime import datetime
import uuid
from functools import wraps
from inspect import iscoroutinefunction

from agent_workflow.providers.llm_observability import (
    LLMObservabilityProvider,
    NoOpLLMObservabilityProvider,
    TraceMetadata,
    SpanMetadata,
    CallMetadata,
    RequestData,
    RunContextMetadata,
    ToolSpan, 
    TraceStatus
)
from agent_workflow.workflow_engine.models import Tool, AgentConfig

# Set up logging
logger = logging.getLogger("workflow-engine.providers.llm_tracing_utils")

# Type definitions
T = TypeVar('T')

F = TypeVar("F", bound=Callable[..., Any])

def with_exception_handler(handler: Callable[[Exception], Any]) -> Callable[[F], F]:
    """
    Decorator to wrap a function with a try/except block and call the handler on exception.
    Supports both synchronous and asynchronous functions.

    :param handler: A function that takes an Exception and handles it.
    :return: A decorator that adds try/except to the decorated function.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return handler(e)
                
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handler(e)
                
        # Use appropriate wrapper based on whether the function is async or not
        if iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)
    return decorator

# Default exception handler for use with the decorator
def default_logging_exception_handler(e: Exception) -> TraceStatus:
    logger.error(f"Exception in decorated method: {e}")
    return TraceStatus.FAILED

class LLMTracer:
    """A class to manage LLM tracing with support for managing group and trace IDs."""

    def __init__(self, providers: Optional[List[LLMObservabilityProvider]] = None):
        """
        Initialize the LLMTracer with support for parallel traces within a group.

        Args:
            providers: The LLM observability provider or a list of providers to use. If None, a NoOp provider is used.
        """
        # Handle single provider or list of providers
        if providers is None:
            self.providers: List[LLMObservabilityProvider] = [NoOpLLMObservabilityProvider()]
        else:
            self.providers = providers

        # Single workflow group ID
        self.group_id: Optional[str] = None
        
        # Active traces and spans - allow multiple concurrent traces
        # Maps trace_id -> set of span_ids
        self.active_traces: Dict[str, Set[str]] = {}
        
        # Maps span_id -> trace_id for efficient lookup
        self.span_traces: Dict[str, str] = {}
        
        # Context stack for async operations
        self._context_stack: List[Dict[str, Optional[str]]] = []
        
    def _get_current_context(self) -> Dict[str, Optional[str]]:
        """
        Get the current execution context with trace/span IDs.
        Creates a new empty context if none exists.
        
        Returns:
            Dictionary containing current trace_id, span_id and group_id
        """
        if not self._context_stack:
            self._context_stack.append({"trace_id": None, "span_id": None, "group_id": self.group_id})
        return self._context_stack[-1]
    
    def _update_current_context(self, trace_id: Optional[str] = None, span_id: Optional[str] = None, group_id: Optional[str] = None) -> None:
        """
        Update the current execution context with new values.
        
        Args:
            trace_id: Optional new trace ID to set in the context
            span_id: Optional new span ID to set in the context
            group_id: Optional new group ID to set in the context
        """
        current = self._get_current_context()
        
        if trace_id is not None:
            current["trace_id"] = trace_id
            
        if span_id is not None:
            current["span_id"] = span_id
            
        if group_id is not None:
            current["group_id"] = group_id
            self.group_id = group_id
                
    def register_trace(self, trace_id: str) -> None:
        """Register a new trace."""
        if trace_id not in self.active_traces:
            self.active_traces[trace_id] = set()
            
    def register_span(self, span_id: str, trace_id: str) -> None:
        """Register a span with its associated trace."""
        if trace_id in self.active_traces:
            self.active_traces[trace_id].add(span_id)
            self.span_traces[span_id] = trace_id
            
    def get_trace_for_span(self, span_id: str) -> Optional[str]:
        """Get the trace ID for a span."""
        return self.span_traces.get(span_id)
        
    def get_spans_for_trace(self, trace_id: str) -> Set[str]:
        """Get all spans for a trace."""
        return self.active_traces.get(trace_id, set())

    @staticmethod
    def _input_dict(system_prompt: Optional[str], prompt: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"prompt": prompt}
        if system_prompt is not None:
            result["system_prompt"] = system_prompt
        return result


    @staticmethod
    def _extract_context_metadata_fields(context_metadata: Optional[Union[Dict[str, Any], RunContextMetadata]]) -> Tuple[datetime, RequestData, int, dict[str, str]]:
        """Extract common metadata fields from context_metadata and apply them to the target metadata.
        
        Args:
            context_metadata: The source context metadata dictionary
            :type context_metadata: Optional[Dict[str, Any]]
        """
        timestamp = datetime.now()
        request_data = RequestData()
        requests = 0
        tags: dict[str, str] = {}

        if context_metadata is not None:
            # Extract timestamp if present
            if "timestamp" in context_metadata:
                timestamp = context_metadata["timestamp"]

            # Extract request_data if present

            if "request_data" in context_metadata:
                context_data = context_metadata["request_data"]
                request_data['input_tokens'] = context_data['input_tokens']
                request_data["output_tokens"] = context_data["output_tokens"]
                request_data["total_tokens"] = context_data["total_tokens"]

            # Extract requests and add to tags

            if "requests" in context_metadata:
                requests = int(context_metadata["requests"])

            if "tags" in context_metadata:
                tags = context_metadata["tags"]

        return timestamp, request_data, requests, tags


    # ---------------------------------------------------------------------
    # Workflow-level tracing (trace groups)
    # ---------------------------------------------------------------------

    # Custom error handler for start_workflow_tracing
    async def _handle_workflow_start_error(self, e: Exception) -> str:
        logger.error(f"Failed to start workflow trace group: {e}")
        group_id = str(uuid.uuid4())
        self._update_current_context(group_id=group_id)
        return group_id

    # No decorator here - we'll implement the try/catch directly
    async def start_workflow_tracing(
        self,
        workflow_id: str,
        workflow_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a trace group for workflow execution.

        Args:
            workflow_id: Unique identifier for the workflow
            workflow_name: Display name of the workflow
            metadata: Additional metadata to include

        Returns:
            The group ID for the workflow trace
        """
        try:
            # Create TraceMetadata with required fields
            # We need to avoid using dict unpacking with TypedDict
            meta = TraceMetadata()
            meta["workflow_id"] = workflow_id
            meta["description"] = f"Workflow execution: {workflow_name}"
            meta["timestamp"] = datetime.now()
            
            # Add tags if provided
            if metadata and metadata.get("tags"):
                meta["tags"] = metadata.get("tags", {})

            # Start trace group with all providers
            group_ids = await asyncio.gather(*[provider.start_trace_group(metadata=meta) for provider in self.providers])
            
            # Use the first provider's group_id as the reference ID
            group_id = group_ids[0] if group_ids else str(uuid.uuid4())
            self._update_current_context(group_id=group_id)
            logger.debug(f"Started workflow trace group: {group_id}")

            return group_id
        except Exception as e:
            # Handle the exception with our custom handler
            return await self._handle_workflow_start_error(e)

    @with_exception_handler(default_logging_exception_handler)
    async def end_workflow_tracing(
        self,
        group_id: Optional[str] = None,
        status: TraceStatus = TraceStatus.SUCCESS,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End a workflow trace group.

        Args:
            group_id: The group ID. If None, uses current_group_id
            status: Completion status ("completed" or "failed")
            error: Exception if the workflow failed
            metadata: Additional metadata to include
        """

        current_context = self._get_current_context()
        group_id = group_id or current_context.get("group_id")
        if not group_id:
            logger.error("No group ID provided and no current group ID set")
            return

        # Create metadata
        tags = {"status": status.name}
        if error:
            tags["error"] = str(error)
            tags["error_type"] = type(error).__name__
            status = TraceStatus.FAILED


        dt, request_data, requests, new_tags = LLMTracer._extract_context_metadata_fields(metadata)
        tags.update(new_tags)

        meta = TraceMetadata(
            description=f"Workflow {status}", timestamp=dt, tags=tags
        )

        # End trace group across all providers
        await asyncio.gather(*[provider.end_trace_group(group_id=group_id, metadata=meta) for provider in self.providers])
        logger.debug(f"Ended workflow trace group: {group_id}")

        # Clear current group ID if it matches
        current_context = self._get_current_context()
        if current_context.get("group_id") == group_id:
            self._update_current_context(group_id=None)


    # ---------------------------------------------------------------------
    # Task-level tracing (spans)
    # ---------------------------------------------------------------------

    @with_exception_handler(default_logging_exception_handler)
    async def start_task_trace(
        self,
        system_prompt: Optional[str],
        prompt: str,
        task_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Start a trace for task execution.

        Args:
            system_prompt: The system prompt to include
            prompt: The prompt text
            task_name: Name of the task
            metadata: Additional metadata to include

        Returns:
            The trace ID for the task trace
        """

        # Create TraceMetadata with required fields
        # We need to avoid using dict unpacking with TypedDict
        dt, request_data, requests, tags = LLMTracer._extract_context_metadata_fields(metadata)
        meta = TraceMetadata()
        meta["description"] = f"Task execution: {task_name}"
        meta["timestamp"] = dt
        meta["tags"] = tags
        meta["request_data"] = request_data

        prompt_dict = self._input_dict(system_prompt, prompt)

        # Get group ID from current context
        current_context = self._get_current_context()
        trace_group_id = current_context.get("group_id")
        
        # Start trace across all providers
        trace_ids = await asyncio.gather(*[provider.start_trace(metadata=meta, group_id=trace_group_id, prompt=prompt_dict, name=task_name) for provider in self.providers])
        
        # Use the first provider's trace_id as the reference ID
        trace_id = trace_ids[0] if trace_ids else str(uuid.uuid4())
        logger.debug(f"Started task trace: {trace_id}")
        
        # Update tracking
        self._update_current_context(trace_id=trace_id)
        self.register_trace(trace_id)

        return trace_id

    @with_exception_handler(default_logging_exception_handler)
    async def end_task_trace(
        self,
        system_prompt: Optional[str],
        prompt: str,
        response: Any,
        trace_id: Optional[str] = None,
        status: TraceStatus = TraceStatus.SUCCESS,
        error: Optional[Exception] = None,
        metadata_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        End a task trace.

        Args:
            system_prompt: The system prompt to include
            prompt: The prompt text
            response: The response from the task execution
            trace_id: The trace ID. If None, uses current_trace_id
            status: Completion status ("completed" or "failed")
            error: Exception if the task failed
            metadata_dict: Additional metadata to include
        """
        current_context = self._get_current_context()
        trace_id = trace_id or current_context.get("trace_id")
        if not trace_id:
            logger.error("No trace ID provided and no current trace ID set")
            return

        # Create metadata
        tags = {"status": status.name}
        if error:
            tags["error"] = str(error)
            tags["error_type"] = type(error).__name__
            status = TraceStatus.FAILED

        if metadata_dict and metadata_dict.get("tags"):
            tags.update(metadata_dict.get("tags", {}))

        dt, request_data, requests, new_tags = LLMTracer._extract_context_metadata_fields(
            metadata_dict
        )
        tags.update(new_tags)

        meta = TraceMetadata()
        meta["description"] = f"Task {status.name}"
        meta["status"] = status
        meta["timestamp"] = dt
        meta["tags"] = tags
        meta["request_data"] = request_data

        prompt_dict = self._input_dict(system_prompt, prompt)

        # End trace across all providers
        await asyncio.gather(*[provider.end_trace(trace_id=trace_id,
                                        metadata=meta,
                                        prompt=prompt_dict,
                                        response=response) for provider in self.providers])
        logger.debug(f"Ended task trace: {trace_id}")

        # Clear current trace ID if it matches
        current_context = self._get_current_context()
        if current_context.get("trace_id") == trace_id:
            self._update_current_context(trace_id=None)

    @with_exception_handler(default_logging_exception_handler)
    async def start_span(
        self,
        parent_trace_id: Optional[str] = None,
        span_name: str = "",
        metadata_dict: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Start a span for a specific operation within a task.

        Args:
            parent_trace_id: Parent task trace ID. If None, uses current_trace_id
            span_name: Name of the span operation
            metadata_dict: Additional metadata to include

        Returns:
            The span ID for the operation span
        """
        # Create RequestData directly
        dt, request_data, requests, tags = LLMTracer._extract_context_metadata_fields(metadata_dict)
        meta = SpanMetadata(timestamp=dt,
                            request_data=request_data,
                            tags=tags)

        # Start span across all providers
        span_ids = await asyncio.gather(*[provider.start_span(
            name=span_name, parent_span_id=parent_trace_id, metadata=meta
        ) for provider in self.providers])
        
        # Use the first provider's span_id as the reference ID
        span_id = span_ids[0] if span_ids else str(uuid.uuid4())
        logger.debug(f"Started operation span: {span_id}")

        self._update_current_context(span_id=span_id)

        return span_id

    @with_exception_handler(default_logging_exception_handler)
    async def end_span(
        self,
        span_id: str,
        status: TraceStatus = TraceStatus.SUCCESS,
        metadata_dict: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> None :
        """
        End an operation span within a task.

        Args:
            span_id: The span ID returned from start_span
            status: TraceStatus enum indicating completion status (SUCCESS, FAILED, IN_PROGRESS)
            metadata_dict: Optional dictionary with additional metadata
            error: Exception if the operation failed
        """
        # Create common metadata with tags
        tags = {"status": status.name}
        if error:
            tags["error"] = str(error)
            tags["error_type"] = type(error).__name__
            status = TraceStatus.FAILED

        # Metadata
        dt, request_data, requests, new_tags = LLMTracer._extract_context_metadata_fields(
            metadata_dict
        )
        meta = SpanMetadata(timestamp=dt, request_data=request_data, tags=tags, status=status)

        tags.update(new_tags)

        current_context = self._get_current_context()
        current_span_id = current_context.get("span_id")

        # End span - ensure we have a valid span_id
        if current_span_id:
            await asyncio.gather(*[provider.end_span(span_id=current_span_id, metadata=meta) for provider in self.providers])
            logger.debug(f"Ended operation span: {span_id}")
            # Clear the span ID from context
            self._update_current_context(span_id=None)
        else:
            logger.warning("Cannot end span - no valid span_id available")

        return None

    @with_exception_handler(default_logging_exception_handler)
    async def trace_llm_call(
        self,
        name: str,
        model: str,
        system_prompt: Optional[str],
        prompt: str,
        response: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an LLM call.

        Args:
            name: The name of the task
            model: The name of the model used
            system_prompt: The system_prompt
            prompt: The prompt sent to the model
            response: The response received from the model
            metadata: Additional metadata to include
        """
        current_context = self._get_current_context()
        task_trace_id = current_context.get("trace_id")
        if task_trace_id is None:
            logger.error(
                "No task trace ID provided and no current trace ID set, and no span ID provided"
            )
            # continue execution

        # Create metadata
        dt, request_data, requests, tags = LLMTracer._extract_context_metadata_fields(metadata)
        meta = CallMetadata(prompt_type="task_execution",
                            request_data=request_data,
                            timestamp=dt,
                            tags=tags)

        prompt_dict = self._input_dict(system_prompt, prompt)

        # Log LLM call across all providers - this will update the trace started with start_trace
        await asyncio.gather(*[provider.log_trace(
            name=name,
            model=model,
            prompt=prompt_dict,
            response=response,
            metadata=meta,
            trace_id=task_trace_id,
        ) for provider in self.providers])
        logger.debug(f"Logged LLM call for trace: {task_trace_id}")

    @with_exception_handler(default_logging_exception_handler)
    async def on_agent_start(
            self,
            name: str,
            system_prompt: str,
            context_metadata: Optional[RunContextMetadata] = None,
            agent_config: Optional[AgentConfig] = None
    ) -> None:
        """Agent starts callback

        Args:
            name: The agent name
            system_prompt: The system prompt used by the agent
            context_metadata: Optional metadata about the execution context
            agent_config: Optional agent configuration object
        """
        # Create metadata with agent info if available
        current_context = self._get_current_context()
        parent_trace_id = current_context.get("trace_id")
        
        await asyncio.gather(*[provider.on_agent_start(
            name=name,
            parent_span_id=parent_trace_id,
            system_prompt=system_prompt,
            context_metadata=context_metadata,
            agent_config=agent_config) for provider in self.providers])

    @with_exception_handler(default_logging_exception_handler)
    async def on_agent_end(
        self,
        name: str,
        system_prompt: str,
        output: object,
        context_metadata: Optional[RunContextMetadata] = None,
        agent_config: Optional[AgentConfig] = None,
    ) -> None:
        """Agent end callback

        Args:
            name: The agent name
            system_prompt: The system prompt used by the agent
            output: The agent output
            context_metadata: Optional metadata about the execution context
            agent_config: Optional agent configuration object
        """
        current_context = self._get_current_context()
        parent_trace_id = current_context.get("trace_id")
        
        await asyncio.gather(*[provider.on_agent_end(
            name=name,
            output=output,
            parent_span_id=parent_trace_id,
            system_prompt=system_prompt,
            context_metadata=context_metadata,
            agent_config=agent_config) for provider in self.providers])

    @with_exception_handler(default_logging_exception_handler)
    async def on_handoff(
            self,
            from_agent_name: str,
            to_agent_name: str,
            from_agent_config: Optional[AgentConfig] = None,
            to_agent_config: Optional[AgentConfig] = None
    ) -> None:
        """Agent handoff callback

        Args:
            from_agent_name: The name of the agent handing off control
            to_agent_name: The name of the agent receiving control
            from_agent_config: Optional configuration of the agent handing off control
            to_agent_config: Optional configuration of the agent receiving control
        """
        current_context = self._get_current_context()
        parent_trace_id = current_context.get("trace_id")
        
        await asyncio.gather(*[provider.on_handoff(
            from_agent_name,
            to_agent_name,
            parent_trace_id,
            from_agent_config,
            to_agent_config,
        ) for provider in self.providers])

    @with_exception_handler(default_logging_exception_handler)
    async def on_tool_start(
        self,
        agent_name: str,
        tool_name: str,
        context_metadata: Optional[RunContextMetadata] = None,
        agent_config: Optional[AgentConfig] = None,
        tool: Optional[Tool] = None,
    ) -> None:
        """Tool start callback
        
        Args:
            agent_name: The name of the agent using the tool
            tool_name: The name of the tool being used
            context_metadata: Optional metadata about the execution context
            agent_config: Optional configuration of the agent
            tool: Optional tool object with additional information
        """
        # Create tags with tool and agent info
        tags = {
            "tool_name": tool_name,
            "agent_name": agent_name,
            "tool_description": "" if not tool else tool.description or ""
        }
        
        # Add agent info if provided
        if agent_config:
            if agent_config.id:
                tags["agent_id"] = str(agent_config.id)
            if hasattr(agent_config, "agent_type") and agent_config.agent_type:
                tags["agent_type"] = str(agent_config.agent_type)
                
        # Create ToolSpan with tool info
        tool_span = ToolSpan(
            tool_type="functional" if not tool else "openai" if tool.type == "openai" else "functional",
            timestamp=datetime.now(),
            tags=tags
        )

        dt, request_data, requests, new_tags = LLMTracer._extract_context_metadata_fields(context_metadata)
        tags.update(new_tags)
        # Create full span metadata
        metadata = SpanMetadata(
            span=tool_span,
            request_data=request_data,
            timestamp=dt,
            tags=tags
        )

        current_context = self._get_current_context()
        parent_trace_id = current_context.get("trace_id")
        
        span_ids = await asyncio.gather(*[provider.start_span(
            name=tool_name,
            parent_span_id=parent_trace_id,
            metadata=metadata
        ) for provider in self.providers])
        
        # Use the first provider's span_id as the reference ID
        span_id = span_ids[0] if span_ids else str(uuid.uuid4())
        
        self._update_current_context(span_id=span_id)

    @with_exception_handler(default_logging_exception_handler)
    async def on_tool_end(
        self,
        agent_name: str,
        tool_name: str,
        result: str,
        context_metadata: Optional[RunContextMetadata] = None,
        agent_config: Optional[AgentConfig] = None,
        tool: Optional[Tool] = None,
    ) -> None:
        """Tool end callback
        
        Args:
            agent_name: The name of the agent using the tool
            tool_name: The name of the tool being used
            result: The result of the tool execution
            context_metadata: Optional metadata about the execution context
            agent_config: Optional configuration of the agent
            tool: Optional tool object with additional information
        """
        current_context = self._get_current_context()
        current_span_id = current_context.get("span_id")
        
        if current_span_id is None:
            logger.warning(f"No span ID found for tool {tool_name}. Cannot log tool completion.")
            return
            
        # Create CallMetadata for log_span
        dt, request_data, requests, new_tags = (
            LLMTracer._extract_context_metadata_fields(context_metadata)
        )
        call_meta = CallMetadata(
            prompt_type="tool_execution",
            timestamp=dt,
            request_data=request_data,
            tags=new_tags
        )

        # Log the result as a span across all providers
        await asyncio.gather(*[provider.log_span(
            name=tool_name,
            model="tool",
            prompt={"tool_name": tool_name, "agent": agent_name},
            response=result,
            span_id=current_span_id,
            metadata=call_meta
        ) for provider in self.providers])

        # Create tags with tool and agent info
        tags = {
            "tool_name": tool_name,
            "agent_name": agent_name,
            "status": TraceStatus.SUCCESS.name,
            "tool_description": "" if not tool else tool.description or ""
        }
        
        # Add context metadata if provided
        if context_metadata and context_metadata.get("tags"):
            tags.update(context_metadata.get("tags", {}))
            
        # Add agent info if provided
        if agent_config:
            if agent_config.id:
                tags["agent_id"] = str(agent_config.id)
            if hasattr(agent_config, "agent_type") and agent_config.agent_type:
                tags["agent_type"] = str(agent_config.agent_type)
        
        # Create tool span for completion
        tool_span = ToolSpan(
            tool_type="functional" if not tool else "openai" if tool.type == "openai" else "functional",
            timestamp=datetime.now(),
            tags=tags
        )

        tags.update(new_tags)
        # Create full span metadata
        span_meta = SpanMetadata(
            span=tool_span,
            request_data=request_data,
            timestamp=dt,
            tags=tags,
            status=TraceStatus.SUCCESS
        )

        # End the span across all providers
        await asyncio.gather(*[provider.end_span(
            span_id=current_span_id,
            metadata=span_meta
        ) for provider in self.providers])

        # Reset the current span ID
        self._update_current_context(span_id=None)