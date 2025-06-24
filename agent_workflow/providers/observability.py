import functools
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

# Type for decorated functions
T = TypeVar("T", bound=Callable)

logger = logging.getLogger("workflow-engine.observability")


class ObservabilityProvider(ABC):
    """Abstract base class for observability providers."""

    @abstractmethod
    def trace_agent(self, agent_id: str, metadata: Dict[str, Any]) -> Any:
        """Create a new trace for an agent execution."""
        pass

    @abstractmethod
    def trace_workflow(self, workflow_id: str, metadata: Dict[str, Any]) -> Any:
        """Create a new trace for a workflow execution."""
        pass

    @abstractmethod
    def trace_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create a new span within a trace."""
        pass

    @abstractmethod
    def end_span(self, span_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """End a span and record its metadata."""
        pass

    @abstractmethod
    def log_llm_call(
        self,
        model: str,
        prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        response: Any,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> None:
        """Log an LLM call."""
        pass

    @abstractmethod
    def capture_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture a score for a trace."""
        pass

    def trace(self, name: Optional[str] = None) -> Callable[[T], T]:
        """Decorator for tracing functions."""

        def decorator(func: T) -> T:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate span name if not provided
                span_name = name or f"{func.__module__}.{func.__name__}"

                # Start span
                span_id = self.trace_span(
                    name=span_name,
                    metadata={
                        "args": str(args),
                        "kwargs": str(kwargs),
                        "function": func.__name__,
                    },
                )

                start_time = time.time()
                try:
                    # Execute the original function
                    result = await func(*args, **kwargs)

                    # End span with success
                    self.end_span(
                        span_id,
                        metadata={
                            "status": "success",
                            "duration_ms": (time.time() - start_time) * 1000,
                            "result_summary": str(result)[:100],  # Just log a summary
                        },
                    )

                    return result
                except Exception as e:
                    # End span with error
                    self.end_span(
                        span_id,
                        metadata={
                            "status": "error",
                            "error": str(e),
                            "duration_ms": (time.time() - start_time) * 1000,
                        },
                    )
                    raise

            return wrapper  # type: ignore

        return decorator


class LangfuseProvider(ObservabilityProvider):
    """Langfuse implementation of the observability provider."""

    def __init__(
        self, public_key: str, secret_key: str, host: str = "https://cloud.langfuse.com"
    ):
        """Initialize the Langfuse provider.

        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL
        """
        try:
            from langfuse import Langfuse

            self.langfuse = Langfuse(
                public_key=public_key, secret_key=secret_key, host=host
            )
            self.enabled = True
        except ImportError:
            logger.warning(
                "Langfuse package not installed. Install with: pip install langfuse"
            )
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            self.enabled = False

        self.active_traces: Dict[str, Any] = {}
        self.active_spans: Dict[str, Any] = {}

    def trace_agent(self, agent_id: str, metadata: Dict[str, Any]) -> str:
        """Create a new trace for an agent execution."""
        if not self.enabled:
            return str(uuid.uuid4())

        trace = self.langfuse.trace(
            name=f"agent:{agent_id}", id=str(uuid.uuid4()), metadata=metadata
        )
        trace_id: str = trace.id
        self.active_traces[trace_id] = trace
        return trace_id

    def trace_workflow(self, workflow_id: str, metadata: Dict[str, Any]) -> str:
        """Create a new trace for a workflow execution."""
        if not self.enabled:
            return str(uuid.uuid4())

        trace = self.langfuse.trace(
            name=f"workflow:{workflow_id}", id=str(uuid.uuid4()), metadata=metadata
        )
        trace_id: str = trace.id
        self.active_traces[trace_id] = trace
        return trace_id

    def trace_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new span within a trace."""
        if not self.enabled:
            return str(uuid.uuid4())

        metadata = metadata or {}

        try:
            if parent_id and parent_id in self.active_traces:
                parent = self.active_traces[parent_id]
                span = parent.span(name=name, metadata=metadata)
            elif parent_id and parent_id in self.active_spans:
                parent = self.active_spans[parent_id]
                span = parent.span(name=name, metadata=metadata)
            else:
                # Create a new trace if no parent
                trace = self.langfuse.trace(name=f"span:{name}", metadata=metadata)
                span = trace.span(name=name, metadata=metadata)

            span_id: str = span.id
            self.active_spans[span_id] = span
            return span_id
        except Exception as e:
            logger.error(f"Error creating Langfuse span: {e}")
            return str(uuid.uuid4())

    def end_span(self, span_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """End a span and record its metadata."""
        if not self.enabled or span_id not in self.active_spans:
            return

        try:
            span = self.active_spans[span_id]
            if metadata:
                for key, value in metadata.items():
                    try:
                        span.add_metadata(key, value)
                    except Exception as e:
                        logger.warning(f"Error ending Langfuse span: {e}")
                        # Skip metadata that can't be serialized
                        pass
            span.end()
            del self.active_spans[span_id]
        except Exception as e:
            logger.error(f"Error ending Langfuse span: {e}")

    def log_llm_call(
        self,
        model: str,
        prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        response: Any,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> None:
        """Log an LLM call to Langfuse."""
        if not self.enabled:
            return

        try:
            # Format prompt and response based on type
            formatted_prompt = prompt
            formatted_response = response

            # Handle different prompt formats
            if isinstance(prompt, str):
                formatted_prompt = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, dict) and "system" in prompt:
                formatted_prompt = [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt.get("user", "")},
                ]

            # Format the generation call
            generation_kwargs = {
                "name": f"llm:{model}",
                "model": model,
                "input": formatted_prompt,
                "output": formatted_response,
                "metadata": metadata or {},
            }

            # Log the generation
            if parent_id and parent_id in self.active_traces:
                parent = self.active_traces[parent_id]
                parent.generation(**generation_kwargs)
            elif parent_id and parent_id in self.active_spans:
                parent = self.active_spans[parent_id]
                parent.generation(**generation_kwargs)
            else:
                # Create a standalone generation
                self.langfuse.generation(**generation_kwargs)

        except Exception as e:
            logger.error(f"Error logging LLM call to Langfuse: {e}")

    def capture_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture a score for a trace."""
        if not self.enabled:
            return

        try:
            if trace_id in self.active_traces:
                trace = self.active_traces[trace_id]
                trace.score(name=name, value=value, metadata=metadata)
            else:
                # Try to find the trace
                self.langfuse.score(
                    trace_id=trace_id, name=name, value=value, metadata=metadata
                )
        except Exception as e:
            logger.error(f"Error capturing Langfuse score: {e}")


class NoopObservabilityProvider(ObservabilityProvider):
    """No-op implementation of the observability provider."""

    def __init__(self) -> None:
        """Initialize the no-op provider."""
        self.enabled = False

    def trace_agent(self, agent_id: str, metadata: Dict[str, Any]) -> str:
        return str(uuid.uuid4())

    def trace_workflow(self, workflow_id: str, metadata: Dict[str, Any]) -> str:
        return str(uuid.uuid4())

    def trace_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return str(uuid.uuid4())

    def end_span(self, span_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass

    def log_llm_call(
        self,
        model: str,
        prompt: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        response: Any,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> None:
        pass

    def capture_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass


class ObservabilityProviderFactory:
    """Factory for creating observability providers."""

    _providers = {"langfuse": LangfuseProvider, "noop": NoopObservabilityProvider}

    @classmethod
    def create_provider(
        cls, provider_type: str = "noop", **kwargs: Any
    ) -> ObservabilityProvider:
        """Create a provider instance based on type."""
        if provider_type not in cls._providers:
            logger.warning(
                f"Unknown observability provider type: {provider_type}, using noop"
            )
            return NoopObservabilityProvider()

        try:
            provider_class = cls._providers[provider_type]
            return provider_class(**kwargs)  # type: ignore
        except Exception as e:
            logger.error(f"Error creating observability provider: {e}")
            return NoopObservabilityProvider()


# Default global instance
default_provider: ObservabilityProvider = NoopObservabilityProvider()


def configure_observability(
    provider_type: str = "noop", **kwargs: Any
) -> ObservabilityProvider:
    """Configure the default observability provider."""
    global default_provider
    default_provider = ObservabilityProviderFactory.create_provider(
        provider_type, **kwargs
    )
    return default_provider


def get_observability_provider() -> ObservabilityProvider:
    """Get the configured observability provider."""
    return default_provider
