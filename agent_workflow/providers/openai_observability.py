from agent_workflow.providers import BaseLLMObservabilityProvider, TraceMetadata, SpanMetadata
from typing import Optional, Dict, Any, List, Union
import agents
import uuid

class OpenaiLLMObservabilityProvider(BaseLLMObservabilityProvider):

    @staticmethod
    def enable_tracing() -> None:
        agents.tracing.set_tracing_disabled(False)

    @staticmethod
    def disable_tracing() -> None:
        agents.tracing.set_tracing_disabled(True)

    def __init__(self) -> None:
       #  enable tracing
        OpenaiLLMObservabilityProvider.enable_tracing()
        self.curr_trace: Optional[agents.tracing.Trace] = None


    async def start_trace_group(
            self,
            metadata: TraceMetadata
    ) -> str:
        """Begin a named group of traces (e.g. all runs in a batch). Return group ID"""
        self.curr_trace = agents.tracing.trace(workflow_name=metadata['workflow_id'])
        self.curr_trace.start(mark_as_current=True)
        return self.curr_trace.trace_id

    async def end_trace_group(self, group_id: str, metadata: TraceMetadata) -> None:
        self.curr_trace.finish()

    # -- Traces --------------------------------------
    async def start_trace(
            self,
            name: str,
            prompt: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]],
            metadata: TraceMetadata,
            group_id: Optional[str],
    ) -> str:
        """Start a new trace; optionally assign it to a group."""
        return str(uuid.uuid4())

    # -- Spans ---------------------------------------
    async def start_span(
            self,
            name: str,
            parent_span_id: Optional[str],
            metadata: SpanMetadata
    ) -> str:
        """Start a new span in a trace; returns the span_id."""
        return agents.tracing.gen_span_id()