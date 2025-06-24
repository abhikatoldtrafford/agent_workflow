"""
Execution Engine abstraction layer to support multiple agent frameworks.
This module re-exports execution engine classes from the execution_engines package.
"""

from agent_workflow.workflow_engine.execution_engines import (
    ExecutionEngine,
    ExecutionEngineFactory,
    OpenAIExecutionEngine,
)
from agent_workflow.workflow_engine.execution_engines.base import ensure_execution_result
from agent_workflow.workflow_engine.models import AgentOutput, ExecutionResult

__all__ = [
    "ExecutionEngine",
    "ExecutionEngineFactory",
    "OpenAIExecutionEngine",
    "ensure_execution_result",
    "ExecutionResult",
    "AgentOutput",
]
