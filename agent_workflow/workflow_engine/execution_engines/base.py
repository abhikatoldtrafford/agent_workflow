"""
Execution Engine abstraction layer to support multiple agent frameworks.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

# Import ProgressCallback from providers.callbacks to avoid circular imports
from agent_workflow.providers.callbacks import ProgressCallback

from agent_workflow.workflow_engine.models import (
    AgentOutput,
    ExecutionResult,
    Workflow,
    WorkflowInput,
    WorkflowStage,
    WorkflowTask,
    TaskExecutionResult,
    StageExecutionResult
)

logger = logging.getLogger("workflow-engine.execution_engine")


# ProgressCallback is imported from providers.callbacks


class ExecutionEngine(ABC):
    """Abstract base class for workflow execution engines."""
    
    @abstractmethod
    async def initialize_workflow(
        self,
        workflow: Workflow,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Workflow:
        """
        Initialize a workflow by creating agents for each task and storing them.
        
        Args:
            workflow: The workflow model to initialize
            progress_callback: Optional callback for reporting execution progress
            
        Returns:
            The initialized workflow with agents created for each task
        """
        pass

    @abstractmethod
    async def execute_workflow(
        self,
        workflow: Workflow,
        inputs: WorkflowInput,
        progress_callback: Optional[ProgressCallback] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute a workflow with the provided workflow model and inputs.

        Args:
            workflow: The workflow model
            inputs: The workflow inputs as WorkflowInput or dict
            progress_callback: Optional callback for reporting execution progress
            kwargs: Additional arguments for the execution engine

        Returns:
            The workflow execution results as an ExecutionResult
        """
        pass

    @abstractmethod
    async def execute_task(
        self,
        task: WorkflowTask,
        inputs: WorkflowInput,
        progress_callback: Optional[ProgressCallback] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> TaskExecutionResult:
        """
        Execute a single task with the provided configuration and inputs.

        Args:
            task: The workflow task
            inputs: The workflow inputs
            progress_callback: Optional callback for reporting execution progress
            kwargs: Additional arguments for the execution engine

        Returns:
            The task execution results
        """
        pass

    @abstractmethod
    async def execute_stage(
        self,
        stage: WorkflowStage,
        inputs: WorkflowInput,
        progress_callback: Optional[ProgressCallback] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> StageExecutionResult:
        """
        Execute a workflow stage with the provided configuration and inputs.

        Args:
            stage: The workflow stage
            inputs: The workflow inputs
            progress_callback: Optional callback for reporting execution progress
            kwargs: Additional arguments for the execution engine

        Returns:
            The stage execution results
        """
        pass


class ExecutionEngineFactory:
    """Factory for creating execution engines."""

    _engines: Dict[str, Type[ExecutionEngine]] = {}

    @classmethod
    def register_engine(
        cls, engine_type: str, engine_class: Type[ExecutionEngine]
    ) -> None:
        """Register a new execution engine type."""
        cls._engines[engine_type] = engine_class

    @classmethod
    def create_engine(cls, engine_type: str, **kwargs: Any) -> ExecutionEngine:
        """Create an execution engine instance based on type."""
        if engine_type not in cls._engines:
            raise ValueError(f"Unknown execution engine type: {engine_type}")

        return cls._engines[engine_type](**kwargs)


# Legacy function maintained for backward compatibility with older execution engines
def ensure_execution_result(
    result: Union[ExecutionResult, Dict[str, Any]],
) -> ExecutionResult:
    """
    Convert a dictionary result to an ExecutionResult if needed.
    This ensures backward compatibility with engines that return dictionaries.

    Args:
        result: The result from an execution engine

    Returns:
        An ExecutionResult object
    """
    # With our structured implementation, this should always be an ExecutionResult
    if isinstance(result, ExecutionResult):
        return result

    # For backward compatibility with legacy engines that might return dictionaries
    logger.warning(
        "Converting dictionary result to ExecutionResult - this is deprecated behavior"
    )

    agent_outputs = []
    all_agents = []

    # Parse agent outputs from the dictionary
    if "agent_outputs" in result:
        for output in result["agent_outputs"]:
            agent = output.get("agent", "unknown")
            all_agents.append(agent)
            agent_outputs.append(
                AgentOutput(
                    agent=agent,
                    output=output.get("output", ""),
                    metadata=output.get("metadata"),
                )
            )

    # Get final result and all agents
    final_result = result.get("final_result", "")
    if "all_agents" in result and isinstance(result["all_agents"], list):
        all_agents = result["all_agents"]

    # Create metadata from remaining fields
    metadata = {
        k: v
        for k, v in result.items()
        if k not in ["agent_outputs", "final_result", "all_agents"]
    }

    return ExecutionResult(
        agent_outputs=agent_outputs,
        final_result=final_result,
        all_agents=all_agents,
        metadata=metadata if metadata else None,
    )
