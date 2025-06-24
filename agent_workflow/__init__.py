"""
Agent Workflow: A flexible framework for orchestrating multi-agent LLM workflows.

This package provides tools for building and executing workflows with different
LLM agents and execution engines.
"""

# All imports use absolute imports with internal package paths
from agent_workflow.expressions import ExpressionEvaluator, create_evaluation_context
from agent_workflow.parsers import YAMLParser
from agent_workflow.providers.tools import Tool, ToolRegistry, register_tool, tool
from agent_workflow.providers.observability import (
    ObservabilityProvider,
    configure_observability,
    get_observability_provider,
)
from agent_workflow.providers.callbacks import ConsoleProgressCallback, ProgressCallback
from agent_workflow.workflow_engine import (
    ExecutionEngine,
    ExecutionEngineFactory,
    AgentOutput,
    ExecutionResult,
    LLMAgent,
    Workflow,
    WorkflowStage,
    WorkflowTask,
    WorkflowManager,
)
from agent_workflow.workflow_engine.models import (
    WorkflowInput,
    WorkflowSource,
    WorkflowSourceDict,
    WorkflowSourceFile,
    WorkflowSourceYAML,
)

__version__ = "0.1.0"

__all__ = [
    # Tools and registry
    "Tool", 
    "ToolRegistry", 
    "register_tool", 
    "tool",
    
    # Observability
    "ObservabilityProvider", 
    "configure_observability",
    "get_observability_provider",
    
    # Progress callbacks
    "ConsoleProgressCallback", 
    "ProgressCallback",
    
    # Expression evaluation
    "ExpressionEvaluator", 
    "create_evaluation_context",
    
    # Parsers
    "YAMLParser",
    
    # Workflow models
    "AgentOutput",
    "ExecutionResult", 
    "Workflow", 
    "WorkflowStage", 
    "WorkflowTask", 
    "LLMAgent",
    "WorkflowInput",
    "WorkflowSource",
    "WorkflowSourceDict",
    "WorkflowSourceFile",
    "WorkflowSourceYAML",
    
    # Engines
    "ExecutionEngine", 
    "ExecutionEngineFactory",
    
    # Manager
    "WorkflowManager",
]