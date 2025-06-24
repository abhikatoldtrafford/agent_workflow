"""
Enhanced WorkflowManager that supports multiple execution engines.
"""

import logging
from typing import Any, Dict, Optional, overload, List, Union

from agent_workflow.parsers import ConfigParser, YAMLParser
from agent_workflow.providers.llm_observability import LLMObservabilityProvider, NoOpLLMObservabilityProvider
from agent_workflow.providers.llm_tracing_utils import LLMTracer
from agent_workflow.providers.mcp import MCPServerRegistry
from agent_workflow.providers import LLMProviderFactory, OpenaiLLMObservabilityProvider
from agent_workflow.providers import ToolRegistry
from agent_workflow.workflow_engine.execution_engines.base import (
    ExecutionEngineFactory,
    ProgressCallback,
)
from agent_workflow.workflow_engine.models import (
    ExecutionResult,
    ProviderConfiguration,
    Workflow,
    WorkflowInput,
    WorkflowSource,
    WorkflowSourceDict,
    WorkflowSourceFile,
    WorkflowSourceYAML
)
import os

logger = logging.getLogger("workflow-engine.workflow_manager")


class WorkflowManager:
    """Manager for loading and executing workflows with different execution engines."""

    def __init__(
        self,
        provider_config: ProviderConfiguration,
        config_parser: Optional[ConfigParser] = None,
        provider_factory: Optional[LLMProviderFactory] = None,
        engine_type: str = "openai",
        engine_options: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        mcp_server_registry: Optional[MCPServerRegistry] = None,
        llm_observability_provider: Optional[Union[LLMObservabilityProvider, List[LLMObservabilityProvider]]] = None,
    ) -> None:
        """
        Initialize the workflow manager.

        Args:
            config_parser: Optional parser for loading configurations
            provider_factory: Optional factory for LLM providers
            engine_type: Type of execution engine to use ("openai")
            engine_options: Optional configuration for the execution engine
            tool_registry: Optional tool registry for function calling
            mcp_server_registry: Optional MCP server registry
            provider_config: Optional configuration for LLM providers
        """
        self.config_parser = config_parser or YAMLParser()
        self.provider_factory = provider_factory or LLMProviderFactory()
        self.engine_type = engine_type
        self.engine_options = engine_options or {}
        self.provider_config = provider_config

        # Set up the tool registry
        if tool_registry is None:
            # Import the global registry if none is provided
            from agent_workflow.providers import global_tool_registry as global_registry

            self.tool_registry = global_registry
        else:
            self.tool_registry = tool_registry

        self.mcp_server_registry: Optional[MCPServerRegistry] = None
        # Set up the MCP server registry
        if mcp_server_registry is None:
            # Import the global registry if none is provided
            try:
                from agent_workflow.providers.mcp import mcp_registry as global_mcp_registry

                self.mcp_server_registry = global_mcp_registry
            except ImportError:
                logger.warning(
                    "MCP server registry not available. MCP server support will be disabled."
                )
        else:
            self.mcp_server_registry = mcp_server_registry

        # handle case of single provider or None
        llm_observability_provider_list: List[LLMObservabilityProvider] = (
            [NoOpLLMObservabilityProvider()]
            if llm_observability_provider is None
            else [llm_observability_provider]
            if not isinstance(llm_observability_provider, list)
            else llm_observability_provider
        )

        # handle the case if none of the observability providers are OpenAI provider.
        # We will diable OpenAI tracing in that case
        if not any(isinstance(obj, OpenaiLLMObservabilityProvider) for obj in llm_observability_provider_list):
            OpenaiLLMObservabilityProvider.disable_tracing()
        else:
            OpenaiLLMObservabilityProvider.enable_tracing()

        # Create the execution engine
        options = self.engine_options.copy()

        if self.tool_registry:
            options["tool_registry"] = self.tool_registry
        if self.mcp_server_registry:
            options["mcp_registry"] = self.mcp_server_registry

        options["llm_tracer"] = LLMTracer(providers=llm_observability_provider_list)

        self.execution_engine = ExecutionEngineFactory.create_engine(
            engine_type, **options
        )

    @overload
    def load_workflow(self, workflow_source: WorkflowSourceDict, provider_mapping: Optional[Dict[str, str]] = None) -> Workflow: ...
    
    @overload
    def load_workflow(self, workflow_source: WorkflowSourceFile, provider_mapping: Optional[Dict[str, str]] = None) -> Workflow: ...

    @overload
    def load_workflow(self, workflow_source: WorkflowSourceYAML, provider_mapping: Optional[Dict[str, str]] = None) -> Workflow: ...

    def load_workflow(
        self,
        workflow_source: WorkflowSource,
        provider_mapping: Optional[Dict[str, str]] = None,
    ) -> Workflow:
        """
        Load a workflow from:
          - a Python dict (WorkflowSourceDict),
          - a file path (WorkflowSourceFile), or
          - a raw YAML/JSON string (WorkflowSourceYAML).

        If `provider_mapping` is provided, it will be applied afterward.
        """
        # 1) Dict input
        if isinstance(workflow_source, dict):
            workflow = self.config_parser.parse_workflow(workflow_source)

        # 2) File path input
        elif isinstance(workflow_source, (str, os.PathLike)) and os.path.exists(workflow_source):
            workflow = self.config_parser.parse_workflow_file(workflow_source)

        # 3) String content input
        elif isinstance(workflow_source, str):
            workflow = self.config_parser.parse_workflow_str(workflow_source)

        else:
            raise TypeError(
                f"Unsupported workflow_source type: {type(workflow_source).__name__}"
            )

        # Apply any provider_mapping if given
        if provider_mapping:
            workflow = self._apply_provider_mapping(workflow, provider_mapping)

        return workflow


    def _apply_provider_mapping(
        self, workflow: Workflow, provider_mapping: Optional[Dict[str, str]] = None
    ) -> Workflow:
        """
        Apply provider mapping to the workflow tasks.

        Args:
            workflow: The workflow to apply mapping to
            provider_mapping: Optional mapping of agent IDs to provider types

        Returns:
            A copy of the workflow with provider mapping applied
        """
        if not provider_mapping:
            return workflow

        # Create a copy of the workflow to modify
        workflow_copy = workflow.copy(deep=True)

        for stages in workflow_copy.stages:
            for task in stages.tasks:
                if task.agent.id in provider_mapping:
                    provider_id = provider_mapping[task.agent.id]
                    # find the provider in provider configuration
                    if provider_id in self.provider_config.providers:
                        # Get the provider type from the mapping
                        task.provider = self.provider_config.providers[provider_id]
                    else:
                        logger.warning(f"Provider ID '{provider_id}' not found in provider mapping. Using default provider.")
                        raise (
                            ValueError(f"Provider ID '{provider_id}' not found in provider mapping.")
                        )

        return workflow_copy

    @overload
    async def initialize_workflow(
        self,
        workflow_source: WorkflowSourceDict,
        provider_mapping: Optional[Dict[str, str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Workflow: ...
    
    @overload
    async def initialize_workflow(
        self,
        workflow_source: WorkflowSourceFile,
        provider_mapping: Optional[Dict[str, str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Workflow: ...
    
    @overload
    async def initialize_workflow(
        self,
        workflow_source: WorkflowSourceYAML,
        provider_mapping: Optional[Dict[str, str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Workflow: ...
    
    async def initialize_workflow(
        self,
        workflow_source: WorkflowSource,
        provider_mapping: Optional[Dict[str, str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Workflow:
        """
        Initialize a workflow without executing it. This creates and initializes agents
        for each task in the workflow.

        Args:
            workflow_source: Path to the workflow YAML file, YAML string content, or dict
            provider_mapping: Optional runtime mapping of agent IDs to provider types
                            (overrides mapping provided during initialization)
            progress_callback: Optional callback for reporting execution progress

        Returns:
            The initialized workflow with agents set up and ready for execution
        """
        # Parse the workflow source into a structured Workflow object with resolved references
        workflow = self.load_workflow(workflow_source, provider_mapping)

        # Initialize the workflow using the execution engine
        initialized_workflow = await self.execution_engine.initialize_workflow(
            workflow, progress_callback
        )
        
        return initialized_workflow

    async def execute(
        self,
        workflow: Workflow,
        inputs: WorkflowInput,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ExecutionResult:
        """
        Execute a previously initialized workflow.

        Args:
            workflow: The initialized Workflow object
            inputs: Optional workflow inputs (as WorkflowInput or Dict)
            progress_callback: Optional callback for reporting execution progress

        Returns:
            The workflow execution results as an ExecutionResult
        """
        # Create new WorkflowInput if none provided
        workflow_inputs = inputs
        
        # Add provider config to inputs
        workflow_inputs.provider_config = self.provider_config

        # Add registries to the context
        if not hasattr(workflow_inputs, "workflow"):
            workflow_inputs.workflow = {}

        # TODO: We need to change this to not be passed through dictionary, maybe move to config object
        workflow_inputs.workflow["tool_registry"] = self.tool_registry

        if self.mcp_server_registry:
            workflow_inputs.workflow["mcp_server_registry"] = self.mcp_server_registry

        # Execute the workflow using the selected engine with structured workflow model
        result = await self.execution_engine.execute_workflow(
            workflow, workflow_inputs, progress_callback
        )

        # Ensure we have a standardized result (already guaranteed by our structured approach)
        return result