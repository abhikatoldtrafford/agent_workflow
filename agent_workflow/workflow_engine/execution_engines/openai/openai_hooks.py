import logging
from datetime import datetime

from agents import RunHooks, RunContextWrapper, Agent, Tool

from agent_workflow.providers.llm_tracing_utils import LLMTracer
from agent_workflow.workflow_engine.models import AgentConfig, Tool as AgentWorkflowTool
from agent_workflow.providers.llm_observability import TraceMetadata, ToolMetadata, RunContextMetadata, RequestData

logger = logging.getLogger("workflow-engine.execution_engine.openai_hooks")
# You can configure handlers/formatters here as needed.


class OpenAIHooks(RunHooks):
    """
    A concrete subclass of RunHooks that logs each lifecycle event.
    Override these methods to add custom behavior (e.g., metrics, tracing, context mutation, etc.).
    """

    def __init__(self, provider: LLMTracer) -> None:
        self.provider = provider

    # Mapping methods for OpenAI Agent SDK objects to agent_workflow classes
    @staticmethod
    def map_agent_to_agentconfig(agent: Agent) -> AgentConfig:
        """

        Maps an OpenAI Agent SDK Agent to AgentConfig.

        Args:
            agent: The OpenAI Agent SDK Agent object

        Returns:
            AgentConfig object with agent information
        """
        try:
            # Extract fields that exist in AgentConfig
            agent_id = getattr(agent, "id", None) or agent.name
            agent_name = agent.name
            agent_description = getattr(agent, "instructions", None) or ""
            agent_version = getattr(agent, "version", None) or "1.0"

            # Create a base AgentConfig object
            agent_config = AgentConfig(
                id=agent_id,
                name=agent_name,
                description=agent_description,
                version=agent_version,
                agent_type="LLMAgent"
            )

            # Map tools if available
            if hasattr(agent, "tools") and agent.tools:
                tools = []
                for tool in agent.tools:
                    tools.append(OpenAIHooks.map_tool_to_agentworkflowtool(tool))
                agent_config.tools = tools

            # Map mcp_servers if present
            if hasattr(agent, "mcp_servers"):
                agent_config.mcp_servers = [mcp.name for mcp in agent.mcp_servers]

            # Map model settings if present
            if hasattr(agent, "model_settings"):
                # For model settings, we'd likely need to create a proper object if needed
                # This depends on how model_settings is structured in AgentConfig
                pass

            return agent_config
        except Exception as e:
            logger.error(f"Error mapping Agent to AgentConfig: {e}")
            return AgentConfig(id="unknown", name="unknown")

    @staticmethod
    def map_tool_to_agentworkflowtool(tool: Tool) -> AgentWorkflowTool:
        """
        Maps an OpenAI Agent SDK Tool to agent_workflow Tool object.

        Args:
            tool: The OpenAI Agent SDK Tool object

        Returns:
            AgentWorkflowTool object with tool information
        """
        try:
            # Get basic tool properties
            tool_name = tool.name
            tool_description = getattr(tool, "description", "")
            tool_type = "function"  # Default type

            # Determine more specific tool type if available
            if hasattr(tool, "type"):
                tool_type = tool.type
            elif hasattr(tool, "__class__") and "FunctionTool" in tool.__class__.__name__:
                tool_type = "function"
            elif hasattr(tool, "__class__") and "OpenAITool" in tool.__class__.__name__:
                tool_type = "openai"

            # Create the Tool object
            workflow_tool = AgentWorkflowTool(
                name=tool_name,
                description=tool_description,
                type=tool_type
            )

            # Map parameters schema if available
            if hasattr(tool, "params_json_schema"):
                workflow_tool.parameters = tool.params_json_schema

            return workflow_tool
        except Exception as e:
            logger.error(f"Error mapping Tool to agent_workflow Tool: {e}")
            return AgentWorkflowTool(name="unknown", description="", type="function")

    @staticmethod
    def map_agent_to_metadata(agent: Agent) -> TraceMetadata:
        """
        Maps an OpenAI Agent SDK Agent to TraceMetadata for LLMObservabilityProvider.

        Args:
            agent: The OpenAI Agent SDK Agent object

        Returns:
            TraceMetadata with agent information
        """
        try:
            agent_id = getattr(agent, "id", None) or agent.name

            metadata = TraceMetadata(
                agent_id=agent_id,
                timestamp=datetime.now(),
                description=getattr(agent, "name", "unknown agent"),
                tags={
                    "version": getattr(agent, "version", "unknown"),
                    "model": str(getattr(agent, "model", "unknown"))
                }
            )
            return metadata
        except Exception as e:
            logger.error(f"Error mapping Agent to TraceMetadata: {e}")
            return TraceMetadata(agent_id="unknown", timestamp=datetime.now())

    @staticmethod
    def map_tool_to_metadata(tool: Tool) -> ToolMetadata:
        """
        Maps an OpenAI Agent SDK Tool to ToolMetadata for LLMObservabilityProvider.

        Args:
            tool: The OpenAI Agent SDK Tool object

        Returns:
            ToolMetadata with tool information
        """
        try:
            tool_type = "function"

            # Determine specific tool type
            if hasattr(tool, "type"):
                tool_type = tool.type
            elif hasattr(tool, "__class__") and "FunctionTool" in tool.__class__.__name__:
                tool_type = "function"
            elif hasattr(tool, "__class__") and "OpenAITool" in tool.__class__.__name__:
                tool_type = "openai"

            metadata = ToolMetadata(
                tool_type=tool_type,
                timestamp=datetime.now(),
                tags={
                    "name": tool.name,
                    "description": getattr(tool, "description", "")
                }
            )
            return metadata
        except Exception as e:
            logger.error(f"Error mapping Tool to ToolMetadata: {e}")
            return ToolMetadata(tool_type="unknown", timestamp=datetime.now())

    @staticmethod
    def map_runcontext_to_metadata(context: RunContextWrapper) -> RunContextMetadata:
        """
        Maps a RunContextWrapper to metadata for tracing.

        Args:
            context: The RunContextWrapper from OpenAI Agent SDK

        Returns:
            RunContextMetadata with context information for tracing
        """
        try:
            metadata = RunContextMetadata(timestamp=datetime.now())

            # Extract usage statistics if available
            if hasattr(context, "usage"):
                usage = context.usage
                
                # Create RequestData object
                request_data = RequestData()
                
                # Build request_data with proper typing
                if hasattr(usage, "input_tokens"):
                    request_data["input_tokens"] = int(usage.input_tokens)
                if hasattr(usage, "output_tokens"):
                    request_data["output_tokens"] = int(usage.output_tokens)
                if hasattr(usage, "total_tokens"):
                    request_data["total_tokens"] = int(usage.total_tokens)
                if hasattr(usage, "latency_ms"):
                    request_data["latency_ms"] = int(usage.latency_ms)
                    
                # Set request_data in metadata
                if request_data:
                    metadata["request_data"] = request_data
                
                # Set requests count
                if hasattr(usage, "requests"):
                    metadata["requests"] = int(usage.requests)

            return metadata
        except Exception as e:
            logger.error(f"Error mapping RunContextWrapper to metadata: {e}")
            return RunContextMetadata(timestamp=datetime.now())

    async def on_agent_start(
            self,
            context: RunContextWrapper,  # The mutable context for the current run
            agent: Agent,                # The agent that is about to begin
    ) -> None:
        """
        Called before the agent is invoked. This happens each time control switches
        to a new agent (including the first agent in a workflow). Override this to
        inspect or modify context, record metrics, etc.
        """
        logger.info(f"[on_agent_start] Starting agent: {agent.name}")

        # Get agent config from agent data
        agent_config = self.map_agent_to_agentconfig(agent)

        name = agent.name
        sys_prompt = agent.instructions
        context_metadata = self.map_runcontext_to_metadata(context)

        # Call the provider with the agent name and config
        await self.provider.on_agent_start(name,
                                     system_prompt=sys_prompt,
                                     context_metadata=context_metadata,
                                     agent_config=agent_config)

    async def on_agent_end(
            self,
            context: RunContextWrapper,
            agent: Agent,
            output: object,
    ) -> None:
        """
        Called when the given agent produces its final output (either structured or plain text).
        You can use `output` here to log results, validate them, or push metrics.
        """
        logger.info(f"[on_agent_end] Agent {agent.name} produced output: {output}")

        # Get agent config from agent data
        agent_config = self.map_agent_to_agentconfig(agent)
        name = agent.name
        sys_prompt = agent.instructions
        context_metadata = self.map_runcontext_to_metadata(context)

        # Call the provider with the agent name, output, and config
        await self.provider.on_agent_end(name= name,
                                   system_prompt=sys_prompt,
                                   context_metadata=context_metadata,
                                   output=output,
                                   agent_config=agent_config)

    async def on_handoff(
            self,
            context: RunContextWrapper,
            from_agent: Agent,
            to_agent: Agent,
    ) -> None:
        """
        Called when control is handed off from one agent to another.
        For example, if an agent decides to invoke a sub-agent/tool, this runs before that handoff.
        """
        logger.info(f"[on_handoff] Handoff from {from_agent.name} -> {to_agent.name}")
        # Get agent configs from both agents
        from_agent_config = self.map_agent_to_agentconfig(from_agent)
        to_agent_config = self.map_agent_to_agentconfig(to_agent)

        # Call the provider with the agent names and configs
        await self.provider.on_handoff(from_agent.name, to_agent.name,
                               from_agent_config=from_agent_config,
                               to_agent_config=to_agent_config)

    async def on_tool_start(
            self,
            context: RunContextWrapper,
            agent: Agent,
            tool: Tool,
    ) -> None:
        """
        Called immediately before the specified tool is invoked by the agent. Use this to
        annotate the context or record that a tool call is about to begin.
        """
        logger.info(f"[on_tool_start] Agent {agent.name} is invoking tool: {tool.name}")
        # Map the tool to our Tool type
        workflow_tool = self.map_tool_to_agentworkflowtool(tool)
        agent_config = self.map_agent_to_agentconfig(agent)
        name = agent.name
        context_metadata = self.map_runcontext_to_metadata(context)

        # Call the provider with the agent name, tool name and tool object
        await self.provider.on_tool_start(agent_name=name,
                                    tool_name=tool.name,
                                    context_metadata=context_metadata,
                                    agent_config=agent_config,
                                    tool=workflow_tool)

    async def on_tool_end(
            self,
            context: RunContextWrapper,
            agent: Agent,
            tool: Tool,
            result: str,
    ) -> None:
        """
        Called immediately after the specified tool has returned. `result` is the raw string
        returned by the tool. Override this to log tool outputs or insert the tool’s result
        back into context for downstream steps.
        """
        logger.info(f"[on_tool_end] Tool {tool.name} returned result: {result}")
        # Map the tool to our Tool type
        workflow_tool = self.map_tool_to_agentworkflowtool(tool)
        agent_config = self.map_agent_to_agentconfig(agent)
        name = agent.name
        context_metadata = self.map_runcontext_to_metadata(context)

        # Call the provider with the agent name, tool name, result and tool object
        await self.provider.on_tool_end(agent_name=name,
                                  tool_name=tool.name,
                                  context_metadata=context_metadata,
                                  agent_config=agent_config,
                                  tool=workflow_tool,
                                  result= result
                                  )