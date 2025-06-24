"""
Example of using MCP servers with a workflow.
"""

import asyncio
import logging
import os

from agent_workflow.providers.callbacks import ConsoleProgressCallback
from agent_workflow.providers.mcp import register_mcp_server
from agent_workflow.workflow_engine import (
    GeminiProviderConfig,
    MCPServerSpec,
    MCPServerType,
    OpenAIProviderConfig,
    ProviderConfiguration,
    ProviderType,
    WorkflowInput,
    BaseProviderConfig
)
from agent_workflow.workflow_engine import WorkflowManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_mcp_workflow():
    """Run an example workflow that uses MCP servers."""

    # Path to the workflow YAML file
    workflow_file = os.path.join(os.path.dirname(__file__), "mcp_workflow_example.yaml")

    # Define available provider configurations
    provider_configs: dict[str, BaseProviderConfig] = {}

    # Add OpenAI providers if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        provider_configs["openai_gpt4"] = OpenAIProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="gpt-4o",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    if os.environ.get("GEMINI_API_KEY"):
        provider_configs["google_gemini_with_tool"] = GeminiProviderConfig(
            provider_type=ProviderType.GEMINI,
            model="gemini-2.5-pro-preview-03-25",
            access_token=os.environ.get("GEMINI_API_KEY"),
            project_id=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_LOCATION"),
            enforce_structured_output=False,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        provider_configs["google_gemini"] = GeminiProviderConfig(
            provider_type=ProviderType.GEMINI,
            model="gemini-2.5-pro-preview-03-25",
            access_token=os.environ.get("GEMINI_API_KEY"),
            project_id=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_LOCATION"),
            enforce_structured_output=True,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    # Create the provider configuration container
    if provider_configs:
        providers_config = ProviderConfiguration(providers=provider_configs)
    else:
        # Fallback to empty dictionary for backward compatibility
        providers_config = {}

    # create MCP server specifications
    mcp_server_spec = MCPServerSpec(
        params={"command": "uvx", "args": ["mcp-server-git"]},
        server_type=MCPServerType.STDIO,
        name="git-mcp-server",
        cache_tools_list=True,
        client_session_timeout=60,
    )

    register_mcp_server(mcp_server_spec)

    # Create a workflow manager
    manager = WorkflowManager(engine_type="openai", provider_config=providers_config)

    # Provider mapping to ensure agents use our MCP-capable provider
    provider_mapping = {"mcp-agent": "google_gemini_with_tool"}

    # User query that would benefit from using MCP server
    user_query = "show the current status for repo ./../.. and then show result of `git show HEAD`, Finally switch to main branch."

    # Create structured workflow inputs
    workflow_inputs = WorkflowInput(
        user_query=user_query, workflow={"inputs": {"user_query": user_query}}
    )

    logger.info(f"Starting workflow execution with query: '{user_query}'")

    # First initialize the workflow
    workflow = await manager.initialize_workflow(
        workflow_file,
        provider_mapping=provider_mapping,
        progress_callback=ConsoleProgressCallback(),
    )
    
    # Then execute the initialized workflow
    result = await manager.execute(
        workflow,
        workflow_inputs,
        progress_callback=ConsoleProgressCallback(),
    )

    # Get the final result
    final_response = result.final_result

    # Print the final response
    logger.info("\n=== Workflow execution completed! ===")
    logger.info("\nFinal response:")
    logger.info(final_response)

    return result


if __name__ == "__main__":
    try:
        asyncio.run(run_mcp_workflow())
    except ValueError as e:
        print(f"Error: {e}")
        if "No LLM providers are configured" in str(e):
            print("\nSetup instructions:")
            print("1. Set OPENAI_API_KEY environment variable")
    except Exception as e:
        print(f"Unexpected error: {e}")
