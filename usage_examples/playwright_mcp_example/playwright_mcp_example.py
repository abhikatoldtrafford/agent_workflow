"""
Example of using Playwright MCP server with an agent workflow.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from agent_workflow import WorkflowSourceFile

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from agent_workflow.providers.callbacks import ConsoleProgressCallback
from agent_workflow.providers.mcp import register_mcp_server
from agent_workflow.workflow_engine import (
    MCPServerSpec,
    MCPServerType,
    OpenAIProviderConfig,
    ProviderConfiguration,
    ProviderType,
    WorkflowInput,
    GeminiProviderConfig,
)
from agent_workflow.workflow_engine import WorkflowManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_playwright_mcp_workflow(user_task: str, website_url: str):
    """Run a workflow that uses Playwright MCP server for web automation.
    
    Args:
        user_task: Task description for web automation
        website_url: URL of the website to automate
    """

    # Path to the workflow YAML file
    workflow_file = os.path.join(
        os.path.dirname(__file__), 
        "playwright_mcp_workflow_example.yaml"
    )

    # Define available provider configurations
    provider_configs: Dict[str, Any] = {}

    # Add OpenAI provider if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        provider_configs["openai_gpt4"] = OpenAIProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="o4-mini-2025-04-16",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    else:
        raise ValueError("OPENAI_API_KEY environment variable is required")

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

    # Create the provider configuration
    providers_config = ProviderConfiguration(providers=provider_configs)

    # Register Playwright MCP server
    playwright_mcp_server = MCPServerSpec(
        params={
            "command": "npx",
            "args": ["@playwright/mcp@latest"]
        },
        server_type=MCPServerType.STDIO,
        name="playwright-mcp-server",
        cache_tools_list=True,
        client_session_timeout=120,  # Longer timeout for web automation
    )

    register_mcp_server(playwright_mcp_server)

    # Create a workflow manager
    manager = WorkflowManager(engine_type="openai", provider_config=providers_config)

    # Provider mapping to ensure agents use our OpenAI provider
    provider_mapping = {"playwright-mcp-agent": "google_gemini_with_tool"}

    # Create structured workflow inputs
    workflow_inputs = WorkflowInput(
        user_query=f"Automate task on {website_url}: {user_task}", 
        workflow={
            "inputs": {
                "user_task": user_task,
                "website_url": website_url
            }
        }
    )

    logger.info(f"Starting web automation workflow for {website_url}")
    logger.info(f"Task: {user_task}")

    # First initialize the workflow
    workflow = await manager.initialize_workflow(
        WorkflowSourceFile(workflow_file),
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
    # Example task and website
    EXAMPLE_TASK = ("""Test the given website, 
                    produce a complete test summary at the end. Use Playright MCP to automate the task.
                    Tets all the links, headers, footers, menus and images and all other web components.
                    Specifically check all the navigation links.
                    Take and save screenshot from website and save. 
                    Keep max turn for tool call to 100.
                    
                    """)
    EXAMPLE_URL = "https://newscorp.com/"
    
    # Get task and URL from command line args if provided
    if len(sys.argv) > 2:
        EXAMPLE_URL = sys.argv[1]
        EXAMPLE_TASK = sys.argv[2]
    
    try:
        asyncio.run(run_playwright_mcp_workflow(EXAMPLE_TASK, EXAMPLE_URL))
    except ValueError as e:
        print(f"Error: {e}")
        if "OPENAI_API_KEY" in str(e):
            print("\nSetup instructions:")
            print("1. Set OPENAI_API_KEY environment variable")
    except Exception as e:
        print(f"Unexpected error: {e}")