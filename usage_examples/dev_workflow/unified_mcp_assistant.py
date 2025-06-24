#!/usr/bin/env python
"""
Unified MCP Development Assistant

This application integrates Git, GitHub, and Jira MCP servers for seamless development tasks.
It provides a command-line interface where users can interact with the different systems
through natural language queries.
"""

import asyncio
import logging
import os
import sys

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
)
from agent_workflow.workflow_engine import WorkflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("mcp-assistant")


class MCPAssistant:
    """Unified assistant that integrates Git, GitHub, and Jira MCP servers."""
    
    def __init__(self):
        """Initialize the MCP Assistant."""
        self.workflow_file = os.path.join(
            os.path.dirname(__file__), "unified_mcp_workflow.yaml"
        )
        self.provider_config = self._setup_provider_config()
        self._register_mcp_servers()
        self.workflow_manager = WorkflowManager(
            engine_type="openai", provider_config=self.provider_config
        )

    async def initialize_workflow(self):
        # Initialize the workflow - this sets up the agents
        # Custom callback for interactive console experience
        callback = ConsoleProgressCallback()

        self.workflow = await self.workflow_manager.initialize_workflow(
            self.workflow_file,
            provider_mapping=self.provider_mapping,
            progress_callback=callback,
        )

    def _setup_provider_config(self) -> ProviderConfiguration:
        """Set up provider configuration based on available API keys."""
        provider_configs = {}

        # Add OpenAI provider if API key is available
        if os.environ.get("OPENAI_API_KEY"):
            provider_configs["openai_gpt4"] = OpenAIProviderConfig(
                provider_type=ProviderType.OPENAI,
                model="gpt-4o",
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        else:
            logger.error("OPENAI_API_KEY environment variable is not set")
            print("Error: OPENAI_API_KEY environment variable must be set")
            sys.exit(1)

        # Add Gemini provider if API key is available
        if os.environ.get("GEMINI_API_KEY"):
            provider_configs["google_gemini"] = GeminiProviderConfig(
                provider_type=ProviderType.GEMINI,
                model="gemini-2.5-pro-preview-03-25",
                access_token=os.environ.get("GEMINI_API_KEY"),
                project_id=os.environ.get("GCP_PROJECT_ID", "default-project"),
                location=os.environ.get("GCP_LOCATION", "us-central1"),
                enforce_structured_output=False,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )

        # Provider mapping to ensure agents use our MCP-capable provider
        self.provider_mapping = {"dev-assistant": "google_gemini"}

        # Create the provider configuration container
        if provider_configs:
            return ProviderConfiguration(providers=provider_configs)
        else:
            raise ValueError("No LLM providers are configured")

    def _register_mcp_servers(self):
        """Register all MCP servers."""
        # Git MCP Server
        git_mcp_server = MCPServerSpec(
            params={"command": "uvx", "args": ["mcp-server-git"]},
            server_type=MCPServerType.STDIO,
            name="git-mcp-server",
            cache_tools_list=True,
            client_session_timeout=120,
        )
        register_mcp_server(git_mcp_server)
        
        # GitHub MCP Server
        github_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        if github_token:
            github_mcp_server = MCPServerSpec(
                params={
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",
                        "--rm",
                        "-e",
                        f"GITHUB_PERSONAL_ACCESS_TOKEN={github_token}",
                        "ghcr.io/github/github-mcp-server"
                    ]
                },
                server_type=MCPServerType.STDIO,
                name="github-mcp-server",
                cache_tools_list=True,
                client_session_timeout=120,
            )
            register_mcp_server(github_mcp_server)
        else:
            logger.warning("GITHUB_PERSONAL_ACCESS_TOKEN not found, GitHub MCP server will not be available")
            
        # Jira MCP Server
        jira_url = os.environ.get("JIRA_URL")
        jira_username = os.environ.get("JIRA_USERNAME")
        jira_token = os.environ.get("JIRA_API_TOKEN")
        
        if jira_url and jira_username and jira_token:
            jira_mcp_server = MCPServerSpec(
                params={
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",
                        "--rm",
                        "-e", f"JIRA_URL={jira_url}",
                        "-e", f"JIRA_USERNAME={jira_username}",
                        "-e", f"JIRA_API_TOKEN={jira_token}",
                        "ghcr.io/sooperset/mcp-atlassian:latest"
                    ]
                },
                server_type=MCPServerType.STDIO,
                name="jira-mcp-server",
                cache_tools_list=True,
                client_session_timeout=120,
            )
            register_mcp_server(jira_mcp_server)
        else:
            logger.warning("Jira credentials not complete, Jira MCP server will not be available")

    async def execute_query(self, user_query: str) -> str:
        """Execute a user query using the workflow.
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            The response from the assistant
        """
        logger.info(f"Executing query: {user_query}")
        
        # Create structured workflow inputs
        workflow_inputs = WorkflowInput(
            user_query=user_query,
            workflow={"inputs": {"user_query": user_query}}
        )

        callback = ConsoleProgressCallback()

        # Execute the workflow with the initialized workflow
        result = await self.workflow_manager.execute(
            self.workflow,
            workflow_inputs,
            progress_callback=callback
        )

        # Return the final result
        return result.final_result

    async def interactive_session(self):
        """Run an interactive session where the user can provide multiple queries."""
        print("\n" + "="*50)
        print("Welcome to the Unified MCP Development Assistant")
        print("This assistant can help with Git, GitHub, and Jira tasks")
        print("Type 'exit' or 'quit' to end the session")
        print("="*50 + "\n")

        while True:
            # Get user input
            user_query = input("\nWhat would you like help with? > ")

            # Check for exit command
            if user_query.lower() in ['exit', 'quit']:
                print("Exiting session. Goodbye!")
                break

            # Skip empty queries
            if not user_query.strip():
                continue

            try:
                # Execute the query
                response = await self.execute_query(user_query)

                # Print the response
                print("\n" + "="*50)
                print("ASSISTANT RESPONSE:")
                print(response)
                print("="*50)

            except Exception as e:
                logger.exception("Error executing query")
                print(f"\nError: {str(e)}")


async def main():
    """Main entry point for the MCP Assistant."""
    try:
        assistant = MCPAssistant()
        await assistant.initialize_workflow()
        await assistant.interactive_session()
    except Exception as e:
        logger.exception("Error in MCP Assistant")
        print(f"Error: {str(e)}")
        if "No LLM providers are configured" in str(e):
            print("\nSetup instructions:")
            print("1. Set OPENAI_API_KEY environment variable")
            print("2. Optional: Set GITHUB_PERSONAL_ACCESS_TOKEN for GitHub integration")
            print("3. Optional: Set JIRA_URL, JIRA_USERNAME, and JIRA_API_TOKEN for Jira integration")


if __name__ == "__main__":
    asyncio.run(main())