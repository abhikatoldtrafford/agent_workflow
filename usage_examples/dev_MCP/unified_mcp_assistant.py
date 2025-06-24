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
from typing import Dict, Optional, Any, Union, TypedDict

from agent_workflow.providers.mcp import register_mcp_server
from agent_workflow.workflow_engine import Workflow
from agent_workflow.workflow_engine.execution_engines import ProgressCallback
from agent_workflow.workflow_engine import (
    AgentOutput,
    ExecutionResult,
    GeminiProviderConfig,
    MCPServerSpec,
    MCPServerType,
    OpenAIProviderConfig,
    ProviderConfiguration,
    ProviderType,
    WorkflowInput,
    WorkflowStage,
    WorkflowTask,
    ModelSettings,
    BaseProviderConfig,
    WorkflowSourceFile
)
from agent_workflow.workflow_engine import WorkflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("mcp-assistant")


class DetailedProgressCallback(ProgressCallback):
    """A progress callback that logs all events with detailed information."""
    
    def __init__(self) -> None:
        """Initialize the progress callback."""
        self.status_log: list[str] = []
        
    async def on_workflow_start(
        self, workflow_name: str, workflow_config: Workflow
    ) -> None:
        """Called when a workflow starts execution."""
        log_entry = f"[WORKFLOW START] {workflow_name}"
        self.status_log.append(log_entry)
        logger.info(log_entry)
        
    async def on_workflow_complete(self, workflow_name: str, result: ExecutionResult) -> None:
        """Called when a workflow completes execution."""
        log_entry = f"[WORKFLOW COMPLETE] {workflow_name}"
        self.status_log.append(log_entry)
        logger.info(log_entry)
        
    async def on_stage_start(
        self, stage_name: str, stage_config: Union[Dict[str, Any], WorkflowStage]
    ) -> None:
        """Called when a stage starts execution."""
        log_entry = f"[STAGE START] {stage_name}"
        self.status_log.append(log_entry)
        logger.info(log_entry)
        
    async def on_stage_complete(self, stage_name: str, stage_result: Dict[str, Any]) -> None:
        """Called when a stage completes execution."""
        log_entry = f"[STAGE COMPLETE] {stage_name}"
        self.status_log.append(log_entry)
        logger.info(log_entry)
        
    async def on_task_start(
        self, task_name: str, task_config: Union[Dict[str, Any], WorkflowTask]
    ) -> None:
        """Called when a task starts execution."""
        log_entry = f"[TASK START] {task_name}"
        self.status_log.append(log_entry)
        logger.info(log_entry)
        
    async def on_task_complete(
        self,
        task_name: str,
        task_result: Dict[str, Any],
        agent_output: Optional[AgentOutput] = None,
    ) -> Optional[AgentOutput]:
        """Called when a task completes execution."""
        log_entry = f"[TASK COMPLETE] {task_name}"
        self.status_log.append(log_entry)
        logger.info(log_entry)
        return None
        
    async def on_task_fail(
        self, task_name: str, error: str, agent_output: Optional[AgentOutput] = None
    ) -> None:
        """Called when a task fails execution."""
        log_entry = f"[TASK FAIL] {task_name}: {error}"
        self.status_log.append(log_entry)
        logger.error(log_entry)
        
    def get_status_log(self) -> list:
        """Get the current status log."""
        return self.status_log


class MCPAssistant:
    """Unified assistant that integrates Git, GitHub, and Jira MCP servers."""
    
    def __init__(self) -> None:
        """Initialize the MCP Assistant."""
        self.workflow_file = os.path.join(
            os.path.dirname(__file__), "unified_mcp_workflow.yaml"
        )
        # Create the progress callback for detailed logging
        self.progress_callback = DetailedProgressCallback()
        # Get filesystem paths to mount
        self.filesystem_paths = self._get_filesystem_paths()
        # Setup provider config
        self.provider_config = self._setup_provider_config()
        # Setup MCP servers with filesystem access
        self._register_mcp_servers()
        # Create workflow manager
        self.workflow_manager = WorkflowManager(
            engine_type="openai", provider_config=self.provider_config
        )

    def _get_filesystem_paths(self) -> Dict[str, str]:
        """
        Get the filesystem paths to mount.
        Users can provide paths at the start of the session.
        
        Returns:
            Dict mapping of local paths to container paths
        """
        print("\n" + "="*50)
        print("FILESYSTEM ACCESS SETUP")
        print("This assistant can access local files via the filesystem MCP server.")
        print("Please specify local paths you want to make available:")
        print("="*50 + "\n")
        
        paths = {}
        
        # Add default working directory
        default_dir = os.getcwd()
        response = input(f"Add current directory ({default_dir})? (y/n): ").lower()
        if response.startswith('y'):
            paths[default_dir] = "/projects/workdir"
            print(f"✅ Added: {default_dir} => /projects/workdir")
        
        # Allow adding more directories
        while True:
            additional_path = input("\nAdd another directory path (leave empty to finish): ").strip()
            if not additional_path:
                break
                
            if os.path.exists(additional_path) and os.path.isdir(additional_path):
                container_path = f"/projects/{os.path.basename(additional_path)}"
                paths[additional_path] = container_path
                print(f"✅ Added: {additional_path} => {container_path}")
            else:
                print(f"⚠️ Error: {additional_path} does not exist or is not a directory")
        
        # Let user know we're done collecting paths
        print("\nFilesystem setup complete. Added", len(paths), "directories.")
        
        return paths
        
    async def initialize_workflow(self) -> None:
        """Initialize the workflow with all agents."""
        logger.info("Initializing workflow with agents...")
        
        # Use our detailed progress callback
        self.workflow = await self.workflow_manager.initialize_workflow(
            WorkflowSourceFile(self.workflow_file),
            provider_mapping=self.provider_mapping,
            progress_callback=self.progress_callback,
        )
        logger.info(f"Workflow '{self.workflow.name}' initialized successfully with {len(self.workflow.stages)} stages")

    def _setup_provider_config(self) -> ProviderConfiguration:
        """Set up provider configuration based on available API keys."""
        provider_configs: dict[str, BaseProviderConfig] = {}

        # Add OpenAI provider if API key is available
        if os.environ.get("OPENAI_API_KEY"):
            provider_configs["openai_gpt4"] = OpenAIProviderConfig(
                provider_type=ProviderType.OPENAI,
                model="gpt-4o",
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_settings=ModelSettings(temperature=0.7)
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
                model_settings=ModelSettings(temperature=0.7),
            )

        # Provider mapping to ensure agents use our MCP-capable provider
        self.provider_mapping = {"dev-assistant": "google_gemini"}

        # Create the provider configuration container
        if provider_configs:
            return ProviderConfiguration(providers=provider_configs)
        else:
            raise ValueError("No LLM providers are configured")

    def _register_mcp_servers(self) -> None:
        """Register all MCP servers."""
        # Git MCP Server
        git_mcp_params: dict = {"command": "uvx", "args": ["mcp-server-git"]}
        git_mcp_server = MCPServerSpec(
            params= git_mcp_params,
            server_type=MCPServerType.STDIO,
            name="git-mcp-server",
            cache_tools_list=True,
            client_session_timeout=120,
        )
        register_mcp_server(git_mcp_server)
        logger.info("Registered git-mcp-server")
        
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
            logger.info("Registered github-mcp-server")
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
            logger.info("Registered jira-mcp-server")
        else:
            logger.warning("Jira credentials not complete, Jira MCP server will not be available")

        # Filesystem MCP Server
        if self.filesystem_paths:
            # Build the docker command with mounted directories
            docker_args = [
                "run",
                "-i",
                "--rm",
            ]
            
            # Add mount points for each path
            for local_path, container_path in self.filesystem_paths.items():
                docker_args.extend([
                    "--mount", f"type=bind,src={local_path},dst={container_path}"
                ])
                
            # Add the filesystem container image and root directory
            docker_args.extend([
                "mcp/filesystem",
                "/projects"  # Root directory inside container
            ])

            #  register Sequential thinking MCP
            filesystem_mcp_server = MCPServerSpec(
                params={
                    "command": "docker",
                    "args": ["run",
                             "--rm",
                             "-i",
                             "mcp/sequentialthinking"
                    ],
                },
                server_type=MCPServerType.STDIO,
                name="filesystem-mcp-server",
                cache_tools_list=True,
                client_session_timeout=120,
            )
            register_mcp_server(filesystem_mcp_server)
            logger.info(f"Registered filesystem-mcp-server with {len(self.filesystem_paths)} mounted directories")

        # add sequential MCP server

        # add local memory
        local_mcp_server = MCPServerSpec(
            params={
                "command": "mcp",
                "args": [
                    "run",
                    "/Users/manish/Documents/WorkingCopy/Agent_workflow/usage_examples/dev_MCP/local_memory_mcp_server.py"
                ]
            },
            server_type=MCPServerType.STDIO,
            name="local-memory-mcp-server",
            cache_tools_list=True,
            client_session_timeout=120,
        )
        register_mcp_server(local_mcp_server)


        # register
        # local_memory_mcp_server = MCPServerSpec(
        #     params={
        #         "command": "docker",
        #         "args": ["run",
        #                  "-i",
        #                  "-v",
        #                  "claude-memory:/app/dist",
        #                  "--rm",
        #                  "mcp/memory"]
        #     },
        #     server_type=MCPServerType.STDIO,
        #     name="local-memory-mcp-server",
        #     cache_tools_list=True,
        #     client_session_timeout=120,
        # )
        # register_mcp_server(local_memory_mcp_server)

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
            workflow={
                "inputs": {
                    "user_query": user_query,
                    "filesystem_paths": list(self.filesystem_paths.keys())
                    # Pass filesystem paths to workflow
                }
            }
        )

        # Use our detailed progress callback to track execution
        logger.info("Executing workflow with query...")
        
        # Execute the workflow with the initialized workflow and progress callback
        result = await self.workflow_manager.execute(
            self.workflow,
            workflow_inputs,
            progress_callback=self.progress_callback
        )
        
        # Log completion status
        log_counts: dict[str, Any] = {}
        for entry in self.progress_callback.get_status_log():
            prefix = entry.split("]")[0] + "]"
            log_counts[prefix] = log_counts.get(prefix, 0) + 1
            
        logger.info(f"Execution stats: {log_counts}")

        # Return the final result
        return result.final_result

    async def interactive_session(self):
        """Run an interactive session where the user can provide multiple queries."""
        print("\n" + "="*50)
        print("Welcome to the Unified MCP Development Assistant")
        print("This assistant can help with Git, GitHub, Jira tasks, and local filesystem access")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'status' to see execution statistics")
        print("="*50 + "\n")

        # Show available filesystem paths
        if self.filesystem_paths:
            print("\nAccessible filesystem paths:")
            for local_path, container_path in self.filesystem_paths.items():
                print(f"  • {local_path} → {container_path}")
        else:
            print("\nNo filesystem paths were configured.")
            
        print("\nThe assistant can work with Git, GitHub, Jira, and local files.")
        print("You can ask questions like:")
        print("  • \"Show me the current git status\"")
        print("  • \"Create a GitHub issue for this bug\"")
        print("  • \"Find Jira tickets assigned to me\"") 
        print("  • \"List files in my project directory\"")
        print()

        while True:
            # Get user input
            user_query = input("\nWhat would you like help with? > ")

            # Check for exit command
            if user_query.lower() in ['exit', 'quit']:
                print("Exiting session. Goodbye!")
                break
                
            # Check for status command
            if user_query.lower() == 'status':
                print("\n" + "="*50)
                print("EXECUTION STATISTICS:")
                if hasattr(self, 'progress_callback') and self.progress_callback.status_log:
                    # Count entries by type
                    log_counts = {}
                    for entry in self.progress_callback.get_status_log():
                        prefix = entry.split("]")[0] + "]"
                        log_counts[prefix] = log_counts.get(prefix, 0) + 1
                    
                    # Print statistics
                    for prefix, count in log_counts.items():
                        print(f"{prefix}: {count}")
                    
                    # Print last 5 entries 
                    print("\nRecent activity:")
                    for entry in self.progress_callback.get_status_log()[-5:]:
                        print(f"  {entry}")
                else:
                    print("No execution statistics available yet.")
                print("="*50)
                continue

            # Skip empty queries
            if not user_query.strip():
                continue

            try:
                # Execute the query
                print("\nProcessing your request... (This might take a moment)")
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
        print("\n🚀 Starting Unified MCP Development Assistant...")
        print("This assistant integrates Git, GitHub, Jira, and filesystem access in one interface.")
        
        # Create and initialize assistant
        assistant = MCPAssistant()
        print("\n⏳ Initializing workflow with MCP servers and agents...")
        await assistant.initialize_workflow()
        print("✅ Initialization complete!")
        
        # Start interactive session
        await assistant.interactive_session()
    except Exception as e:
        logger.exception("Error in MCP Assistant")
        print(f"Error: {str(e)}")
        if "No LLM providers are configured" in str(e):
            print("\nSetup instructions:")
            print("1. Required: Set OPENAI_API_KEY environment variable")
            print("2. Optional: Set GITHUB_PERSONAL_ACCESS_TOKEN for GitHub integration")
            print("3. Optional: Set JIRA_URL, JIRA_USERNAME, and JIRA_API_TOKEN for Jira integration")
            print("\nExample setup commands:")
            print("  export OPENAI_API_KEY=your_api_key_here")
            print("  export GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token")
            print("  export JIRA_URL=https://your-company.atlassian.net")
            print("  export JIRA_USERNAME=your_email@example.com")
            print("  export JIRA_API_TOKEN=your_jira_api_token")


if __name__ == "__main__":
    asyncio.run(main())