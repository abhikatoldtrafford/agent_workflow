"""
Example of using progress callbacks to get intermediate results from workflows.
"""

import argparse
import asyncio
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agent_workflow.providers.callbacks import ConsoleProgressCallback, StreamingProgressCallback
from agent_workflow.providers.observability import configure_observability
from agent_workflow.workflow_engine import WorkflowManager


def on_task_complete(task_name, task_result, agent_output):
    """Custom handler for task completion events."""
    print(f"\n📊 Task Completed: {task_name}")
    print(f"📋 Output summary: {task_result}")

    # Print a snippet of the agent's response if available
    if agent_output and hasattr(agent_output, "output"):
        response_text = ""
        if isinstance(agent_output.output, dict) and "response" in agent_output.output:
            response_text = agent_output.output["response"]
        elif isinstance(agent_output.output, str):
            response_text = agent_output.output

        if response_text:
            print(f"📝 Agent response snippet: {response_text[:150]}...")


def stream_to_client(event):
    """
    In a real app, this would stream data to the client through a websocket
    or other stream mechanism. For this example, we just print.
    """
    print(f"\n🔄 STREAM EVENT: {event['event_type']}")
    if "task_name" in event:
        print(f"   Task: {event['task_name']}")
    if "output" in event and isinstance(event["output"], str):
        snippet = (
            event["output"][:100] + "..."
            if len(event["output"]) > 100
            else event["output"]
        )
        print(f"   Output snippet: {snippet}")


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run a workflow with progress callbacks"
    )
    parser.add_argument(
        "--engine", choices=["openai"], default="openai", help="Execution engine to use"
    )
    parser.add_argument(
        "--callback",
        choices=["console", "streaming"],
        default="console",
        help="Callback type to use",
    )
    parser.add_argument(
        "--workflow",
        default="tool_workflow_example.yaml",
        help="Workflow YAML file path",
    )
    args = parser.parse_args()

    # Configure observability (optional)
    observability_provider = configure_observability(provider_type="noop")

    # Create the appropriate callback
    if args.callback == "streaming":
        progress_callback = StreamingProgressCallback(
            stream_fn=stream_to_client, on_task_callback=on_task_complete
        )
    else:
        progress_callback = ConsoleProgressCallback(on_task_callback=on_task_complete)

    # Engine-specific options
    engine_options = {}
    if args.engine == "openai":
        # Get OpenAI API key from environment
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            print(
                "Error: OPENAI_API_KEY environment variable is required for OpenAI engine."
            )
            return
        engine_options["api_key"] = openai_api_key

    # Create workflow manager
    manager = WorkflowManager(engine_type=args.engine, engine_options=engine_options)

    # Define workflow inputs
    inputs = {
        "feature_request": "Create a user dashboard with metrics for revenue, app usage, and user engagement",
        "constraints": "Use a responsive design and existing UI components, ensure accessibility compliance",
    }

    # Add provider configurations for the Agentic engine
    if args.engine == "agentic":
        inputs["provider_config"] = {
            "bedrock_sonnet_3.5": {
                "provider_type": "bedrock_sonnet_3.5",
                "credentials": {
                    "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "region_name": "us-east-1",
                },
                "observability": {"provider": observability_provider},
            }
        }

        # Provider mapping for specific agents
        inputs["provider_mapping"] = {
            "tool-agent": "bedrock_sonnet_3.5",
            "response-agent": "bedrock_sonnet_3.5",
        }

    # Execute the workflow with the progress callback
    print(
        f"\n🚀 Starting workflow execution with {args.engine} engine and {args.callback} callbacks..."
    )
    # First initialize the workflow
    workflow = await manager.initialize_workflow(
        args.workflow, 
        progress_callback=progress_callback
    )
    
    # Then execute the initialized workflow
    results = await manager.execute(
        workflow, 
        inputs, 
        progress_callback=progress_callback
    )

    # Print final results summary
    print("\n✅ Workflow execution completed!")
    print(f"  Agent count: {len(results.all_agents)}")
    print(f"  Agents used: {', '.join(results.all_agents)}")


if __name__ == "__main__":
    asyncio.run(main())
