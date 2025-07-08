"""
Example of a workflow with user input between agent steps.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Union

from agent_workflow.providers.callbacks import UserInputCallback
from agent_workflow.workflow_engine import ExecutionResult
from agent_workflow.workflow_engine import (
    AgentOutput,
    OpenAIProviderConfig,
    ProviderConfiguration,
    ProviderType,
    WorkflowInput,
)
from agent_workflow.workflow_engine import WorkflowManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_user_input(
    agent_name: str, task_result: Dict[str, Any], agent_output: AgentOutput
) -> Union[str, None]:
    """
    Handler function to get user input after an agent completes.

    Args:
        agent_name: The name of the agent that just completed
        task_result: The task result dictionary
        agent_output: The agent's output

    Returns:
        Modified output if user makes changes, or None to keep original
    """
    print("\n" + "=" * 80)
    print(f"AGENT '{agent_name}' OUTPUT:")
    print(f"{agent_output.output}")
    print("=" * 80)

    while True:
        choice = input("\nDo you want to modify this output? (y/n): ").lower()
        if choice == "n":
            return None
        elif choice == "y":
            print("\nEnter your modified output (type 'DONE' on a new line to finish):")
            lines = []
            while True:
                line = input()
                if line == "DONE":
                    break
                lines.append(line)
            modified_output = "\n".join(lines)
            return modified_output
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")


async def run_user_input_workflow() -> ExecutionResult:
    """Run a workflow with user input enabled."""

    # Path to the workflow YAML file
    workflow_file = os.path.join(
        os.path.dirname(__file__), "user_input_workflow_example.yaml"
    )

    # Define available provider configurations based on available credentials
    provider_configs = {}

    # Add OpenAI provider if API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning(
            "OPENAI_API_KEY not found in environment. Using placeholder value for demo purposes."
        )
        openai_api_key = "your-openai-api-key-here"  # Placeholder for demo

    provider_configs["openai_gpt4"] = OpenAIProviderConfig(
        provider_type=ProviderType.OPENAI,
        model="o3-mini-2025-01-31",
        api_key=openai_api_key,
    )

    # Create the provider configuration container
    providers_config = ProviderConfiguration(providers=provider_configs)

    # Define a sample content for analysis
    user_query = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence 
    displayed by animals including humans. The field of AI research was founded on the assumption that
    human intelligence can be precisely described in terms of symbol manipulation and can be simulated
    by a machine. However, many AI experts have struggled to explain the capabilities of large language
    models such as GPT-4, which have shown impressive abilities across various tasks but don't work like
    traditional AI systems. Modern machine learning systems use algorithmic techniques like deep learning
    to achieve specific goals, but researchers continue to debate whether these systems can achieve
    general intelligence comparable to humans.
    """

    # Provider mapping to ensure all agents use our provider
    provider_mapping = {
        "summarizer": "openai_gpt4",
        "key-points": "openai_gpt4",
        "response-generator": "openai_gpt4",
    }

    # Create workflow input
    inputs = WorkflowInput(
        user_query=user_query, workflow={"inputs": {"user_query": user_query}}
    )

    # Create a user input callback
    callback = UserInputCallback(input_handler=get_user_input)

    # Create a workflow manager with the OpenAI engine
    engine_type = "openai"  # Using OpenAI engine for user input support
    manager = WorkflowManager(engine_type=engine_type, provider_config=providers_config)

    logger.info("Starting workflow execution with user input enabled")

    # First initialize the workflow
    workflow = await manager.initialize_workflow(
        workflow_file,
        provider_mapping=provider_mapping,
        progress_callback=callback,
    )
    
    # Then execute the initialized workflow
    result = await manager.execute(
        workflow,
        inputs,
        progress_callback=callback,
    )

    # Get the final result
    final_response = result.final_result

    # Print the final response
    logger.info("\n=== Workflow execution completed! ===")
    logger.info("\nFinal response:")
    logger.info(final_response)

    # Print a summary of each agent's contributions
    logger.info("\n=== Agent Contributions ===")
    for agent_output in result.agent_outputs:
        logger.info(f"\nAgent: {agent_output.agent}")
        # Print a short excerpt of the output (first 100 chars)
        output_excerpt = str(agent_output.output)
        if len(output_excerpt) > 100:
            output_excerpt = output_excerpt[:100] + "... (truncated)"
        logger.info(f"Output: {output_excerpt}")

    return result


if __name__ == "__main__":
    try:
        asyncio.run(run_user_input_workflow())
    except ValueError as e:
        print(f"Error: {e}")
        if "No LLM providers are configured" in str(e):
            print("\nPlease set the OPENAI_API_KEY environment variable.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
