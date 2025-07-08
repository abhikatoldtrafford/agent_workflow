"""
Example of using tools with a workflow.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any

from agent_workflow.providers.callbacks import ConsoleProgressCallback
from agent_workflow.providers import FunctionTool, Tool, register_tool, tool
from agent_workflow.workflow_engine import ExecutionResult
from agent_workflow.workflow_engine import (
    GeminiProviderConfig,
    OpenAIProviderConfig,
    ProviderConfiguration,
    ProviderType,
    WorkflowInput,
    BaseProviderConfig,
)
from agent_workflow.workflow_engine import WorkflowManager
from agent_workflow.providers import LLMObservabilityProvider
from usage_examples.tool_usage.langfuse_observability import LangfuseLLMObservabilityProvider
from agent_workflow.providers.openai_observability import OpenaiLLMObservabilityProvider


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example tool defined as a regular function
@tool(
    name="calculate",
    description="basic Math Calculation, supported operations are `add`, `subtract`, `multiply` and `divide`",
)
def calculate(operation: str, a: float, b: float) -> float:
    """
    Perform a mathematical calculation between two numbers.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number

    Returns:
        The result of the calculation
    """
    logger.info(
        "Calculate tool called with parameters: operation=%s, a=%s, b=%s",
        operation,
        a,
        b,
    )
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b
    else:
        raise NotImplementedError(f"Operation '{operation}' is not supported.")


# Example tool defined as a class
class WeatherTool(Tool):
    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "Get current weather information for a location"

    @property
    def type(self) -> str:
        return "functional"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name or location"}
            },
            "required": ["location"],
        }

    async def execute(self, location: str) -> dict[str, Any]:  # type: ignore
        """Get weather information for a location (simulated)."""
        # This is a simulated response since we're not making an actual API call

        logger.info("executing weather tool called for location: %s", location)

        weather_data = {
            "location": location,
            "temperature": 82,
            "condition": "sunny",
            "humidity": 45,
            "wind_speed": 10,
            "timestamp": str(datetime.now()),
        }
        return weather_data

    @property
    def asFunctionalTool(self) -> FunctionTool:
        """Convert the class to a FunctionTool for compatibility."""
        return FunctionTool(
            name=self.name, description=self.description, func=self.execute
        )


# Register the class-based tool
register_tool(WeatherTool().asFunctionalTool)


# Example tool for searching information
@tool(name="search", description="Search for information on a topic")
def search_tool(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search for information on a given topic.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Search results
    """
    # Simulated search results
    results = [
        {
            "title": f"Result {i} for: {query}",
            "snippet": f"This is a snippet of information about {query}",
        }
        for i in range(1, max_results + 1)
    ]
    return {"query": query, "results": results}


# Initialize observability (can be controlled via environment variables)

def setup_observability() -> LLMObservabilityProvider:
    provider = LangfuseLLMObservabilityProvider(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY_V3", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY_V3", ""),
        host=os.environ.get("LANGFUSE_HOST_V3", "https://cloud.langfuse.com"),
    )

    return provider


async def run_tool_workflow() -> ExecutionResult:
    """Run an example workflow that uses tools."""
    # Path to the workflow YAML file
    workflow_file = os.path.join(
        os.path.dirname(__file__), "tool_workflow_example.yaml"
    )

    # read the file
    file = open(workflow_file, 'r')
    content = file.read()

    import yaml
    # Parse the content to a dictionary
    content_dict = yaml.safe_load(content)


    # Configure observability
    observability_provider = setup_observability()
    
    # Create a workflow trace group
    # workflow_group_id = tracer.start_workflow_tracing(
    #     workflow_id=f"Tool_workflow_{engine_type}",
    #     workflow_name="Tool Example Workflow",
    #     metadata={
    #         "tags": {
    #             "workflow_name": "Tool Example Workflow",
    #             "version": "1.0.0",
    #             "engine_type": engine_type
    #         }
    #     }
    # )

    # Define a query that requires both calculate and weather tools
    user_query = "What's the current temperature in San Francisco?"
    # user_query = "What's the current temperature in San Francisco? Convert the temperature to Celsius and add 10 degrees, use calculate tool."

    # Define available provider configurations based on available credentials
    # Now using structured provider configuration models
    provider_configs: dict[str, BaseProviderConfig] = {}

    # Add OpenAI providers if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        provider_configs["openai_gpt4"] = OpenAIProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="o3-mini-2025-01-31",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    if os.environ.get("GEMINI_API_KEY"):
        provider_configs["google_gemini_with_tool"] = GeminiProviderConfig(
            provider_type=ProviderType.GEMINI,
            model="gemini-2.5-pro-preview-03-25",
            access_token=os.environ.get("GEMINI_API_KEY", ""),
            project_id=os.environ.get("GCP_PROJECT_ID", ""),
            location=os.environ.get("GCP_LOCATION", ""),
            enforce_structured_output=False,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        provider_configs["google_gemini"] = GeminiProviderConfig(
            provider_type=ProviderType.GEMINI,
            model="gemini-2.5-pro-preview-03-25",
            access_token=os.environ.get("GEMINI_API_KEY", ""),
            project_id=os.environ.get("GCP_PROJECT_ID", ""),
            location=os.environ.get("GCP_LOCATION", ""),
            enforce_structured_output=True,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    # Create the provider configuration container
    providers_config = ProviderConfiguration(providers=provider_configs)


    # Create a workflow manager
    manager = WorkflowManager(engine_type=engine_type,
                              provider_config=providers_config,
                              llm_observability_provider=[observability_provider, OpenaiLLMObservabilityProvider()])

    # default_provider = "openai_gpt4"
    # default_provider = "google_gemini"
    # Provider mapping to ensure all agents use our tool-capable provider
    provider_mapping = {
        "tool-agent": "openai_gpt4",
        "response-agent": "openai_gpt4",
    }

    # Create structured workflow inputs
    workflow_inputs = WorkflowInput(
        user_query=user_query,
        workflow={
            "inputs": {
                "user_query": user_query  # Make sure this is available for ${workflow.inputs.user_query}
            }
        },
        provider_config=providers_config,
        provider_mapping=provider_mapping
    )

    # Verify the input is correctly set
    logger.info(f"Input user_query is: '{workflow_inputs.user_query}'")
    logger.info(
        f"Workflow inputs user_query is: '{workflow_inputs.workflow['inputs']['user_query']}'"
    )

    logger.info(f"Starting workflow execution with query: '{user_query}'")

    # First initialize the workflow
    workflow = await manager.initialize_workflow(
        # WorkflowSourceFile(workflow_file),
        # WorkflowSourceYAML(content),
        content_dict,
        provider_mapping=provider_mapping,
        progress_callback=ConsoleProgressCallback(),
    )
    
    # Then execute the initialized workflow
    result = await manager.execute(
        workflow,
        workflow_inputs,
        progress_callback=ConsoleProgressCallback(),
    )

    # Get the final result directly from the ExecutionResult
    final_response = result.final_result

    # Print the final response
    logger.info("\n=== Workflow execution completed! ===")
    logger.info("\nFinal response:")
    logger.info(final_response)

    # Print detailed tool usage information
    print_tool_usage_report(result)
    return result


def print_tool_usage_report(result: Any) -> None:
    """Print detailed information about tool usage in the workflow result."""
    logger.info("\n=== Tool Usage Details ===")
    tool_usage_found = False

    # Look for tool usage in the metadata of agent outputs
    for agent_output in result.agent_outputs:
        if agent_output.metadata and "_tool_usage" in agent_output.metadata:
            tool_usage = agent_output.metadata["_tool_usage"]
            tool_usage_found = True

            logger.info(f"\nAgent: {agent_output.agent}")
            logger.info(f"Total tool calls: {len(tool_usage)}")

            # Print details for each tool call
            for i, usage in enumerate(tool_usage):
                logger.info(f"\n  Tool Call {i + 1}:")
                logger.info(f"  Tool: {usage.get('name', 'unknown')}")
                logger.info(
                    f"  Parameters: {json.dumps(usage.get('parameters', {}), indent=2)}"
                )

                # Format the result nicely
                result_data = usage.get("result")
                if result_data:
                    if not isinstance(result_data, str):
                        result_str = json.dumps(result_data, indent=2)
                    else:
                        result_str = result_data

                    # Truncate long results
                    if len(result_str) > 100:
                        result_str = result_str[:100] + "... (truncated)"

                    logger.info(f"  Result: {result_str}")

                # Show execution time if available
                if "execution_time" in usage:
                    exec_time = usage["execution_time"]
                    logger.info(f"  Execution time: {exec_time:.2f} seconds")

    if not tool_usage_found:
        logger.info("No tool usage information found in the workflow results")


if __name__ == "__main__":
    # Using OpenAI engine type
    engine_type = "openai"
    if engine_type:
        print(f"Using model override from environment: {engine_type}")

    try:
        asyncio.run(run_tool_workflow())
    except ValueError as e:
        print(f"Error: {e}")
        if "No LLM providers are configured" in str(e):
            print("\nSetup instructions:")
    except Exception as e:
        print(f"Unexpected error: {e}")
