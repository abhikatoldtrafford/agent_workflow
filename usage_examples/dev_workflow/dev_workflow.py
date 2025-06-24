import asyncio
import json
import logging
import os

from agent_workflow.providers.observability import configure_observability
from agent_workflow.workflow_engine import (
    GeminiProviderConfig,
    OpenAIProviderConfig,
    ProviderConfiguration,
    ProviderType,
    WorkflowInput,
)
from agent_workflow.workflow_engine import WorkflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Initialize observability (can be controlled via environment variables)
def setup_observability():
    """Configure observability based on environment variables."""
    return configure_observability(
        provider_type="langfuse",
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY_AGENT_WF"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY_AGENT_WF"),
        host=os.environ.get("LANGFUSE_HOST"),
    )


async def run_product_dev_workflow():
    """Run the product development workflow example."""
    # Configure observability
    observability_provider = setup_observability()

    # Create a workflow manager with the specified engine
    # engine_type = os.environ.get("ENGINE_TYPE", "agentic")

    engine_type = "openai"

    # Start a workflow trace
    workflow_trace_id = observability_provider.trace_workflow(
        workflow_id=f"product_dev_workflow_{engine_type}",
        metadata={
            "workflow_name": "Product Development Workflow",
            "version": "1.0.0",
            "engine_type": engine_type,
        },
    )

    # Define available provider configurations
    provider_configs = {}

    # Add OpenAI providers if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        provider_configs["openai_gpt4"] = OpenAIProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="gpt-4-turbo",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    # Add Gemini providers if API key is available
    if os.environ.get("GEMINI_API_KEY"):
        provider_configs["google_gemini"] = GeminiProviderConfig(
            provider_type=ProviderType.GEMINI,
            model="gemini-1.5-pro",
            access_token=os.environ.get("GEMINI_API_KEY"),
            project_id=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_LOCATION"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    # Create the provider configuration container
    if provider_configs:
        providers_config = ProviderConfiguration(providers=provider_configs)
    else:
        # Fallback to empty dictionary for backward compatibility
        providers_config = {}

    manager = WorkflowManager(engine_type=engine_type, provider_config=providers_config)

    default_provider = "google_gemini"  # Default provider for agents

    # Provider mapping for agents - this makes agent YAML files engine-agnostic
    provider_mapping = {
        # Map agent IDs to provider types
        "implementation-plan-agent": default_provider,
        "api-design-agent": default_provider,
        "db-design-agent": default_provider,
        "ui-design-agent": default_provider,
        "technical-review-agent": default_provider,
    }

    # Define workflow inputs using the structured WorkflowInput format
    inputs = WorkflowInput(
        workflow={
            "inputs": {
                "feature_request": """
                We need a new feature that allows users to create custom dashboards.
                Users should be able to:
                - Add different types of widgets (charts, tables, metrics)
                - Arrange widgets in a grid layout
                - Schedule dashboard exports as PDFs
                """,
                "constraints": """
                Use the existing charting library and ensure the layout is responsive.
                Use good software engineering practices and document the code.
                """,
            }
        }
    )

    # Execute the workflow
    try:
        # Create a span for workflow execution
        span_id = observability_provider.trace_span(
            name="workflow_execution",
            parent_id=workflow_trace_id,
            metadata={"workflow_file": "product_dev_workflow_example.yaml"},
        )

        # First initialize the workflow
        workflow = await manager.initialize_workflow(
            os.path.join(
                os.path.dirname(__file__), "product_dev_workflow_example.yaml"
            ),
            provider_mapping=provider_mapping,
        )
        
        # Then execute the initialized workflow
        result = await manager.execute(
            workflow,
            inputs,
        )

        # End the execution span with success status
        observability_provider.end_span(span_id=span_id, metadata={"status": "success"})

        # Print workflow results
        print("\nWorkflow execution completed!")
        print_workflow_results(result)

        # Print detailed tool usage information if present
        print_tool_usage_report(result)

        # Capture an overall score for the workflow
        if observability_provider.enabled:
            observability_provider.capture_score(
                trace_id=workflow_trace_id, name="workflow_completed", value=1.0
            )

        return result
    except Exception as e:
        # End the execution span with error status
        if "span_id" in locals():
            observability_provider.end_span(
                span_id=span_id, metadata={"status": "error", "error": str(e)}
            )

        logger.error(f"Error executing workflow: {e}")
        raise


def print_workflow_results(results):
    """Print the workflow results in a readable format."""
    print("\n=== Workflow Results ===")

    # Handle standardized ExecutionResult format
    if hasattr(results, "all_agents"):
        print(f"\nAgents executed in sequence: {', '.join(results.all_agents)}\n")

        # Print each agent's output
        for idx, agent_output in enumerate(results.agent_outputs):
            print(f"\n== Agent {idx + 1}: {agent_output.agent} ==")

            # Truncate long outputs for display
            output_text = str(agent_output.output)
            if len(output_text) > 500:
                output_text = output_text[:500] + "... [truncated]"
            print(output_text)

        # Print final result
        print(f"\nFinal result: {results.final_result}")
        return

    # Handle legacy formats
    if isinstance(results, dict):
        if "stages" in results:
            for stage_name, stage_data in results.get("stages", {}).items():
                print(f"\nStage: {stage_name}")

                # Check for agent flow information
                if "agent_flow" in stage_data:
                    print(f"  Agents used: {', '.join(stage_data['agent_flow'])}")

                # Check for individual agent information
                if "agent" in stage_data:
                    print(f"  Agent: {stage_data.get('agent')}")

                for task_name, task_data in stage_data.get("tasks", {}).items():
                    print(f"  Task: {task_name}")

                    # Print task outputs
                    for output_name, output_value in task_data.get(
                        "outputs", {}
                    ).items():
                        print(f"    Output: {output_name}")

                        # Format output value based on type
                        if isinstance(output_value, list):
                            for i, item in enumerate(output_value):
                                print(f"      {i + 1}. {item}")
                        elif isinstance(output_value, dict):
                            for k, v in output_value.items():
                                print(f"      {k}: {v}")
                        else:
                            print(f"      {output_value}")


def print_tool_usage_report(result):
    """Print detailed information about tool usage in the workflow result."""
    logger.info("\n=== Tool Usage Details ===")
    tool_usage_found = False

    # Look for tool usage in the metadata of agent outputs
    if hasattr(result, "agent_outputs"):
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
    # Allow engine type override via environment variable
    engine_type = os.environ.get("ENGINE_TYPE", "openai")
    print(f"Using engine type: {engine_type}")

    # Allow model override via environment variable
    model_override = os.environ.get("MODEL_OVERRIDE")
    if model_override:
        print(f"Using model override: {model_override}")

    try:
        asyncio.run(run_product_dev_workflow())
    except ValueError as e:
        print(f"Error: {e}")
        if "No LLM providers are configured" in str(e):
            print("\nSetup instructions:")
            print("For AWS Bedrock: export AWS_ACCESS_KEY_ID=your_key_id")
            print("                 export AWS_SECRET_ACCESS_KEY=your_secret_key")
            print(
                "                 export AWS_REGION_NAME=your_region (default: us-east-1)"
            )
            print("For OpenAI: export OPENAI_API_KEY=your_api_key")
            print("For Gemini: export GEMINI_API_KEY=your_api_key")
    except Exception as e:
        print(f"Unexpected error: {e}")
