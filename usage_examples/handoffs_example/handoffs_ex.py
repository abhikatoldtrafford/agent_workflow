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
from agent_workflow.providers.observability import configure_observability
from agent_workflow.providers.observability import ObservabilityProvider

# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_observability() -> ObservabilityProvider:
    """Configure observability based on environment variables."""
    return configure_observability(
        provider_type="langfuse",
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY_AGENT_WF", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY_AGENT_WF", ""),
        host=os.environ.get("LANGFUSE_HOST", ""),
    )

async def run_tool_workflow() -> ExecutionResult:
    import yaml
    
    """Run an example workflow that uses tools."""
    # Path to the workflow YAML file
    #CONFIG
    handoffs_config_workflow = os.path.join(
        os.path.dirname(__file__), "handoffs_config2.yaml"
    )
    handoff_config_file = open(handoffs_config_workflow, 'r')
    handoff_config_content = handoff_config_file.read()
    handoff_config_content_dict = yaml.safe_load(handoff_config_content)

    #user_query = "Translate this question: Who is the CEO of OpenAI and what is their background?"
    user_query = "Answer the question: Who is the CEO of OpenAI and what is their background?"

    provider_configs: dict[str, BaseProviderConfig] = {}

    # Add OpenAI providers if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        provider_configs["openai_gpt4"] = OpenAIProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="gpt-4.1-2025-04-14",
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
    manager = WorkflowManager(engine_type="openai", provider_config=providers_config)
    provider_mapping = {
        "qa_agent": "openai_gpt4",
        "translation_agent": "openai_gpt4"
    }

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

    logger.info(f"Input user_query is: '{workflow_inputs.user_query}'")
    logger.info(
        f"Workflow inputs user_query is: '{workflow_inputs.workflow['inputs']['user_query']}'"
    )

    logger.info(f"Starting workflow execution with query: '{user_query}'")


    workflow = await manager.initialize_workflow(
        # WorkflowSourceFile(workflow_file),
        # WorkflowSourceYAML(content),
        handoff_config_content_dict,
        provider_mapping=provider_mapping,
        progress_callback=ConsoleProgressCallback(),
    )

    result = await manager.execute(
        workflow,
        workflow_inputs,
        progress_callback=ConsoleProgressCallback(),
    )
    # Print the final response
    logger.info("\n=== Workflow execution completed! ===")
    logger.info("\nFinal response:")
    final_output = result.final_result
    logger.info(final_output)

    # Print detailed tool usage information
    return result

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
