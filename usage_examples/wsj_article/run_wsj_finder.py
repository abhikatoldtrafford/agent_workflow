"""
WSJ Article Finder

This script runs a workflow that finds digital versions of WSJ print articles.
It uses a LLM-powered agent to search and match articles based on content similarity,
not just titles.

Usage:
    python run_wsj_finder.py [--headline "print headline"] [--date "YYYY-MM-DD"]
                          [--excerpt "excerpt from print"] [--author "author name"]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Union

from agent_workflow.providers.observability import configure_observability
from agent_workflow.workflow_engine import (
    BedrockProviderConfig,
    ExecutionResult,
    GeminiProviderConfig,
    OpenAIProviderConfig,
    ProviderConfiguration,
    ProviderType,
    WorkflowInput,
)
from agent_workflow.workflow_engine import WorkflowManager

sys.path.append("../..")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def setup_observability():
    """Configure observability based on environment variables."""
    return configure_observability(
        provider_type="langfuse",
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY_AGENT_WF"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY_AGENT_WF"),
        host=os.environ.get("LANGFUSE_HOST"),
    )


async def run_wsj_article_workflow(
    print_headline: str,
    print_date: Optional[str] = None,
    print_excerpt: Optional[str] = None,
    author: Optional[str] = None,
    engine_type: str = "openai",
    model_name: Optional[str] = None,
) -> Union[ExecutionResult, Dict[str, Any]]:
    """
    Run the WSJ article finder workflow.

    Args:
        print_headline: Headline of the print article
        print_date: Publication date (YYYY-MM-DD)
        print_excerpt: Excerpt from the print article
        author: Author name (optional)
        engine_type: Type of engine to use (openai, agentic)
        model_name: Optional model name to override

    Returns:
        ExecutionResult containing the workflow output
    """
    # Configure observability
    observability_provider = setup_observability()

    # Start a workflow trace
    workflow_trace_id = observability_provider.trace_workflow(
        workflow_id=f"wsj_article_finder_{datetime.now().strftime('%Y%m%d')}",
        metadata={
            "workflow_name": "WSJ Article Finder",
            "version": "1.0.0",
            "engine_type": engine_type,
            "print_headline": print_headline,
            "print_date": print_date,
        },
    )

    # Define available provider configurations
    provider_configs = {}

    # Add Bedrock Claude providers if AWS credentials are available
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        # Claude 3.5 Sonnet
        provider_configs["bedrock_sonnet_3.5"] = BedrockProviderConfig(
            provider_type=ProviderType.BEDROCK,
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            AWS_REGION=os.environ.get("AWS_REGION_NAME", "us-east-1"),
            AWS_ACCESS_KEY_ID=os.environ.get("AWS_ACCESS_KEY_ID"),
            AWS_SECRET_ACCESS_KEY=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

        # Claude 3.7 Sonnet
        provider_configs["bedrock_sonnet_3.7"] = BedrockProviderConfig(
            provider_type=ProviderType.BEDROCK,
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            AWS_REGION=os.environ.get("AWS_REGION_NAME", "us-east-1"),
            AWS_ACCESS_KEY_ID=os.environ.get("AWS_ACCESS_KEY_ID"),
            AWS_SECRET_ACCESS_KEY=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

    # Add OpenAI providers if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        provider_configs["openai_gpt4"] = OpenAIProviderConfig(
            provider_type=ProviderType.OPENAI,
            model=model_name or "gpt-4o-2024-08-06",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    # Add Gemini providers if API key is available
    if os.environ.get("GEMINI_API_KEY"):
        provider_configs["google_gemini"] = GeminiProviderConfig(
            provider_type=ProviderType.GEMINI,
            model="gemini-2.5-pro-preview-03-25",
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

    # Create a workflow manager with the specified engine
    manager = WorkflowManager(engine_type=engine_type, provider_config=providers_config)

    default_provider = "openai_gpt4"
    # default_provider = "google_gemini"

    # Provider mapping for agents
    provider_mapping = {"wsj-finder-agent": default_provider}

    # Setup workflow inputs
    input_data = {
        "print_headline": print_headline,
        "print_date": print_date,
        "print_excerpt": print_excerpt,
    }

    if author:
        input_data["author"] = author

    # Create structured workflow inputs
    workflow_inputs = WorkflowInput(
        **input_data,
        provider_config=providers_config,
        provider_mapping=provider_mapping,
        workflow={"inputs": input_data},
    )

    workflow_file = os.path.join(os.path.dirname(__file__), "wsj_article_workflow.yaml")

    # Execute the workflow
    try:
        # Create a span for workflow execution
        span_id = observability_provider.trace_span(
            name="workflow_execution",
            parent_id=workflow_trace_id,
            metadata={"workflow_file": workflow_file},
        )

        # First initialize the workflow
        workflow = await manager.initialize_workflow(
            workflow_file,
            provider_mapping=provider_mapping,
        )
        
        # Then execute the initialized workflow
        result = await manager.execute(
            workflow,
            workflow_inputs,
        )

        # End the execution span with success status
        observability_provider.end_span(span_id=span_id, metadata={"status": "success"})

        # Print results
        print_wsj_match_result(result)

        # Print detailed tool usage information if present
        print_tool_usage_report(result)

        # Capture an overall score for the workflow
        if observability_provider.enabled:
            confidence_score = get_confidence_score(result)
            observability_provider.capture_score(
                trace_id=workflow_trace_id,
                name="match_confidence",
                value=confidence_score / 10.0,  # Normalize to 0-1 range
                metadata={
                    "print_headline": print_headline,
                    "digital_headline": get_digital_headline(result),
                },
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


def print_wsj_match_result(result: Union[ExecutionResult, Dict[str, Any]]) -> None:
    """
    Print the WSJ article match results in a readable format.
    Supports both ExecutionResult and dictionary formats.
    """
    print("\n" + "=" * 60)
    print("WSJ ARTICLE FINDER RESULTS")
    print("=" * 60)

    # Extract information from result based on its type
    if isinstance(result, ExecutionResult):
        # Access from standardized ExecutionResult format
        agent_output = None
        if result.agent_outputs and len(result.agent_outputs) > 0:
            agent_output = result.agent_outputs[0].output

        if isinstance(agent_output, dict):
            # Extract fields from agent output (structured dictionary format)
            digital_url = agent_output.get("digital_url", "URL not found")
            digital_headline = agent_output.get(
                "digital_headline", "Headline not found"
            )
            author = agent_output.get("author", "Author not found")
            publication_date = agent_output.get("publication_date", "Date not found")
            confidence_score = agent_output.get("confidence_score", 0)
            matching_content = agent_output.get(
                "matching_content", "No matching content identified"
            )
            article_text = agent_output.get("article_text", "Article text not found")

            # Print structured output
            print(f"DIGITAL HEADLINE: {digital_headline}")
            print(f"URL: {digital_url}")
            print(f"AUTHOR: {author}")
            print(f"PUBLICATION DATE: {publication_date}")
            print(f"MATCH CONFIDENCE: {confidence_score}/10")

            print("\n" + "-" * 60)
            print("MATCHING CONTENT SECTIONS:")
            print("-" * 60)
            print(matching_content)

            print("\n" + "-" * 60)
            print("ARTICLE TEXT PREVIEW:")
            print("-" * 60)
            # Show a preview of the article text (first 500 chars)
            preview_length = 500
            if len(article_text) > preview_length:
                print(f"{article_text[:preview_length]}...\n[truncated]")
            else:
                print(article_text)

        elif isinstance(agent_output, str):
            # Handle text format output (unstructured)
            print("\n" + "-" * 60)
            print("ARTICLE OUTPUT:")
            print("-" * 60)
            print(agent_output)

            # Try to extract headline from text format
            lines = agent_output.strip().split("\n")
            headline = ""
            for i, line in enumerate(lines):
                if "headline" in line.lower() or "###" in line:
                    if i + 1 < len(lines) and lines[i + 1].strip():
                        headline = lines[i + 1].strip()
                        break

            if headline:
                print("\n" + "-" * 60)
                print(f"EXTRACTED HEADLINE: {headline}")
        else:
            # Unknown format - just dump the data
            print(f"\nAGENT OUTPUT (type: {type(agent_output).__name__}):")
            print(agent_output)

        # Also print final result if it differs from agent output
        if result.final_result != agent_output and result.final_result:
            print("\n" + "-" * 60)
            print("FINAL RESULT:")
            print("-" * 60)
            print(result.final_result)
    else:
        # Try to extract from dictionary format (legacy)
        try:
            outputs = result["stages"]["article-search"]["tasks"][
                "Find Digital Article"
            ]["outputs"]
            digital_url = outputs.get("digital_url", "URL not found")
            digital_headline = outputs.get("digital_headline", "Headline not found")
            author = outputs.get("author", "Author not found")
            publication_date = outputs.get("publication_date", "Date not found")
            confidence_score = outputs.get("confidence_score", 0)
            matching_content = outputs.get(
                "matching_content", "No matching content identified"
            )
            article_text = outputs.get("article_text", "Article text not found")

            # Print structured output
            print(f"DIGITAL HEADLINE: {digital_headline}")
            print(f"URL: {digital_url}")
            print(f"AUTHOR: {author}")
            print(f"PUBLICATION DATE: {publication_date}")
            print(f"MATCH CONFIDENCE: {confidence_score}/10")

            print("\n" + "-" * 60)
            print("MATCHING CONTENT SECTIONS:")
            print("-" * 60)
            print(matching_content)

            print("\n" + "-" * 60)
            print("ARTICLE TEXT PREVIEW:")
            print("-" * 60)
            # Show a preview of the article text (first 500 chars)
            preview_length = 500
            if len(article_text) > preview_length:
                print(f"{article_text[:preview_length]}...\n[truncated]")
            else:
                print(article_text)
        except (KeyError, TypeError) as e:
            logger.error(f"Error extracting data from result structure: {e}")
            print("Raw result:")
            print(result)

    print("=" * 60)


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


def get_confidence_score(result: Union[ExecutionResult, Dict[str, Any]]) -> int:
    """Extract the confidence score from the result."""
    if isinstance(result, ExecutionResult):
        if result.agent_outputs and len(result.agent_outputs) > 0:
            agent_output = result.agent_outputs[0].output
            if isinstance(agent_output, dict):
                return agent_output.get("confidence_score", 0)
            elif isinstance(agent_output, str):
                # Try to extract confidence score from text
                for line in agent_output.lower().split("\n"):
                    if "confidence" in line and ":" in line:
                        try:
                            # Extract numeric part after the colon
                            score_part = line.split(":")[1].strip()
                            # Extract just the number
                            score = int("".join(c for c in score_part if c.isdigit()))
                            if 0 <= score <= 10:  # Validate range
                                return score
                        except (ValueError, IndexError):
                            pass
    else:
        try:
            return result["stages"]["article-search"]["tasks"]["Find Digital Article"][
                "outputs"
            ].get("confidence_score", 0)
        except (KeyError, TypeError):
            pass
    return 0


def get_digital_headline(result: Union[ExecutionResult, Dict[str, Any]]) -> str:
    """Extract the digital headline from the result."""
    if isinstance(result, ExecutionResult):
        if result.agent_outputs and len(result.agent_outputs) > 0:
            agent_output = result.agent_outputs[0].output
            if isinstance(agent_output, dict):
                return agent_output.get("digital_headline", "")
            elif isinstance(agent_output, str):
                # Try to extract headline from text format
                lines = agent_output.strip().split("\n")
                for i, line in enumerate(lines):
                    if "headline" in line.lower() or "###" in line:
                        if i + 1 < len(lines) and lines[i + 1].strip():
                            return lines[i + 1].strip()
    else:
        try:
            return result["stages"]["article-search"]["tasks"]["Find Digital Article"][
                "outputs"
            ].get("digital_headline", "")
        except (KeyError, TypeError):
            pass

    # If we get here, try to extract from final_result if it's a string
    if isinstance(result, ExecutionResult) and isinstance(result.final_result, str):
        lines = result.final_result.strip().split("\n")
        for i, line in enumerate(lines):
            if "headline" in line.lower() or "###" in line:
                if i + 1 < len(lines) and lines[i + 1].strip():
                    return lines[i + 1].strip()

    return ""


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find digital versions of WSJ print articles"
    )

    parser.add_argument(
        "--headline", type=str, required=False, help="Headline of the print article"
    )

    parser.add_argument(
        "--date", type=str, required=False, help="Publication date (YYYY-MM-DD)"
    )

    parser.add_argument("--excerpt", type=str, help="Excerpt from the print article")

    parser.add_argument("--author", type=str, help="Author name (optional)")

    parser.add_argument(
        "--engine",
        "-e",
        type=str,
        default="openai",
        choices=["openai"],
        help="Execution engine type (openai)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Override model name (e.g., gpt-4o, gpt-4-turbo, bedrock_sonnet_3.7, google_gemini)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Set model override environment variable if specified
    if args.model:
        os.environ["MODEL_OVERRIDE"] = args.model
        print(f"Using model override: {args.model}")

    # Use provided or default headline
    headline = args.headline or "Hungary to Withdraw From ICC as Netanyahu Visits"
    date = args.date or "04022025"

    try:
        asyncio.run(
            run_wsj_article_workflow(
                print_headline=headline,
                print_date=date,
                print_excerpt=args.excerpt,
                author=args.author,
                engine_type=args.engine,
                model_name=args.model,
            )
        )
    except ValueError as e:
        logger.error(f"Error: {e}")
        if "API key not found" in str(e) or "No LLM providers are configured" in str(e):
            logger.error("\nSetup instructions:")
            logger.error("For OpenAI: export OPENAI_API_KEY=your_api_key_here")
            logger.error("For AWS Bedrock: export AWS_ACCESS_KEY_ID=your_key_id")
            logger.error(
                "                 export AWS_SECRET_ACCESS_KEY=your_secret_key"
            )
            logger.error(
                "                 export AWS_REGION_NAME=your_region (default: us-east-1)"
            )
            logger.error("For Gemini: export GEMINI_API_KEY=your_api_key")
            logger.error("            export GCP_PROJECT_ID=your_project_id (optional)")
            logger.error("            export GCP_LOCATION=your_location (optional)")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
