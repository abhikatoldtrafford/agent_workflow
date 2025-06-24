"""
Journalism Workflow Runner

This script runs a workflow that helps journalists build raw material for stories including:
- Finding relevant images
- Creating headlines
- Generating SEO tags
- Finding related articles from RSS feeds
- Creating a story summary

Usage:
    python run_journalism_workflow.py [--topic "topic"] [--keywords "keyword1,keyword2"]
"""

import argparse
import asyncio
import logging
import os
from typing import List, Optional

from agent_workflow.providers.callbacks import ConsoleProgressCallback
from agent_workflow.providers.observability import configure_observability
from agent_workflow.workflow_engine import (
    ExecutionResult,
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


def setup_observability():
    """Configure observability based on environment variables."""
    return configure_observability(
        provider_type="langfuse",
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY_AGENT_WF"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY_AGENT_WF"),
        host=os.environ.get("LANGFUSE_HOST"),
    )


def print_workflow_results(result: ExecutionResult):
    """
    Print the journalism workflow results in a formatted way.

    Args:
        result: The ExecutionResult from the workflow execution
    """
    agent_outputs = {}

    # Collect outputs by agent name
    for agent_output in result.agent_outputs:
        agent_outputs[agent_output.agent] = agent_output.output

    # Print headline
    if "Headline Agent" in agent_outputs:
        print("\n📰 HEADLINES:")
        headlines = agent_outputs["Headline Agent"]
        if isinstance(headlines, dict) and "headlines" in headlines:
            for i, headline in enumerate(headlines["headlines"], 1):
                print(f"  {i}. {headline}")
        else:
            print(f"  {headlines}")

    # Print summary
    if "Summary Agent" in agent_outputs:
        print("\n📝 SUMMARY:")
        summary = agent_outputs["Summary Agent"]
        if isinstance(summary, dict) and "summary" in summary:
            print(f"  {summary['summary']}")
        else:
            print(f"  {summary}")

    # Print SEO tags
    if "SEO Tag Agent" in agent_outputs:
        print("\n🔍 SEO TAGS:")
        seo_tags = agent_outputs["SEO Tag Agent"]
        if isinstance(seo_tags, dict) and "tags" in seo_tags:
            print("  " + ", ".join(seo_tags["tags"]))
        else:
            print(f"  {seo_tags}")

    # Print image suggestions
    if "Image Search Agent" in agent_outputs:
        print("\n🖼️ SUGGESTED IMAGES:")
        images = agent_outputs["Image Search Agent"]
        if isinstance(images, dict) and "images" in images:
            for i, image in enumerate(images["images"], 1):
                print(f"  {i}. {image['description']}")
                if "query" in image:
                    print(f"     Search query: {image['query']}")
        else:
            print(f"  {images}")

    # Print related articles
    if "Related Articles Agent" in agent_outputs:
        print("\n📚 RELATED ARTICLES:")
        articles = agent_outputs["Related Articles Agent"]
        if isinstance(articles, dict) and "articles" in articles:
            for i, article in enumerate(articles["articles"], 1):
                print(f"  {i}. {article['title']}")
                if "source" in article and article["source"]:
                    print(f"     Source: {article['source']}")
        else:
            print(f"  {articles}")

    print("\n============================================================")


async def run_journalism_workflow(
    story_topic: str,
    keywords: List[str],
    audience: str = "general public",
    tone: str = "informative",
    image_count: int = 3,
    article_count: int = 5,
    rss_feeds: Optional[List[str]] = None,
    engine_type: str = "openai",  # Default to OpenAI engine, can be overridden
):
    """Run the journalism workflow to build raw material for a story."""

    # Configure observability
    observability_provider = setup_observability()

    # Start a workflow trace
    workflow_trace_id = observability_provider.trace_workflow(
        workflow_id=f"journalism_workflow_{engine_type}",
        metadata={
            "workflow_name": "Journalism Story Preparation",
            "version": "1.0.0",
            "engine_type": engine_type,
            "story_topic": story_topic,
        },
    )

    # Default RSS feeds if none provided
    if rss_feeds is None:
        rss_feeds = [
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
            "https://feeds.washingtonpost.com/rss/world",
            "https://www.theguardian.com/world/rss",
            "https://www.wsj.com/news/rss-news-and-feeds",
        ]

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
            model="gemini-1.5-flash-002",
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

    # standard_model = "bedrock_sonnet_3.5" if "bedrock_sonnet_3.5" in provider_keys else large_model

    standard_model = "openai_gpt4"
    large_model = "google_gemini"

    # Provider mapping for agents - this makes agent YAML files engine-agnostic
    provider_mapping = {
        # Map agent IDs to provider types
        "image-search-agent": large_model,  # Use standard model for image search
        "headline-agent": large_model,  # Use standard model for headlines
        "seo-tag-agent": large_model,  # Use standard model for SEO tags
        "related-articles-agent": large_model,  # Use standard model for related articles
        "summary-agent": large_model,  # Use large model for the summary (more complex task)
    }

    # Create structured workflow inputs
    workflow_inputs = WorkflowInput(
        provider_config=providers_config,
        provider_mapping=provider_mapping,
        workflow={
            "inputs": {
                "story_topic": story_topic,
                "keywords": keywords,
                "audience": audience,
                "tone": tone,
                "image_count": image_count,
                "article_count": article_count,
                "rss_feeds": rss_feeds,
            }
        },
    )

    # Execute the workflow
    try:
        # Create a span for workflow execution
        span_id = observability_provider.trace_span(
            name="workflow_execution",
            parent_id=workflow_trace_id,
            metadata={"workflow_file": "journalism_workflow.yaml"},
        )

        # Create console progress callback
        progress_callback = ConsoleProgressCallback()

        # First initialize the workflow
        workflow = await manager.initialize_workflow(
            os.path.join(os.path.dirname(__file__), "journalism_workflow.yaml"),
            provider_mapping=provider_mapping,
            progress_callback=progress_callback,
        )
        
        # Then execute the initialized workflow
        result = await manager.execute(
            workflow,
            workflow_inputs,
            progress_callback=progress_callback,
        )

        # End the execution span with success status
        observability_provider.end_span(span_id=span_id, metadata={"status": "success"})

        # Print journalism story package with results
        print("\n============== Journalism Story Package Completed! ==============")
        print_workflow_results(result)

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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run journalism workflow for story preparation"
    )

    parser.add_argument("--topic", type=str, help="Main topic of the news story")

    parser.add_argument(
        "--keywords", type=str, help="Comma-separated keywords related to the story"
    )

    parser.add_argument(
        "--audience",
        type=str,
        default="general public",
        help="Target audience for the story",
    )

    parser.add_argument(
        "--tone",
        type=str,
        default="informative",
        help="Tone for the story (e.g., informative, investigative, human interest)",
    )

    parser.add_argument(
        "--image-count", type=int, default=3, help="Number of images to find"
    )

    parser.add_argument(
        "--article-count",
        type=int,
        default=5,
        help="Number of related articles to find",
    )

    parser.add_argument(
        "--engine",
        "-e",
        type=str,
        default="openai",
        choices=["openai"],
        help="Execution engine type (openai)",
    )

    # Add model override option
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Override model for all agents (e.g., bedrock_sonnet_3.7, openai_gpt4, google_gemini)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Set model override environment variable if specified
    if args.model:
        os.environ["MODEL_OVERRIDE"] = args.model
        print(f"Using model override: {args.model}")

    # Use default topic and keywords if not provided
    story_topic = args.topic or "The impact of artificial intelligence on journalism"
    keywords_raw = (
        args.keywords
        or "AI, journalism, news automation, media transformation, technology"
    )
    keywords = [k.strip() for k in keywords_raw.split(",")]

    try:
        asyncio.run(
            run_journalism_workflow(
                story_topic=story_topic,
                keywords=keywords,
                audience=args.audience,
                tone=args.tone,
                image_count=args.image_count,
                article_count=args.article_count,
                engine_type="openai",
            )
        )
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
            print("                 export GCP_PROJECT_ID=your_project_id (optional)")
            print("                 export GCP_LOCATION=your_location (optional)")
            print(
                "\nTo override the model: --model bedrock_sonnet_3.7 or --model google_gemini"
            )
    except Exception as e:
        print(f"Unexpected error: {e}")
