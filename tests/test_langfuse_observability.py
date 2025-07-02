"""
Test Langfuse Observability Integration.

Tests the integration of observability providers with the Langfuse execution engine,
including tracing of workflows, stages, tasks, agents, and tool calls.
"""
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock
import time
import logging
import random

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agent_workflow.parsers import YAMLParser
from agent_workflow.workflow_engine import OpenAIProviderConfig, ProviderConfiguration, ProviderType
from agent_workflow.providers.callbacks import ConsoleProgressCallback
# Import Langfuse provider from the correct location
from usage_examples.tool_usage.langfuse_observability import LangfuseLLMObservabilityProvider
# Import TraceStatus for error handling
from agent_workflow.providers.llm_observability import TraceStatus
from agent_workflow.providers.tools import ToolRegistry, tool, register_tool, Tool, global_tool_registry
from agent_workflow.workflow_engine import WorkflowManager
from agent_workflow.workflow_engine.models import WorkflowInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Module-level cleanup
# ============================================================================

def pytest_sessionfinish(session, exitstatus):
    """Clean up after all tests are done."""
    # Langfuse doesn't require the same cleanup as OpenAI
    try:
        import time
        time.sleep(0.5)
    except:
        pass


# ============================================================================
# Helper Functions
# ============================================================================

def get_provider_mapping_for_workflow(workflow_dict: Dict[str, Any]) -> Dict[str, str]:
    """Extract all agent IDs from a workflow and create a provider mapping."""
    agent_ids = set()
    for stage in workflow_dict.get("stages", []):
        for task in stage.get("tasks", []):
            if "agent" in task and "id" in task["agent"]:
                agent_ids.add(task["agent"]["id"])
    # Map all agents to openai provider
    return {agent_id: "openai" for agent_id in agent_ids}


# ============================================================================
# Mock Tools for Testing
# ============================================================================

@tool(
    name="test_weather",
    description="Get weather information for testing"
)
def mock_weather(location: str) -> str:
    """Mock weather tool for testing."""
    # Simulate some processing time
    time.sleep(0.1)
    temps = {"New York": 72, "London": 65, "Tokyo": 78, "Sydney": 82}
    temp = temps.get(location, 70)
    return f"Weather in {location}: Sunny, {temp}°F"

@tool(
    name="test_calculator",
    description="Perform calculations for testing"
)
def mock_calculator(expression: str) -> float:
    """Mock calculator tool for testing."""
    # Simulate some processing time
    time.sleep(0.05)
    try:
        # Check for division by zero
        if "/0" in expression.replace(" ", ""):
            raise ZeroDivisionError("Division by zero")
        return eval(expression)
    except Exception as e:
        raise ValueError(f"Calculation error: {str(e)}")

@tool(
    name="test_database_query",
    description="Query mock database"
)
def mock_database_query(query: str, table: str) -> Dict[str, Any]:
    """Mock database query tool."""
    time.sleep(0.2)  # Simulate DB latency
    return {
        "query": query,
        "table": table,
        "results": [
            {"id": 1, "name": "Item 1", "value": 100},
            {"id": 2, "name": "Item 2", "value": 200}
        ],
        "count": 2
    }

@tool(
    name="test_api_call",
    description="Make mock API calls"
)
def mock_api_call(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
    """Mock API call tool."""
    time.sleep(0.15)  # Simulate network latency
    return {
        "endpoint": endpoint,
        "method": method,
        "status": 200,
        "response": {"message": "Success", "data": data or {}}
    }

@tool(
    name="test_file_processor",
    description="Process files"
)
def mock_file_processor(filename: str, operation: str) -> Dict[str, Any]:
    """Mock file processing tool."""
    time.sleep(0.1)
    operations = ["read", "write", "analyze", "compress"]
    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")
    return {
        "filename": filename,
        "operation": operation,
        "status": "completed",
        "size_kb": random.randint(10, 1000)
    }

@tool(
    name="test_sentiment_analyzer",
    description="Analyze sentiment of text"
)
def mock_sentiment_analyzer(text: str) -> Dict[str, Any]:
    """Mock sentiment analysis tool."""
    time.sleep(0.05)
    # Simple mock sentiment based on keywords
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "hate", "awful", "horrible"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = "positive"
        score = 0.7 + (pos_count * 0.1)
    elif neg_count > pos_count:
        sentiment = "negative"
        score = 0.3 - (neg_count * 0.1)
    else:
        sentiment = "neutral"
        score = 0.5
    
    return {
        "text": text[:50] + "..." if len(text) > 50 else text,
        "sentiment": sentiment,
        "score": max(0, min(1, score)),
        "confidence": 0.85
    }

class TestDataAnalysisTool(Tool):
    """Test tool for data analysis operations."""
    
    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "test_data_analysis"
    
    @property
    def description(self) -> str:
        return "Analyze data for testing"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "Data to analyze"},
                "operation": {"type": "string", "description": "Analysis operation"}
            },
            "required": ["data", "operation"]
        }
    
    @property
    def type(self) -> str:
        return "functional"
    
    async def execute(self, data: str, operation: str) -> Dict[str, Any]:
        """Perform mock data analysis."""
        await asyncio.sleep(0.1)  # Simulate async processing
        return {
            "result": f"Analysis of {data} using {operation}",
            "confidence": 0.95,
            "metrics": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.91
            }
        }

class TestTranslationTool(Tool):
    """Test tool for translation operations."""
    
    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "test_translator"
    
    @property
    def description(self) -> str:
        return "Translate text between languages"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "source_lang": {"type": "string", "description": "Source language"},
                "target_lang": {"type": "string", "description": "Target language"}
            },
            "required": ["text", "target_lang"]
        }
    
    @property
    def type(self) -> str:
        return "functional"
    
    async def execute(self, text: str, target_lang: str, source_lang: str = "en") -> Dict[str, Any]:
        """Perform mock translation."""
        await asyncio.sleep(0.15)  # Simulate translation time
        # Mock translation by adding language prefix
        translations = {
            "es": "ES: " + text,
            "fr": "FR: " + text,
            "de": "DE: " + text,
            "ja": "JA: " + text,
            "zh": "ZH: " + text
        }
        return {
            "original": text,
            "translated": translations.get(target_lang, f"{target_lang.upper()}: {text}"),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "confidence": 0.93
        }

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_tracing(request):
    """Ensure Langfuse tracing is properly cleaned up after tests."""
    def finalizer():
        try:
            # Langfuse client cleanup if needed
            import time
            time.sleep(0.5)
        except Exception:
            pass
    
    request.addfinalizer(finalizer)


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key


@pytest.fixture
def langfuse_credentials():
    """Get Langfuse credentials from environment."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "https://langfuse-v3.ncgt.mpp-kwatee.com")
    
    # Remove quotes if present in host
    if host and (host.startswith('"') and host.endswith('"')):
        host = host[1:-1]
    
    # Debug print to verify credentials are loaded
    logger.info(f"Langfuse credentials loaded:")
    logger.info(f"  Public Key: {public_key[:10]}..." if public_key else "  Public Key: NOT FOUND")
    logger.info(f"  Secret Key: {secret_key[:10]}..." if secret_key else "  Secret Key: NOT FOUND")
    logger.info(f"  Host: {host}")
    
    if not public_key or not secret_key:
        pytest.skip("LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set")
    
    return {
        "public_key": public_key,
        "secret_key": secret_key,
        "host": host
    }


@pytest.fixture
def observability_provider(langfuse_credentials):
    """Create Langfuse observability provider."""
    provider = LangfuseLLMObservabilityProvider(
        public_key=langfuse_credentials["public_key"],
        secret_key=langfuse_credentials["secret_key"],
        host=langfuse_credentials["host"]
    )
    yield provider
    # Ensure all traces are flushed before test ends
    try:
        provider.client.flush()
        time.sleep(1)  # Give it a moment to send
    except Exception as e:
        logger.warning(f"Error flushing Langfuse client: {e}")


@pytest.fixture
def tool_registry():
    """Create and populate tool registry."""
    # Use the global tool registry which already has tools registered by @tool decorator
    from agent_workflow.providers.tools import global_tool_registry
    
    # Register class-based tools
    register_tool(TestDataAnalysisTool())
    register_tool(TestTranslationTool())
    
    return global_tool_registry


@pytest.fixture
def provider_config(openai_api_key):
    """Create provider configuration."""
    provider_configs = {
        "openai": OpenAIProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key=openai_api_key,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            enforce_structured_output=False
        )
    }
    return ProviderConfiguration(providers=provider_configs)


@pytest.fixture
def workflow_manager(provider_config, tool_registry, observability_provider):
    """Create workflow manager with OpenAI engine."""
    # Note: This is a regular fixture, not async
    manager = WorkflowManager(
        engine_type="openai",
        provider_config=provider_config,
        tool_registry=tool_registry,
        llm_observability_provider=observability_provider
    )
    yield manager
    # Cleanup: Ensure Langfuse flushes all data
    try:
        observability_provider.client.flush()
        time.sleep(1)
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


# ============================================================================
# Complex Test Workflows
# ============================================================================

def create_simple_sequential_workflow() -> Dict[str, Any]:
    """Create a simple sequential workflow."""
    return {
        "name": "test_simple_sequential",
        "description": "Test sequential execution with tracing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "Analysis",
                "description": "Analyze user request",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "Analyze",
                        "description": "Analyze the request",
                        "agent": {
                            "id": "analyzer",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "You are an analyst. Analyze the request briefly.",
                            "user_prompt": "Analyze: ${workflow.inputs.user_query}"
                        },
                        "outputs": {"analysis": "Analysis complete"}
                    }
                ]
            }
        ]
    }


def create_tool_usage_workflow() -> Dict[str, Any]:
    """Create workflow that uses tools."""
    return {
        "name": "test_tool_usage",
        "description": "Test tool execution tracing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "ToolUsage",
                "description": "Use various tools",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "UseTools",
                        "description": "Execute tools",
                        "agent": {
                            "id": "tool_user",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [
                                {"name": "test_weather", "type": "function"},
                                {"name": "test_calculator", "type": "function"}
                            ],
                            "system_prompt": "You are a helpful assistant. Use the weather tool to check weather in London, then use the calculator to add 10+20.",
                            "user_prompt": "${workflow.inputs.user_query}"
                        },
                        "outputs": {"result": "Tool execution complete"}
                    }
                ]
            }
        ]
    }


def create_complex_handoff_workflow() -> Dict[str, Any]:
    """Create complex workflow with multiple handoff patterns."""
    return {
        "name": "test_complex_handoff",
        "description": "Complex multi-level handoff routing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "InitialRouting",
                "description": "Initial query classification and routing",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "QueryClassifier",
                        "description": "Classify the query type",
                        "agent": {
                            "id": "classifier_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": """You are a query classifier. Classify queries into categories:
                            - technical: programming, software, IT issues
                            - financial: billing, pricing, payments
                            - support: general help, account issues
                            - sales: product info, purchases
                            Output only the category name.""",
                            "user_prompt": "Classify this query: ${workflow.inputs.query}"
                        },
                        "outputs": {"category": "Query category"}
                    }
                ]
            },
            {
                "name": "PrimaryHandoff",
                "description": "Route to primary specialist",
                "execution_type": "handoff",
                "tasks": [
                    {
                        "name": "TechnicalSpecialist",
                        "description": "Handle technical queries",
                        "agent": {
                            "id": "tech_specialist",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [
                                {"name": "test_database_query", "type": "function"},
                                {"name": "test_api_call", "type": "function"}
                            ],
                            "system_prompt": "You are a technical specialist. Use tools to diagnose technical issues.",
                            "user_prompt": "Technical query: ${workflow.inputs.query}"
                        },
                        "outputs": {"tech_response": "Technical analysis"}
                    },
                    {
                        "name": "FinancialSpecialist",
                        "description": "Handle financial queries",
                        "agent": {
                            "id": "financial_specialist",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [
                                {"name": "test_calculator", "type": "function"},
                                {"name": "test_database_query", "type": "function"}
                            ],
                            "system_prompt": "You are a financial specialist. Calculate costs and check billing.",
                            "user_prompt": "Financial query: ${workflow.inputs.query}"
                        },
                        "outputs": {"financial_response": "Financial analysis"}
                    },
                    {
                        "name": "SupportSpecialist",
                        "description": "Handle support queries",
                        "agent": {
                            "id": "support_specialist",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_sentiment_analyzer", "type": "function"}],
                            "system_prompt": "You are a support specialist. Provide helpful support.",
                            "user_prompt": "Support query: ${workflow.inputs.query}"
                        },
                        "outputs": {"support_response": "Support response"}
                    },
                    {
                        "name": "SalesSpecialist",
                        "description": "Handle sales queries",
                        "agent": {
                            "id": "sales_specialist",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "You are a sales specialist. Provide product information.",
                            "user_prompt": "Sales query: ${workflow.inputs.query}"
                        },
                        "outputs": {"sales_response": "Sales information"}
                    }
                ]
            },
            {
                "name": "SecondaryHandoff",
                "description": "Escalation or specialized handling",
                "execution_type": "handoff",
                "tasks": [
                    {
                        "name": "SeniorTechnical",
                        "description": "Senior technical escalation",
                        "agent": {
                            "id": "senior_tech",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [
                                {"name": "test_data_analysis", "type": "functional"},
                                {"name": "test_file_processor", "type": "function"}
                            ],
                            "system_prompt": "Senior tech expert. Analyze complex technical issues.",
                            "user_prompt": "Escalated: ${stages.[PrimaryHandoff].outputs}"
                        },
                        "outputs": {"senior_tech_resolution": "Advanced resolution"}
                    },
                    {
                        "name": "AccountManager",
                        "description": "Account management escalation",
                        "agent": {
                            "id": "account_manager",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Account manager. Handle VIP and complex account issues.",
                            "user_prompt": "Account issue: ${stages.[PrimaryHandoff].outputs}"
                        },
                        "outputs": {"account_resolution": "Account resolution"}
                    }
                ]
            }
        ]
    }


def create_massive_parallel_workflow() -> Dict[str, Any]:
    """Create workflow with many parallel agents to stress test tracing."""
    parallel_tasks = []
    for i in range(10):  # 10 parallel agents
        parallel_tasks.append({
            "name": f"ParallelAgent{i}",
            "description": f"Parallel processing agent {i}",
            "agent": {
                "id": f"parallel_agent_{i}",
                "agent_type": "LLMAgent",
                "llm_type": "openai",
                "tools": [
                    {"name": "test_calculator", "type": "function"},
                    {"name": "test_weather", "type": "function"}
                ] if i % 2 == 0 else [],  # Half the agents have tools
                "system_prompt": f"You are agent {i}. Process data segment {i}.",
                "user_prompt": f"Process segment {i}: ${{workflow.inputs.data_segment_{i}}}"
            },
            "outputs": {f"result_{i}": f"Processing result {i}"}
        })
    
    return {
        "name": "test_massive_parallel",
        "description": "Massive parallel execution test",
        "version": "1.0.0",
        "stages": [
            {
                "name": "DataPreparation",
                "description": "Prepare data for parallel processing",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "DataSplitter",
                        "description": "Split data into segments",
                        "agent": {
                            "id": "data_splitter",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Split the input data into 10 segments for parallel processing.",
                            "user_prompt": "Split this data: ${workflow.inputs.large_dataset}"
                        },
                        "outputs": {"segments": "Data segments created"}
                    }
                ]
            },
            {
                "name": "ParallelProcessing",
                "description": "Process all segments in parallel",
                "execution_type": "parallel",
                "tasks": parallel_tasks
            },
            {
                "name": "ResultAggregation",
                "description": "Aggregate all parallel results",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "Aggregator",
                        "description": "Aggregate all results",
                        "agent": {
                            "id": "aggregator_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_data_analysis", "type": "functional"}],
                            "system_prompt": "Aggregate all parallel processing results into a final report.",
                            "user_prompt": "Aggregate results from all parallel agents"
                        },
                        "outputs": {"final_report": "Aggregated results"}
                    }
                ]
            }
        ]
    }


def create_multi_tool_orchestration_workflow() -> Dict[str, Any]:
    """Create workflow where multiple agents coordinate tool usage."""
    return {
        "name": "test_multi_tool_orchestration",
        "description": "Complex multi-agent tool orchestration",
        "version": "1.0.0",
        "stages": [
            {
                "name": "DataCollection",
                "description": "Collect data from multiple sources",
                "execution_type": "parallel",
                "tasks": [
                    {
                        "name": "WeatherCollector",
                        "description": "Collect weather data",
                        "agent": {
                            "id": "weather_collector",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_weather", "type": "function"}],
                            "system_prompt": "Collect weather data for New York, London, Tokyo, and Sydney.",
                            "user_prompt": "Get weather for major cities"
                        },
                        "outputs": {"weather_data": "Weather information"}
                    },
                    {
                        "name": "DatabaseCollector",
                        "description": "Collect database data",
                        "agent": {
                            "id": "db_collector",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_database_query", "type": "function"}],
                            "system_prompt": "Query user data and transaction data from database.",
                            "user_prompt": "Get user and transaction data"
                        },
                        "outputs": {"db_data": "Database results"}
                    },
                    {
                        "name": "APICollector",
                        "description": "Collect API data",
                        "agent": {
                            "id": "api_collector",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_api_call", "type": "function"}],
                            "system_prompt": "Call multiple APIs to gather system status.",
                            "user_prompt": "Check system status from APIs"
                        },
                        "outputs": {"api_data": "API results"}
                    }
                ]
            },
            {
                "name": "DataProcessing",
                "description": "Process collected data",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "DataAnalyzer",
                        "description": "Analyze all collected data",
                        "agent": {
                            "id": "data_analyzer",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [
                                {"name": "test_data_analysis", "type": "functional"},
                                {"name": "test_calculator", "type": "function"}
                            ],
                            "system_prompt": "Analyze the collected data and calculate statistics.",
                            "user_prompt": "Analyze: ${stages.[DataCollection].outputs}"
                        },
                        "outputs": {"analysis": "Data analysis results"}
                    },
                    {
                        "name": "SentimentAnalyzer",
                        "description": "Analyze sentiment of results",
                        "agent": {
                            "id": "sentiment_analyzer_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_sentiment_analyzer", "type": "function"}],
                            "system_prompt": "Analyze sentiment of the analysis results.",
                            "user_prompt": "Check sentiment: ${stages.[DataProcessing].tasks.[DataAnalyzer].outputs.analysis}"
                        },
                        "outputs": {"sentiment": "Sentiment analysis"}
                    }
                ]
            },
            {
                "name": "ReportGeneration",
                "description": "Generate final reports",
                "execution_type": "parallel",
                "tasks": [
                    {
                        "name": "TechnicalReporter",
                        "description": "Generate technical report",
                        "agent": {
                            "id": "tech_reporter",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_file_processor", "type": "function"}],
                            "system_prompt": "Generate technical report from analysis.",
                            "user_prompt": "Create tech report: ${stages.[DataProcessing].outputs}"
                        },
                        "outputs": {"tech_report": "Technical report"}
                    },
                    {
                        "name": "ExecutiveReporter",
                        "description": "Generate executive summary",
                        "agent": {
                            "id": "exec_reporter",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Generate executive summary.",
                            "user_prompt": "Create summary: ${stages.[DataProcessing].outputs}"
                        },
                        "outputs": {"exec_summary": "Executive summary"}
                    }
                ]
            }
        ]
    }


def create_conditional_routing_workflow() -> Dict[str, Any]:
    """Create workflow with complex conditional routing between agents."""
    return {
        "name": "test_conditional_routing",
        "description": "Complex conditional routing patterns",
        "version": "1.0.0",
        "stages": [
            {
                "name": "InitialAssessment",
                "description": "Initial assessment and scoring",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "RiskAssessor",
                        "description": "Assess risk level",
                        "agent": {
                            "id": "risk_assessor",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_calculator", "type": "function"}],
                            "system_prompt": "Assess risk level (high/medium/low) and calculate risk score.",
                            "user_prompt": "Assess risk for: ${workflow.inputs.scenario}"
                        },
                        "outputs": {"risk_level": "Risk assessment", "risk_score": "Risk score"}
                    }
                ]
            },
            {
                "name": "RiskBasedRouting",
                "description": "Route based on risk level",
                "execution_type": "handoff",
                "tasks": [
                    {
                        "name": "HighRiskHandler",
                        "description": "Handle high risk cases",
                        "agent": {
                            "id": "high_risk_handler",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [
                                {"name": "test_api_call", "type": "function"},
                                {"name": "test_database_query", "type": "function"}
                            ],
                            "system_prompt": "Handle HIGH risk cases with immediate action.",
                            "user_prompt": "High risk case: ${workflow.inputs.scenario}"
                        },
                        "outputs": {"high_risk_action": "Action taken"}
                    },
                    {
                        "name": "MediumRiskHandler",
                        "description": "Handle medium risk cases",
                        "agent": {
                            "id": "medium_risk_handler",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_data_analysis", "type": "functional"}],
                            "system_prompt": "Handle MEDIUM risk cases with analysis.",
                            "user_prompt": "Medium risk case: ${workflow.inputs.scenario}"
                        },
                        "outputs": {"medium_risk_action": "Action taken"}
                    },
                    {
                        "name": "LowRiskHandler",
                        "description": "Handle low risk cases",
                        "agent": {
                            "id": "low_risk_handler",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Handle LOW risk cases with standard procedure.",
                            "user_prompt": "Low risk case: ${workflow.inputs.scenario}"
                        },
                        "outputs": {"low_risk_action": "Action taken"}
                    }
                ]
            },
            {
                "name": "ComplianceCheck",
                "description": "Verify compliance",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "ComplianceVerifier",
                        "description": "Verify all actions meet compliance",
                        "agent": {
                            "id": "compliance_verifier",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_file_processor", "type": "function"}],
                            "system_prompt": "Verify compliance of all actions taken.",
                            "user_prompt": "Check compliance: ${stages.[RiskBasedRouting].outputs}"
                        },
                        "outputs": {"compliance_status": "Compliance check result"}
                    }
                ]
            }
        ]
    }


def create_recursive_handoff_workflow() -> Dict[str, Any]:
    """Create workflow with recursive handoff patterns."""
    return {
        "name": "test_recursive_handoff",
        "description": "Recursive handoff between specialists",
        "version": "1.0.0",
        "stages": [
            {
                "name": "InitialHandoff",
                "description": "Initial specialist selection",
                "execution_type": "handoff",
                "tasks": [
                    {
                        "name": "LanguageDetector",
                        "description": "Detect language and route",
                        "agent": {
                            "id": "language_detector",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Detect language of query. If English proceed, else needs translation.",
                            "user_prompt": "Detect language: ${workflow.inputs.multilingual_query}"
                        },
                        "outputs": {"language": "Detected language", "needs_translation": "true/false"}
                    },
                    {
                        "name": "DirectProcessor",
                        "description": "Process English queries directly",
                        "agent": {
                            "id": "direct_processor",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Process English queries directly.",
                            "user_prompt": "Process: ${workflow.inputs.multilingual_query}"
                        },
                        "outputs": {"processed": "Direct processing result"}
                    }
                ]
            },
            {
                "name": "TranslationHandoff",
                "description": "Translation if needed",
                "execution_type": "handoff",
                "tasks": [
                    {
                        "name": "Translator",
                        "description": "Translate non-English queries",
                        "agent": {
                            "id": "translator_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_translator", "type": "functional"}],
                            "system_prompt": "Translate query to English.",
                            "user_prompt": "Translate: ${workflow.inputs.multilingual_query}"
                        },
                        "outputs": {"translated_query": "Translated text"}
                    },
                    {
                        "name": "PostTranslationProcessor",
                        "description": "Process after translation",
                        "agent": {
                            "id": "post_translation_processor",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Process translated query.",
                            "user_prompt": "Process: ${stages.[TranslationHandoff].tasks.[Translator].outputs.translated_query}"
                        },
                        "outputs": {"processed_translation": "Processed result"}
                    }
                ]
            },
            {
                "name": "FinalHandoff",
                "description": "Final processing based on content",
                "execution_type": "handoff",
                "tasks": [
                    {
                        "name": "ContentAnalyzer",
                        "description": "Analyze processed content",
                        "agent": {
                            "id": "content_analyzer",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_sentiment_analyzer", "type": "function"}],
                            "system_prompt": "Analyze content sentiment and determine response type.",
                            "user_prompt": "Analyze: ${stages.[TranslationHandoff].outputs}"
                        },
                        "outputs": {"content_analysis": "Content analysis"}
                    },
                    {
                        "name": "PositiveResponseHandler",
                        "description": "Handle positive sentiment",
                        "agent": {
                            "id": "positive_handler",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Generate enthusiastic positive response.",
                            "user_prompt": "Respond positively to: ${stages.[FinalHandoff].tasks.[ContentAnalyzer].outputs}"
                        },
                        "outputs": {"positive_response": "Positive response"}
                    },
                    {
                        "name": "NegativeResponseHandler",
                        "description": "Handle negative sentiment",
                        "agent": {
                            "id": "negative_handler",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Generate empathetic response for negative sentiment.",
                            "user_prompt": "Respond to negative: ${stages.[FinalHandoff].tasks.[ContentAnalyzer].outputs}"
                        },
                        "outputs": {"negative_response": "Negative response"}
                    }
                ]
            }
        ]
    }


def create_error_cascade_workflow() -> Dict[str, Any]:
    """Create workflow to test error propagation in complex scenarios."""
    return {
        "name": "test_error_cascade",
        "description": "Test error handling in cascading failures",
        "version": "1.0.0",
        "stages": [
            {
                "name": "InitialProcessing",
                "description": "Initial processing that might fail",
                "execution_type": "parallel",
                "tasks": [
                    {
                        "name": "SuccessfulTask",
                        "description": "Task that succeeds",
                        "agent": {
                            "id": "success_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_calculator", "type": "function"}],
                            "system_prompt": "Calculate 100 + 200",
                            "user_prompt": "Add numbers"
                        },
                        "outputs": {"success_result": "Calculation result"}
                    },
                    {
                        "name": "FailingTask",
                        "description": "Task that fails",
                        "agent": {
                            "id": "failing_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_calculator", "type": "function"}],
                            "system_prompt": "You must calculate 100/0 to test error handling.",
                            "user_prompt": "Divide by zero"
                        },
                        "outputs": {"fail_result": "Should fail"}
                    },
                    {
                        "name": "AnotherSuccessfulTask",
                        "description": "Another task that succeeds",
                        "agent": {
                            "id": "another_success_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Process data successfully",
                            "user_prompt": "Process: ${workflow.inputs.data}"
                        },
                        "outputs": {"another_success": "Processing complete"}
                    }
                ]
            },
            {
                "name": "ErrorRecovery",
                "description": "Attempt to recover from errors",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "ErrorHandler",
                        "description": "Handle errors from previous stage",
                        "agent": {
                            "id": "error_handler",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_api_call", "type": "function"}],
                            "system_prompt": "Check for errors and attempt recovery.",
                            "user_prompt": "Check status and recover from: ${stages.[InitialProcessing].outputs}"
                        },
                        "outputs": {"recovery_status": "Recovery attempt result"}
                    }
                ]
            },
            {
                "name": "FinalProcessing",
                "description": "Final processing with partial results",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "ResultAggregator",
                        "description": "Aggregate available results",
                        "agent": {
                            "id": "result_aggregator",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_data_analysis", "type": "functional"}],
                            "system_prompt": "Aggregate all available results, handling missing data gracefully.",
                            "user_prompt": "Aggregate partial results from all stages"
                        },
                        "outputs": {"final_aggregation": "Final results with error handling"}
                    }
                ]
            }
        ]
    }


def create_performance_stress_workflow() -> Dict[str, Any]:
    """Create workflow to stress test performance with many operations."""
    # Create multiple parallel stages with different patterns
    stages = []
    
    # Stage 1: Heavy parallel processing
    parallel_tasks = []
    for i in range(5):
        parallel_tasks.append({
            "name": f"HeavyProcessor{i}",
            "description": f"Heavy processing task {i}",
            "agent": {
                "id": f"heavy_processor_{i}",
                "agent_type": "LLMAgent",
                "llm_type": "openai",
                "tools": [
                    {"name": "test_calculator", "type": "function"},
                    {"name": "test_data_analysis", "type": "functional"},
                    {"name": "test_file_processor", "type": "function"}
                ],
                "system_prompt": f"Process complex calculations and data analysis for segment {i}.",
                "user_prompt": f"Process heavy workload {i}: ${{workflow.inputs.workload_{i}}}"
            },
            "outputs": {f"heavy_result_{i}": f"Heavy processing result {i}"}
        })
    
    stages.append({
        "name": "HeavyParallelStage",
        "description": "Heavy parallel processing",
        "execution_type": "parallel",
        "tasks": parallel_tasks
    })
    
    # Stage 2: Sequential with multiple tool calls
    stages.append({
        "name": "SequentialToolStage",
        "description": "Sequential processing with many tools",
        "execution_type": "sequential",
        "tasks": [
            {
                "name": "MultiToolUser1",
                "description": "Use multiple tools sequentially",
                "agent": {
                    "id": "multi_tool_user_1",
                    "agent_type": "LLMAgent",
                    "llm_type": "openai",
                    "tools": [
                        {"name": "test_weather", "type": "function"},
                        {"name": "test_database_query", "type": "function"},
                        {"name": "test_api_call", "type": "function"},
                        {"name": "test_sentiment_analyzer", "type": "function"}
                    ],
                    "system_prompt": "Use all available tools to gather comprehensive data.",
                    "user_prompt": "Gather data about cities, databases, APIs, and analyze sentiment"
                },
                "outputs": {"multi_tool_result_1": "Multi-tool results"}
            },
            {
                "name": "MultiToolUser2",
                "description": "Process results with more tools",
                "agent": {
                    "id": "multi_tool_user_2",
                    "agent_type": "LLMAgent",
                    "llm_type": "openai",
                    "tools": [
                        {"name": "test_calculator", "type": "function"},
                        {"name": "test_translator", "type": "functional"},
                        {"name": "test_file_processor", "type": "function"}
                    ],
                    "system_prompt": "Process previous results with calculations and translations.",
                    "user_prompt": "Process: ${stages.[SequentialToolStage].tasks.[MultiToolUser1].outputs}"
                },
                "outputs": {"multi_tool_result_2": "Processed results"}
            }
        ]
    })
    
    # Stage 3: Complex handoff with many agents
    handoff_tasks = []
    for i in range(8):
        handoff_tasks.append({
            "name": f"HandoffAgent{i}",
            "description": f"Handoff agent {i}",
            "agent": {
                "id": f"handoff_agent_{i}",
                "agent_type": "LLMAgent",
                "llm_type": "openai",
                "tools": [{"name": "test_calculator", "type": "function"}] if i % 3 == 0 else [],
                "system_prompt": f"Specialist {i}: Handle specific case type {i}.",
                "user_prompt": f"Handle case type {i}: ${{workflow.inputs.case_type}}"
            },
            "outputs": {f"handoff_result_{i}": f"Handoff result {i}"}
        })
    
    stages.append({
        "name": "ComplexHandoffStage",
        "description": "Complex handoff routing",
        "execution_type": "handoff",
        "tasks": handoff_tasks
    })
    
    # Final aggregation stage
    stages.append({
        "name": "FinalAggregation",
        "description": "Aggregate all results",
        "execution_type": "sequential",
        "tasks": [
            {
                "name": "MasterAggregator",
                "description": "Aggregate everything",
                "agent": {
                    "id": "master_aggregator",
                    "agent_type": "LLMAgent",
                    "llm_type": "openai",
                    "tools": [
                        {"name": "test_data_analysis", "type": "functional"},
                        {"name": "test_file_processor", "type": "function"}
                    ],
                    "system_prompt": "Create comprehensive report from all previous stages.",
                    "user_prompt": "Aggregate all results into final report"
                },
                "outputs": {"final_report": "Comprehensive final report"}
            }
        ]
    })
    
    return {
        "name": "test_performance_stress",
        "description": "Performance stress test with many operations",
        "version": "1.0.0",
        "stages": stages
    }


# ============================================================================
# Base Test Class with Helper Methods
# ============================================================================

class BaseLangfuseTest:
    """Base class with helper methods for Langfuse tests."""
    
    async def _flush_traces(self, workflow_manager):
        """Helper to flush traces to Langfuse."""
        try:
            if hasattr(workflow_manager.execution_engine, '_ctx'):
                if hasattr(workflow_manager.execution_engine._ctx, 'llm_tracer'):
                    for provider in workflow_manager.execution_engine._ctx.llm_tracer.providers:
                        if isinstance(provider, LangfuseLLMObservabilityProvider):
                            provider.client.flush()
                            await asyncio.sleep(2)  # Give time for flush
                            logger.info("Flushed Langfuse traces")
        except Exception as e:
            logger.warning(f"Error flushing traces: {e}")
    
    def _log_workflow_summary(self, workflow_name: str, result: Any, execution_time: float):
        """Log workflow execution summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Workflow: {workflow_name}")
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        logger.info(f"Agents Executed: {len(result.all_agents) if result and hasattr(result, 'all_agents') else 0}")
        logger.info(f"Success: {result is not None}")
        logger.info(f"{'='*60}\n")


# ============================================================================
# Test Classes
# ============================================================================

@pytest.mark.asyncio
class TestLangfuseConnection(BaseLangfuseTest):
    """Test Langfuse connection and basic tracing."""
    
    async def test_langfuse_client_connection(self, observability_provider):
        """Test that Langfuse client is properly initialized and can create traces."""
        # Create a simple trace directly
        trace = observability_provider.client.trace(
            name="test_connection_trace",
            metadata={"test": "connection"}
        )
        
        # Flush to ensure data is sent
        observability_provider.client.flush()
        
        # Give it a moment to send
        await asyncio.sleep(2)
    
    async def test_workflow_trace_creation(self, workflow_manager):
        """Test that workflow execution creates a trace group in Langfuse."""
        workflow_dict = {
            "name": "test_trace_creation",
            "description": "Verify trace creation",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "SimpleStage",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "SimpleTask",
                            "agent": {
                                "id": "trace_test_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Say hello",
                                "user_prompt": "Hello"
                            }
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(workflow_dict)
        
        inputs = WorkflowInput(
            user_query="Test trace",
            workflow={"inputs": {}}
        )
        
        # Execute workflow
        result = await workflow_manager.execute(workflow, inputs)
        
        # Explicitly flush Langfuse
        workflow_manager.execution_engine._ctx.llm_tracer.providers[0].client.flush()
        await asyncio.sleep(2)  # Wait for flush to complete
        
        assert result is not None
        logger.info("Workflow execution completed - check Langfuse for traces")


@pytest.mark.asyncio
class TestLangfuseObservabilityHandoffs(BaseLangfuseTest):
    """Enhanced tests for agent handoff tracing with rigorous validation."""
    
    async def test_complex_handoff_workflow(self, workflow_manager):
        """Test complex multi-level handoff routing."""
        workflow_dict = create_complex_handoff_workflow()
        
        # Get provider mapping for all agents
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        # Test different query types to trigger different handoff paths
        test_queries = [
            ("How do I debug a Python error?", "technical"),
            ("What's my current bill amount?", "financial"),
            ("I can't log into my account", "support"),
            ("Tell me about your enterprise plan", "sales")
        ]
        
        for query, expected_category in test_queries:
            logger.info(f"\nTesting handoff for: {query}")
            
            inputs = WorkflowInput(
                user_query=query,
                workflow={"inputs": {"query": query}}
            )
            
            start_time = time.time()
            result = await workflow_manager.execute(
                workflow,
                inputs,
                progress_callback=ConsoleProgressCallback()
            )
            execution_time = time.time() - start_time
            
            assert result is not None
            self._log_workflow_summary(f"Complex Handoff - {expected_category}", result, execution_time)
            
            # Flush traces after each execution
            await self._flush_traces(workflow_manager)
    
    async def test_recursive_handoff_workflow(self, workflow_manager):
        """Test recursive handoff patterns with language detection."""
        workflow_dict = create_recursive_handoff_workflow()
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        # Test with different languages
        test_queries = [
            "Hello, how are you today?",  # English
            "Bonjour, comment allez-vous?",  # French
            "Hola, ¿cómo estás?",  # Spanish
            "This makes me very happy!",  # Positive sentiment
            "This is terrible and disappointing."  # Negative sentiment
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting recursive handoff for: {query}")
            
            inputs = WorkflowInput(
                user_query=query,
                workflow={"inputs": {"multilingual_query": query}}
            )
            
            start_time = time.time()
            result = await workflow_manager.execute(
                workflow,
                inputs,
                progress_callback=ConsoleProgressCallback()
            )
            execution_time = time.time() - start_time
            
            assert result is not None
            self._log_workflow_summary(f"Recursive Handoff - {query[:20]}...", result, execution_time)
            
            await self._flush_traces(workflow_manager)
    
    async def test_conditional_routing_workflow(self, workflow_manager):
        """Test conditional routing based on assessment."""
        workflow_dict = create_conditional_routing_workflow()
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        # Test different risk scenarios
        test_scenarios = [
            "Critical security breach detected in production",  # High risk
            "Unusual login pattern from new location",  # Medium risk
            "User requested password reset"  # Low risk
        ]
        
        for scenario in test_scenarios:
            logger.info(f"\nTesting conditional routing for: {scenario}")
            
            inputs = WorkflowInput(
                user_query=scenario,
                workflow={"inputs": {"scenario": scenario}}
            )
            
            start_time = time.time()
            result = await workflow_manager.execute(
                workflow,
                inputs,
                progress_callback=ConsoleProgressCallback()
            )
            execution_time = time.time() - start_time
            
            assert result is not None
            self._log_workflow_summary(f"Conditional Routing - {scenario[:30]}...", result, execution_time)
            
            await self._flush_traces(workflow_manager)


@pytest.mark.asyncio
class TestLangfuseObservabilityMultiAgent(BaseLangfuseTest):
    """Test complex multi-agent interactions."""
    
    async def test_massive_parallel_execution(self, workflow_manager):
        """Test massive parallel agent execution."""
        workflow_dict = create_massive_parallel_workflow()
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        # Create input data for all parallel agents
        inputs_data = {
            "large_dataset": "Dataset with 10 segments for parallel processing"
        }
        for i in range(10):
            inputs_data[f"data_segment_{i}"] = f"Segment {i} data: [values {i*100} to {(i+1)*100}]"
        
        inputs = WorkflowInput(
            user_query="Process large dataset in parallel",
            workflow={"inputs": inputs_data}
        )
        
        start_time = time.time()
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        execution_time = time.time() - start_time
        
        assert result is not None
        # Should have executed data splitter + 10 parallel agents + aggregator = 12 agents minimum
        assert len(result.all_agents) >= 12
        self._log_workflow_summary("Massive Parallel Execution", result, execution_time)
        
        await self._flush_traces(workflow_manager)
    
    async def test_multi_tool_orchestration(self, workflow_manager):
        """Test complex multi-agent tool orchestration."""
        workflow_dict = create_multi_tool_orchestration_workflow()
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Collect and analyze comprehensive data",
            workflow={"inputs": {}}
        )
        
        start_time = time.time()
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        execution_time = time.time() - start_time
        
        assert result is not None
        # Should have multiple agents using various tools
        self._log_workflow_summary("Multi-Tool Orchestration", result, execution_time)
        
        await self._flush_traces(workflow_manager)
    
    async def test_performance_stress_workflow(self, workflow_manager):
        """Test performance under stress with many operations."""
        workflow_dict = create_performance_stress_workflow()
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        # Create complex input data
        inputs_data = {"case_type": "complex_technical_issue"}
        for i in range(5):
            inputs_data[f"workload_{i}"] = f"Heavy workload {i}: process large dataset with complex calculations"
        
        inputs = WorkflowInput(
            user_query="Stress test with heavy workload",
            workflow={"inputs": inputs_data}
        )
        
        start_time = time.time()
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        execution_time = time.time() - start_time
        
        assert result is not None
        self._log_workflow_summary("Performance Stress Test", result, execution_time)
        
        # Performance should complete within reasonable time even under stress
        assert execution_time < 120  # 2 minutes max
        
        await self._flush_traces(workflow_manager)


@pytest.mark.asyncio
class TestLangfuseObservabilityErrorHandling(BaseLangfuseTest):
    """Test error handling and recovery scenarios."""
    
    async def test_error_cascade_workflow(self, workflow_manager):
        """Test error propagation and recovery in complex scenarios."""
        workflow_dict = create_error_cascade_workflow()
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test error cascade and recovery",
            workflow={"inputs": {"data": "Important data to process"}}
        )
        
        start_time = time.time()
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        execution_time = time.time() - start_time
        
        # Workflow should complete even with errors
        assert result is not None
        self._log_workflow_summary("Error Cascade Test", result, execution_time)
        
        await self._flush_traces(workflow_manager)
    
    async def test_partial_failure_recovery(self, workflow_manager):
        """Test handling of partial failures in parallel execution."""
        workflow_dict = {
            "name": "test_partial_failure",
            "description": "Test partial failure recovery",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "ParallelWithFailures",
                    "description": "Parallel execution with some failures",
                    "execution_type": "parallel",
                    "tasks": [
                        {
                            "name": "SuccessTask1",
                            "agent": {
                                "id": "success_1",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [{"name": "test_weather", "type": "function"}],
                                "system_prompt": "Get weather for New York",
                                "user_prompt": "Check NYC weather"
                            },
                            "outputs": {"weather": "Weather data"}
                        },
                        {
                            "name": "FailTask",
                            "agent": {
                                "id": "fail_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [{"name": "test_calculator", "type": "function"}],
                                "system_prompt": "Calculate 50/0",
                                "user_prompt": "Divide by zero"
                            },
                            "outputs": {"fail": "Should fail"}
                        },
                        {
                            "name": "SuccessTask2",
                            "agent": {
                                "id": "success_2",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [{"name": "test_api_call", "type": "function"}],
                                "system_prompt": "Call status API",
                                "user_prompt": "Check system status"
                            },
                            "outputs": {"status": "System status"}
                        }
                    ]
                },
                {
                    "name": "RecoveryStage",
                    "description": "Recover from partial failures",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "RecoveryAgent",
                            "agent": {
                                "id": "recovery_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Process available results and handle failures gracefully",
                                "user_prompt": "Process partial results: ${stages.[ParallelWithFailures].outputs}"
                            },
                            "outputs": {"recovery": "Recovery complete"}
                        }
                    ]
                }
            ]
        }
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test partial failure recovery",
            workflow={"inputs": {}}
        )
        
        start_time = time.time()
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        execution_time = time.time() - start_time
        
        assert result is not None
        self._log_workflow_summary("Partial Failure Recovery", result, execution_time)
        
        await self._flush_traces(workflow_manager)


@pytest.mark.asyncio
class TestLangfuseObservabilityEdgeCases(BaseLangfuseTest):
    """Test edge cases and special scenarios."""
    
    async def test_rapid_sequential_executions(self, workflow_manager):
        """Test rapid sequential workflow executions."""
        workflow_dict = create_simple_sequential_workflow()
        
        workflow = await workflow_manager.initialize_workflow(workflow_dict)
        
        # Execute same workflow rapidly multiple times
        for i in range(5):
            inputs = WorkflowInput(
                user_query=f"Rapid test {i}",
                workflow={"inputs": {"user_query": f"Rapid execution test {i}"}}
            )
            
            start_time = time.time()
            result = await workflow_manager.execute(workflow, inputs)
            execution_time = time.time() - start_time
            
            assert result is not None
            logger.info(f"Rapid execution {i} completed in {execution_time:.2f}s")
            
            # Small delay between executions
            await asyncio.sleep(0.5)
        
        await self._flush_traces(workflow_manager)
    
    async def test_deep_nested_data_references(self, workflow_manager):
        """Test workflows with deep nested data references."""
        workflow_dict = {
            "name": "test_deep_nesting",
            "description": "Test deep nested data references",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "Stage1",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "Task1",
                            "agent": {
                                "id": "agent1",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Generate structured data",
                                "user_prompt": "Create nested data structure"
                            },
                            "outputs": {"data": {"level1": {"level2": {"level3": "deep_value"}}}}
                        }
                    ]
                },
                {
                    "name": "Stage2",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "Task2",
                            "agent": {
                                "id": "agent2",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Process nested data",
                                "user_prompt": "Process: ${stages.[Stage1].tasks.[Task1].outputs.data}"
                            },
                            "outputs": {"processed": "Processed nested data"}
                        }
                    ]
                },
                {
                    "name": "Stage3",
                    "execution_type": "parallel",
                    "tasks": [
                        {
                            "name": "Task3A",
                            "agent": {
                                "id": "agent3a",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Further process A",
                                "user_prompt": "Process A: ${stages.[Stage2].tasks.[Task2].outputs.processed}"
                            },
                            "outputs": {"result_a": "Result A"}
                        },
                        {
                            "name": "Task3B",
                            "agent": {
                                "id": "agent3b",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Further process B",
                                "user_prompt": "Process B: ${stages.[Stage2].tasks.[Task2].outputs.processed}"
                            },
                            "outputs": {"result_b": "Result B"}
                        }
                    ]
                }
            ]
        }
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping
        )
        
        inputs = WorkflowInput(
            user_query="Test deep nesting",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(workflow, inputs)
        assert result is not None
        
        await self._flush_traces(workflow_manager)
    
    async def test_empty_tool_responses(self, workflow_manager):
        """Test handling of empty or null tool responses."""
        workflow_dict = {
            "name": "test_empty_responses",
            "description": "Test empty tool responses",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "EmptyResponseStage",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "EmptyToolUser",
                            "agent": {
                                "id": "empty_tool_user",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [
                                    {"name": "test_database_query", "type": "function"},
                                    {"name": "test_api_call", "type": "function"}
                                ],
                                "system_prompt": "Query empty tables and call non-existent endpoints",
                                "user_prompt": "Query: SELECT * FROM empty_table"
                            },
                            "outputs": {"empty_result": "Empty response handling"}
                        }
                    ]
                }
            ]
        }
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping
        )
        
        inputs = WorkflowInput(
            user_query="Test empty responses",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(workflow, inputs)
        assert result is not None
        
        await self._flush_traces(workflow_manager)
    
    async def test_race_condition_scenario(self, workflow_manager):
        """Test potential race conditions in parallel execution."""
        workflow_dict = {
            "name": "test_race_conditions",
            "description": "Test race conditions",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "SharedResourceStage",
                    "execution_type": "parallel",
                    "tasks": [
                        {
                            "name": f"RaceAgent{i}",
                            "agent": {
                                "id": f"race_agent_{i}",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [{"name": "test_calculator", "type": "function"}],
                                "system_prompt": f"Agent {i}: Calculate and update shared counter",
                                "user_prompt": f"Add {i} to counter"
                            },
                            "outputs": {f"counter_{i}": f"Counter value from agent {i}"}
                        } for i in range(5)
                    ]
                },
                {
                    "name": "VerificationStage",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "Verifier",
                            "agent": {
                                "id": "verifier_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Verify all counter updates completed correctly",
                                "user_prompt": "Verify: ${stages.[SharedResourceStage].outputs}"
                            },
                            "outputs": {"verification": "Verification result"}
                        }
                    ]
                }
            ]
        }
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping
        )
        
        inputs = WorkflowInput(
            user_query="Test race conditions",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(workflow, inputs)
        assert result is not None
        
        await self._flush_traces(workflow_manager)

