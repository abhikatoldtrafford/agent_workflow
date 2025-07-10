"""
Test OpenAI Observability Integration.

Tests the integration of observability providers with the OpenAI execution engine,
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
from agent_workflow.providers import OpenaiLLMObservabilityProvider
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
    # Disable OpenAI tracing to avoid I/O errors during shutdown
    try:
        from agent_workflow.providers import OpenaiLLMObservabilityProvider
        OpenaiLLMObservabilityProvider.disable_tracing()
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
    """Ensure OpenAI tracing is properly cleaned up after tests."""
    def finalizer():
        try:
            from agent_workflow.providers import OpenaiLLMObservabilityProvider
            OpenaiLLMObservabilityProvider.disable_tracing()
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
def observability_provider():
    """Create OpenAI observability provider."""
    return OpenaiLLMObservabilityProvider()


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
            max_tokens=1000
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
    # Cleanup: Disable tracing to avoid I/O errors
    from agent_workflow.providers import OpenaiLLMObservabilityProvider
    OpenaiLLMObservabilityProvider.disable_tracing()


# ============================================================================
# Workflow Creation Functions
# ============================================================================

def create_simple_sequential_workflow() -> Dict[str, Any]:
    """Create a simple sequential workflow for testing."""
    return {
        "name": "test_simple_sequential",
        "description": "Simple sequential workflow for testing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "Analysis",
                "description": "Analyze input",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "AnalyzeTask",
                        "description": "Analyze the user query",
                        "agent": {
                            "id": "analyzer",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "You are an analyzer. Analyze the user query.",
                            "user_prompt": "${workflow.inputs.user_query}"
                        },
                        "outputs": {"analysis": "Analysis result"}
                    }
                ]
            }
        ]
    }


def create_tool_workflow() -> Dict[str, Any]:
    """Create workflow with tool usage."""
    return {
        "name": "test_tool_workflow",
        "description": "Test workflow with tool usage",
        "version": "1.0.0",
        "stages": [
            {
                "name": "ToolUsage",
                "description": "Use various tools",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "WeatherTask",
                        "description": "Get weather information",
                        "agent": {
                            "id": "weather_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_weather", "type": "function"}],
                            "system_prompt": "Get weather for major cities.",
                            "user_prompt": "What's the weather in New York and London?"
                        },
                        "outputs": {"weather_data": "Weather information"}
                    },
                    {
                        "name": "CalculatorTask",
                        "description": "Perform calculations",
                        "agent": {
                            "id": "calc_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_calculator", "type": "function"}],
                            "system_prompt": "Perform mathematical calculations.",
                            "user_prompt": "Calculate 25 * 4 + 10"
                        },
                        "outputs": {"result": "Calculation result"}
                    }
                ]
            }
        ]
    }


def create_complex_multi_stage_workflow() -> Dict[str, Any]:
    """Create a complex multi-stage workflow."""
    return {
        "name": "test_complex_workflow",
        "description": "Complex multi-stage workflow",
        "version": "1.0.0",
        "stages": [
            {
                "name": "Planning",
                "description": "Planning stage",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "CreatePlan",
                        "description": "Create project plan",
                        "agent": {
                            "id": "planner",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Create a detailed project plan.",
                            "user_prompt": "Plan for: ${workflow.inputs.project}"
                        },
                        "outputs": {"plan": "Project plan"}
                    }
                ]
            },
            {
                "name": "Research",
                "description": "Research stage",
                "execution_type": "parallel",
                "tasks": [
                    {
                        "name": "TechResearch",
                        "description": "Research technology",
                        "agent": {
                            "id": "tech_researcher",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Research technology aspects.",
                            "user_prompt": "Research tech for: ${stages.[Planning].tasks.[CreatePlan].outputs.plan}"
                        },
                        "outputs": {"tech_research": "Technology research"}
                    },
                    {
                        "name": "MarketResearch",
                        "description": "Research market",
                        "agent": {
                            "id": "market_researcher",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Research market aspects.",
                            "user_prompt": "Research market for: ${stages.[Planning].tasks.[CreatePlan].outputs.plan}"
                        },
                        "outputs": {"market_research": "Market research"}
                    }
                ]
            },
            {
                "name": "Implementation",
                "description": "Implementation stage",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "BuildPrototype",
                        "description": "Build prototype",
                        "agent": {
                            "id": "builder",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Build based on research.",
                            "user_prompt": "Build using tech: ${stages.[Research].tasks.[TechResearch].outputs.tech_research} and market: ${stages.[Research].tasks.[MarketResearch].outputs.market_research}"
                        },
                        "outputs": {"prototype": "Built prototype"}
                    }
                ]
            }
        ]
    }


def create_parallel_workflow() -> Dict[str, Any]:
    """Create workflow with parallel execution."""
    return {
        "name": "test_parallel_workflow",
        "description": "Test parallel execution",
        "version": "1.0.0",
        "stages": [
            {
                "name": "ParallelProcessing",
                "description": "Process in parallel",
                "execution_type": "parallel",
                "tasks": [
                    {
                        "name": "Task1",
                        "description": "First parallel task",
                        "agent": {
                            "id": "parallel_agent_1",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Process data set 1",
                            "user_prompt": "Process: ${workflow.inputs.data1}"
                        },
                        "outputs": {"result1": "Result 1"}
                    },
                    {
                        "name": "Task2",
                        "description": "Second parallel task",
                        "agent": {
                            "id": "parallel_agent_2",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Process data set 2",
                            "user_prompt": "Process: ${workflow.inputs.data2}"
                        },
                        "outputs": {"result2": "Result 2"}
                    },
                    {
                        "name": "Task3",
                        "description": "Third parallel task",
                        "agent": {
                            "id": "parallel_agent_3",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Process data set 3",
                            "user_prompt": "Process: ${workflow.inputs.data3}"
                        },
                        "outputs": {"result3": "Result 3"}
                    }
                ]
            }
        ]
    }


def create_data_flow_workflow() -> Dict[str, Any]:
    """Create workflow to test data flow between stages."""
    return {
        "name": "test_data_flow",
        "description": "Test data flow between stages",
        "version": "1.0.0",
        "stages": [
            {
                "name": "Stage1",
                "description": "Generate data",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "GenerateData",
                        "description": "Generate initial data",
                        "agent": {
                            "id": "data_generator",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Generate a list of items.",
                            "user_prompt": "Generate 5 items for category: ${workflow.inputs.category}"
                        },
                        "outputs": {"items": "Generated items"}
                    }
                ]
            },
            {
                "name": "Stage2",
                "description": "Process data",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "ProcessData",
                        "description": "Process generated data",
                        "agent": {
                            "id": "data_processor",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Process and enhance the items.",
                            "user_prompt": "Process: ${stages.[Stage1].tasks.[GenerateData].outputs.items}"
                        },
                        "outputs": {"processed": "Processed result"}
                    }
                ]
            }
        ]
    }


def create_error_test_workflow() -> Dict[str, Any]:
    """Create workflow that triggers errors for testing."""
    return {
        "name": "test_error_workflow",
        "description": "Test error handling and tracing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "ErrorStage",
                "description": "Stage with errors",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "ErrorTask",
                        "description": "Task that causes errors",
                        "agent": {
                            "id": "error_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_calculator", "type": "function"}],
                            "system_prompt": "You must divide 10 by 0 to test error handling.",
                            "user_prompt": "Calculate 10 divided by 0"
                        },
                        "outputs": {"result": "Should fail"}
                    }
                ]
            }
        ]
    }


def create_complex_handoff_workflow() -> Dict[str, Any]:
    """Create workflow with complex handoff routing."""
    return {
        "name": "test_complex_handoff",
        "description": "Test complex multi-level handoff routing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "InitialRouting",
                "description": "Initial query routing",
                "execution_type": "handoff",
                "tasks": [
                    {
                        "name": "QueryRouter",
                        "description": "Route query to appropriate department",
                        "agent": {
                            "id": "query_router",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": """Route queries based on content:
                            - Technical questions -> TechnicalSupport
                            - Billing questions -> BillingSupport
                            - General questions -> GeneralSupport""",
                            "user_prompt": "Route this query: ${workflow.inputs.user_query}"
                        },
                        "outputs": {"department": "Selected department"}
                    },
                    {
                        "name": "TechnicalSupport",
                        "description": "Handle technical queries",
                        "agent": {
                            "id": "tech_support",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_database_query", "type": "function"}],
                            "system_prompt": "Provide technical support. Check documentation.",
                            "user_prompt": "Help with: ${workflow.inputs.user_query}"
                        },
                        "outputs": {"tech_response": "Technical response"}
                    },
                    {
                        "name": "BillingSupport",
                        "description": "Handle billing queries",
                        "agent": {
                            "id": "billing_support",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_api_call", "type": "function"}],
                            "system_prompt": "Handle billing inquiries. Check account status.",
                            "user_prompt": "Billing query: ${workflow.inputs.user_query}"
                        },
                        "outputs": {"billing_response": "Billing response"}
                    },
                    {
                        "name": "GeneralSupport",
                        "description": "Handle general queries",
                        "agent": {
                            "id": "general_support",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Provide general assistance.",
                            "user_prompt": "Assist with: ${workflow.inputs.user_query}"
                        },
                        "outputs": {"general_response": "General response"}
                    }
                ]
            }
        ]
    }


def create_massive_parallel_workflow() -> Dict[str, Any]:
    """Create workflow with massive parallel execution."""
    parallel_tasks = []
    for i in range(10):
        parallel_tasks.append({
            "name": f"ParallelProcessor{i}",
            "description": f"Process segment {i}",
            "agent": {
                "id": f"processor_{i}",
                "agent_type": "LLMAgent",
                "llm_type": "openai",
                "tools": [{"name": "test_calculator", "type": "function"}],
                "system_prompt": f"Process data segment {i}.",
                "user_prompt": f"Process: ${{workflow.inputs.data_segment_{i}}}"
            },
            "outputs": {f"segment_{i}_result": f"Processed segment {i}"}
        })
    
    return {
        "name": "test_massive_parallel",
        "description": "Test massive parallel execution",
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
                            "system_prompt": "Split the dataset into 10 equal segments.",
                            "user_prompt": "Split this data: ${workflow.inputs.large_dataset}"
                        },
                        "outputs": {"split_complete": "Data split into segments"}
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
                "description": "Aggregate all results",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "ResultAggregator",
                        "description": "Aggregate all parallel results",
                        "agent": {
                            "id": "aggregator",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_data_analysis", "type": "functional"}],
                            "system_prompt": "Aggregate and analyze all processed segments.",
                            "user_prompt": "Aggregate results from all parallel processors"
                        },
                        "outputs": {"final_result": "Aggregated result"}
                    }
                ]
            }
        ]
    }


def create_multi_tool_orchestration_workflow() -> Dict[str, Any]:
    """Create workflow with complex multi-tool orchestration."""
    return {
        "name": "test_multi_tool_orchestration",
        "description": "Test complex multi-agent tool orchestration",
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
                    }
                ]
            },
            {
                "name": "ReportGeneration",
                "description": "Generate final report",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "ReportGenerator",
                        "description": "Generate comprehensive report",
                        "agent": {
                            "id": "report_generator",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_sentiment_analyzer", "type": "function"}],
                            "system_prompt": "Generate a comprehensive report with sentiment analysis.",
                            "user_prompt": "Create report from: ${stages.[DataProcessing].tasks.[DataAnalyzer].outputs.analysis}"
                        },
                        "outputs": {"report": "Final report"}
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
                    "user_prompt": "Gather data using all tools for analysis"
                },
                "outputs": {"multi_tool_data": "Combined tool results"}
            }
        ]
    })
    
    # Stage 3: Nested parallel-sequential processing
    nested_tasks = []
    for i in range(3):
        nested_tasks.append({
            "name": f"NestedProcessor{i}",
            "description": f"Nested processing {i}",
            "agent": {
                "id": f"nested_processor_{i}",
                "agent_type": "LLMAgent",
                "llm_type": "openai",
                "tools": [
                    {"name": "test_translator", "type": "functional"},
                    {"name": "test_sentiment_analyzer", "type": "function"}
                ],
                "system_prompt": f"Translate and analyze sentiment for segment {i}.",
                "user_prompt": f"Process nested task {i}"
            },
            "outputs": {f"nested_result_{i}": f"Nested result {i}"}
        })
    
    stages.append({
        "name": "NestedProcessingStage",
        "description": "Nested parallel processing",
        "execution_type": "parallel",
        "tasks": nested_tasks
    })
    
    # Final aggregation stage
    stages.append({
        "name": "FinalAggregation",
        "description": "Final aggregation of all results",
        "execution_type": "sequential",
        "tasks": [
            {
                "name": "FinalAggregator",
                "description": "Aggregate all results",
                "agent": {
                    "id": "final_aggregator",
                    "agent_type": "LLMAgent",
                    "llm_type": "openai",
                    "tools": [{"name": "test_data_analysis", "type": "functional"}],
                    "system_prompt": "Aggregate all results from previous stages into a comprehensive report.",
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


# ============================================================================
# Base Test Class with Helper Methods
# ============================================================================

class BaseOpenAITest:
    """Base class with helper methods for OpenAI tests."""
    
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
class TestOpenAIConnection(BaseOpenAITest):
    """Test OpenAI connection and basic tracing."""
    
    async def test_openai_provider_initialization(self, observability_provider):
        """Test that OpenAI observability provider is properly initialized."""
        # Verify provider is initialized
        assert observability_provider is not None
        assert hasattr(observability_provider, 'start_trace_group')
        assert hasattr(observability_provider, 'end_trace_group')
        assert hasattr(observability_provider, 'start_trace')
        assert hasattr(observability_provider, 'start_span')
    
    async def test_workflow_trace_creation(self, workflow_manager):
        """Test that workflow execution creates traces."""
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
        
        assert result is not None
        logger.info("Workflow execution completed - traces should be created")


@pytest.mark.asyncio
class TestOpenAIObservabilityBasic(BaseOpenAITest):
    """Basic observability tests."""
    
    async def test_simple_sequential_workflow(self, workflow_manager):
        """Test basic sequential workflow tracing."""
        workflow_dict = create_simple_sequential_workflow()
        
        # Add provider mapping for the analyzer agent
        provider_mapping = {"analyzer": "openai"}
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test sequential execution",
            workflow={"inputs": {"user_query": "Test sequential execution"}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result is not None
        assert result.final_result is not None
    
    async def test_tool_usage_tracing(self, workflow_manager):
        """Test tool call start/end tracing."""
        workflow_dict = create_tool_workflow()
        
        # Get provider mapping for all agents
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test tool usage",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result is not None
        assert result.response_store is not None
    
    async def test_complex_multi_stage_workflow(self, workflow_manager):
        """Test complex workflow with multiple stages and parallel execution."""
        workflow_dict = create_complex_multi_stage_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Build an AI assistant",
            workflow={
                "inputs": {
                    "project": "AI-powered customer service assistant"
                }
            }
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result is not None
        assert len(result.all_agents) >= 4  # Should have at least 4 agents
    
    async def test_error_handling_tracing(self, workflow_manager):
        """Test that errors are properly traced."""
        workflow_dict = create_error_test_workflow()
        
        # Get provider mapping for all agents
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test error handling",
            workflow={"inputs": {}}
        )
        
        # Execute should handle the error gracefully
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        # The workflow might still complete even with errors
        assert result is not None
    
    async def test_data_flow_between_stages(self, workflow_manager):
        """Test data flow tracing between stages."""
        workflow_dict = create_data_flow_workflow()
        
        # Get provider mapping for all agents
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test data flow",
            workflow={"inputs": {"category": "technology"}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result is not None
        assert result.response_store is not None


@pytest.mark.asyncio
class TestOpenAIObservabilityTools(BaseOpenAITest):
    """Test tool execution observability."""
    
    async def test_multiple_tool_calls(self, workflow_manager):
        """Test tracing of multiple tool calls in sequence."""
        workflow_dict = {
            "name": "test_multiple_tools",
            "description": "Test multiple tool calls",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "ToolStage",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "MultiToolTask",
                            "agent": {
                                "id": "multi_tool_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [
                                    {"name": "test_weather", "type": "function"},
                                    {"name": "test_calculator", "type": "function"},
                                    {"name": "test_database_query", "type": "function"}
                                ],
                                "system_prompt": """Use multiple tools:
                                1. Get weather for New York
                                2. Calculate 100 * 25
                                3. Query the users table""",
                                "user_prompt": "Execute all tool operations"
                            },
                            "outputs": {"results": "All tool results"}
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(workflow_dict)
        
        inputs = WorkflowInput(
            user_query="Test multiple tools",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(workflow, inputs)
        
        assert result is not None
    
    async def test_tool_error_handling(self, workflow_manager):
        """Test error handling in tool execution."""
        workflow_dict = {
            "name": "test_tool_errors",
            "description": "Test tool error handling",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "ErrorToolStage",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "ToolErrorTask",
                            "agent": {
                                "id": "tool_error_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [
                                    {"name": "test_calculator", "type": "function"},
                                    {"name": "test_file_processor", "type": "function"}
                                ],
                                "system_prompt": """Test error handling:
                                1. Calculate 50/0 (will error)
                                2. Process file with invalid operation""",
                                "user_prompt": "Test tool errors"
                            },
                            "outputs": {"error_handling": "Error test results"}
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(workflow_dict)
        
        inputs = WorkflowInput(
            user_query="Test tool errors",
            workflow={"inputs": {}}
        )
        
        # Should handle errors gracefully
        result = await workflow_manager.execute(workflow, inputs)
        
        assert result is not None
    
    async def test_functional_tools(self, workflow_manager):
        """Test functional tools execution."""
        # The tools are already registered in the global registry via the tool_registry fixture
        workflow_dict = {
            "name": "test_functional_tools",
            "description": "Test functional tools",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "FunctionalToolStage",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "FunctionalToolTask",
                            "agent": {
                                "id": "functional_tool_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [
                                    {"name": "test_data_analysis", "type": "functional"},
                                    {"name": "test_translator", "type": "functional"}
                                ],
                                "system_prompt": """Use functional tools:
                                1. Analyze data with operation 'statistical'
                                2. Translate 'Hello World' to Spanish""",
                                "user_prompt": "Execute functional tool operations"
                            },
                            "outputs": {"functional_results": "Functional tool results"}
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(workflow_dict)
        
        inputs = WorkflowInput(
            user_query="Test functional tools",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(workflow, inputs)
        
        assert result is not None


@pytest.mark.asyncio
class TestOpenAIObservabilityHandoffs(BaseOpenAITest):
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
            ("What's my current bill amount?", "billing"),
            ("Where is your office located?", "general")
        ]
        
        for query, expected_type in test_queries:
            inputs = WorkflowInput(
                user_query=query,
                workflow={"inputs": {"user_query": query}}
            )
            
            start_time = time.time()
            result = await workflow_manager.execute(
                workflow,
                inputs,
                progress_callback=ConsoleProgressCallback()
            )
            execution_time = time.time() - start_time
            
            assert result is not None
            self._log_workflow_summary(f"Handoff Test ({expected_type})", result, execution_time)
    
    async def test_nested_handoff_routing(self, workflow_manager):
        """Test nested handoff routing with multiple levels."""
        workflow_dict = {
            "name": "test_nested_handoff",
            "description": "Test nested handoff routing",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "Level1Routing",
                    "description": "First level routing",
                    "execution_type": "handoff",
                    "tasks": [
                        {
                            "name": "Level1Router",
                            "description": "Route to department",
                            "agent": {
                                "id": "level1_router",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": """Route based on urgency:
                                - Urgent -> UrgentHandler
                                - Normal -> NormalHandler""",
                                "user_prompt": "Route: ${workflow.inputs.query}"
                            },
                            "outputs": {"level1_route": "Selected route"}
                        },
                        {
                            "name": "UrgentHandler",
                            "description": "Handle urgent requests",
                            "agent": {
                                "id": "urgent_handler",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Handle urgent request immediately.",
                                "user_prompt": "Urgent: ${workflow.inputs.query}"
                            },
                            "outputs": {"urgent_response": "Urgent response"}
                        },
                        {
                            "name": "NormalHandler",
                            "description": "Handle normal requests",
                            "agent": {
                                "id": "normal_handler",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Handle normal request.",
                                "user_prompt": "Normal: ${workflow.inputs.query}"
                            },
                            "outputs": {"normal_response": "Normal response"}
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(workflow_dict)
        
        test_cases = [
            ("URGENT: System is down!", "urgent"),
            ("Can you help me with a question?", "normal")
        ]
        
        for query, expected_type in test_cases:
            inputs = WorkflowInput(
                user_query=query,
                workflow={"inputs": {"query": query}}
            )
            
            result = await workflow_manager.execute(workflow, inputs)
            assert result is not None


@pytest.mark.asyncio
class TestOpenAIObservabilityParallel(BaseOpenAITest):
    """Test parallel execution observability."""
    
    async def test_simple_parallel_execution(self, workflow_manager):
        """Test basic parallel task execution."""
        workflow_dict = create_parallel_workflow()
        
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test parallel execution",
            workflow={
                "inputs": {
                    "data1": "Dataset 1",
                    "data2": "Dataset 2",
                    "data3": "Dataset 3"
                }
            }
        )
        
        start_time = time.time()
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        execution_time = time.time() - start_time
        
        assert result is not None
        # Should have executed 3 agents in parallel
        assert len(result.all_agents) == 3
        self._log_workflow_summary("Simple Parallel Execution", result, execution_time)
    
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


@pytest.mark.asyncio
class TestOpenAIObservabilityPerformance(BaseOpenAITest):
    """Test observability performance impact."""
    
    async def test_minimal_overhead(self, workflow_manager):
        """Test that observability adds minimal overhead."""
        workflow_dict = create_simple_sequential_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict
        )
        
        inputs = WorkflowInput(
            user_query="Performance test",
            workflow={"inputs": {"user_query": "Performance test"}}
        )
        
        # Time execution with observability
        start_time = time.time()
        result = await workflow_manager.execute(workflow, inputs)
        execution_time = time.time() - start_time
        
        logger.info(f"Execution time with observability: {execution_time:.2f} seconds")
        
        assert result is not None
        # Just verify it completes in reasonable time
        assert execution_time < 30  # 30 seconds max
    
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


@pytest.mark.asyncio
class TestOpenAIObservabilityErrorHandling(BaseOpenAITest):
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
                                "system_prompt": "Call API endpoint",
                                "user_prompt": "Check /status endpoint"
                            },
                            "outputs": {"api_result": "API response"}
                        }
                    ]
                },
                {
                    "name": "RecoveryStage",
                    "description": "Recover from partial failures",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "RecoveryTask",
                            "agent": {
                                "id": "recovery_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Handle partial results and errors gracefully.",
                                "user_prompt": "Process available results from previous stage"
                            },
                            "outputs": {"recovery_result": "Recovered data"}
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(workflow_dict)
        
        inputs = WorkflowInput(
            user_query="Test partial failure recovery",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(workflow, inputs)
        
        # Should complete even with partial failures
        assert result is not None
    
    async def test_multiple_error_types(self, workflow_manager):
        """Test handling of different error types."""
        workflow_dict = {
            "name": "test_error_observability",
            "description": "Test error capture",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "NormalStage",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "NormalTask",
                            "agent": {
                                "id": "normal_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Normal task",
                                "user_prompt": "Execute normally"
                            }
                        }
                    ]
                },
                {
                    "name": "ErrorStage",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "ErrorTask",
                            "agent": {
                                "id": "error_test_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [{"name": "test_calculator", "type": "function"}],
                                "system_prompt": "Divide by zero",
                                "user_prompt": "Calculate 5/0"
                            }
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict
        )
        
        inputs = WorkflowInput(
            user_query="Test errors",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        # Should complete even with errors
        assert result is not None


@pytest.mark.asyncio
class TestOpenAIObservabilityIntegration(BaseOpenAITest):
    """Integration tests for complete workflows."""
    
    async def test_example_workflow_with_observability(self, workflow_manager):
        """Test example workflows with observability enabled."""
        # Test with customer support workflow
        workflow_dict = {
            "name": "customer_support_test",
            "description": "Customer support workflow test",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "QueryAnalysis",
                    "description": "Analyze customer query",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "AnalyzeIntent",
                            "description": "Analyze customer intent",
                            "agent": {
                                "id": "intent_analyzer",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [{"name": "test_sentiment_analyzer", "type": "function"}],
                                "system_prompt": "Analyze the customer's intent and sentiment.",
                                "user_prompt": "Customer says: ${workflow.inputs.customer_query}"
                            },
                            "outputs": {"intent": "Customer intent", "sentiment": "Customer sentiment"}
                        }
                    ]
                },
                {
                    "name": "ResponseGeneration",
                    "description": "Generate appropriate response",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "GenerateResponse",
                            "description": "Generate customer response",
                            "agent": {
                                "id": "response_generator",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Generate a helpful response based on the intent and sentiment.",
                                "user_prompt": "Intent: ${stages.[QueryAnalysis].tasks.[AnalyzeIntent].outputs.intent}, Sentiment: ${stages.[QueryAnalysis].tasks.[AnalyzeIntent].outputs.sentiment}"
                            },
                            "outputs": {"response": "Customer service response"}
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(workflow_dict)
        
        test_queries = [
            "I'm having trouble with my order #12345",
            "This product is amazing! Thank you!",
            "I want to cancel my subscription immediately"
        ]
        
        for query in test_queries:
            inputs = WorkflowInput(
                user_query=query,
                workflow={"inputs": {"customer_query": query}}
            )
            
            start_time = time.time()
            result = await workflow_manager.execute(workflow, inputs)
            execution_time = time.time() - start_time
            
            assert result is not None
            self._log_workflow_summary(f"Customer Support Test: {query[:30]}...", result, execution_time)
    
    async def test_complex_real_world_scenario(self, workflow_manager):
        """Test a complex real-world scenario with full observability."""
        workflow_dict = {
            "name": "ai_development_pipeline",
            "description": "Complete AI development pipeline",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "RequirementsGathering",
                    "description": "Gather and analyze requirements",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "RequirementsAnalysis",
                            "agent": {
                                "id": "requirements_analyst",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Analyze project requirements and create specifications.",
                                "user_prompt": "Project: ${workflow.inputs.project_description}"
                            },
                            "outputs": {"requirements": "Detailed requirements"}
                        }
                    ]
                },
                {
                    "name": "DesignPhase",
                    "description": "Design system architecture",
                    "execution_type": "parallel",
                    "tasks": [
                        {
                            "name": "ArchitectureDesign",
                            "agent": {
                                "id": "architect",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [{"name": "test_database_query", "type": "function"}],
                                "system_prompt": "Design system architecture based on requirements.",
                                "user_prompt": "Design for: ${stages.[RequirementsGathering].tasks.[RequirementsAnalysis].outputs.requirements}"
                            },
                            "outputs": {"architecture": "System architecture"}
                        },
                        {
                            "name": "DataModelDesign",
                            "agent": {
                                "id": "data_modeler",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Design data models and schemas.",
                                "user_prompt": "Create data model for: ${stages.[RequirementsGathering].tasks.[RequirementsAnalysis].outputs.requirements}"
                            },
                            "outputs": {"data_model": "Data model design"}
                        }
                    ]
                },
                {
                    "name": "Implementation",
                    "description": "Implement the solution",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "CodeGeneration",
                            "agent": {
                                "id": "code_generator",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [
                                    {"name": "test_file_processor", "type": "function"},
                                    {"name": "test_api_call", "type": "function"}
                                ],
                                "system_prompt": "Generate implementation based on architecture and data model.",
                                "user_prompt": "Implement based on architecture: ${stages.[DesignPhase].tasks.[ArchitectureDesign].outputs.architecture} and data model: ${stages.[DesignPhase].tasks.[DataModelDesign].outputs.data_model}"
                            },
                            "outputs": {"implementation": "Generated code and configuration"}
                        }
                    ]
                },
                {
                    "name": "Testing",
                    "description": "Test the implementation",
                    "execution_type": "parallel",
                    "tasks": [
                        {
                            "name": "UnitTesting",
                            "agent": {
                                "id": "unit_tester",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [{"name": "test_calculator", "type": "function"}],
                                "system_prompt": "Create and execute unit tests.",
                                "user_prompt": "Test implementation: ${stages.[Implementation].tasks.[CodeGeneration].outputs.implementation}"
                            },
                            "outputs": {"unit_test_results": "Unit test results"}
                        },
                        {
                            "name": "IntegrationTesting",
                            "agent": {
                                "id": "integration_tester",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [{"name": "test_api_call", "type": "function"}],
                                "system_prompt": "Perform integration testing.",
                                "user_prompt": "Integration test: ${stages.[Implementation].tasks.[CodeGeneration].outputs.implementation}"
                            },
                            "outputs": {"integration_test_results": "Integration test results"}
                        }
                    ]
                },
                {
                    "name": "Deployment",
                    "description": "Deploy the solution",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "DeploymentPrep",
                            "agent": {
                                "id": "deployment_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [{"name": "test_data_analysis", "type": "functional"}],
                                "system_prompt": "Prepare deployment based on test results.",
                                "user_prompt": "Deploy based on tests: Unit=${stages.[Testing].tasks.[UnitTesting].outputs.unit_test_results}, Integration=${stages.[Testing].tasks.[IntegrationTesting].outputs.integration_test_results}"
                            },
                            "outputs": {"deployment_status": "Deployment complete"}
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(workflow_dict)
        
        inputs = WorkflowInput(
            user_query="Build AI chatbot system",
            workflow={
                "inputs": {
                    "project_description": "Build an AI-powered customer service chatbot with natural language understanding, multi-language support, and integration with existing CRM system"
                }
            }
        )
        
        start_time = time.time()
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        execution_time = time.time() - start_time
        
        assert result is not None
        # Should have executed all stages
        assert len(result.all_agents) >= 7  # At least 7 agents in the pipeline
        self._log_workflow_summary("AI Development Pipeline", result, execution_time)


# ============================================================================
# Example Workflow Tests
# ============================================================================

@pytest.mark.asyncio
class TestOpenAIObservabilityExampleWorkflows(BaseOpenAITest):
    """Test existing example workflows."""
    
    async def test_example_workflow_yaml(self, workflow_manager):
        """Test loading and executing example workflow from YAML."""
        # This would normally load from a YAML file
        # For testing, we'll create a workflow dict directly
        workflow_dict = {
            "name": "example_workflow_test",
            "description": "Test example workflow",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "ExampleStage",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "ExampleTask",
                            "agent": {
                                "id": "example_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "You are a helpful assistant.",
                                "user_prompt": "Help with: ${workflow.inputs.request}"
                            },
                            "outputs": {"result": "Task result"}
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(workflow_dict)
        
        inputs = WorkflowInput(
            user_query="Test example workflow",
            workflow={"inputs": {"request": "Test request"}}
        )
        
        result = await workflow_manager.execute(workflow, inputs)
        
        assert result is not None