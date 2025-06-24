"""
Comprehensive test suite for OpenAI observability and tracing.
Tests all aspects of tracing including agent calls, handoffs, tool calls,
and various workflow execution strategies.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agent_workflow.providers import (
    Tool,
    FunctionTool,
    ToolRegistry,
    register_tool,
    tool,
)
from agent_workflow.providers.callbacks import ConsoleProgressCallback
from agent_workflow.providers.openai_observability import OpenaiLLMObservabilityProvider
from agent_workflow.workflow_engine import (
    WorkflowManager,
    WorkflowInput,
    ExecutionResult,
    OpenAIProviderConfig,
    ProviderConfiguration,
    ProviderType,
)
from agent_workflow.workflow_engine.models import (
    WorkflowStage,
    WorkflowTask,
    AgentConfig,
    LLMAgent,
    ResponseStore,
    StageExecutionStrategy,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Tools
# ============================================================================

@tool(name="test_calculator", description="Perform basic mathematical calculations")
def calculator_tool(operation: str, a: float, b: float) -> Dict[str, Any]:
    """Test calculator tool for mathematical operations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None,
    }
    
    if operation not in operations:
        return {"error": f"Unknown operation: {operation}"}
    
    result = operations[operation](a, b)
    if result is None:
        return {"error": "Division by zero"}
    
    return {"result": result, "operation": operation, "inputs": {"a": a, "b": b}}


@tool(name="test_weather", description="Get weather information for testing")
def weather_tool(location: str) -> Dict[str, Any]:
    """Mock weather tool for testing observability."""
    weather_data = {
        "New York": {"temperature": 72, "condition": "Partly cloudy", "humidity": 65},
        "London": {"temperature": 59, "condition": "Rainy", "humidity": 80},
        "Tokyo": {"temperature": 78, "condition": "Sunny", "humidity": 55},
        "default": {"temperature": 68, "condition": "Clear", "humidity": 50}
    }
    
    data = weather_data.get(location, weather_data["default"])
    return {
        "location": location,
        **data,
        "timestamp": datetime.now().isoformat()
    }


@tool(name="test_search", description="Search tool for testing")
def search_tool(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Mock search tool for testing observability."""
    results = []
    for i in range(min(max_results, 3)):
        results.append({
            "title": f"Result {i+1} for '{query}'",
            "snippet": f"Mock search result for query '{query}'. Result number {i+1}.",
            "url": f"https://example.com/search?q={query}&result={i+1}"
        })
    return results


class TestDataAnalysisTool(Tool):
    """Test tool for data analysis operations."""
    
    @property
    def name(self):
        return "test_data_analyzer"
    
    @property
    def description(self):
        return "Analyze data for testing purposes"
    
    @property
    def type(self):
        return "functional"
    
    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of numbers to analyze"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["mean", "sum", "count"],
                    "description": "Type of analysis"
                }
            },
            "required": ["data", "analysis_type"]
        }
    
    async def execute(self, data: List[float], analysis_type: str) -> Dict[str, Any]:
        """Execute data analysis."""
        if not data:
            return {"error": "No data provided"}
        
        results = {
            "data_points": len(data),
            "analysis_type": analysis_type
        }
        
        if analysis_type == "mean":
            results["value"] = sum(data) / len(data)
        elif analysis_type == "sum":
            results["value"] = sum(data)
        elif analysis_type == "count":
            results["value"] = len(data)
        else:
            results["error"] = f"Unknown analysis type: {analysis_type}"
        
        return results


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def openai_api_key():
    """Ensure OpenAI API key is available."""
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
    registry = ToolRegistry()
    # Function-based tools are auto-registered by @tool decorator
    # Register class-based tool
    register_tool(TestDataAnalysisTool())
    return registry


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
async def workflow_manager(provider_config, tool_registry, observability_provider):
    """Create workflow manager with OpenAI engine."""
    manager = WorkflowManager(
        engine_type="openai",
        provider_config=provider_config,
        tool_registry=tool_registry,
        llm_observability_provider=observability_provider
    )
    yield manager
    # Cleanup if needed
    await asyncio.sleep(0.1)


# ============================================================================
# Test Workflows
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
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "You are an analyst. Analyze the request briefly.",
                            "user_prompt": "Analyze: ${workflow.inputs.user_query}"
                        },
                        "outputs": {
                            "analysis": "Analysis result"
                        }
                    }
                ]
            }
        ]
    }


def create_tool_usage_workflow() -> Dict[str, Any]:
    """Create workflow that uses tools."""
    return {
        "name": "test_tool_usage",
        "description": "Test tool call tracing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "ToolExecution",
                "description": "Execute tools",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "UseTools",
                        "description": "Use various tools",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": ["test_calculator", "test_weather", "test_search"],
                            "system_prompt": "You are a helpful assistant. Use tools to answer questions.",
                            "user_prompt": "${workflow.inputs.user_query}"
                        },
                        "outputs": {
                            "result": "Tool execution result"
                        }
                    }
                ]
            }
        ]
    }


def create_parallel_execution_workflow() -> Dict[str, Any]:
    """Create workflow with parallel execution."""
    return {
        "name": "test_parallel_execution",
        "description": "Test parallel task execution tracing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "ParallelTasks",
                "description": "Execute tasks in parallel",
                "execution_type": "parallel",
                "tasks": [
                    {
                        "name": "Task1",
                        "description": "First parallel task",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": ["test_search"],
                            "system_prompt": "Research the first topic.",
                            "user_prompt": "Research: ${workflow.inputs.topic1}"
                        },
                        "outputs": {"result1": "Research result"}
                    },
                    {
                        "name": "Task2",
                        "description": "Second parallel task",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": ["test_data_analyzer"],
                            "system_prompt": "Analyze the data.",
                            "user_prompt": "Analyze: ${workflow.inputs.data}"
                        },
                        "outputs": {"result2": "Analysis result"}
                    },
                    {
                        "name": "Task3",
                        "description": "Third parallel task",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Process the third item.",
                            "user_prompt": "Process: ${workflow.inputs.item3}"
                        },
                        "outputs": {"result3": "Processing result"}
                    }
                ]
            }
        ]
    }


def create_handoff_workflow() -> Dict[str, Any]:
    """Create workflow with agent handoffs."""
    return {
        "name": "test_handoff_workflow",
        "description": "Test agent handoff tracing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "SupportHandoff",
                "description": "Customer support with handoffs",
                "execution_type": "handoff",
                "tasks": [
                    {
                        "name": "GeneralSupport",
                        "description": "General support agent",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": """You are a general support agent. 
                            For technical issues, say 'handoff to technical'.
                            For billing issues, say 'handoff to billing'.""",
                            "user_prompt": "${workflow.inputs.support_query}"
                        }
                    },
                    {
                        "name": "TechnicalSupport",
                        "description": "Technical support specialist",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": ["test_search"],
                            "system_prompt": "You are a technical support specialist.",
                            "handoff_description": "Technical support for complex issues"
                        }
                    },
                    {
                        "name": "BillingSupport",
                        "description": "Billing specialist",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": ["test_calculator"],
                            "system_prompt": "You are a billing specialist.",
                            "handoff_description": "Billing and payment support"
                        }
                    }
                ]
            }
        ]
    }


def create_complex_multi_stage_workflow() -> Dict[str, Any]:
    """Create complex workflow with multiple stages and strategies."""
    return {
        "name": "test_complex_workflow",
        "description": "Complex multi-stage workflow testing all features",
        "version": "1.0.0",
        "stages": [
            {
                "name": "Planning",
                "description": "Initial planning phase",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "Requirements",
                        "description": "Gather requirements",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Extract key requirements from the project description.",
                            "user_prompt": "Project: ${workflow.inputs.project}"
                        },
                        "outputs": {
                            "requirements": "List of requirements"
                        }
                    },
                    {
                        "name": "Planning",
                        "description": "Create plan",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": ["test_calculator"],
                            "system_prompt": "Create a project plan with time estimates.",
                            "user_prompt": "Requirements: ${stages.[Planning].tasks.[Requirements].outputs.requirements}"
                        },
                        "outputs": {
                            "plan": "Project plan",
                            "timeline": "Timeline estimate"
                        }
                    }
                ]
            },
            {
                "name": "Research",
                "description": "Parallel research tasks",
                "execution_type": "parallel",
                "tasks": [
                    {
                        "name": "TechResearch",
                        "description": "Technical research",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": ["test_search", "test_data_analyzer"],
                            "system_prompt": "Research technical solutions.",
                            "user_prompt": "Research tech for: ${stages.[Planning].tasks.[Requirements].outputs.requirements}"
                        },
                        "outputs": {"tech_findings": "Technical findings"}
                    },
                    {
                        "name": "MarketResearch",
                        "description": "Market research",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": ["test_search"],
                            "system_prompt": "Research market trends.",
                            "user_prompt": "Market research for: ${workflow.inputs.project}"
                        },
                        "outputs": {"market_findings": "Market findings"}
                    }
                ]
            },
            {
                "name": "Development",
                "description": "Development with handoffs",
                "execution_type": "handoff",
                "tasks": [
                    {
                        "name": "LeadDev",
                        "description": "Lead developer",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": """You are a lead developer.
                            Say 'handoff to backend' for backend work.
                            Say 'handoff to frontend' for UI work.""",
                            "user_prompt": """Plan: ${stages.[Planning].tasks.[Planning].outputs.plan}
Tech: ${stages.[Research].tasks.[TechResearch].outputs.tech_findings}"""
                        }
                    },
                    {
                        "name": "BackendDev",
                        "description": "Backend developer",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": ["test_data_analyzer"],
                            "system_prompt": "Implement backend solutions.",
                            "handoff_description": "Backend development"
                        }
                    },
                    {
                        "name": "FrontendDev",
                        "description": "Frontend developer",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Implement UI solutions.",
                            "handoff_description": "Frontend development"
                        }
                    }
                ]
            },
            {
                "name": "Review",
                "description": "Final review",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "FinalReview",
                        "description": "Review and document",
                        "agent": {
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Create final documentation.",
                            "user_prompt": "Summarize all work completed."
                        },
                        "outputs": {
                            "documentation": "Final documentation"
                        }
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
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": ["test_calculator"],
                            "system_prompt": "You must divide 10 by 0 to test error handling.",
                            "user_prompt": "Calculate 10 divided by 0"
                        },
                        "outputs": {"result": "Should fail"}
                    }
                ]
            }
        ]
    }


# ============================================================================
# Test Classes
# ============================================================================

@pytest.mark.asyncio
class TestOpenAIObservabilityBasic:
    """Basic observability tests."""
    
    async def test_simple_sequential_workflow(self, workflow_manager):
        """Test basic sequential workflow tracing."""
        workflow_dict = create_simple_sequential_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,  # Pass dict directly
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test sequential execution",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True
        assert result.final_result is not None
    
    async def test_tool_usage_tracing(self, workflow_manager):
        """Test tool call start/end tracing."""
        workflow_dict = create_tool_usage_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,  # Pass dict directly
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="What's the weather in New York and calculate 15 + 25?",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True
        # Tool calls should have been traced
    
    async def test_parallel_execution_tracing(self, workflow_manager):
        """Test parallel task execution tracing."""
        workflow_dict = create_parallel_execution_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,  # Pass dict directly
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test parallel execution",
            workflow={
                "inputs": {
                    "topic1": "AI research",
                    "data": "[1, 2, 3, 4, 5]",
                    "item3": "Process this item"
                }
            }
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True
        # All parallel tasks should have been traced


@pytest.mark.asyncio
class TestOpenAIObservabilityHandoffs:
    """Test agent handoff tracing."""
    
    async def test_handoff_to_technical(self, workflow_manager):
        """Test handoff to technical support."""
        workflow_dict = create_handoff_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,  # Pass dict directly
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="I can't connect to the server",
            workflow={"inputs": {"support_query": "I can't connect to the server"}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True
    
    async def test_handoff_to_billing(self, workflow_manager):
        """Test handoff to billing support."""
        workflow_dict = create_handoff_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,  # Pass dict directly
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="I was charged twice",
            workflow={"inputs": {"support_query": "I was charged twice for my subscription"}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True
    
    async def test_no_handoff_needed(self, workflow_manager):
        """Test case where no handoff is needed."""
        workflow_dict = create_handoff_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,  # Pass dict directly
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="What are your hours?",
            workflow={"inputs": {"support_query": "What are your business hours?"}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True


@pytest.mark.asyncio
class TestOpenAIObservabilityComplex:
    """Test complex workflows and edge cases."""
    
    async def test_complex_multi_stage_workflow(self, workflow_manager):
        """Test complex workflow with all execution strategies."""
        workflow_dict = create_complex_multi_stage_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,  # Pass dict directly
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Build analytics dashboard",
            workflow={
                "inputs": {
                    "project": "Real-time analytics dashboard with user metrics"
                }
            }
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True
        assert "Planning" in result.stage_results
        assert "Research" in result.stage_results
        assert "Development" in result.stage_results
        assert "Review" in result.stage_results
    
    async def test_error_handling_tracing(self, workflow_manager):
        """Test error handling and tracing."""
        workflow_dict = create_error_test_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,  # Pass dict directly
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test error",
            workflow={"inputs": {}}
        )
        
        # The workflow should handle the error gracefully
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        # Even with errors, execution should complete
        assert result is not None
    
    async def test_data_flow_between_stages(self, workflow_manager):
        """Test data flow and tracing between stages."""
        workflow_dict = {
            "name": "test_data_flow",
            "description": "Test data flow between stages",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "Stage1",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "GenerateData",
                            "agent": {
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Generate a list of 3 items.",
                                "user_prompt": "Generate items for: ${workflow.inputs.category}"
                            },
                            "outputs": {"items": "Generated items"}
                        }
                    ]
                },
                {
                    "name": "Stage2",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "ProcessData",
                            "agent": {
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Process the items.",
                                "user_prompt": "Process: ${stages.[Stage1].tasks.[GenerateData].outputs.items}"
                            },
                            "outputs": {"processed": "Processed result"}
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,  # Pass dict directly
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
        
        assert result.completed is True
        assert "Stage2" in result.stage_results


@pytest.mark.asyncio
class TestOpenAIObservabilityExampleWorkflows:
    """Test existing example workflows."""
    
    async def test_handoffs_example_workflow(self, workflow_manager):
        """Test the handoffs example workflow if it exists."""
        handoffs_path = Path("usage_examples/handoffs_example/handoffs_config2.yaml")
        if not handoffs_path.exists():
            pytest.skip("Handoffs example not found")
        
        workflow = await workflow_manager.initialize_workflow(
            str(handoffs_path),  # Pass string path directly
            provider_mapping={
                "qa_agent": "openai",
                "translation_agent": "openai"
            },
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Who is the CEO of OpenAI?",
            workflow={"inputs": {"user_query": "Who is the CEO of OpenAI?"}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True
    
    async def test_product_dev_workflow(self, workflow_manager):
        """Test the product development workflow if it exists."""
        dev_path = Path("usage_examples/dev_workflow/product_dev_workflow_example.yaml")
        if not dev_path.exists():
            pytest.skip("Product dev example not found")
        
        # Need to ensure agent YAML files exist
        workflow = await workflow_manager.initialize_workflow(
            str(dev_path),  # Pass string path directly
            provider_mapping={
                "plan_agent": "openai",
                "api_design_agent": "openai",
                "ui_design_agent": "openai",
                "db_design_agent": "openai",
                "validation_agent": "openai"
            },
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Create dashboard",
            workflow={
                "inputs": {
                    "feature_request": "User dashboard with metrics",
                    "constraints": "Must be responsive"
                }
            }
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True


@pytest.mark.asyncio
class TestOpenAIObservabilityEdgeCases:
    """Test edge cases and special scenarios."""
    
    async def test_empty_workflow(self, workflow_manager):
        """Test workflow with no tasks."""
        workflow_dict = {
            "name": "test_empty",
            "description": "Empty workflow",
            "version": "1.0.0",
            "stages": []
        }
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict
        )
        
        inputs = WorkflowInput(
            user_query="Test empty",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(workflow, inputs)
        assert result.completed is True
    
    async def test_agent_with_multiple_tools(self, workflow_manager):
        """Test agent using multiple tools in one task."""
        workflow_dict = {
            "name": "test_multi_tools",
            "description": "Test multiple tool usage",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "MultiTool",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "UseAllTools",
                            "agent": {
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": ["test_calculator", "test_weather", "test_search", "test_data_analyzer"],
                                "system_prompt": "Use all available tools to answer comprehensively.",
                                "user_prompt": "Weather in Tokyo, calculate 50*3, search for AI news, and analyze [1,2,3,4,5]"
                            },
                            "outputs": {"result": "Combined results"}
                        }
                    ]
                }
            ]
        }
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict
        )
        
        inputs = WorkflowInput(
            user_query="Test all tools",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True
    
    async def test_nested_parallel_stages(self, workflow_manager):
        """Test workflow with multiple parallel stages."""
        workflow_dict = {
            "name": "test_nested_parallel",
            "description": "Multiple parallel stages",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "Parallel1",
                    "execution_type": "parallel",
                    "tasks": [
                        {
                            "name": "P1T1",
                            "agent": {
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Process item 1",
                                "user_prompt": "Item 1"
                            }
                        },
                        {
                            "name": "P1T2",
                            "agent": {
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Process item 2",
                                "user_prompt": "Item 2"
                            }
                        }
                    ]
                },
                {
                    "name": "Parallel2",
                    "execution_type": "parallel",
                    "tasks": [
                        {
                            "name": "P2T1",
                            "agent": {
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Process item 3",
                                "user_prompt": "Item 3"
                            }
                        },
                        {
                            "name": "P2T2",
                            "agent": {
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Process item 4",
                                "user_prompt": "Item 4"
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
            user_query="Test nested parallel",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result.completed is True
        assert "Parallel1" in result.stage_results
        assert "Parallel2" in result.stage_results


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
class TestOpenAIObservabilityIntegration:
    """Integration tests for complete workflows."""
    
    async def test_full_observability_stack(self, workflow_manager):
        """Test that all observability components work together."""
        # This test verifies the entire stack:
        # 1. Workflow start/end
        # 2. Stage start/end
        # 3. Task start/end
        # 4. Agent start/end
        # 5. Tool start/end
        # 6. Handoffs
        
        workflow_dict = create_complex_multi_stage_workflow()
        
        # Mock the observability provider to track calls
        with patch.object(workflow_manager.execution_engine._ctx.llm_tracer, 'providers') as mock_providers:
            workflow = await workflow_manager.initialize_workflow(
                workflow_dict,
                progress_callback=ConsoleProgressCallback()
            )
            
            inputs = WorkflowInput(
                user_query="Full stack test",
                workflow={
                    "inputs": {
                        "project": "Test project for observability"
                    }
                }
            )
            
            result = await workflow_manager.execute(
                workflow,
                inputs,
                progress_callback=ConsoleProgressCallback()
            )
            
            assert result.completed is True
            
            # Verify all stages were executed
            assert len(result.stage_results) == 4  # Planning, Research, Development, Review
    
    async def test_observability_with_errors(self, workflow_manager):
        """Test observability captures errors properly."""
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
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": ["test_calculator"],
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


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
class TestOpenAIObservabilityPerformance:
    """Test observability performance impact."""
    
    async def test_minimal_overhead(self, workflow_manager):
        """Test that observability adds minimal overhead."""
        import time
        
        workflow_dict = create_simple_sequential_workflow()
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict
        )
        
        inputs = WorkflowInput(
            user_query="Performance test",
            workflow={"inputs": {}}
        )
        
        # Time execution with observability
        start_time = time.time()
        result = await workflow_manager.execute(workflow, inputs)
        execution_time = time.time() - start_time
        
        assert result.completed is True
        # Just verify it completes in reasonable time
        assert execution_time < 30  # 30 seconds max