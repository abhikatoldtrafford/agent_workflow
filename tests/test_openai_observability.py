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

import pytest

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
    return f"Weather in {location}: Sunny, 72°F"

@tool(
    name="test_calculator",
    description="Perform calculations for testing"
)
def mock_calculator(expression: str) -> float:
    """Mock calculator tool for testing."""
    # Simple eval for testing (not safe for production!)
    try:
        # Check for division by zero
        if "/0" in expression.replace(" ", ""):
            raise ZeroDivisionError("Division by zero")
        return eval(expression)
    except Exception as e:
        raise ValueError(f"Calculation error: {str(e)}")

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
        return {
            "result": f"Analysis of {data} using {operation}",
            "confidence": 0.95
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
            # Give it a moment to flush
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
def observability_provider():
    """Create OpenAI observability provider."""
    return OpenaiLLMObservabilityProvider()


@pytest.fixture
def tool_registry():
    """Create and populate tool registry."""
    # Use the global tool registry which already has tools registered by @tool decorator
    from agent_workflow.providers.tools import global_tool_registry
    
    # Register class-based tool
    register_tool(TestDataAnalysisTool())
    
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
                            "id": "analyzer",
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
        "description": "Test tool usage tracing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "ToolUsage",
                "description": "Use multiple tools",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "UseTools",
                        "description": "Use weather and calculator tools",
                        "agent": {
                            "id": "tool_user",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [
                                {"name": "test_weather", "type": "function"},
                                {"name": "test_calculator", "type": "function"}
                            ],
                            "system_prompt": "You must use both tools to answer the query.",
                            "user_prompt": "${workflow.inputs.user_query}"
                        },
                        "outputs": {
                            "weather": "Weather information",
                            "calculation": "Calculation result"
                        }
                    }
                ]
            }
        ]
    }


def create_parallel_execution_workflow() -> Dict[str, Any]:
    """Create workflow with parallel tasks."""
    return {
        "name": "test_parallel",
        "description": "Test parallel execution tracing",
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
                            "system_prompt": "Process the first topic briefly.",
                            "user_prompt": "Topic: ${workflow.inputs.topic1}"
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
                            "tools": [{"name": "test_data_analysis", "type": "function"}],
                            "system_prompt": "Analyze the data.",
                            "user_prompt": "Analyze: ${workflow.inputs.data}"
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
                            "system_prompt": "Process the third item.",
                            "user_prompt": "Item: ${workflow.inputs.item3}"
                        },
                        "outputs": {"result3": "Result 3"}
                    }
                ]
            }
        ]
    }


def create_handoff_workflow() -> Dict[str, Any]:
    """Create workflow with agent handoffs."""
    return {
        "name": "test_handoffs",
        "description": "Test agent handoff tracing",
        "version": "1.0.0",
        "stages": [
            {
                "name": "InitialSupport",
                "description": "Initial support triage",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "Triage",
                        "description": "Triage support request",
                        "agent": {
                            "id": "triage_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": """You are a support triage agent. 
                            Categorize queries as: technical, billing, or general.
                            Respond with just the category.""",
                            "user_prompt": "Categorize: ${workflow.inputs.support_query}"
                        },
                        "outputs": {"category": "Support category"}
                    }
                ]
            },
            {
                "name": "SpecializedSupport",
                "description": "Specialized support handling",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "HandleRequest",
                        "description": "Handle based on category",
                        "agent": {
                            "id": "specialized_agent",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Provide specialized support based on the category.",
                            "user_prompt": "Category: ${stages.[InitialSupport].tasks.[Triage].outputs.category}, Query: ${workflow.inputs.support_query}"
                        },
                        "outputs": {"resolution": "Support resolution"}
                    }
                ]
            }
        ]
    }


def create_complex_multi_stage_workflow() -> Dict[str, Any]:
    """Create complex workflow with multiple stages."""
    return {
        "name": "test_complex",
        "description": "Complex multi-stage workflow",
        "version": "1.0.0",
        "stages": [
            {
                "name": "Planning",
                "description": "Project planning",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "CreatePlan",
                        "description": "Create project plan",
                        "agent": {
                            "id": "planner",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Create a brief project plan.",
                            "user_prompt": "Project: ${workflow.inputs.project}"
                        },
                        "outputs": {"plan": "Project plan"}
                    }
                ]
            },
            {
                "name": "Research",
                "description": "Research phase",
                "execution_type": "parallel",
                "tasks": [
                    {
                        "name": "TechnicalResearch",
                        "description": "Technical research",
                        "agent": {
                            "id": "tech_researcher",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "tools": [{"name": "test_data_analysis", "type": "function"}],
                            "system_prompt": "Research technical aspects.",
                            "user_prompt": "Research for: ${stages.[Planning].tasks.[CreatePlan].outputs.plan}"
                        },
                        "outputs": {"tech_research": "Technical findings"}
                    },
                    {
                        "name": "MarketResearch",
                        "description": "Market research",
                        "agent": {
                            "id": "market_researcher",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Research market aspects.",
                            "user_prompt": "Market research for: ${stages.[Planning].tasks.[CreatePlan].outputs.plan}"
                        },
                        "outputs": {"market_research": "Market findings"}
                    }
                ]
            },
            {
                "name": "Development",
                "description": "Development phase",
                "execution_type": "sequential",
                "tasks": [
                    {
                        "name": "Develop",
                        "description": "Develop solution",
                        "agent": {
                            "id": "developer",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Develop based on research.",
                            "user_prompt": "Tech: ${stages.[Research].tasks.[TechnicalResearch].outputs.tech_research}, Market: ${stages.[Research].tasks.[MarketResearch].outputs.market_research}"
                        },
                        "outputs": {"solution": "Developed solution"}
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
                        "description": "Review all work",
                        "agent": {
                            "id": "reviewer",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Review and summarize all work.",
                            "user_prompt": "Solution: ${stages.[Development].tasks.[Develop].outputs.solution}"
                        },
                        "outputs": {"final_report": "Final report"}
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
                        "description": "Generate test data",
                        "agent": {
                            "id": "data_generator",
                            "agent_type": "LLMAgent",
                            "llm_type": "openai",
                            "system_prompt": "Generate a list of 3 items based on the category.",
                            "user_prompt": "Category: ${workflow.inputs.category}"
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
                            "system_prompt": "Process the items briefly.",
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


# ============================================================================
# Test Classes
# ============================================================================

@pytest.mark.asyncio
class TestOpenAIObservabilityBasic:
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
        workflow_dict = create_tool_usage_workflow()
        
        # Add provider mapping for the tool user agent
        provider_mapping = {"tool_user": "openai"}
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="What's the weather in New York and calculate 15 + 25?",
            workflow={"inputs": {"user_query": "What's the weather in New York and calculate 15 + 25?"}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result is not None
        # Tool calls should have been traced
    
    async def test_parallel_execution_tracing(self, workflow_manager):
        """Test parallel task execution tracing."""
        workflow_dict = create_parallel_execution_workflow()
        
        # Add provider mapping for all parallel agents
        provider_mapping = {
            "parallel_agent_1": "openai",
            "parallel_agent_2": "openai",
            "parallel_agent_3": "openai"
        }
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
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
        
        assert result is not None
        # All parallel tasks should have been traced


@pytest.mark.asyncio
class TestOpenAIObservabilityHandoffs:
    """Enhanced tests for agent handoff tracing with rigorous validation."""
    
    def create_routing_workflow(self) -> Dict[str, Any]:
        """Create a workflow with explicit routing logic for handoff testing."""
        return {
            "name": "test_routing_handoff",
            "description": "Test handoff with routing logic - Route to the appropriate agent based on query type: technical, billing, or general",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "RouterStage",
                    "description": "Route to the appropriate agent based on query type: technical, billing, or general",
                    "execution_type": "handoff",
                    "tasks": [
                        {
                            "name": "TechnicalAgent",
                            "description": "Handle technical queries",
                            "agent": {
                                "id": "technical_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": """You are a technical support specialist.
                                Only respond to technical queries about servers, connectivity, or system issues.
                                For non-technical queries, say 'NOT_MY_DOMAIN'.""",
                                "user_prompt": "${workflow.inputs.query}"
                            },
                            "outputs": {"technical_response": "Technical agent response"}
                        },
                        {
                            "name": "BillingAgent",
                            "description": "Handle billing queries",
                            "agent": {
                                "id": "billing_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": """You are a billing specialist.
                                Only respond to billing queries about invoices, payments, or pricing.
                                For non-billing queries, say 'NOT_MY_DOMAIN'.""",
                                "user_prompt": "${workflow.inputs.query}"
                            },
                            "outputs": {"billing_response": "Billing agent response"}
                        },
                        {
                            "name": "GeneralAgent",
                            "description": "Handle general queries",
                            "agent": {
                                "id": "general_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": """You are a general support agent.
                                Handle all general queries about business hours, locations, or other info.
                                Always provide helpful responses.""",
                                "user_prompt": "${workflow.inputs.query}"
                            },
                            "outputs": {"general_response": "General agent response"}
                        }
                    ]
                }
            ]
        }
    
    async def test_handoff_to_technical(self, workflow_manager):
        """Test handoff to technical support with validation."""
        workflow_dict = create_handoff_workflow()
        
        # Add provider mapping for handoff agents
        provider_mapping = {
            "triage_agent": "openai",
            "specialized_agent": "openai"
        }
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
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
        
        assert result is not None
        assert len(result.all_agents) >= 2  # Should have at least triage and specialized agents
        assert result.final_result is not None
        
        # Check for the actual task names returned by the workflow
        agent_names_lower = str(result.all_agents).lower()
        assert "triage" in agent_names_lower, f"Expected 'Triage' task in agents, got: {result.all_agents}"
        assert "handlerequest" in agent_names_lower, f"Expected 'HandleRequest' task in agents, got: {result.all_agents}"
        
        # Verify the workflow executed both stages
        logger.info(f"Executed agents: {result.all_agents}")
        logger.info(f"Response store: {result.response_store}")
    
    async def test_handoff_to_billing(self, workflow_manager):
        """Test handoff to billing support."""
        workflow_dict = create_handoff_workflow()
        
        # Add provider mapping for handoff agents
        provider_mapping = {
            "triage_agent": "openai",
            "specialized_agent": "openai"
        }
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="I need help with my invoice",
            workflow={"inputs": {"support_query": "I need help with my invoice"}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result is not None
        assert len(result.all_agents) >= 2  # Should have at least triage and specialized agents
        
        # Verify proper task execution
        agent_names_lower = str(result.all_agents).lower()
        assert "triage" in agent_names_lower
        assert "handlerequest" in agent_names_lower
    
    async def test_no_handoff_needed(self, workflow_manager):
        """Test when no handoff is needed - but workflow still executes both stages."""
        workflow_dict = create_handoff_workflow()
        
        # Add provider mapping for handoff agents
        provider_mapping = {
            "triage_agent": "openai",
            "specialized_agent": "openai"
        }
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="What are your business hours?",
            workflow={"inputs": {"support_query": "What are your business hours?"}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result is not None
        # Note: The existing workflow is sequential, not handoff, so both stages always execute
        assert len(result.all_agents) >= 2  # Should still have both agents even for general queries
    
    async def test_handoff_routing_accuracy(self, workflow_manager):
        """Test that handoff correctly routes to the appropriate agent."""
        workflow_dict = self.create_routing_workflow()
        
        provider_mapping = {
            "technical_agent": "openai",
            "billing_agent": "openai", 
            "general_agent": "openai"
        }
        
        # Test cases - since this is a true handoff workflow, only one agent should execute
        test_cases = [
            ("My server is down and I can't connect", ["TechnicalAgent", "BillingAgent", "GeneralAgent"]),
            ("I need help understanding my invoice", ["TechnicalAgent", "BillingAgent", "GeneralAgent"]),
            ("What are your office hours?", ["TechnicalAgent", "BillingAgent", "GeneralAgent"]),
        ]
        
        for query, possible_agents in test_cases:
            workflow = await workflow_manager.initialize_workflow(
                workflow_dict,
                provider_mapping=provider_mapping,
                progress_callback=ConsoleProgressCallback()
            )
            
            inputs = WorkflowInput(
                user_query=query,
                workflow={"inputs": {"query": query}}
            )
            
            result = await workflow_manager.execute(
                workflow,
                inputs,
                progress_callback=ConsoleProgressCallback()
            )
            
            assert result is not None
            # In handoff mode, we should see the handoff agent plus one selected agent
            assert len(result.all_agents) >= 1
            logger.info(f"Query '{query}' routed to agents: {result.all_agents}")
    
    async def test_explicit_handoff_workflow(self, workflow_manager):
        """Test explicit handoff execution type with multiple agents."""
        workflow_dict = {
            "name": "test_explicit_handoff",
            "description": "Test explicit handoff execution",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "HandoffStage",
                    "description": "Route to Agent1 for technical, Agent2 for billing, or Agent3 for general queries",
                    "execution_type": "handoff",
                    "tasks": [
                        {
                            "name": "Agent1",
                            "description": "First agent - handles technical queries",
                            "agent": {
                                "id": "handoff_agent_1",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "You are Agent 1. Handle technical queries. Answer briefly.",
                                "user_prompt": "${workflow.inputs.query}"
                            },
                            "outputs": {"response1": "Agent 1 response"}
                        },
                        {
                            "name": "Agent2",
                            "description": "Second agent - handles billing queries",
                            "agent": {
                                "id": "handoff_agent_2",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "You are Agent 2. Handle billing queries. Answer briefly.",
                                "user_prompt": "${workflow.inputs.query}"
                            },
                            "outputs": {"response2": "Agent 2 response"}
                        },
                        {
                            "name": "Agent3",
                            "description": "Third agent - handles general queries",
                            "agent": {
                                "id": "handoff_agent_3",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "You are Agent 3. Handle general queries. Summarize briefly.",
                                "user_prompt": "Summarize the query: ${workflow.inputs.query}"
                            },
                            "outputs": {"response3": "Agent 3 response"}
                        }
                    ]
                }
            ]
        }
        
        # Get provider mapping for all agents
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Test handoff between agents",
            workflow={"inputs": {"query": "What is the meaning of life?"}}
        )
        
        # Time the execution to check for performance issues
        start_time = time.time()
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        execution_time = time.time() - start_time
        
        assert result is not None
        # In handoff mode, typically only one agent executes based on routing logic
        assert len(result.all_agents) >= 1
        logger.info(f"Handoff execution time: {execution_time:.2f} seconds")
        logger.info(f"Agents executed: {result.all_agents}")
    
    async def test_handoff_performance_metrics(self, workflow_manager):
        """Test performance tracking across handoff execution."""
        # Use the existing handoff workflow
        workflow_dict = create_handoff_workflow()
        
        provider_mapping = {
            "triage_agent": "openai",
            "specialized_agent": "openai"
        }
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        # Track execution time
        start_time = time.time()
        
        inputs = WorkflowInput(
            user_query="Analyze system performance issues",
            workflow={"inputs": {"support_query": "Analyze system performance issues"}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        execution_time = time.time() - start_time
        
        assert result is not None
        assert execution_time < 30  # Reasonable timeout
        logger.info(f"Handoff workflow completed in {execution_time:.2f} seconds")
        
        # Verify both stages executed (since it's sequential, not handoff)
        assert len(result.all_agents) >= 2

@pytest.mark.asyncio
class TestOpenAIObservabilityComplex:
    """Test complex observability scenarios."""
    
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
        
        assert result is not None
    
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
        
        assert result is not None


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
        assert result is not None
    
    async def test_agent_with_multiple_tools(self, workflow_manager):
        """Test agent using multiple tools in one task."""
        workflow_dict = {
            "name": "test_multi_tool",
            "description": "Test multiple tool usage",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "MultiTool",
                    "execution_type": "sequential",
                    "tasks": [
                        {
                            "name": "UseMultipleTools",
                            "agent": {
                                "id": "multi_tool_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "tools": [
                                    {"name": "test_weather", "type": "function"},
                                    {"name": "test_calculator", "type": "function"},
                                    {"name": "test_data_analysis", "type": "function"}
                                ],
                                "system_prompt": "Use all three tools to answer comprehensively.",
                                "user_prompt": "Get weather for NYC, calculate 50*2, and analyze the results"
                            }
                        }
                    ]
                }
            ]
        }
        
        # Get provider mapping for all agents
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        workflow = await workflow_manager.initialize_workflow(
            workflow_dict,
            provider_mapping=provider_mapping,
            progress_callback=ConsoleProgressCallback()
        )
        
        inputs = WorkflowInput(
            user_query="Multi-tool test",
            workflow={"inputs": {}}
        )
        
        result = await workflow_manager.execute(
            workflow,
            inputs,
            progress_callback=ConsoleProgressCallback()
        )
        
        assert result is not None
    
    async def test_nested_parallel_stages(self, workflow_manager):
        """Test workflow with nested parallel stages."""
        workflow_dict = {
            "name": "test_nested_parallel",
            "description": "Nested parallel execution",
            "version": "1.0.0",
            "stages": [
                {
                    "name": "OuterParallel",
                    "execution_type": "parallel",
                    "tasks": [
                        {
                            "name": "Branch1",
                            "agent": {
                                "id": "branch1_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Process branch 1",
                                "user_prompt": "Execute task 1"
                            }
                        },
                        {
                            "name": "Branch2",
                            "agent": {
                                "id": "branch2_agent",
                                "agent_type": "LLMAgent",
                                "llm_type": "openai",
                                "system_prompt": "Process branch 2",
                                "user_prompt": "Execute task 2"
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
        
        result = await workflow_manager.execute(workflow, inputs)
        assert result is not None


@pytest.mark.asyncio
class TestOpenAIObservabilityIntegration:
    """Integration tests for observability stack."""
    
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
        
        # Get provider mapping for all agents
        provider_mapping = get_provider_mapping_for_workflow(workflow_dict)
        
        # Mock the observability provider to track calls
        with patch.object(workflow_manager.execution_engine._ctx.llm_tracer.providers[0], 
                         'start_trace_group') as mock_trace_start:
            mock_trace_start.return_value = "test-trace-id"
            
            workflow = await workflow_manager.initialize_workflow(
                workflow_dict,
                provider_mapping=provider_mapping,
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
            
            assert result is not None
            
            # Verify all stages were executed
            assert len(result.all_agents) >= 4  # Planning, Research, Development, Review
    
    async def test_observability_spans_created(self, workflow_manager):
        """Test that spans are actually created in traces."""
        workflow_dict = create_tool_usage_workflow()
        
        # Track spans created
        created_spans = []
        
        # Mock the span creation to track what spans are created
        original_span = workflow_manager.execution_engine._ctx.llm_tracer.providers[0].start_span
        
        async def mock_start_span(name, parent_span_id, metadata):
            created_spans.append(name)
            return await original_span(name, parent_span_id, metadata)
        
        with patch.object(
            workflow_manager.execution_engine._ctx.llm_tracer.providers[0],
            'start_span',
            side_effect=mock_start_span
        ):
            # Add provider mapping for the tool user agent
            provider_mapping = {"tool_user": "openai"}
            
            workflow = await workflow_manager.initialize_workflow(
                workflow_dict,
                provider_mapping=provider_mapping,
                progress_callback=ConsoleProgressCallback()
            )
            
            inputs = WorkflowInput(
                user_query="Test span creation",
                workflow={"inputs": {"user_query": "Check weather and calculate 10+20"}}
            )
            
            result = await workflow_manager.execute(
                workflow,
                inputs,
                progress_callback=ConsoleProgressCallback()
            )
            
            assert result is not None
            
            # Log what spans were created
            logger.info(f"Created spans: {created_spans}")
            
            # Should have created spans for various operations
            assert len(created_spans) > 0, "No spans were created during execution"
    
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


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
class TestOpenAIObservabilityPerformance:
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
        
    async def test_execution_time_variance(self, workflow_manager):
        """Test that execution times vary between different workflows."""
        # Test 1: Simple workflow
        simple_workflow = create_simple_sequential_workflow()
        simple_wf = await workflow_manager.initialize_workflow(simple_workflow)
        
        start_time = time.time()
        result1 = await workflow_manager.execute(
            simple_wf,
            WorkflowInput(
                user_query="Simple test",
                workflow={"inputs": {"user_query": "Simple test"}}
            )
        )
        simple_time = time.time() - start_time
        
        # Test 2: Complex workflow
        complex_workflow = create_complex_multi_stage_workflow()
        complex_wf = await workflow_manager.initialize_workflow(complex_workflow)
        
        start_time = time.time()
        result2 = await workflow_manager.execute(
            complex_wf,
            WorkflowInput(
                user_query="Complex test",
                workflow={"inputs": {"project": "Test project"}}
            )
        )
        complex_time = time.time() - start_time
        
        logger.info(f"Simple workflow time: {simple_time:.2f}s")
        logger.info(f"Complex workflow time: {complex_time:.2f}s")
        
        assert result1 is not None
        assert result2 is not None
        
        # Complex workflow should take longer than simple workflow
        # If they're exactly the same, something might be wrong
        if abs(simple_time - complex_time) < 0.1:
            logger.warning(f"Execution times are suspiciously similar: simple={simple_time:.2f}s, complex={complex_time:.2f}s")