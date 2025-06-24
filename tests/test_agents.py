from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_workflow.providers import LLMServiceProvider
from agent_workflow.workflow_engine import Agent, AgentFactory, LLMAgent


class TestLLMAgent:
    """Unit tests for the LLMAgent class."""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock(spec=LLMServiceProvider)
        provider.generate_response = AsyncMock(
            return_value='{"result": "test response", "confidence": 0.9}'
        )
        return provider

    @pytest.fixture
    def config_with_schema(self):
        return {
            "agent_type": "LLMAgent",
            "system_prompt": "You are a helpful assistant. ${context}",
            "user_prompt": "Answer this question: ${question}",
            "resources": {"max_tokens": 1000},
            "input_schema": {
                "question": {"type": "str", "description": "The question to answer"},
                "context": {
                    "type": "str",
                    "description": "Additional context for the question",
                },
            },
            "output_schema": {
                "result": {"type": "str", "description": "The answer to the question"},
                "confidence": {
                    "type": "float",
                    "description": "Confidence score between 0 and 1",
                },
            },
            "retry": {"max_attempts": 2, "backoff_seconds": 1},
        }

    @pytest.mark.asyncio
    async def test_execute_with_schema(self, mock_provider, config_with_schema):
        """Test executing an LLMAgent with input and output schemas."""
        # Create the agent
        agent = LLMAgent(config_with_schema, mock_provider)

        # Execute the agent
        inputs = {
            "question": "What is the capital of France?",
            "context": "Focus on geography.",
        }
        result = await agent.execute(inputs)

        # Verify the result
        assert result["result"] == "test response"
        assert result["confidence"] == 0.9

        # Verify the provider was called with the correct prompts
        mock_provider.generate_response.assert_called_once()
        call_args = mock_provider.generate_response.call_args[1]

        assert (
            "You are a helpful assistant. Focus on geography."
            in call_args["system_prompt"]
        )
        assert (
            "Answer this question: What is the capital of France?"
            in call_args["user_prompt"]
        )
        assert call_args["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_execute_with_retry(self, config_with_schema):
        """Test retry logic when the provider fails."""
        # Create a provider that fails on first attempt but succeeds on second
        provider = MagicMock(spec=LLMServiceProvider)
        provider.generate_response = AsyncMock(
            side_effect=[
                Exception("Network error"),
                '{"result": "test response after retry", "confidence": 0.8}',
            ]
        )

        # Create the agent
        agent = LLMAgent(config_with_schema, provider)

        # Execute the agent
        inputs = {
            "question": "What is the capital of France?",
            "context": "Focus on geography.",
        }
        result = await agent.execute(inputs)

        # Verify the result from the second attempt
        assert result["result"] == "test response after retry"
        assert result["confidence"] == 0.8

        # Verify the provider was called twice
        assert provider.generate_response.call_count == 2

    @pytest.mark.asyncio
    async def test_parse_response(self, mock_provider):
        """Test parsing different response formats."""
        # Create an agent with a simple configuration
        config = {
            "agent_type": "LLMAgent",
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "Answer this question: ${question}",
            "output_schema": {"answer": {"type": "str", "description": "The answer"}},
        }
        agent = LLMAgent(config, mock_provider)

        # Test parsing a clean JSON response
        response = '{"answer": "Paris"}'
        result = agent._parse_response(response)
        assert result == {"answer": "Paris"}

        # Test parsing a JSON response within code blocks
        response = 'Here is the answer:\n```json\n{"answer": "London"}\n```\nThe answer is provided above.'
        result = agent._parse_response(response)
        assert result == {"answer": "London"}

        # Test parsing a JSON response without code blocks but within text
        response = 'The answer is: {"answer": "Berlin"} as provided by sources.'
        result = agent._parse_response(response)
        assert result == {"answer": "Berlin"}


class TestAgentFactory:
    """Unit tests for the AgentFactory."""

    def test_create_agent(self):
        """Test creating an agent with the factory."""
        # Create a config for an LLMAgent
        config = {
            "agent_type": "LLMAgent",
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "Answer this question: ${question}",
        }

        # Create a mock provider
        provider = MagicMock(spec=LLMServiceProvider)

        # Create the agent
        agent = AgentFactory.create_agent(config, provider)

        # Verify the agent type
        assert isinstance(agent, LLMAgent)
        assert agent.config == config
        assert agent.provider == provider

    def test_create_agent_unknown_type(self):
        """Test creating an agent with an unknown type."""
        config = {
            "agent_type": "UnknownAgent",
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "Answer this question: ${question}",
        }

        # Attempt to create the agent should raise an error
        with pytest.raises(ValueError) as excinfo:
            AgentFactory.create_agent(config)

        assert "Unknown agent type: UnknownAgent" in str(excinfo.value)

    def test_register_agent_type(self):
        """Test registering a new agent type."""

        # Create a mock agent class
        class MockAgent(Agent):
            async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                return {"mock_result": "test"}

        # Register the mock agent type
        AgentFactory.register_agent_type("MockAgent", MockAgent)

        # Create an agent with the new type
        config = {"agent_type": "MockAgent"}
        agent = AgentFactory.create_agent(config)

        # Verify the agent type
        assert isinstance(agent, MockAgent)
