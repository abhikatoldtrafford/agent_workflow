from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_workflow.parsers import YAMLParser
from agent_workflow.providers import LLMProviderFactory
from agent_workflow.workflow_engine import ExecutionEngine, ExecutionEngineFactory
from agent_workflow.workflow_engine import WorkflowManager


class TestExecutionEngine:
    """Unit tests for the ExecutionEngine classes."""

    @pytest.fixture
    def mock_provider_factory(self):
        return MagicMock(spec=LLMProviderFactory)

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.generate_response = AsyncMock(return_value="Test response")
        return provider


class TestExecutionEngineFactory:
    """Unit tests for the ExecutionEngineFactory."""

    def test_create_engine(self):
        """Test creating execution engine instances."""
        # Test creating an OpenAI engine
        engine = ExecutionEngineFactory.create_engine("openai")
        assert isinstance(engine, ExecutionEngine)

        # Test error for unknown engine type
        with pytest.raises(ValueError):
            ExecutionEngineFactory.create_engine("unknown_engine")

    def test_register_engine(self):
        """Test registering a new engine type."""
        # Create a mock engine class
        mock_engine_class = MagicMock(spec=ExecutionEngine)

        # Register it
        ExecutionEngineFactory.register_engine("mock_engine", mock_engine_class)

        # Create an instance and verify
        ExecutionEngineFactory.create_engine("mock_engine")
        mock_engine_class.assert_called_once()


class TestWorkflowManager:
    """Unit tests for the WorkflowManager."""

    @pytest.fixture
    def mock_config_parser(self):
        parser = MagicMock(spec=YAMLParser)
        parser.load_config.return_value = {
            "name": "test_workflow",
            "version": "1.0",
            "stages": [
                {
                    "name": "stage1",
                    "tasks": [{"name": "task1", "agent": {"agent_type": "LLMAgent"}}],
                }
            ],
        }
        return parser

    @pytest.fixture
    def mock_execution_engine(self):
        engine = MagicMock(spec=ExecutionEngine)
        engine.execute_workflow = AsyncMock(return_value={"result": "success"})
        return engine

    def test_init_default_values(self):
        """Test initializing WorkflowManager with default values."""
        manager = WorkflowManager()
        assert isinstance(manager.config_parser, YAMLParser)
        assert isinstance(manager.provider_factory, LLMProviderFactory)
        assert manager.engine_type == "openai"
        assert isinstance(manager.execution_engine, ExecutionEngine)

    def test_init_custom_values(self):
        """Test initializing WorkflowManager with custom values."""
        mock_parser = MagicMock(spec=YAMLParser)
        mock_provider_factory = MagicMock(spec=LLMProviderFactory)

        # Since we can't test OpenAI engine directly (not implemented),
        # we'll patch the factory to return a mock
        with patch.object(
            ExecutionEngineFactory, "create_engine"
        ) as mock_create_engine:
            mock_engine = MagicMock(spec=ExecutionEngine)
            mock_create_engine.return_value = mock_engine

            manager = WorkflowManager(
                config_parser=mock_parser,
                provider_factory=mock_provider_factory,
                engine_type="custom_engine",
                engine_options={"api_key": "test_key"},
            )

            assert manager.config_parser == mock_parser
            assert manager.provider_factory == mock_provider_factory
            assert manager.engine_type == "custom_engine"
            assert manager.execution_engine == mock_engine
            mock_create_engine.assert_called_once_with(
                "custom_engine", api_key="test_key"
            )

    def test_load_workflow(self, mock_config_parser):
        """Test loading a workflow configuration."""
        manager = WorkflowManager(config_parser=mock_config_parser)
        workflow = manager.load_workflow("test_workflow.yaml")

        mock_config_parser.load_config.assert_called_once_with("test_workflow.yaml")
        assert workflow["name"] == "test_workflow"
        assert workflow["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_execute_workflow(self, mock_config_parser):
        """Test executing a workflow."""
        with patch.object(
            ExecutionEngineFactory, "create_engine"
        ) as mock_create_engine:
            mock_engine = MagicMock(spec=ExecutionEngine)
            mock_engine.execute_workflow = AsyncMock(return_value={"result": "success"})
            mock_create_engine.return_value = mock_engine

            manager = WorkflowManager(config_parser=mock_config_parser)
            result = await manager.execute_workflow(
                "test_workflow.yaml", inputs={"param1": "value1"}
            )

            mock_config_parser.load_config.assert_called_once_with("test_workflow.yaml")
            mock_engine.execute_workflow.assert_called_once()
            assert result == {"result": "success"}

            # Check that engine_type is added to inputs
            inputs_arg = mock_engine.execute_workflow.call_args[0][1]
            assert inputs_arg["_execution"]["engine_type"] == "openai"
