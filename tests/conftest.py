"""
Pytest configuration for OpenAI observability tests.
This file is automatically loaded by pytest and provides shared fixtures and configuration.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Ensure we're in test mode
    os.environ["TESTING"] = "true"
    
    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set", allow_module_level=True)
    
    yield
    
    # Cleanup
    os.environ.pop("TESTING", None)


@pytest.fixture
def clean_tool_registry():
    """Ensure tool registry is clean for each test."""
    from agent_workflow.providers import ToolRegistry
    
    # Store original state
    original_tools = ToolRegistry._instance._tools.copy() if ToolRegistry._instance else {}
    
    yield
    
    # Restore original state
    if ToolRegistry._instance:
        ToolRegistry._instance._tools = original_tools


@pytest.fixture
def mock_openai_response():
    """Provide mock OpenAI responses for testing without API calls."""
    return {
        "choices": [
            {
                "message": {
                    "content": '{"result": "Mocked response", "confidence": 0.95}'
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


# Pytest markers for categorizing tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: Integration tests that may take longer"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: Tests that require a valid OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than a few seconds"
    )


# Command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--skip-integration",
        action="store_true",
        default=False,
        help="Skip integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and options."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if config.getoption("--skip-integration"):
        skip_integration = pytest.mark.skip(reason="Skipping integration tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)