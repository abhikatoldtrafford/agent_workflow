import asyncio
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_workflow.providers import BedrockProvider
from agent_workflow.providers import FunctionTool, Tool, ToolRegistry


class SimpleCalculationTool(Tool):
    """A simple calculation tool for testing."""

    @property
    def name(self) -> str:
        return "calculate"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        }

    async def execute(self, **kwargs):
        """Execute the calculation."""
        expression = kwargs.get("expression")
        if not expression:
            return {"error": "No expression provided"}

        try:
            # Use eval with limited scope for security
            allowed_names = {"__builtins__": {}}
            result = eval(expression, allowed_names, {})
            return {"result": result}
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}


class TestTools(unittest.TestCase):
    """Test the tools implementation."""

    def setUp(self):
        # Create a clean registry for each test
        self.test_registry = ToolRegistry()

    def test_tool_registration(self):
        """Test registering a tool."""
        calc_tool = SimpleCalculationTool()
        self.test_registry.register_tool(calc_tool)

        # Verify the tool was registered
        self.assertEqual(len(self.test_registry.list_tools()), 1)
        self.assertEqual(self.test_registry.list_tools()[0], "calculate")

        # Verify we can retrieve the tool
        retrieved_tool = self.test_registry.get_tool("calculate")
        self.assertIsNotNone(retrieved_tool)
        self.assertEqual(retrieved_tool.name, "calculate")

    def test_function_tool(self):
        """Test creating and using a function tool."""

        # Define a test function
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together.

            Args:
                a: The first number
                b: The second number

            Returns:
                The sum of a and b
            """
            return a + b

        # Create a function tool
        tool = FunctionTool(add_numbers)

        # Verify the tool properties
        self.assertEqual(tool.name, "add_numbers")
        self.assertTrue("Add two numbers together" in tool.description)

        # Verify the parameters schema
        params = tool.parameters
        self.assertEqual(params["type"], "object")
        self.assertTrue("a" in params["properties"])
        self.assertTrue("b" in params["properties"])
        self.assertEqual(params["properties"]["a"]["type"], "integer")
        self.assertEqual(params["properties"]["b"]["type"], "integer")

        # Test executing the tool
        async def run_tool():
            result = await tool.execute(a=5, b=7)
            return result

        result = asyncio.run(run_tool())
        self.assertEqual(result, 12)

    def test_tool_decorator(self):
        """Test registering a tool with the decorator."""

        @self.test_registry.register_function
        def multiply(x: float, y: float):
            """Multiply two numbers.

            Args:
                x: First number
                y: Second number
            """
            return x * y

        # Verify the tool was registered
        self.assertEqual(len(self.test_registry.list_tools()), 1)
        self.assertEqual(self.test_registry.list_tools()[0], "multiply")

        # Test executing the tool
        async def run_tool():
            return await self.test_registry.execute_tool("multiply", {"x": 3, "y": 4})

        result = asyncio.run(run_tool())
        self.assertEqual(result, 12)

    @patch("boto3.client")
    def test_bedrock_provider_with_tools(self, mock_boto3_client):
        """Test the BedrockProvider with tool calling."""
        # Setup mock response from AWS Bedrock
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client

        # First response with tool call
        first_response_body = {
            "content": [
                {"type": "text", "text": "I'll help you calculate that."},
                {
                    "type": "tool_use",
                    "id": "call1",
                    "name": "calculate",
                    "input": {"expression": "5 * 7"},
                },
            ]
        }

        # Final response with result
        final_response_body = {
            "content": [{"type": "text", "text": "The result of 5 * 7 is 35."}]
        }

        # Setup mock responses
        mock_response_1 = MagicMock()
        mock_response_1.get.return_value.read.return_value = json.dumps(
            first_response_body
        )

        mock_response_2 = MagicMock()
        mock_response_2.get.return_value.read.return_value = json.dumps(
            final_response_body
        )

        # Configure the mock to return different responses on consecutive calls
        mock_client.invoke_model.side_effect = [mock_response_1, mock_response_2]

        # Create BedrockProvider
        provider = BedrockProvider(model_name="anthropic.claude-3-sonnet-20240229-v1:0")

        # Create tool registry with calculation tool
        test_registry = ToolRegistry()
        calc_tool = SimpleCalculationTool()
        test_registry.register_tool(calc_tool)

        # Define tool schema
        tools = [
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            }
        ]

        # Test the generate_with_tools method
        async def test_generate():
            response, tool_usage = await provider.generate_with_tools(
                system_prompt="You are a helpful assistant that uses tools when needed.",
                user_prompt="Calculate 5 * 7",
                tools=tools,
                tool_registry=test_registry,
                max_tokens=1024,
            )
            return response, tool_usage

        response, tool_usage = asyncio.run(test_generate())

        # Verify the method was called with correct parameters
        mock_client.invoke_model.assert_called()

        # Check that we got the expected final response
        self.assertEqual(response, "The result of 5 * 7 is 35.")

        # Check that we have tool usage information
        self.assertEqual(len(tool_usage), 1)
        self.assertEqual(tool_usage[0]["name"], "calculate")
        self.assertEqual(tool_usage[0]["parameters"], {"expression": "5 * 7"})


if __name__ == "__main__":
    unittest.main()
