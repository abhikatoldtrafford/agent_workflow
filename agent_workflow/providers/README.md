# Provider Modules for Agent Workflow

This directory contains the implementation of various service providers for the Agent Workflow system.

## LLM Service Providers

The `providers.py` file implements classes for different LLM services:

- `AnthropicProvider`: Integration with Anthropic Claude API
- `BedrockProvider`: Integration with AWS Bedrock for Claude models

## Tool Registry

The `tools.py` file provides a registry for function-calling capabilities that can be used by LLMs in workflows:

### Registering Tools

You can register tools in two ways:

1. Using the `@tool` decorator:

```python
from agent_workflow.providers import tool


@tool(name="calculate", description="Perform calculations")
def calculator(operation: str, a: float, b: float):
    """Calculate results based on operation type."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    # ... more operations
```

2. Creating a class-based tool:

```python
from agent_workflow.providers import Tool, register_tool


class WeatherTool(Tool):
    @property
    def name(self):
        return "get_weather"

    @property
    def description(self):
        return "Get weather information for a location"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }

    async def execute(self, location: str):
        # Implementation
        return {"location": location, "temperature": 72}


# Register the tool
register_tool(WeatherTool())
```

### Using Tools in Workflows

In your workflow YAML file, specify the tools that an agent can use:

```yaml
agent:
  agent_type: "LLMAgent"
  llm_type: "bedrock_sonnet_3.5"
  tools: ["calculate", "get_weather", "search"]
  system_prompt: |+
    You are an assistant with access to tools.
    Use the appropriate tools to help answer the user's question.
```

The LLM will automatically be instructed on how to use the available tools and will generate tool calls when needed.

## Observability

The `observability.py` file provides infrastructure for tracing and monitoring LLM calls within the workflow system.

### Overview

The observability module allows for:
- Tracing workflow and agent execution
- Creating spans for detailed performance monitoring
- Logging LLM calls with request/response data
- Capturing custom scores for evaluation

### Available Providers

1. **LangfuseProvider**: Integration with [Langfuse](https://langfuse.com/) for AI observability
2. **NoopObservabilityProvider**: No-op implementation when observability is disabled

### Setup

To use observability in your workflow:

1. Configure the provider in your script:

```python
from agent_workflow.providers.observability import configure_observability

# Configure Langfuse
observability_provider = configure_observability(
    provider_type="langfuse",
    public_key="your_langfuse_public_key",
    secret_key="your_langfuse_secret_key"
)

# Or use the no-op provider
observability_provider = configure_observability(provider_type="noop")
```

2. Pass the provider to your workflow:

```python
# Create a workflow trace
workflow_trace_id = observability_provider.trace_workflow(
    workflow_id="example_workflow",
    metadata={"version": "1.0.0"}
)

# Add the observability provider to your inputs
inputs = {
    # Your other inputs...
    "provider_config": {
        "bedrock_sonnet_2": {
            # Provider configuration...
            "observability": {
                "provider": observability_provider,
                "trace_id": workflow_trace_id
            }
        }
    },
    "observability": {
        "provider": observability_provider,
        "trace_id": workflow_trace_id
    }
}
```

### Environment Variables

You can control observability using environment variables:

```bash
# Enable/disable observability
export ENABLE_OBSERVABILITY=true

# Langfuse credentials
export LANGFUSE_PUBLIC_KEY=your_public_key
export LANGFUSE_SECRET_KEY=your_secret_key
export LANGFUSE_HOST=https://cloud.langfuse.com  # Optional
```

### Tracing Functions

You can decorate your functions with the `@trace` decorator:

```python
async def run_task():
    # Create a span manually
    span_id = observability_provider.trace_span(
        name="task_execution", 
        metadata={"task_type": "example"}
    )
    
    try:
        # Task logic
        result = await some_operation()
        
        # End span with success
        observability_provider.end_span(
            span_id=span_id,
            metadata={"status": "success"}
        )
        return result
    except Exception as e:
        # End span with error
        observability_provider.end_span(
            span_id=span_id,
            metadata={"status": "error", "error": str(e)}
        )
        raise
```

Or use the decorator pattern:

```python
@observability_provider.trace("custom_operation")
async def my_function(arg1, arg2):
    # Function logic
    return result
```