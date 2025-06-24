# Agent Workflow

A flexible, scalable framework for orchestrating multi-agent LLM workflows through YAML configurations.

## Installation

You can install the package directly from GitHub using pip:

```bash
pip install git+https://github.com/your-org/Agent_workflow.git
```

Or add it to your requirements.txt:

```
git+https://github.com/your-org/Agent_workflow.git@main
```

For development installations:

```bash
git clone https://github.com/your-org/Agent_workflow.git
cd Agent_workflow
pip install -e .
```

Once installed, you can import the package using:

```python
from agent_workflow import WorkflowManager, Tool, ObservabilityProvider
```

## Basic Usage

Here's a simple example of how to use Agent Workflow:

```python
import os
import asyncio
from agent_workflow import (
    WorkflowManager,
    WorkflowInput,
    ProviderConfiguration,
    OpenAIProviderConfig,
    ProviderType,
    ConsoleProgressCallback,
    WorkflowSourceFile
)

async def run_workflow():
    # Configure your provider
    provider_configs = {
        "openai_gpt4": OpenAIProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="gpt-4-turbo",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    }
    
    # Create provider configuration
    providers_config = ProviderConfiguration(providers=provider_configs)
    
    # Create workflow manager
    manager = WorkflowManager(
        provider_config=providers_config,
        engine_type="openai"
    )
    
    # Load and initialize workflow from YAML file
    workflow_path = "path/to/your/workflow.yaml"
    workflow = await manager.initialize_workflow(
        WorkflowSourceFile(workflow_path), 
        provider_mapping={"agent-id": "openai_gpt4"},
        progress_callback=ConsoleProgressCallback()
    )
    
    # Create workflow inputs
    inputs = WorkflowInput(
        user_query="Your query here",
        workflow={
            "inputs": {
                "user_query": "Your query here"
            }
        }
    )
    
    # Execute the workflow
    result = await manager.execute(
        workflow, 
        inputs, 
        progress_callback=ConsoleProgressCallback()
    )
    
    # Get and print the results
    final_output = result.final_result
    print(f"Final result: {final_output}")
    
    return result

if __name__ == "__main__":
    asyncio.run(run_workflow())
```

## Main Components

- **WorkflowManager**: Manages the loading and execution of workflows
- **ToolRegistry**: Manages the tools available to agents
- **ObservabilityProvider**: Provides observability for workflows and agents
- **ExecutionEngine**: Handles the execution of workflows with different backends

## Package Structure

The package exposes the following modules:

- **expressions**: Expression evaluation for workflow configurations
- **parsers**: YAML/JSON parsing for workflow configurations
- **providers**: Tool, observability, and callback providers
- **workflow_engine**: Core workflow execution functionality 
- **yaml**: YAML schemas for workflow and agent configurations

## Creating Tools

Register a custom tool function:

```python
from agent_workflow import tool

@tool(name="calculate", description="Performs mathematical operations")
def calculate(operation: str, a: float, b: float) -> float:
    """
    Perform a mathematical calculation between two numbers.
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    # Add more operations as needed
    else:
        raise ValueError(f"Unknown operation: {operation}")
```

## License

MIT