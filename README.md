# Agent Workflow

A flexible, scalable framework for orchestrating multi-agent LLM workflows through YAML configurations.

## Current Status

The library is under active development with our primary focus on the OpenAI execution engine using the OpenAI Agent SDK. Work on the custom Agentic execution engine is in progress.

## Features

- 🔄 **Declarative Workflow Design**: Define complex workflows using simple YAML configurations
- 🤖 **Multi-Agent Orchestration**: Coordinate multiple specialized LLM agents in sequential or parallel execution
- 🔌 **Multiple Execution Engines**: Support for both custom Agentic framework (in progress) and OpenAI Assistants API
- 🧩 **Multiple LLM Providers**: Support for OpenAI, AWS Bedrock (Claude), and Google Gemini
- 📊 **Observability**: Built-in tracing and monitoring using Langfuse
- 📝 **Schema Validation**: Input/output validation using Pydantic models
- ⚙️ **Dynamic Expressions**: Template-based expressions for flexible data flow between workflow stages
- 🔁 **Conditional Execution**: Control flow based on dynamic conditions
- 🛠️ **Tool Integration**: Function calling support for all compatible models
- 🧪 **Structured Output**: Support for schema-based structured outputs

## Installation

### As a Development Dependency

Add the package to your project using uv:

```bash
# Install directly from GitHub
uv pip install git+https://github.com/newscorp-ghfb/agent_workflow.git
```

or 

```bash
uv add git+https://github.com/newscorp-ghfb/agent_workflow.git
```

### For Development

```bash
# Clone the repository
git clone <repository-url>
cd Agent_workflow

# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
# .venv\Scripts\activate  # On Windows

# Install dependencies
uv sync --all-extras

# Alternative: Use the setup script
chmod +x setup_dev_with_uv.sh
./setup_dev_with_uv.sh
```

## Quick Start

1. Create a workflow YAML file:

```yaml
# example-workflow.yaml
name: "Simple Workflow"
description: "A two-stage workflow example"
version: "1.0.0"

stages:
  - name: "Planning"
    description: "Generate a plan"
    execution_type: "sequential"
    tasks:
      - name: "Create Plan"
        description: "Create an implementation plan"
        agent:
          ref: "plan_agent.yaml"
        inputs:
          "${agent.input_schema.requirements}": "${workflow.inputs.feature_request}"
          "${agent.input_schema.constraints}": "${workflow.inputs.constraints}"
        outputs:
          "implementation_plan": "${agent.output_schema.implementation_plan}"

  - name: "Implementation"
    description: "Implement the plan"
    execution_type: "sequential"
    tasks:
      - name: "Implement"
        description: "Implement based on plan"
        agent:
          agent_type: "LLMAgent"
          parameters:
            llm_type: "bedrock_sonnet_2"
            system_prompt: "You are a developer implementing a plan."
        inputs:
          plan: "${stages.[Planning].tasks.[Create Plan].outputs.implementation_plan}"
        outputs:
          - implementation_result
```

2. Run your workflow using the OpenAI execution engine:

```python
from agent_workflow.workflow_engine import WorkflowManager
import asyncio
from agent_workflow.providers.observability import configure_observability
import os

# Configure observability
observability_provider = configure_observability(provider_type="langfuse")
workflow_trace_id = observability_provider.trace_workflow("example_workflow")

# Set up the OpenAI engine
engine_type = "openai"
engine_options = {
  "api_key": os.environ.get("OPENAI_API_KEY")
}

# Create workflow manager with the OpenAI engine
manager = WorkflowManager(
  engine_type=engine_type,
  engine_options=engine_options
)

# Define inputs
inputs = {
  "feature_request": "Create a user dashboard with metrics and charts",
  "constraints": "Use existing libraries and ensure responsive design",
}

# Add observability configuration
inputs["observability"] = {
  "provider": observability_provider,
  "trace_id": workflow_trace_id
}

# Execute workflow with the OpenAI engine
results = await manager.execute_workflow("example-workflow.yaml", inputs)
print(results)
```

You can also run the workflow with the Agentic engine (currently in progress):

```python
# Choose the Agentic engine
engine_type = "agentic"
engine_options = {}

# Create workflow manager with the Agentic engine
manager = WorkflowManager(
    engine_type=engine_type,
    engine_options=engine_options
)

# Add provider configurations for the Agentic engine
inputs["provider_config"] = {
    "bedrock_sonnet_2": {
        "provider_type": "bedrock_sonnet_2",
        "credentials": {
            "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region_name": "us-east-1"
        },
        "observability": {
            "provider": observability_provider,
            "trace_id": workflow_trace_id
        }
    }
}

# Execute workflow with the Agentic engine
results = await manager.execute_workflow("example-workflow.yaml", inputs)
print(results)
```

## Usage Examples

The repository includes several usage examples in the `usage_examples` directory:

- **Product Development Workflow**: A multi-stage workflow for planning, designing, and validating a new product feature
- **API Design Agent**: Specialized agent for API design tasks
- **Database Design Agent**: Agent for designing database schemas
- **UI Design Agent**: Agent for creating UI wireframes and flows

Run the example with:

```bash
# Run with the default Agentic engine
python usage_examples/dev_workflow/dev_workflow.py

# Run with engine selection
python usage_examples/dev_workflow/engine_selection_example.py --engine agentic
python usage_examples/dev_workflow/engine_selection_example.py --engine openai
```

## Architecture

The framework consists of several key components:

- **Workflow Engine**: Core orchestration system that executes workflow definitions
- **Execution Engines**: Pluggable execution engines supporting different backends:
  - **Agentic Engine**: Custom implementation with direct LLM integration
  - **OpenAI Engine**: Integration with OpenAI's Assistants API
- **Agent Framework**: Implements different agent types and their execution logic
- **LLM Providers**: Adapters for various LLM services
- **Observability**: Monitoring and tracing infrastructure
- **Expression System**: Template-based expression evaluation for dynamic data flow

## Configuration

### Workflow Schema

Workflows are defined using a structured YAML schema:

- **name**: Workflow identifier
- **description**: Human-readable description
- **version**: Workflow version
- **stages**: List of execution stages
  - **name**: Stage identifier
  - **description**: Stage description
  - **execution_type**: "sequential" or "parallel"
  - **condition**: Optional expression to determine execution
  - **tasks**: List of tasks in the stage
    - **name**: Task identifier
    - **description**: Task description
    - **agent**: Agent configuration (reference or inline)
    - **inputs**: Mapping of input parameters
    - **outputs**: Mapping of output parameters
    - **condition**: Optional task execution condition

### Agent Schema

Agents are defined with their own YAML schema:

- **id**: Unique identifier
- **name**: Human-readable name
- **description**: Agent description
- **version**: Agent version
- **agent_type**: Implementation class (e.g., "LLMAgent")
- **llm_type**: LLM service type (e.g., "bedrock_sonnet_2")
- **input_schema**: Expected input parameters
- **output_schema**: Output structure definition
- **system_prompt**: System instructions template
- **user_prompt**: User query template
- **resources**: Resource requirements (tokens, timeout)
- **retry**: Retry configuration

## Observability

The framework integrates with Langfuse for comprehensive LLM observability:

- **Trace Workflows**: Monitor entire workflow execution
- **Span Details**: Track performance of individual components
- **Log LLM Calls**: Capture request/response details
- **Custom Scores**: Evaluate results based on custom metrics

Configure observability with environment variables:

```bash
export LANGFUSE_HOST=your_host
export LANGFUSE_PUBLIC_KEY=your_public_key
export LANGFUSE_SECRET_KEY=your_secret_key
```

## Execution Engines & LLM Providers

The framework supports multiple execution engines and LLM providers:

### Execution Engines

#### OpenAI Engine

Currently our primary focus is on the OpenAI Engine, which integrates with OpenAI's Assistants API. This engine leverages advanced capabilities like function calling and structured outputs, while supporting different model providers through adapter layers.

To specify the OpenAI engine:

```python
# In code
manager = WorkflowManager(engine_type="openai", engine_options={"api_key": "your_openai_key"})

# Or via command line
python engine_selection_example.py --engine openai
```

#### Agentic Engine

The Agentic engine is under development and will provide a custom implementation with direct LLM integrations. This engine will offer full control over LLM interactions and support for multiple providers. Note: this implementation is currently in progress and may not have full functionality.

### Supported LLM Providers

The framework now supports the following providers:

#### OpenAI

Direct integration with OpenAI's API.

```bash
export OPENAI_API_KEY=your_openai_api_key
```

#### AWS Bedrock

Support for Claude and other models via AWS Bedrock.

```bash
export AWS_ACCESS_KEY_ID=your_aws_access_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret_key
export AWS_REGION=your_aws_region
```

#### Google Gemini

Integration with Google's Gemini models.

```bash
export GOOGLE_API_KEY=your_google_api_key
export GOOGLE_PROJECT_ID=your_project_id  # Optional
```

Example of using AWS Bedrock:

```python
from agent_workflow.workflow_engine import BedrockProviderConfig, ProviderType, ProviderConfiguration

# Create provider configuration
provider_configs = {
  "bedrock_claude_3_7": BedrockProviderConfig(
    provider_type=ProviderType.BEDROCK,
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    AWS_REGION=os.environ.get("AWS_REGION", "us-east-1"),
    AWS_ACCESS_KEY_ID=os.environ.get("AWS_ACCESS_KEY_ID"),
    AWS_SECRET_ACCESS_KEY=os.environ.get("AWS_SECRET_ACCESS_KEY")
  )
}

# Create container and set up workflow inputs
providers_config = ProviderConfiguration(providers=provider_configs)
```

## Testing

This project uses pytest for unit testing. To run the tests:

```bash
# For Poetry version 1.2.0 and newer:
poetry install --with dev

# For older Poetry versions:
poetry install -E dev

# Run all tests
pytest

# Run specific test file
pytest tests/test_workflow_engine.py

# Run tests with verbose output
pytest -v
```

Test files are located in the `tests` directory and follow the naming pattern `test_*.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

When contributing code, please make sure to:
1. Add appropriate unit tests for new functionality
2. Ensure all tests pass before submitting a PR
3. Follow the existing code style and patterns

## Documentation & Resources

The framework includes comprehensive documentation and examples:

- **Schema Definitions**: YAML schemas for workflow and agent definitions in the `yaml/schema` directory
- **Usage Examples**: Real-world workflow examples in the `usage_examples` directory
- **Test Suite**: Example implementations and tests in the `tests` directory

### YAML Schemas

The framework uses two main YAML schemas that define the structure of workflows and agents:

- **Workflow Schema**: Located at `yaml/schema/workflow_schema.yaml`
- **Agent Schema**: Located at `yaml/schema/agent_schema.yaml`

These schemas provide a reference for creating valid workflow and agent configurations.
