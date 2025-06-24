# Workflow Builder

A drag-and-drop editor for creating and managing agent workflows, built with React, ReactFlow, and Material UI.

## Features

- **Workflow Editor**: Create and configure workflows with multiple stages and tasks
- **Agent Editor**: Define and manage agents with customizable prompts, tools, and schemas
- **Visual Canvas**: Drag-and-drop interface for designing workflow connections
- **YAML Import/Export**: Convert workflows to/from YAML format
- **Execution**: Execute workflows directly from the interface
- **Input/Output Mapping**: Define dynamic connections between tasks 

## Getting Started

### Installation

```bash
# Install dependencies
npm install

# Start the development server
npm run dev
```

### Usage

1. **Create Agents**: 
   - Go to the Agent Editor tab
   - Define agents with system prompts, schemas, and tools

2. **Build Workflows**:
   - Create stages (sequential or parallel)
   - Add tasks within stages
   - Connect tasks with dependencies
   - Map inputs and outputs

3. **Export/Import**:
   - Export your workflow to YAML
   - Import existing YAML definitions

## Technologies

- **React**: UI framework
- **ReactFlow**: Workflow diagram visualization
- **Material UI**: Component library
- **js-yaml**: YAML parsing and generation

## Project Structure

- `/components`: React UI components
  - `/nodes`: Custom ReactFlow node components
  - `/dialogs`: Modal dialogs for adding/editing elements
  - `/forms`: Form components for data entry
  - `/panels`: Side panels and property editors
- `/utils`: Utility functions
- `/services`: Business logic services

## Workflow YAML Structure

```yaml
name: "Example Workflow"
description: "A sample workflow"
version: "1.0.0"

stages:
  - name: "Stage 1"
    description: "First stage of the workflow"
    execution_type: "sequential"
    tasks:
      - name: "Task 1"
        description: "First task"
        agent:
          ref: "agent1.yaml"
        inputs:
          parameter1: "${workflow.inputs.value}"
        outputs:
          result: "${agent.output_schema.response}"
```

## Agent YAML Structure

```yaml
name: "Example Agent"
description: "A sample agent"
agent_type: "LLMAgent"

system_prompt: "You are a helpful assistant..."
user_prompt: "Please process this request: ${input}"

tools:
  - name: "calculator"
    description: "Perform calculations"
    type: "function"
    parameters:
      type: "object"
      properties:
        expression:
          type: "string"
          description: "Math expression to evaluate"
      required: ["expression"]

input_schema:
  query:
    type: "str"
    description: "The user's query"
    required: true

output_schema:
  response:
    type: "str"
    description: "The generated response"
```