"""
Callbacks for workflow execution progress reporting.
"""

import logging
from typing import Any, Callable, Dict, Optional, Union, TypeVar, ForwardRef

logger = logging.getLogger("workflow-engine.callbacks")


# Define the ProgressCallback class here to avoid import cycle
class ProgressCallback:
    """Interface for workflow progress callbacks."""

    async def on_workflow_start(self, workflow_name: str, workflow_config: Any) -> None:
        """Called when a workflow starts execution."""
        pass

    async def on_workflow_complete(self, workflow_name: str, result: Any) -> None:
        """Called when a workflow completes execution."""
        pass

    async def on_stage_start(self, stage_name: str, stage_config: Any) -> None:
        """Called when a stage starts execution."""
        pass

    async def on_stage_complete(self, stage_name: str, stage_result: Dict[str, Any]) -> None:
        """Called when a stage completes execution."""
        pass

    async def on_task_start(self, task_name: str, task_config: Any) -> None:
        """Called when a task starts execution."""
        pass

    async def on_task_complete(self, task_name: str, task_result: Dict[str, Any], agent_output: Optional[Any] = None) -> Optional[Any]:
        """Called when a task completes execution."""
        pass

    async def on_task_fail(self, task_name: str, error: str, agent_output: Optional[Any] = None) -> None:
        """Called when a task fails execution."""
        pass


class UserInputCallback(ProgressCallback):
    """
    Implementation of ProgressCallback that requests user input after each agent execution.
    This allows the user to review and potentially modify the agent's output before it's
    passed to the next agent in the workflow.
    """

    def __init__(
        self,
        input_handler: Callable[
            [str, Dict[str, Any], Any], Union[str, Dict[str, Any], None]
        ],
        stream_handler: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize UserInputCallback.

        Args:
            input_handler: Function that handles requesting input from the user.
                          Should take agent name, task result, and agent output,
                          and return modified output or None to keep existing output.
            stream_handler: Optional handler function for streaming output
        """
        self.input_handler = input_handler
        self.stream = stream_handler if stream_handler else print

    async def on_workflow_start(
        self, workflow_name: str, workflow_config: Union[Dict[str, Any], Any]
    ) -> None:
        """Called when a workflow starts execution."""
        self.stream(f"Starting workflow: {workflow_name}")
        if hasattr(workflow_config, "get") and workflow_config.get("description"):
            self.stream(f"Description: {workflow_config['description']}")

    async def on_workflow_complete(self, workflow_name: str, result: Any) -> None:
        """Called when a workflow completes execution."""
        self.stream(f"Workflow complete: {workflow_name}")

    async def on_stage_start(
        self, stage_name: str, stage_config: Union[Dict[str, Any], Any]
    ) -> None:
        """Called when a stage starts execution."""
        self.stream(f"\n🔄 Starting stage: {stage_name}")
        if hasattr(stage_config, "get") and stage_config.get("description"):
            self.stream(f"  Description: {stage_config['description']}")

    async def on_stage_complete(self, stage_name: str, stage_result: Dict[str, Any]) -> None:
        """Called when a stage completes execution."""
        self.stream(f"✅ Stage complete: {stage_name}")

    async def on_task_start(
        self, task_name: str, task_config: Union[Dict[str, Any], Any]
    ) -> None:
        """Called when a task starts execution."""
        self.stream(f"\n  🔹 Starting task: {task_name}")
        if hasattr(task_config, "get") and task_config.get("description"):
            self.stream(f"    Description: {task_config['description']}")

    async def on_task_complete(
        self,
        task_name: str,
        task_result: Dict[str, Any],
        agent_output: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        Called when a task completes execution. Requests user input for the output.

        This method will:
        1. Show the user the agent's output
        2. Request input from the user
        3. Return the modified output if the user provided changes

        Returns the potentially modified agent_output
        """
        self.stream(f"  ✅ Task complete: {task_name}")

        if agent_output:
            # Request user input to review and potentially modify the output
            modified_output = self.input_handler(task_name, task_result, agent_output)

            # If user provided modifications, update the agent output
            if modified_output is not None:
                if isinstance(modified_output, str):
                    agent_output.output = modified_output
                    task_result["outputs"]["result"] = modified_output
                elif isinstance(modified_output, dict):
                    agent_output.output = modified_output
                    task_result["outputs"]["result"] = modified_output

                # Return the modified agent_output to indicate changes were made
                return agent_output

        return None

    async def on_task_fail(
        self, task_name: str, error: str, agent_output: Optional[Any] = None
    ) -> None:
        """Called when a task fails execution."""
        self.stream(f"❌ Task failed: {task_name}")
        self.stream(f"  Error: {error}")


class ConsoleProgressCallback(ProgressCallback):
    """Implementation of ProgressCallback that prints progress to the console."""

    def __init__(
        self,
        on_task_callback: Optional[
            Callable[[str, Dict[str, Any], Optional[Any]], None]
        ] = None,
        stream_handler: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize ConsoleProgressCallback.

        Args:
            on_task_callback: Optional callback function to execute when a task completes
            stream_handler: Optional handler function for streaming output
        """
        self.on_task_callback = on_task_callback
        self.stream = stream_handler if stream_handler else print

    async def on_workflow_start(
        self, workflow_name: str, workflow_config: Any
    ) -> None:
        """Called when a workflow starts execution."""
        self.stream(f"Starting workflow: {workflow_name}")
        if hasattr(workflow_config, "description"):
            self.stream(f"Description: {workflow_config.description}")

    async def on_workflow_complete(
        self, workflow_name: str, result: Any
    ) -> None:
        """Called when a workflow completes execution."""
        self.stream(f"Workflow complete: {workflow_name}")

    async def on_stage_start(self, stage_name: str, stage_config: Any) -> None:
        """Called when a stage starts execution."""
        self.stream(f"\n🔄 Starting stage: {stage_name}")
        if hasattr(stage_config, "description"):
            self.stream(f"  Description: {stage_config.description}")

    async def on_stage_complete(self, stage_name: str, stage_result: Dict[str, Any]) -> None:
        """Called when a stage completes execution."""
        self.stream(f"✅ Stage complete: {stage_name}")

    async def on_task_start(self, task_name: str, task_config: Any) -> None:
        """Called when a task starts execution."""
        self.stream(f"\n  🔹 Starting task: {task_name}")
        if hasattr(task_config, "description"):
            self.stream(f"    Description: {task_config.description}")

    async def on_task_complete(
        self,
        task_name: str,
        task_result: Dict[str, Any],
        agent_output: Optional[Any] = None,
    ) -> Optional[Any]:
        """Called when a task completes execution."""
        self.stream(f"  ✅ Task complete: {task_name}")

        # Use the callback if provided
        if self.on_task_callback and agent_output:
            self.on_task_callback(task_name, task_result, agent_output)

        return None


class StreamingProgressCallback(ConsoleProgressCallback):
    """Progress callback that also streams intermediate results to a web client."""

    def __init__(
        self,
        stream_fn: Callable[[Dict[str, Any]], None],
        on_task_callback: Optional[
            Callable[[str, Dict[str, Any], Optional[Any]], None]
        ] = None,
    ) -> None:
        """
        Initialize StreamingProgressCallback.

        Args:
            stream_fn: Function to send streaming updates (should accept a dict)
            on_task_callback: Optional callback function to execute when a task completes
        """
        super().__init__(on_task_callback=on_task_callback)
        self.stream_fn = stream_fn

    async def on_task_complete(
        self,
        task_name: str,
        task_result: Dict[str, Any],
        agent_output: Optional[Any] = None,
    ) -> Optional[Any]:
        """Stream task completion events to the client."""
        modified_output = await super().on_task_complete(
            task_name, task_result, agent_output
        )

        # Create a streamable event
        event = {
            "event_type": "task_complete",
            "task_name": task_name,
            "result": task_result,
        }

        # Use the potentially modified output
        output_to_use = modified_output if modified_output else agent_output

        if output_to_use:
            event["agent"] = output_to_use.agent if hasattr(output_to_use, "agent") else task_name
            event["output"] = output_to_use.output if hasattr(output_to_use, "output") else str(output_to_use)
            if hasattr(output_to_use, "metadata") and output_to_use.metadata:
                event["metadata"] = output_to_use.metadata

        # Stream the event
        self.stream_fn(event)

        return modified_output

    async def on_workflow_start(
        self, workflow_name: str, workflow_config: Any
    ) -> None:
        """Stream workflow start event."""
        await super().on_workflow_start(workflow_name, workflow_config)

        event_data = {
            "event_type": "workflow_start",
            "workflow_name": workflow_name,
        }
        
        # Add additional fields if available
        if hasattr(workflow_config, "description"):
            event_data["description"] = workflow_config.description
        if hasattr(workflow_config, "version"):
            event_data["version"] = workflow_config.version
            
        self.stream_fn(event_data)

    async def on_workflow_complete(
        self, workflow_name: str, result: Any
    ) -> None:
        """Stream workflow completion event."""
        await super().on_workflow_complete(workflow_name, result)

        # Convert result to dict if possible
        result_dict = result
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        elif not isinstance(result, dict):
            result_dict = {"result": str(result)}

        self.stream_fn(
            {
                "event_type": "workflow_complete",
                "workflow_name": workflow_name,
                "result": result_dict,
            }
        )

    async def on_stage_start(self, stage_name: str, stage_config: Any) -> None:
        """Stream stage start event."""
        await super().on_stage_start(stage_name, stage_config)

        # Build event data
        event_data = {
            "event_type": "stage_start",
            "stage_name": stage_name,
        }
        
        # Add additional fields if available
        if hasattr(stage_config, "description"):
            event_data["description"] = stage_config.description
        if hasattr(stage_config, "execution_type"):
            event_data["execution_type"] = stage_config.execution_type
            
        self.stream_fn(event_data)

    async def on_stage_complete(self, stage_name: str, stage_result: Dict[str, Any]) -> None:
        """Stream stage completion event."""
        await super().on_stage_complete(stage_name, stage_result)

        self.stream_fn(
            {
                "event_type": "stage_complete",
                "stage_name": stage_name,
                "task_count": len(stage_result.get("tasks", {})),
            }
        )


class LoggingProgressCallback(ProgressCallback):
    """Implementation of ProgressCallback that logs progress to the logger."""

    def __init__(self, logger_name: str = "workflow-engine.progress"):
        """Initialize LoggingProgressCallback with a specific logger."""
        self.logger = logging.getLogger(logger_name)

    async def on_workflow_start(
        self, workflow_name: str, workflow_config: Any
    ) -> None:
        """Called when a workflow starts execution."""
        self.logger.info(f"Starting workflow: {workflow_name}")
        if hasattr(workflow_config, "description"):
            self.logger.info(f"Description: {workflow_config.description}")

    async def on_workflow_complete(
        self, workflow_name: str, result: Any
    ) -> None:
        """Called when a workflow completes execution."""
        self.logger.info(f"Workflow complete: {workflow_name}")

    async def on_stage_start(self, stage_name: str, stage_config: Any) -> None:
        """Called when a stage starts execution."""
        self.logger.info(f"Starting stage: {stage_name}")
        if hasattr(stage_config, "description"):
            self.logger.info(f"Stage description: {stage_config.description}")

    async def on_stage_complete(self, stage_name: str, stage_result: Dict[str, Any]) -> None:
        """Called when a stage completes execution."""
        self.logger.info(f"Stage complete: {stage_name}")

    async def on_task_start(self, task_name: str, task_config: Any) -> None:
        """Called when a task starts execution."""
        self.logger.info(f"Starting task: {task_name}")
        if hasattr(task_config, "description"):
            self.logger.info(f"Task description: {task_config.description}")

    async def on_task_complete(
        self,
        task_name: str,
        task_result: Dict[str, Any],
        agent_output: Optional[Any] = None,
    ) -> Optional[Any]:
        """Called when a task completes execution."""
        self.logger.info(f"Task complete: {task_name}")
        return None