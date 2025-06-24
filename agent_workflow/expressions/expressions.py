import logging
import re
from typing import Any, Dict, Optional

import jinja2

from agent_workflow.workflow_engine.models import ResponseStore, WorkflowInput

logger = logging.getLogger("workflow-engine.expressions")


class ExpressionEvaluator:
    """Evaluate template expressions and conditions."""

    @staticmethod
    def _get_value_from_path(data: Dict[str, Any], path: str) -> Optional[Any]:
        """Extract a value from nested dictionaries using a path expression."""
        parts = re.findall(r"\[([^\]]+)\]|([^.\[\]]+)", path)
        current = data

        # Debug - log the path and current data structure
        logger.debug(f"Getting value from path: {path}")
        logger.debug(f"Available top-level keys: {list(data.keys())}")

        for i, (bracket, dot) in enumerate(parts):
            key = bracket if bracket else dot
            if key in current:
                current = current[key]
                logger.debug(
                    f"Found key '{key}' at path segment {i + 1}, continuing..."
                )
            else:
                logger.debug(
                    f"Key '{key}' not found at path segment {i + 1}. Available keys: {list(current.keys()) if isinstance(current, dict) else 'not a dict'}"
                )
                return None

        return current

    @classmethod
    def evaluate_template(cls, template: str, context: Dict[str, Any]) -> str:
        """Evaluate a Jinja2 template with the given context."""
        env = jinja2.Environment()

        # Configure Jinja2 to use ${ and } as variable delimiters
        # env = jinja2.Environment(
        #     variable_start_string='${',
        #     variable_end_string='}',
        #     autoescape=False
        # )

        template_obj = env.from_string(template)
        return template_obj.render(**context)

    @classmethod
    def evaluate_expression(cls, expr: Any, context: Dict[str, Any]) -> Any:
        """Evaluate a template expression that references values in the context."""
        # If expr is not a string, return it directly (handles non-string inputs)
        if not isinstance(expr, str):
            return expr

        # Handle ${...} expression syntax
        if expr.startswith("${") and expr.endswith("}"):
            path = expr[2:-1]
            result = cls._get_value_from_path(context, path)

            # Special handling for task outputs - if we get None from regular path
            # Try to infer if this is a task output path and try alternative path formats
            if result is None and "tasks." in path:
                logger.info(
                    f"Path not resolved directly, attempting fallbacks for: {path}"
                )
                parts = path.split(".")

                # Check for the pattern: stages.StageName.tasks.TaskName.outputs.field_name
                if (
                    len(parts) >= 6
                    and parts[0] == "stages"
                    and parts[2] == "tasks"
                    and parts[4] == "outputs"
                ):
                    stage_name = parts[1]
                    task_name = parts[3]
                    output_field = parts[5]

                    logger.info(
                        f"Looking for stage='{stage_name}', task='{task_name}', output='{output_field}'"
                    )

                    # Try direct path to task outputs
                    if (
                        stage_name in context.get("stages", {})
                        and "tasks" in context["stages"][stage_name]
                    ):
                        if task_name in context["stages"][stage_name]["tasks"]:
                            task_data = context["stages"][stage_name]["tasks"][
                                task_name
                            ]

                            # Print all available outputs for debugging
                            logger.info(
                                f"Task data structure: {list(task_data.keys())}"
                            )
                            if "outputs" in task_data:
                                if isinstance(task_data["outputs"], dict):
                                    logger.info(
                                        f"Available output fields: {list(task_data['outputs'].keys())}"
                                    )
                                else:
                                    logger.info(
                                        f"Outputs is not a dict: {type(task_data['outputs'])}"
                                    )

                            # Case 1: outputs is a dict with field_name as key
                            if (
                                "outputs" in task_data
                                and isinstance(task_data["outputs"], dict)
                                and output_field in task_data["outputs"]
                            ):
                                result = task_data["outputs"][output_field]
                                logger.info(
                                    f"Found result via direct task outputs reference: {type(result).__name__}"
                                )
                                return result

                            # Case 2: field is directly in outputs
                            if (
                                "outputs" in task_data
                                and output_field in task_data["outputs"]
                            ):
                                result = task_data["outputs"][output_field]
                                logger.info(
                                    f"Found result directly in outputs: {type(result).__name__}"
                                )
                                return result
                        else:
                            logger.info(
                                f"Task '{task_name}' not found in stage '{stage_name}'"
                            )
                            logger.info(
                                f"Available tasks: {list(context['stages'][stage_name]['tasks'].keys())}"
                            )
                    else:
                        logger.info(f"Stage '{stage_name}' not found or has no tasks")
                        if "stages" in context:
                            logger.info(
                                f"Available stages: {list(context['stages'].keys())}"
                            )

                # Check for the pattern: tasks.TaskName.outputs.field_name (shorter form)
                if len(parts) >= 4 and parts[0] == "tasks" and parts[2] == "outputs":
                    task_name = parts[1]
                    output_field = parts[3]

                    # Find the task in any stage
                    for stage_name, stage_data in context.get("stages", {}).items():
                        if "tasks" in stage_data and task_name in stage_data["tasks"]:
                            task_data = stage_data["tasks"][task_name]

                            # Try to find the output field
                            if "outputs" in task_data and isinstance(
                                task_data["outputs"], dict
                            ):
                                if output_field in task_data["outputs"]:
                                    result = task_data["outputs"][output_field]
                                    logger.info(
                                        f"Found result in stage '{stage_name}', task '{task_name}', field '{output_field}'"
                                    )
                                    return result

                # Log all available paths for debugging
                logger.debug("Available paths in context:")
                for stage_name, stage_data in context.get("stages", {}).items():
                    if "tasks" in stage_data:
                        for task_name, task_data in stage_data["tasks"].items():
                            if "outputs" in task_data:
                                if isinstance(task_data["outputs"], dict):
                                    logger.debug(
                                        f"  stages.{stage_name}.tasks.{task_name}.outputs: {list(task_data['outputs'].keys())}"
                                    )

            return result

        # Otherwise, assume it's a literal value
        return expr

    @classmethod
    def evaluate_condition(cls, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a boolean condition expression."""
        if not condition:
            return True

        # Replace ${...} references with their values
        def replace_refs(match: re.Match[str]) -> str:
            path = match.group(1)
            value = cls._get_value_from_path(context, path)
            if isinstance(value, str):
                return f"'{value}'"
            return str(value)

        expr = re.sub(r"\${([^}]+)}", replace_refs, condition)

        try:
            # Safe eval of the expression
            return bool(eval(expr, {"__builtins__": {}}, context))
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False


def create_evaluation_context(
        response_store: ResponseStore,
        inputs: WorkflowInput
) -> Dict[str, Any]:
    """
    Create a context for expression evaluation from response store and inputs.
    This utility function can be used by all execution engines.

    Args:
        response_store: The response store containing previous task outputs
        inputs: The workflow inputs as WorkflowInput or dict

    Returns:
        A context dictionary that can be used with ExpressionEvaluator
    """
    # test for input correctness
    workflow_data: Dict[str, Any] = {}

    if not (isinstance(inputs, WorkflowInput) or inputs.workflow is not None):
        logger.error(
            f"Invalid inout type found {type(inputs)} - expected WorkflowInput"
        )
    else:
        workflow_data = inputs.workflow

    context: Dict[str, Any] = {"workflow": workflow_data, "stages": {}}

    responses = response_store.responses

    # Add all previous task outputs from response_store to the context
    for stage_name, stage_data in responses.items():
        if stage_name not in context["stages"]:
            context["stages"][stage_name] = {"tasks": {}}

        for task_name, task_outputs in stage_data.items():
            context["stages"][stage_name]["tasks"][task_name] = {
                "outputs": task_outputs.result
            }

    return context
