import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Type, TypeVar, Optional

import yaml
from agent_workflow.workflow_engine.models import (
    BaseModel,
    DynamicModelGenerator,
    LLMAgent,
    ModelGenerator,
    Workflow,
)

logger = logging.getLogger("workflow-engine.parsers")

T = TypeVar("T", bound=BaseModel)


class ConfigParser(ABC, Generic[T]):
    """Abstract base class for configuration parsers."""

    @abstractmethod
    def parse_workflow(self, raw_wf: dict[str, Any]) -> Workflow:
        """Parse a workflow configuration file into a Workflow model."""
        pass

    @abstractmethod
    def parse_workflow_file(self, file_path: str) -> Workflow:
        """Parse a workflow YAML file into a Workflow model."""
        pass

    @abstractmethod
    def parse_workflow_str(self, raw_yaml_string: str) -> Workflow:
        """Parse a workflow configuration file into a Workflow model."""
        pass

    @abstractmethod
    def parse_agent(self, file_path: str) -> LLMAgent:
        """Parse an agent configuration file into a LLMAgent model."""
        pass

class YAMLParser(ConfigParser):
    """Parse YAML configuration files."""

    def __init__(self, model_generator: Optional[ModelGenerator] = None):
        self.model_generator = model_generator or DynamicModelGenerator()

    def load_config(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file as a dictionary."""
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to load YAML file '{file_path}': {e}")
            raise

    def resolve_references(
        self, config: Dict[str, Any], base_dir: str
    ) -> Dict[str, Any]:
        """
        Recursively resolve 'ref' references in the configuration.

        Args:
            config: The configuration dictionary
            base_dir: The base directory for resolving relative paths

        Returns:
            Updated configuration with references resolved
        """
        resolved_config: dict[str, Any] = {}

        for key, value in config.items():
            # If we find a 'ref' key, load and merge the referenced file
            if key == "ref" and isinstance(value, str):
                # Resolve the referenced file path relative to the base directory
                ref_path = os.path.join(base_dir, value)

                # Check if the referenced file exists
                if not os.path.exists(ref_path):
                    raise FileNotFoundError(f"Referenced file not found: {ref_path}")

                try:
                    # Load the referenced file
                    ref_config = self.load_config(ref_path)

                    # Recursively resolve any nested references in the loaded file
                    ref_config = self.resolve_references(
                        ref_config, os.path.dirname(ref_path)
                    )

                    # Return the resolved referenced config (maintaining the 'ref' key for documentation)
                    resolved_config[key] = value

                    # Merge the referenced config with the current one
                    for ref_key, ref_value in ref_config.items():
                        # Only add keys that don't already exist in the parent config
                        if ref_key not in resolved_config:
                            resolved_config[ref_key] = ref_value

                except Exception as e:
                    logger.error(f"Failed to resolve reference '{value}': {e}")
                    raise ValueError(f"Failed to resolve reference '{value}': {e}")
            elif isinstance(value, dict):
                # Recursively resolve nested dictionaries
                resolved_config[key] = self.resolve_references(value, base_dir)
            elif isinstance(value, list):
                # Recursively resolve items in lists
                resolved_config[key] = [
                    self.resolve_references(item, base_dir)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                # Keep other values as is
                resolved_config[key] = value

        return resolved_config

    def load_schema_models(self, schema_file: str) -> Dict[str, Type[BaseModel]]:
        """Load YAML schema and create Pydantic models."""
        yaml_content = self.load_config(schema_file)
        models = {}

        # Create models from schema definitions
        for model_name, model_schema in yaml_content.items():
            model = self.model_generator.create_model(model_name, model_schema)
            models[model_name] = model

        return models

    def parse_config(self, raw_config: dict[str, Any], model_class: Type[T]) -> T:
        """Parse configuration file into a structured model instance."""
        try:
            # Parse the resolved config into the model
            return model_class.parse_obj(raw_config)
        except Exception as e:
            logger.error(
                f"Failed to parse config '{raw_config}' as {model_class.__name__}: {e}"
            )
            raise ValueError(
                f"Invalid configuration provided: {str(e)}"
            )

    def parse_config_file(self, file_path: str, model_class: Type[T]) -> T:
        """Parse configuration file into a structured model instance."""
        try:
            # Load the raw configuration
            raw_config = self.load_config(file_path)

            # Resolve any references in the configuration
            base_dir = os.path.dirname(file_path)
            resolved_config = self.resolve_references(raw_config, base_dir)

            # Parse the resolved config into the model
            return model_class.parse_obj(resolved_config)
        except FileNotFoundError as e:
            logger.error(f"Referenced file not found in '{file_path}': {e}")
            raise ValueError(
                f"Referenced file not found in {os.path.basename(file_path)}: {str(e)}"
            )
        except Exception as e:
            logger.error(
                f"Failed to parse config '{file_path}' as {model_class.__name__}: {e}"
            )
            raise ValueError(
                f"Invalid configuration in {os.path.basename(file_path)}: {str(e)}"
            )

    def parse_workflow_file(self, file_path: str) -> Workflow:
        """Parse a workflow YAML file into a Workflow model."""
        return self.parse_config_file(file_path, Workflow)

    def parse_workflow(self, raw_wf: dict[str, Any]) -> Workflow:
        """Parse a workflow YAML file into a Workflow model."""
        return self.parse_config(raw_wf, Workflow)

    def parse_workflow_str(self, raw_yaml_string: str) -> Workflow:
        """Parse a workflow configuration file into a Workflow model."""
        raw_dict = yaml.safe_load(raw_yaml_string)
        return self.parse_workflow(raw_dict)

    def parse_agent(self, file_path: str) -> LLMAgent:
        """Parse an agent YAML file into a PlanAgent model."""
        return self.parse_config_file(file_path, LLMAgent)
