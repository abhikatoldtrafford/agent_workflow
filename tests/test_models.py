from typing import List, Optional

import pytest
from pydantic import BaseModel

from agent_workflow.workflow_engine import (
    AgentConfig,
    DynamicModelGenerator,
    ModelRegistry,
    ResponseStore,
    StageExecutionStrategy,
    WorkflowStage,
    WorkflowTask,
)


class TestModelRegistry:
    """Unit tests for the ModelRegistry."""

    def setup_method(self):
        # Reset the registry before each test
        ModelRegistry._models = {}

    def test_register_and_get(self):
        """Test registering and retrieving models."""

        # Create a simple Pydantic model
        class TestModel(BaseModel):
            name: str
            value: int

        # Register the model
        ModelRegistry.register("TestModel", TestModel)

        # Retrieve the model
        retrieved_model = ModelRegistry.get("TestModel")

        assert retrieved_model == TestModel

    def test_has(self):
        """Test checking if a model exists."""

        # Create a simple Pydantic model
        class TestModel(BaseModel):
            name: str
            value: int

        # Initially the registry should be empty
        assert not ModelRegistry.has("TestModel")

        # Register the model
        ModelRegistry.register("TestModel", TestModel)

        # Now it should exist
        assert ModelRegistry.has("TestModel")

    def test_get_nonexistent(self):
        """Test retrieving a non-existent model."""
        # Try to retrieve a model that doesn't exist
        retrieved_model = ModelRegistry.get("NonExistentModel")

        # Should return None
        assert retrieved_model is None


class TestDynamicModelGenerator:
    """Unit tests for the DynamicModelGenerator."""

    def setup_method(self):
        # Reset the registry before each test
        ModelRegistry._models = {}
        self.generator = DynamicModelGenerator()

    def test_get_field_type(self):
        """Test converting type strings to Python types."""
        # Test basic types
        assert DynamicModelGenerator._get_field_type("str") is str
        assert DynamicModelGenerator._get_field_type("int") is int
        assert DynamicModelGenerator._get_field_type("float") is float
        assert DynamicModelGenerator._get_field_type("bool") is bool
        assert DynamicModelGenerator._get_field_type("dict") is dict
        assert DynamicModelGenerator._get_field_type("list") is list

        # Test list with item type
        list_type = DynamicModelGenerator._get_field_type("list", "str")
        assert list_type == List[str]

    def test_create_model(self):
        """Test creating a model from a schema definition."""
        schema = {
            "name": {"type": "str", "description": "The name", "required": True},
            "age": {"type": "int", "description": "The age", "required": True},
            "email": {
                "type": "str",
                "description": "The email address",
                "required": False,
            },
            "tags": {
                "type": "list",
                "item_type": "str",
                "description": "A list of tags",
                "required": True,
            },
        }

        # Create the model
        model = self.generator.create_model("Person", schema)

        # Verify the model was created successfully
        assert issubclass(model, BaseModel)
        assert ModelRegistry.has("Person")

        # Verify the model has the expected fields
        field_info = model.__annotations__
        assert "name" in field_info and field_info["name"] is str
        assert "age" in field_info and field_info["age"] is int
        assert "email" in field_info and field_info["email"] is Optional[str]
        assert "tags" in field_info

        # Create an instance of the model
        person = model(name="John", age=30, tags=["developer", "python"])
        assert person.name == "John"
        assert person.age == 30
        assert person.email is None
        assert person.tags == ["developer", "python"]

    def test_create_model_with_default_values(self):
        """Test creating a model with default values."""
        schema = {
            "name": {"type": "str", "description": "The name", "required": True},
            "age": {
                "type": "int",
                "description": "The age",
                "value": 25,  # Default value
            },
            "active": {
                "type": "bool",
                "description": "Whether the user is active",
                "value": True,  # Default value
            },
        }

        # Create the model
        model = self.generator.create_model("User", schema)

        # Create an instance with minimal values
        user = model(name="Alice")

        # Verify default values were applied
        assert user.name == "Alice"
        assert user.age == 25
        assert user.active is True

    def test_nested_model_references(self):
        """Test creating models with references to other models."""
        # address_model = self.generator.create_model("Address", address_schema)

        # Now create a Person model that references the Address model
        person_schema = {
            "name": {"type": "str", "description": "The name", "required": True},
            "address": {
                "type": "Address",  # Reference to the Address model
                "description": "The person's address",
                "required": True,
            },
        }

        person_model = self.generator.create_model("Person", person_schema)

        # Create a person with a nested address
        person = person_model(
            name="John",
            address={"street": "123 Main St", "city": "Anytown", "zip_code": "12345"},
        )

        # Verify the nested model works correctly
        assert person.name == "John"
        assert person.address.street == "123 Main St"
        assert person.address.city == "Anytown"
        assert person.address.zip_code == "12345"


class TestWorkflowStageExecutionType:
    """Unit tests for the WorkflowStage execution_type validation."""

    def setup_method(self):
        # Create a minimal agent config for testing
        self.agent_config = AgentConfig(id="test_agent", ref="test_agent.yaml")

    def test_valid_enum_value(self):
        """Test with valid enum value."""
        stage = WorkflowStage(
            name="Test Stage",
            execution_type=StageExecutionStrategy.SEQUENTIAL,
            tasks=[WorkflowTask(name="Test Task", agent=self.agent_config)],
        )
        assert stage.execution_type == StageExecutionStrategy.SEQUENTIAL

    def test_valid_string_value_lowercase(self):
        """Test with valid string value (lowercase)."""
        stage = WorkflowStage(
            name="Test Stage",
            execution_type="parallel",
            tasks=[WorkflowTask(name="Test Task", agent=self.agent_config)],
        )
        assert stage.execution_type == StageExecutionStrategy.PARALLEL

    def test_valid_string_value_uppercase(self):
        """Test with valid string value (uppercase)."""
        stage = WorkflowStage(
            name="Test Stage",
            execution_type="SEQUENTIAL",
            tasks=[WorkflowTask(name="Test Task", agent=self.agent_config)],
        )
        assert stage.execution_type == StageExecutionStrategy.SEQUENTIAL

    def test_invalid_string_value(self):
        """Test with invalid string value."""
        with pytest.raises(ValueError):
            WorkflowStage(
                name="Test Stage",
                execution_type="invalid_value",
                tasks=[WorkflowTask(name="Test Task", agent=self.agent_config)],
            )

    def test_execution_type_enum(self):
        """Test the ExecutionType enum."""
        assert StageExecutionStrategy.SEQUENTIAL.value == "sequential"
        assert StageExecutionStrategy.PARALLEL.value == "parallel"
        assert StageExecutionStrategy.HANDOFF.value == "handoff"

        # Test string conversion
        assert str(StageExecutionStrategy.SEQUENTIAL) == "sequential"
        assert str(StageExecutionStrategy.PARALLEL) == "parallel"
        assert str(StageExecutionStrategy.HANDOFF) == "handoff"

        # Test equality
        assert StageExecutionStrategy.SEQUENTIAL == StageExecutionStrategy.SEQUENTIAL
        assert StageExecutionStrategy.SEQUENTIAL != StageExecutionStrategy.PARALLEL
        assert (
            StageExecutionStrategy.SEQUENTIAL == "sequential"
        )  # String comparison works too


class TestResponseStore:
    """Unit tests for the ResponseStore."""

    def test_add_and_get(self):
        """Test adding and retrieving responses."""
        store = ResponseStore()

        # Add response data
        store.add("stage1", "task1", {"result": "hello", "confidence": 0.9})

        # Retrieve the whole task data
        task_data = store.get("stage1", "task1")
        assert task_data == {"result": "hello", "confidence": 0.9}

        # Retrieve specific key
        assert store.get("stage1", "task1", "result") == "hello"
        assert store.get("stage1", "task1", "confidence") == 0.9

    def test_non_existent_entries(self):
        """Test retrieving non-existent entries."""
        store = ResponseStore()

        # Add some data
        store.add("stage1", "task1", {"result": "hello"})

        # Test non-existent stage
        with pytest.raises(ValueError):
            store.get("stage2", "task1")

        # Test non-existent task
        with pytest.raises(ValueError):
            store.get("stage1", "task2")

        # Test non-existent key
        with pytest.raises(ValueError):
            store.get("stage1", "task1", "non_existent_key")

    def test_has_methods(self):
        """Test the methods to check existence of entries."""
        store = ResponseStore()

        # Add response data
        store.add("stage1", "task1", {"result": "hello"})

        # Test has_stage
        assert store.has_stage("stage1") is True
        assert store.has_stage("stage2") is False

        # Test has_task
        assert store.has_task("stage1", "task1") is True
        assert store.has_task("stage1", "task2") is False
        assert store.has_task("stage2", "task1") is False

        # Test has_key
        assert store.has_key("stage1", "task1", "result") is True
        assert store.has_key("stage1", "task1", "non_existent") is False
        assert store.has_key("stage1", "task2", "result") is False
        assert store.has_key("stage2", "task1", "result") is False

    def test_get_lists(self):
        """Test methods to get lists of stages, tasks, and keys."""
        store = ResponseStore()

        # Add response data
        store.add("stage1", "task1", {"key1": "value1", "key2": "value2"})
        store.add("stage1", "task2", {"key3": "value3"})
        store.add("stage2", "task1", {"key4": "value4"})

        # Test get_stages
        stages = store.get_stages()
        assert len(stages) == 2
        assert "stage1" in stages
        assert "stage2" in stages

        # Test get_tasks
        tasks1 = store.get_tasks("stage1")
        assert len(tasks1) == 2
        assert "task1" in tasks1
        assert "task2" in tasks1

        tasks2 = store.get_tasks("stage2")
        assert len(tasks2) == 1
        assert "task1" in tasks2

        # Test get_tasks with non-existent stage
        with pytest.raises(ValueError):
            store.get_tasks("stage3")

        # Test get_keys
        keys1 = store.get_keys("stage1", "task1")
        assert len(keys1) == 2
        assert "key1" in keys1
        assert "key2" in keys1

        keys2 = store.get_keys("stage1", "task2")
        assert len(keys2) == 1
        assert "key3" in keys2

        # Test get_keys with non-existent stage/task
        with pytest.raises(ValueError):
            store.get_keys("stage3", "task1")

        with pytest.raises(ValueError):
            store.get_keys("stage1", "task3")

    def test_to_dict(self):
        """Test converting ResponseStore to a dictionary."""
        store = ResponseStore()

        # Add response data
        store.add("stage1", "task1", {"key1": "value1"})
        store.add("stage2", "task2", {"key2": "value2"})

        # Convert to dict
        result = store.to_dict()

        # Verify structure
        assert isinstance(result, dict)
        assert "stage1" in result
        assert "stage2" in result
        assert "task1" in result["stage1"]
        assert "task2" in result["stage2"]
        assert result["stage1"]["task1"]["key1"] == "value1"
        assert result["stage2"]["task2"]["key2"] == "value2"
