import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from agent_workflow.providers import LLMServiceProvider

logger = logging.getLogger("workflow-engine.agents")


class Agent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self, config: Dict[str, Any], provider: Optional[LLMServiceProvider] = None
    ):
        self.config = config
        self.provider = provider

    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with the given inputs.

        Args:
            inputs: Dictionary of input parameters for the agent

        Returns:
            Dictionary of output values from the agent execution
        """
        pass
