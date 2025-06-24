import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger("workflow-engine.providers")


class LLMServiceProvider(ABC):
    """Abstract interface for LLM service providers."""

    def __init__(self) -> None:
        # Default observability provider reference (will be set by the factory)
        self.observability_provider: Optional[Any] = None

    def set_observability_provider(self, provider: Any) -> None:
        """Set the observability provider for this LLM provider."""
        self.observability_provider = provider

    @abstractmethod
    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        max_tokens: int = 4096,
    ) -> Any:
        """Generate a response from the LLM.

        Args:
            system_prompt: The system prompt for the LLM
            user_prompt: The user prompt for the LLM
            model_name: The name of the model to use
            max_tokens: Maximum number of tokens in the response

        Returns:
            The generated response from the LLM
        """
        pass

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_registry: Any,
        model_name: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response using tools.

        This method should be implemented by providers that support tool calling.

        Args:
            system_prompt: The system prompt for the model
            user_prompt: The user prompt for the model
            tools: List of tool definitions in provider-specific format
            tool_registry: The tool registry for executing tools
            model_name: Optional model name
            max_tokens: Maximum tokens in the response

        Returns:
            A tuple containing:
            - The final text response
            - List of tool call records

        Raises:
            NotImplementedError: If the provider doesn't support tool calling
        """
        raise NotImplementedError("This provider does not support tool calling")


class LLMProviderFactory:
    """Factory for creating LLM service providers."""

    _providers: Dict[str, Type[LLMServiceProvider]] = {
        # Add other providers here
    }

    @classmethod
    def register_provider(
        cls, provider_type: str, provider_class: Type[LLMServiceProvider]
    ) -> None:
        """Register a new provider type."""
        cls._providers[provider_type] = provider_class

    @classmethod
    def create_provider(cls, provider_type: str, **kwargs: Any) -> LLMServiceProvider:
        """Create a provider instance based on type."""
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown provider type: {provider_type}")

        return cls._providers[provider_type](**kwargs)
