import logging
from enum import Enum
from typing import Optional, Union, cast

import openai
from agents import OpenAIChatCompletionsModel
from mypy.nodes import CastExpr
from openai import AsyncOpenAI

from agent_workflow.workflow_engine.execution_engines.openai.GeminiAdapter import GeminiModelAdapter
from agent_workflow.workflow_engine.models import BaseProviderConfig, GeminiProviderConfig

logger = logging.getLogger(
    "workflow_engine.execution_engines.openai.OpenAIAgentAdapter"
)


class ProviderType(str, Enum):
    """Supported LLM provider types"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    GEMINI = "gemini"


class OpenAIAgentAdapter:
    """
    Adapter for connecting different LLM providers to OpenAI Agents SDK.
    """

    def __init__(
        self,
        provider: BaseProviderConfig,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the adapter with a specific provider.
        Args:
            provider: the provider configuration
            base_url: Base URL for the provider, optional parameter not needed for OpenAI
            api_key: API key for the provider
        """
        self.provider = provider
        self.provider_type = provider.provider_type
        self.api_key = api_key
        self.base_url = base_url
        self.provider_kwargs: dict[str, str] = {}

    def _initialize_client(self) -> AsyncOpenAI:
        """Initialize the appropriate client based on provider type."""
        if self.provider_type == ProviderType.OPENAI:
            raise NotImplementedError(
                "OpenAI provider is not supported by the Adapter."
            )

        elif self.provider_type == ProviderType.ANTHROPIC:
            return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)  # type: ignore

        elif self.provider_type == ProviderType.GEMINI:
            try:  # type: ignore
                # now cast for MyPy
                gemini_cfg = cast(GeminiProviderConfig, self.provider)

                return GeminiModelAdapter.openai_client(gemini_cfg)
            except CastExpr:
                logger.error("A `GeminiProviderConfig` is required to Use Gemini.")
                raise
        else:
            raise ValueError(f"Unsupported provider type: {self.provider_type}")

    def chat_completion_model(
        self, model: str
    ) -> Union[str, OpenAIChatCompletionsModel]:
        """
        Create a chat completion model.
        Args:
            model: The model name
        """
        if self.provider_type == ProviderType.OPENAI:
            openai.api_key = self.api_key  # type: ignore
            return model
        else:
            if self.provider_type == ProviderType.GEMINI:
                # For Gemini, we need to store the model name in provider_kwargs
                # we really don't need it now, consider removing it
                self.provider_kwargs["model"] = model  # type: ignore

            # Initialize the client for the appropriate provider
            client = self._initialize_client()

            # Return the OpenAIChatCompletionsModel with the initialized client
            return OpenAIChatCompletionsModel(model=model, openai_client=client)
