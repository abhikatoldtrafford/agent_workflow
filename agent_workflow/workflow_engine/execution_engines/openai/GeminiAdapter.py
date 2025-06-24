"""
Adapter for connecting Google's Gemini models to the OpenAI Agents SDK.
"""

import logging

from openai import AsyncOpenAI

from agent_workflow.workflow_engine.models import GeminiProviderConfig

logger = logging.getLogger("workflow-engine.execution_engines.openai.GeminiAdapter")


class GeminiModelAdapter:
    """Adapter to create Gemini models compatible with the OpenAI Agents SDK."""

    @staticmethod
    def openai_client(config: GeminiProviderConfig) -> AsyncOpenAI:
        """
        Create a Gemini model for use with OpenAI Agents SDK.

        Args:
            config: GeminiProviderConfig object containing configuration details

        Returns:
            A Gemini model adapter compatible with OpenAI Agents SDK
        """

        api_key = config.api_key
        base_url = config.base_url

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        return client
