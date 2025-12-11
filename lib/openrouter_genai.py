from openai import OpenAI
import os
from lib.openai_genai import OpenAIGenAIClient


class OpenRouterGenAIClient(OpenAIGenAIClient):
    """Drop-in replacement for google.genai.Client using OpenRouter."""

    def __init__(self, api_key: str | None = None):
        """Initialize client for OpenRouter.

        Args:
            api_key: OpenRouter API key
        """
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError("OpenRouter API key is required")

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self._default_model = os.getenv(
            "OPENROUTER_MODEL", "google/gemma-3-27b-it:free"
        )

        super().__init__()
