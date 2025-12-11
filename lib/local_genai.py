from openai import OpenAI
import os
from lib.openai_genai import OpenAIGenAIClient


class LocalGenAIClient(OpenAIGenAIClient):
    """Drop-in replacement for google.genai.Client using local OpenAI-compatible servers."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize client for local LLM servers.

        Args:
            api_key: Not used, kept for compatibility with genai.Client
            base_url: Local server URL (default: http://localhost:1234/v1)
        """
        if base_url is None:
            base_url = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")

        self._client = OpenAI(base_url=base_url, api_key="local-model")
        self._default_model = "local-model"

        super().__init__()
