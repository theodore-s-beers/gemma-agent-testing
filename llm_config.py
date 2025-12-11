import os
from typing import Any, Callable, ClassVar, Optional, Protocol, cast
from dotenv import load_dotenv


from lib.local_genai import LocalGenAIClient
from lib.openrouter_genai import OpenRouterGenAIClient

load_dotenv()


class GenAIProtocol(Protocol):
    types: ClassVar[Any]
    Client: Callable[[Optional[str]], Any]


# Determine which client to use
client_class = None
base_url = None
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

if LLM_PROVIDER == "lmstudio":
    try:
        base_url = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
        client_class = LocalGenAIClient
        print(f"[LLM Config] Using LM Studio at {base_url}")
    except ImportError:
        print(
            "[LLM Config] Error: local_genai module not found. Falling back to Google API."
        )
        LLM_PROVIDER = "google"

elif LLM_PROVIDER == "llamacpp":
    try:
        base_url = os.getenv("LLAMACPP_URL", "http://localhost:8080/v1")
        client_class = LocalGenAIClient
        print(f"[LLM Config] Using llama.cpp server at {base_url}")
    except ImportError:
        print(
            "[LLM Config] Error: local_genai module not found. Falling back to Google API."
        )
        LLM_PROVIDER = "google"

elif LLM_PROVIDER == "ollama":
    try:
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
        client_class = LocalGenAIClient
        print(f"[LLM Config] Using Ollama at {base_url}")
    except ImportError:
        print(
            "[LLM Config] Error: local_genai module not found. Falling back to Google API."
        )
        LLM_PROVIDER = "google"

elif LLM_PROVIDER == "openrouter":
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("[LLM Config] Warning: OPENROUTER_API_KEY not set in .env file")
        client_class = OpenRouterGenAIClient
        print("[LLM Config] Using OpenRouter")
    except ImportError:
        print(
            "[LLM Config] Error: openrouter_genai module not found. Falling back to Google API."
        )
        LLM_PROVIDER = "google"


if client_class is None:
    if LLM_PROVIDER != "google":
        print(
            f"[LLM Config] Warning: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Using Google Generative AI"
        )
    else:
        print("[LLM Config] Using Google Generative AI")

    if not os.getenv("GEMINI_API_KEY"):
        print("[LLM Config] Warning: GEMINI_API_KEY not set in .env file")

    import google.genai as _google_genai

    genai: GenAIProtocol = cast(GenAIProtocol, _google_genai)

    LLM_PROVIDER = "google"
else:
    import google.genai as _google_genai

    class GenAIShim:
        types: ClassVar[Any] = _google_genai.types

        @staticmethod
        def Client(api_key=None):
            if client_class == LocalGenAIClient:
                return LocalGenAIClient(api_key=api_key, base_url=base_url)
            elif client_class == OpenRouterGenAIClient:
                return OpenRouterGenAIClient(
                    api_key=api_key or os.getenv("OPENROUTER_API_KEY")
                )
            else:
                raise RuntimeError("Unsupported client class")

    genai: GenAIProtocol = GenAIShim()

ClientType = Any  # TODO: Replace `Any` with the actual client class
ContentType = Any  # TODO: Replace `Any` with the actual content class

__all__ = ["genai", "LLM_PROVIDER"]
