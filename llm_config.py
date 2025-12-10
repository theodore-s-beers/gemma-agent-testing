import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

if LLM_PROVIDER == "lmstudio":
    try:
        from lib.local_genai import LocalGenAIClient
        import google.genai as google_genai

        base_url = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
        print(f"[LLM Config] Using LM Studio at {base_url}")

        class genai:
            types = google_genai.types

            @staticmethod
            def Client(api_key=None):
                return LocalGenAIClient(api_key=api_key, base_url=base_url)

    except ImportError:
        print("[LLM Config] Error: local_genai module not found. Falling back to Google API.")

        import google.genai as genai
        LLM_PROVIDER = "google"

elif LLM_PROVIDER == "llamacpp":
    try:
        from lib.local_genai import LocalGenAIClient
        import google.genai as google_genai

        base_url = os.getenv("LLAMACPP_URL", "http://localhost:8080/v1")
        print(f"[LLM Config] Using llama.cpp server at {base_url}")
        
        class genai:
            types = google_genai.types

            @staticmethod
            def Client(api_key=None):
                return LocalGenAIClient(api_key=api_key, base_url=base_url)

    except ImportError:
        print("[LLM Config] Error: local_genai module not found. Falling back to Google API.")
        import google.genai as genai
        LLM_PROVIDER = "google"

elif LLM_PROVIDER == "openrouter":
    try:
        from lib.openrouter_genai import OpenRouterGenAIClient
        import google.genai as google_genai

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("[LLM Config] Warning: OPENROUTER_API_KEY not set in .env file")

        print("[LLM Config] Using OpenRouter")

        class genai:
            types = google_genai.types

            @staticmethod
            def Client(api_key=None):
                return OpenRouterGenAIClient(api_key=api_key or os.getenv("OPENROUTER_API_KEY"))

    except ImportError:
        print("[LLM Config] Error: openrouter_genai module not found. Falling back to Google API.")

        import google.genai as genai
        LLM_PROVIDER = "google"

else:
    if LLM_PROVIDER != "google":
        print(f"[LLM Config] Warning: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Using Google Generative AI")
    else:
        print("[LLM Config] Using Google Generative AI")
    import google.genai as genai

    if not os.getenv("GEMINI_API_KEY"):
        print("[LLM Config] Warning: GEMINI_API_KEY not set in .env file")
    LLM_PROVIDER = "google"

__all__ = ["genai", "LLM_PROVIDER"]
