import os
from dotenv import load_dotenv

load_dotenv()

USE_LOCAL = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"

if USE_LOCAL:
    try:
        from lib.local_genai import LocalGenAIClient
        import google.genai as google_genai

        base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        print(f"[LLM Config] Using local LLM at {base_url}")

        class genai:
            types = google_genai.types

            @staticmethod
            def Client(api_key=None):
                return LocalGenAIClient(api_key=api_key, base_url=base_url)

    except ImportError:
        print(
            "[LLM Config] Error: local_genai module not found. Falling back to Google API."
        )
        import google.genai as genai

        USE_LOCAL = False

elif USE_OPENROUTER:
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
        print(
            "[LLM Config] Error: openrouter_genai module not found. Falling back to Google API."
        )
        import google.genai as genai
        USE_OPENROUTER = False

else:
    print("[LLM Config] Using Google Generative AI")
    import google.genai as genai

    if not os.getenv("GEMINI_API_KEY"):
        print("[LLM Config] Warning: GEMINI_API_KEY not set in .env file")

__all__ = ["genai", "USE_LOCAL", "USE_OPENROUTER"]
