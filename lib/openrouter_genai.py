from openai import OpenAI
import os


class OpenRouterGenAIClient:
    """Drop-in replacement for google.genai.Client using OpenRouter."""

    def __init__(self, api_key: str | None = None):
        """Initialize client.
        
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
        self._default_model = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")
        self.models = self.ModelsAPI(self._client, self._default_model)

    class ModelsAPI:
        """Mimics the genai.Client().models interface"""

        def __init__(self, client: OpenAI, default_model: str):
            self._client = client
            self._default_model = default_model

        def generate_content(
            self,
            model: str,
            contents: str | list | dict,
            temperature: float = 0.7,
            max_tokens: int | None = None,
            **kwargs,
        ) -> "OpenRouterGenAIClient.Response":
            """Generate content matching genai interface.

            Args:
                model: Model name (can use OpenRouter model names or pass through).
                contents: Prompt string or structured contents.
                temperature: Sampling temperature.
                max_tokens: Maximum tokens to generate (None = use model's default)
                **kwargs: Additional parameters.

            Returns:
                Response object with .text attribute.
            """
            messages = self._convert_contents_to_messages(contents)

            actual_model = self._default_model
            
            params = {
                "model": actual_model,
                "messages": messages,
                "temperature": temperature,
            }

            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            params.update(kwargs)

            response = self._client.chat.completions.create(**params)

            return OpenRouterGenAIClient.Response(response)

        def _convert_contents_to_messages(self, contents: any) -> list[dict[str, str]]:
            """Convert various content formats to OpenAI messages format."""
            if isinstance(contents, str):
                return [{"role": "user", "content": contents}]

            elif isinstance(contents, list):
                messages = []
                for content in contents:
                    if isinstance(content, dict):
                        if "role" in content and "content" in content:
                            messages.append(content)
                        elif "text" in content:
                            messages.append(
                                {"role": "user", "content": content["text"]}
                            )
                        else:
                            messages.append({"role": "user", "content": str(content)})
                    else:
                        messages.append({"role": "user", "content": str(content)})
                return messages

            elif isinstance(contents, dict):
                if "role" in contents and "content" in contents:
                    return [contents]
                elif "text" in contents:
                    return [{"role": "user", "content": contents["text"]}]
                else:
                    return [{"role": "user", "content": str(contents)}]

            else:
                return [{"role": "user", "content": str(contents)}]

    class UsageMetadata:
        """Mimics google.genai usage metadata."""

        def __init__(self, openai_usage):
            if openai_usage:
                self.prompt_token_count = openai_usage.prompt_tokens
                self.candidates_token_count = openai_usage.completion_tokens
                self.total_token_count = openai_usage.total_tokens
            else:
                self.prompt_token_count = 0
                self.candidates_token_count = 0
                self.total_token_count = 0

        def __repr__(self):
            return (
                f"UsageMetadata(prompt_token_count={self.prompt_token_count}, "
                f"candidates_token_count={self.candidates_token_count}, "
                f"total_token_count={self.total_token_count})"
            )

    class Response:
        """Mimics google.genai response object."""

        def __init__(self, openai_response):
            self._response = openai_response
            self.text = openai_response.choices[0].message.content
            self.usage_metadata = OpenRouterGenAIClient.UsageMetadata(
                openai_response.usage
            )

        def __str__(self):
            return self.text or ""

        def __repr__(self):
            return f"Response(text={self.text!r})"

        def __bool__(self):
            return bool(self.text)
