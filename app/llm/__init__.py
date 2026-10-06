"""The chat model, built in one place.

Local Ollama, so the project keeps the property its README leads with: no API
keys, no rate limits, no cloud dependency. Embeddings were already local; this
keeps generation local too.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel

from app.config import settings


def get_chat_model(**overrides: Any) -> BaseChatModel:
    """A configured ChatOllama. Overrides win, for tests and integration runs.

    Imported lazily so that importing app.main does not pull langchain_ollama
    and httpx into every process that only wants to serve static files.
    """
    from langchain_ollama import ChatOllama

    params: dict[str, Any] = {
        "model": settings.OLLAMA_MODEL,
        "base_url": settings.OLLAMA_BASE_URL,
        "temperature": settings.OLLAMA_TEMPERATURE,
        # Defaults from the setting so a Qwen3-family model doesn't emit its
        # chain of thought as the answer, with no way for the UI to tell the
        # difference. An explicit reasoning= override below deliberately
        # takes precedence, for tests and integration runs.
        "reasoning": not settings.OLLAMA_DISABLE_THINKING,
    }
    params.update(overrides)
    return ChatOllama(**params)
