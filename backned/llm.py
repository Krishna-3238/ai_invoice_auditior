"""Small LiteLLM facade retaining the OpenAI-shaped calls used by agents."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from litellm import completion, embedding


class LiteLLMClient:
    """OpenAI-shaped facade; model routing and credentials are owned by LiteLLM."""
    def __init__(self, **_: Any):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._completion))
        self.embeddings = SimpleNamespace(create=self._embedding)

    @staticmethod
    def _completion(**kwargs):
        kwargs.pop("azure_endpoint", None)
        kwargs.pop("api_key", None)
        kwargs.pop("api_version", None)
        return completion(**kwargs)

    @staticmethod
    def _embedding(**kwargs):
        return embedding(**kwargs)
