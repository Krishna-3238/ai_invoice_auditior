"""LangChain model helpers for Amazon Bedrock chat and embeddings."""
from __future__ import annotations

from functools import lru_cache
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backned.config.settings import AWS_REGION_NAME


@lru_cache(maxsize=16)
def _chat_model(model_id: str, max_tokens: int, temperature: float):
    return ChatBedrock(
        model_id=model_id,
        region_name=AWS_REGION_NAME,
        model_kwargs={"max_tokens": max_tokens, "temperature": temperature},
    )


@lru_cache(maxsize=4)
def _embedding_model(model_id: str):
    return BedrockEmbeddings(model_id=model_id, region_name=AWS_REGION_NAME)


def invoke_bedrock_chat(
    model_id: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    """Invoke a Bedrock chat model with LangChain message objects."""
    langchain_messages = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        else:
            langchain_messages.append(HumanMessage(content=content))
    response = _chat_model(model_id, max_tokens, temperature).invoke(langchain_messages)
    return str(response.content or "")


def embed_bedrock_documents(model_id: str, texts: list[str]) -> list[list[float]]:
    """Create document vectors with Amazon Bedrock Embeddings."""
    return _embedding_model(model_id).embed_documents(texts)
