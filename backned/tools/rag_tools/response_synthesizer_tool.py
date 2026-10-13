"""Response Synthesizer Tool — generates a natural language answer using
ranked context chunks, the user query, and conversation history.

Exposes `response_synthesizer_tool` as a LangChain @tool for the Generation Agent.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from backned.llm import invoke_bedrock_chat
from backned.config.settings import BEDROCK_RAG_MODEL


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert **Invoice Audit Support Agent** helping human support staff
understand invoice processing decisions made by an AI system.

You have access to:
- Detailed AI-generated audit reports for each invoice (verdict, issues, discrepancies)
- ERP master data (vendors, purchase orders, SKU catalogue)
- Validation rules applied during auditing

Guidelines:
1. Answer ONLY from the provided context.  If the context does not contain
   enough information, say so clearly — do not hallucinate facts.
2. When explaining a rejection or manual-review decision, quote the exact
   issues found (missing fields, discrepancies, ERP mismatches).
3. Use structured responses: use **bold** for key terms, and bullet points
   for lists of issues.
4. Keep answers concise but complete — no padding.
5. If asked about a specific invoice, identify it by filename or Invoice #
   and give a focused answer.
6. Maintain conversation context from chat history when answering follow-ups.
"""


@tool
def response_synthesizer_tool(
    query: str,
    ranked_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> dict:
    """Generate a contextual answer from ranked evidence chunks and conversation history.

    Args:
        query:         The user's current question.
        ranked_chunks: Top-N ranked context chunks from chunk_ranker_tool.
        chat_history:  Prior conversation turns as list of
                       {'role': 'user'|'assistant', 'content': str}.

    Returns:
        dict with keys: answer (str), sources_used (list[str]),
        context_chunks (int), status (str).
    """
    if not ranked_chunks:
        return {
            "answer":         "I could not find any relevant information in the indexed data. "
                              "Please ensure the pipeline has been run and the index is built.",
            "sources_used":   [],
            "context_chunks": 0,
            "status":         "no_context",
        }

    #  Build context block 
    context_parts = []
    sources_used: list[str] = []

    for i, chunk in enumerate(ranked_chunks, 1):
        src  = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        score = chunk.get("final_score", chunk.get("score", 0))
        context_parts.append(
            f"--- Context [{i}] | Source: {src} | Relevance: {score:.3f} ---\n{text}"
        )
        if src not in sources_used:
            sources_used.append(src)

    context_block = "\n\n".join(context_parts)

    #  Build message list 
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Include last 6 turns of conversation history (3 exchanges)
    if chat_history:
        for turn in chat_history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    # Current user message with context
    user_message = (
        f"Context from invoice audit system:\n\n{context_block}\n\n"
        f"---\nUser question: {query}"
    )
    messages.append({"role": "user", "content": user_message})

    #  Call LLM 
    answer = invoke_bedrock_chat(
        model_id=BEDROCK_RAG_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    answer = answer or "No response generated."

    logger.info(
        "Synthesized answer for query='%.60s' | %d chunks | %d sources",
        query, len(ranked_chunks), len(sources_used),
    )

    return {
        "answer":         answer,
        "sources_used":   sources_used,
        "context_chunks": len(ranked_chunks),
        "status":         "ok",
    }
