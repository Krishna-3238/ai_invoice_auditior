"""Generation Agent — uses the Response Synthesizer Tool to generate a
contextual answer from ranked chunks and conversation history.

LangGraph flow:  generate_node → END
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from backned.config.logging_setup import setup_logging
from backned.tool_dispatch import invoke_tool_sync
from backned.config.observability import observe, langfuse_context

logger = logging.getLogger(__name__)


#  State 

class GenerationState(TypedDict):
    query:         str
    ranked_chunks: list[dict[str, Any]]
    chat_history:  list[dict[str, str]]
    result:        dict[str, Any]
    error:         str


#  Node 

def generate_node(state: GenerationState) -> GenerationState:
    """Call response_synthesizer_tool to produce the final answer."""
    logger.info(
        "Generation Agent: synthesizing answer | %d chunks | %d history turns",
        len(state["ranked_chunks"]), len(state["chat_history"]),
    )
    try:
        # Invoke MCP tool (async-only tools are wrapped by invoke_tool_sync)
        result = invoke_tool_sync("response_synthesizer", {
            "query":         state["query"],
            "ranked_chunks": state["ranked_chunks"],
            "chat_history":  state["chat_history"],
        })
        logger.info("Answer generated (%d chars)", len(result.get("answer", "")))
        return {**state, "result": result, "error": ""}
    except Exception as exc:
        logger.error("ERROR: Generation failed: %s", exc)
        return {
            **state,
            "result": {
                "answer":         f"Generation error: {exc}",
                "sources_used":   [],
                "context_chunks": 0,
                "status":         "error",
            },
            "error": str(exc),
        }


#  Graph 

def build_generation_graph() -> StateGraph:
    g = StateGraph(GenerationState)
    g.add_node("generate", generate_node)
    g.set_entry_point("generate")
    g.add_edge("generate", END)
    return g.compile()


class GenerationAgent:
    """Generation Agent — synthesizes a final LLM answer from context."""

    def __init__(self) -> None:
        self._graph = build_generation_graph()

    @observe(name="agent-rag-generation")
    def run(
        self,
        query: str,
        ranked_chunks: list[dict],
        chat_history: list[dict] | None = None,
        callbacks: list | None = None,
    ) -> dict[str, Any]:
        """Run the generation agent.

        Returns dict with answer (str), sources_used (list), context_chunks (int), status (str).
        """
        state = GenerationState(
            query=query,
            ranked_chunks=ranked_chunks,
            chat_history=chat_history or [],
            result={},
            error="",
        )
        cfg = {"callbacks": callbacks} if callbacks else {}
        final = self._graph.invoke(state, config=cfg)
        return final["result"]
