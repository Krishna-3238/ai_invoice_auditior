"""Augmentation Agent — re-ranks raw retrieved chunks using the Chunk Ranker Tool
to improve context quality before generation.

LangGraph flow:  rank_node → END
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

class AugmentationState(TypedDict):
    query:   str
    chunks:  list[dict[str, Any]]
    top_n:   int
    result:  dict[str, Any]
    error:   str


#  Node 

def rank_node(state: AugmentationState) -> AugmentationState:
    """Call chunk_ranker_tool to re-rank and deduplicate candidate chunks."""
    logger.info(
        "Augmentation Agent: re-ranking %d chunk(s) for top-%d",
        len(state["chunks"]), state["top_n"],
    )
    try:
        # Invoke MCP tool (async-only tools are wrapped by invoke_tool_sync)
        result = invoke_tool_sync("chunk_ranker", {
            "query":  state["query"],
            "chunks": state["chunks"],
            "top_n":  state["top_n"],
        })
        n = len(result.get("ranked_chunks", []))
        logger.info("Re-ranking complete: %d chunk(s) selected", n)
        return {**state, "result": result, "error": ""}
    except Exception as exc:
        logger.error("ERROR: Augmentation failed: %s", exc)
        return {
            **state,
            "result": {"query": state["query"], "ranked_chunks": state["chunks"][:state["top_n"]], "status": "fallback"},
            "error":  str(exc),
        }


#  Graph 

def build_augmentation_graph() -> StateGraph:
    g = StateGraph(AugmentationState)
    g.add_node("rank", rank_node)
    g.set_entry_point("rank")
    g.add_edge("rank", END)
    return g.compile()


class AugmentationAgent:
    """Augmentation Agent — re-ranks retrieved chunks for maximum relevance."""

    def __init__(self) -> None:
        self._graph = build_augmentation_graph()

    @observe(name="agent-rag-augmentation")
    def run(self, query: str, chunks: list[dict], top_n: int = 5, callbacks: list | None = None) -> dict[str, Any]:
        """Run the augmentation agent.

        Returns a dict with ranked_chunks (list), query (str), status (str).
        """
        state = AugmentationState(
            query=query, chunks=chunks, top_n=top_n, result={}, error=""
        )
        cfg = {"callbacks": callbacks} if callbacks else {}
        final = self._graph.invoke(state, config=cfg)
        return final["result"]
