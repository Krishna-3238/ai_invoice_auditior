"""Indexing Agent — parses and indexes all invoice documents into the FAISS vector store.

LangGraph flow:  index_node → END
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from backned.config.logging_setup import setup_logging
from backned.tool_dispatch import invoke_tool_sync

logger = logging.getLogger(__name__)


#  State 

class IndexingState(TypedDict):
    force_reindex: bool
    result: dict[str, Any]
    error: str


#  Nodes 

def index_node(state: IndexingState) -> IndexingState:
    """Call the vector_indexer_tool and capture the result."""
    logger.info("Indexing Agent: building / updating FAISS index ...")
    try:
        result = invoke_tool_sync("vector_indexer", {"force_reindex": state["force_reindex"]})
        logger.info(
            "Indexing complete: %d chunk(s) indexed, status=%s",
            result.get("chunks_indexed", 0),
            result.get("status", "?"),
        )
        return {**state, "result": result, "error": ""}
    except Exception as exc:
        logger.error("ERROR: Indexing failed: %s", exc)
        return {**state, "result": {}, "error": str(exc)}


#  Graph 

def build_indexing_graph() -> StateGraph:
    """Build the Indexing Agent LangGraph."""
    g = StateGraph(IndexingState)
    g.add_node("index", index_node)
    g.set_entry_point("index")
    g.add_edge("index", END)
    return g.compile()


class IndexingAgent:
    """Indexing Agent — builds/updates the FAISS vector index."""

    def __init__(self) -> None:
        self._graph = build_indexing_graph()

    def run(self, force_reindex: bool = False) -> dict[str, Any]:
        """Run the indexing agent.

        Returns the indexing result dict.
        """
        state = IndexingState(force_reindex=force_reindex, result={}, error="")
        final = self._graph.invoke(state)
        if final["error"]:
            raise RuntimeError(final["error"])
        return final["result"]


#  CLI 

if __name__ == "__main__":
    setup_logging()
    agent = IndexingAgent()
    res = agent.run(force_reindex=True)
    print(f"Indexed {res.get('chunks_indexed', 0)} chunks from {len(res.get('sources', []))} source(s)")
    print(f"Index path: {res.get('index_path')}")
