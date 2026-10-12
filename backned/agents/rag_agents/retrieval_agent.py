"""Retrieval Agent — accepts a user query, generates its embedding, and performs
semantic similarity search over the FAISS index.

LangGraph flow:  retrieve_node → END
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

class RetrievalState(TypedDict):
    query:   str
    top_k:   int
    invoice_id: str | None
    result:  dict[str, Any]
    error:   str


#  Node 

def retrieve_node(state: RetrievalState) -> RetrievalState:
    """Call the semantic_retriever_tool and capture raw candidates."""
    logger.info("Retrieval Agent: query='%.80s'", state["query"])
    try:
        # Invoke MCP tool (async-only tools are wrapped by invoke_tool_sync)
        result = invoke_tool_sync("semantic_retriever", {
            "query": state["query"],
            "top_k": state["top_k"],
            "invoice_id": state.get("invoice_id"),
        })
        n = len(result.get("chunks", []))
        logger.info("Retrieved %d candidate chunk(s)", n)
        return {**state, "result": result, "error": ""}
    except Exception as exc:
        logger.error("ERROR: Retrieval failed: %s", exc)
        return {**state, "result": {"query": state["query"], "chunks": [], "status": "error"}, "error": str(exc)}


#  Graph 

def build_retrieval_graph() -> StateGraph:
    g = StateGraph(RetrievalState)
    g.add_node("retrieve", retrieve_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", END)
    return g.compile()


class RetrievalAgent:
    """Retrieval Agent — semantic search over the invoice FAISS index."""

    def __init__(self) -> None:
        self._graph = build_retrieval_graph()

    @observe(name="agent-rag-retrieval")
    def run(self, query: str, top_k: int = 10, callbacks: list | None = None, invoice_id: str | None = None) -> dict[str, Any]:
        """Run the retrieval agent.

        Returns the retrieval result dict (query, chunks, total_indexed, status).
        """
        state = RetrievalState(query=query, top_k=top_k, invoice_id=invoice_id, result={}, error="")
        cfg = {"callbacks": callbacks} if callbacks else {}
        final = self._graph.invoke(state, config=cfg)
        return final["result"]


#  CLI 

if __name__ == "__main__":
    setup_logging()
    import sys
    q = " ".join(sys.argv[1:]) or "Why was invoice INV_EN_006 rejected?"
    agent = RetrievalAgent()
    res = agent.run(q)
    print(f"Query: {res['query']}")
    print(f"Retrieved {len(res['chunks'])} chunks (index has {res['total_indexed']} total)")
    for c in res["chunks"][:3]:
        print(f"[{c['score']:.3f}] {c['source']}: {c['text'][:120]}")
