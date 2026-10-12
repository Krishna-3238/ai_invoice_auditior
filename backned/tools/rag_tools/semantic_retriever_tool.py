"""Retrieve relevant invoice/report chunks from Qdrant."""
from langchain_core.tools import tool
from backned.services.qdrant_store import retrieve


@tool
def semantic_retriever_tool(query: str, top_k: int = 10, invoice_id: str | None = None) -> dict:
    """Retrieve indexed chunks, optionally scoped to an invoice UUID."""
    chunks = retrieve(query, limit=top_k, invoice_id=invoice_id)
    return {"query": query, "chunks": chunks, "total_indexed": len(chunks), "status": "ok" if chunks else "empty_index"}
