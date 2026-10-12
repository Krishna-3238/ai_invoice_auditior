"""Index PostgreSQL invoice text and reports into the derived Qdrant store."""
from langchain_core.tools import tool
from backned.services.qdrant_store import index_all


@tool
def vector_indexer_tool(force_reindex: bool = False) -> dict:
    """Index all persisted invoice and report text in Qdrant."""
    return index_all()
