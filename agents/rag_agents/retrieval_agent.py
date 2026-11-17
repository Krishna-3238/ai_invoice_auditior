"""
RAG Retrieval Agent
===================
Persona: Information Retriever
Role: Retrieve relevant document chunks based on user queries

This agent:
1. Converts user queries to embeddings
2. Performs similarity search in FAISS vector database
3. Retrieves top-k most relevant chunks
4. Ranks and filters results
5. Returns context for generation

The agent uses the same embedding model as the indexing agent.
Prompts are loaded dynamically from config/persona_invoice_agent.yaml.
"""

import logging
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.persona_loader import AgentPersona
from integrations.storage import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RetrievalAgent")


class RetrievalAgent:
    """
    Retrieval Agent that performs semantic search over indexed documents.
    Persona: Information Retriever

    Returns relevant chunks for RAG-based question answering.
    """

    def __init__(
        self, indexing_agent, top_k: int = 5, similarity_threshold: float = 0.5
    ):
        """
        Initialize the Retrieval Agent.

        Args:
            indexing_agent: IndexingAgent instance with loaded index
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.indexing_agent = indexing_agent
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self.persona = AgentPersona("retrieval_agent")

        self.embedding_model = indexing_agent.embedding_model
        self.qdrant = getattr(indexing_agent, "qdrant", QdrantVectorStore())

        logger.info(f"Retrieval Agent initialized (top_k={top_k})")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity.

        Args:
            query: User query string
            top_k: Number of results to return (overrides default)
            filter_metadata: Metadata filters (e.g., {"invoice_no": "INV-001"})

        Returns:
            List of relevant document chunks with scores
        """
        k = top_k if top_k is not None else self.top_k

        logger.info(f"=== RETRIEVAL SEARCH START ===")
        logger.info(f"Searching for: '{query[:100]}...'")
        logger.info(f"Documents in index: {len(self.indexing_agent.documents)}")
        logger.info(f"Requesting top_k: {k}")

        if len(self.indexing_agent.documents) == 0:
            logger.warning("No documents indexed yet. Cannot search.")
            logger.info("=== RETRIEVAL SEARCH END (EMPTY INDEX) ===")
            return []

        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)

        if self.qdrant.enabled:
            try:
                results = self.qdrant.search(
                    query_embedding[0].tolist(), k, filter_metadata
                )
                results = [r for r in results if r["similarity_score"] >= self.similarity_threshold]
                if results or not self.indexing_agent.documents:
                    return results
            except Exception as exc:
                logger.warning("Qdrant retrieval failed; falling back to FAISS: %s", exc)

        distances, indices = self.indexing_agent.index.search(
            query_embedding, min(k * 2, len(self.indexing_agent.documents))
        )                                                       

        if len(distances[0]) == 0 or len(indices[0]) == 0:
            logger.warning("No results from FAISS search")
            return []

        similarities = 1 / (1 + distances[0])

        results = []
        for idx, (doc_idx, similarity) in enumerate(zip(indices[0], similarities)):
            if doc_idx < 0 or doc_idx >= len(self.indexing_agent.documents):
                continue

            doc = self.indexing_agent.documents[doc_idx]

            if filter_metadata:
                if not self._matches_filter(doc["metadata"], filter_metadata):
                    continue

            if similarity < self.similarity_threshold:
                continue

            results.append(
                {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "similarity_score": float(similarity),
                    "rank": len(results) + 1,
                }
            )

            if len(results) >= k:
                break

        logger.info(f"Retrieved {len(results)} relevant chunks")
        return results

    def _matches_filter(
        self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_dict.items():
            if metadata.get(key) != value:
                return False
        return True

    def search_by_invoice(
        self, invoice_no: str, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific invoice.

        Args:
            invoice_no: Invoice number to filter by
            query: User query
            top_k: Number of results

        Returns:
            Relevant chunks from the specified invoice
        """
        return self.search(
            query, top_k=top_k, filter_metadata={"invoice_no": invoice_no}
        )

    def search_by_vendor(
        self, vendor_id: str, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search within invoices from a specific vendor.

        Args:
            vendor_id: Vendor ID to filter by
            query: User query
            top_k: Number of results

        Returns:
            Relevant chunks from the specified vendor
        """
        return self.search(query, top_k=top_k, filter_metadata={"vendor_id": vendor_id})

    def get_invoice_summary(self, invoice_no: str) -> Optional[Dict[str, Any]]:
        """
        Get all chunks for a specific invoice.

        Args:
            invoice_no: Invoice number

        Returns:
            All chunks for the invoice
        """
        chunks = []
        for doc in self.indexing_agent.documents:
            if doc["metadata"].get("invoice_no") == invoice_no:
                chunks.append({"text": doc["text"], "metadata": doc["metadata"]})

        if chunks:
            return {
                "invoice_no": invoice_no,
                "chunks": chunks,
                "total_chunks": len(chunks),
            }
        return None

    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute retrieval for a query.

        Args:
            query: User query
            filters: Optional metadata filters
            top_k: Number of results

        Returns:
            Retrieval results with metadata
        """
        logger.info("=" * 60)
        logger.info("RETRIEVAL AGENT: Starting retrieval")
        logger.info("=" * 60)

        results = self.search(query, top_k=top_k, filter_metadata=filters)

        context_texts = [r["text"] for r in results]
        context = "\n\n---\n\n".join(context_texts)

        logger.info("=" * 60)
        logger.info(f"RETRIEVAL COMPLETE: {len(results)} chunks retrieved")
        logger.info("=" * 60)

        return {
            "query": query,
            "results": results,
            "context": context,
            "num_results": len(results),
            "success": True,
        }


