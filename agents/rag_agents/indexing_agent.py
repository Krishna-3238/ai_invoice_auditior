"""
RAG Indexing Agent
==================
Persona: Knowledge Base Manager
Role: Index and chunk invoice documents and reports for efficient retrieval

This agent:
1. Chunks invoice data and reports into manageable pieces
2. Generates embeddings for each chunk
3. Stores chunks and embeddings in FAISS vector database
4. Maintains metadata for traceability
5. Supports incremental indexing

The agent uses sentence transformers for embeddings and FAISS for vector storage.
Prompts are loaded dynamically from config/persona_invoice_agent.yaml.
"""

import logging
import os
import sys
from typing import Dict, Any, List
from pathlib import Path
import pickle
from datetime import datetime

from sentence_transformers import SentenceTransformer
import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.persona_loader import AgentPersona
from integrations.storage import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("IndexingAgent")


class IndexingAgent:
    """
    Indexing Agent that chunks documents and creates vector embeddings.
    Persona: Knowledge Base Manager

    Stores data in FAISS vector database for RAG retrieval.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_path: str = "./data/vector_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the Indexing Agent.

        Args:
            embedding_model: Sentence transformer model name
            index_path: Directory to store FAISS index
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.embedding_model_name = embedding_model
        self.index_path = Path(index_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.persona = AgentPersona("indexing_agent")

        self.index_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        self.index = None
        self.documents = []                                       
        self.qdrant = QdrantVectorStore()
        self.load_index()
        if self.documents and self.qdrant.enabled:
            self._sync_qdrant(self.documents)

    def _sync_qdrant(self, chunks: List[Dict[str, Any]]):
        """Mirror indexed chunks to Qdrant when QDRANT_URL is configured."""
        if not self.qdrant.enabled:
            return
        try:
            self.qdrant.upsert(chunks, self.embedding_dim)
        except Exception as exc:
            logger.warning("Qdrant indexing failed; FAISS remains available: %s", exc)

    def load_index(self):
        """Load existing FAISS index or create new one"""
        index_file = self.index_path / "faiss_index.bin"
        docs_file = self.index_path / "documents.pkl"

        if index_file.exists() and docs_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(docs_file, "rb") as f:
                    self.documents = pickle.load(f)
                logger.info(
                    f"Loaded existing index with {len(self.documents)} documents"
                )
            except Exception as e:
                logger.error(f"Failed to load index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []
        logger.info("Created new FAISS index")

    def save_index(self):
        """Save FAISS index and documents to disk"""
        try:
            index_file = self.index_path / "faiss_index.bin"
            docs_file = self.index_path / "documents.pkl"

            faiss.write_index(self.index, str(index_file))
            with open(docs_file, "wb") as f:
                pickle.dump(self.documents, f)

            logger.info(f"Saved index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        text = text.strip()

        if len(text) <= self.chunk_size:
            chunks.append(
                {
                    "text": text,
                    "metadata": {**metadata, "chunk_index": 0, "total_chunks": 1},
                }
            )
            return chunks

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                for punct in [". ", "! ", "? ", "\n\n"]:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct > self.chunk_size // 2:                            
                        end = start + last_punct + len(punct)
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": chunk_index,
                            "start_char": start,
                            "end_char": end,
                        },
                    }
                )
                chunk_index += 1

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

        return chunks

    def index_invoice_data(self, validation_result: Dict[str, Any]) -> int:
        """
        Index invoice data from validation results.

        Args:
            validation_result: Output from ValidationAgent

        Returns:
            Number of chunks indexed
        """
        invoice_data = validation_result.get("invoice_data", {})
        invoice_no = invoice_data.get("invoice_no", "UNKNOWN")

        logger.info(f"Indexing invoice: {invoice_no}")

        base_metadata = {
            "source": "invoice",
            "invoice_no": invoice_no,
            "vendor_id": invoice_data.get("vendor_id"),
            "po_number": invoice_data.get("po_number"),
            "date": invoice_data.get("invoice_date"),
            "indexed_at": datetime.now().isoformat(),
        }

        invoice_text = self._format_invoice_text(validation_result)

        chunks = self.chunk_text(invoice_text, base_metadata)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        self.index.add(embeddings)

        for chunk, embedding in zip(chunks, embeddings):
            self.documents.append(
                {
                    **chunk,
                    "embedding": embedding.tolist(),
                    "doc_id": len(self.documents),
                }
            )

        self._sync_qdrant([
            {**chunk, "embedding": embedding.tolist()}
            for chunk, embedding in zip(chunks, embeddings)
        ])

        logger.info(f"Indexed {len(chunks)} chunks for invoice {invoice_no}")
        return len(chunks)

    def index_report(self, report: Dict[str, Any]) -> int:
        """
        Index validation report.

        Args:
            report: JSON report from ReportingAgent

        Returns:
            Number of chunks indexed
        """
        invoice_summary = report.get("invoice_summary", {})
        invoice_no = invoice_summary.get("invoice_number", "UNKNOWN")

        logger.info(f"Indexing report for invoice: {invoice_no}")

        base_metadata = {
            "source": "report",
            "invoice_no": invoice_no,
            "report_id": report.get("report_metadata", {}).get("report_id"),
            "recommendation": report.get("validation_summary", {}).get(
                "recommendation"
            ),
            "indexed_at": datetime.now().isoformat(),
        }

        report_text = self._format_report_text(report)

        chunks = self.chunk_text(report_text, base_metadata)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        self.index.add(embeddings)

        for chunk, embedding in zip(chunks, embeddings):
            self.documents.append(
                {
                    **chunk,
                    "embedding": embedding.tolist(),
                    "doc_id": len(self.documents),
                }
            )

        self._sync_qdrant([
            {**chunk, "embedding": embedding.tolist()}
            for chunk, embedding in zip(chunks, embeddings)
        ])

        logger.info(f"Indexed {len(chunks)} chunks for report {invoice_no}")
        return len(chunks)

    def _format_invoice_text(self, validation_result: Dict[str, Any]) -> str:
        """Format invoice data as searchable text"""
        invoice_data = validation_result.get("invoice_data", {})
        line_items = validation_result.get("line_items", [])

        text_parts = [
            f"Invoice Number: {invoice_data.get('invoice_no', 'N/A')}",
            f"Invoice Date: {invoice_data.get('invoice_date', 'N/A')}",
            f"Vendor ID: {invoice_data.get('vendor_id', 'N/A')}",
            f"Vendor Name: {invoice_data.get('vendor_name', 'N/A')}",
            f"PO Number: {invoice_data.get('po_number', 'N/A')}",
            f"Currency: {invoice_data.get('currency', 'N/A')}",
            f"Total Amount: {invoice_data.get('total_amount', 'N/A')}",
            f"Tax Amount: {invoice_data.get('tax_amount', 'N/A')}",
            "\nLine Items:",
        ]

        for idx, item in enumerate(line_items, 1):
            text_parts.append(
                f"\nItem {idx}: {item.get('item_code', 'N/A')} - "
                f"{item.get('description', 'N/A')}, "
                f"Quantity: {item.get('qty', 'N/A')}, "
                f"Unit Price: {item.get('unit_price', 'N/A')}, "
                f"Total: {item.get('total', 'N/A')}"
            )

        return "\n".join(text_parts)

    def _format_report_text(self, report: Dict[str, Any]) -> str:
        """Format report as searchable text"""
        invoice_summary = report.get("invoice_summary", {})
        validation_summary = report.get("validation_summary", {})
        discrepancies = report.get("discrepancy_details", {})

        text_parts = [
            f"Invoice Validation Report",
            f"Invoice Number: {invoice_summary.get('invoice_number', 'N/A')}",
            f"Vendor: {invoice_summary.get('vendor_id', 'N/A')} - {invoice_summary.get('vendor_name', 'N/A')}",
            f"PO Number: {invoice_summary.get('po_number', 'N/A')}",
            f"Total Amount: {invoice_summary.get('total_amount', 'N/A')} {invoice_summary.get('currency', 'N/A')}",
            f"\nValidation Status: {validation_summary.get('recommendation', 'N/A')}",
            f"Errors: {validation_summary.get('errors_count', 0)}",
            f"Warnings: {validation_summary.get('warnings_count', 0)}",
            f"\nExecutive Summary: {report.get('executive_summary', 'N/A')}",
        ]

        errors = discrepancies.get("errors", [])
        warnings = discrepancies.get("warnings", [])

        if errors:
            text_parts.append("\nErrors Found:")
            for error in errors:
                text_parts.append(f"- {error.get('message', 'No message')}")

        if warnings:
            text_parts.append("\nWarnings Found:")
            for warning in warnings:
                text_parts.append(f"- {warning.get('message', 'No message')}")

        return "\n".join(text_parts)

    def index_reviewer_feedback(
        self, invoice_data: Dict[str, Any], feedback: Dict[str, Any]
    ) -> int:
        """
        Index reviewer feedback/notes for an invoice.

        Args:
            invoice_data: Invoice data dictionary
            feedback: Feedback dictionary containing reviewer_notes, decision, reviewer_id, etc.

        Returns:
            Number of chunks indexed
        """
        invoice_no = invoice_data.get("invoice_no", "UNKNOWN")
        reviewer_notes = feedback.get("reviewer_notes", "")
        decision = feedback.get("decision", "")
        reviewer_id = feedback.get("reviewer_id", "unknown")
        reviewed_at = feedback.get("reviewed_at", datetime.now().isoformat())

        if not reviewer_notes and not decision:
            logger.info(f"No reviewer notes or decision to index for invoice {invoice_no}")
            return 0

        logger.info(f"Indexing reviewer feedback for invoice: {invoice_no}")

        base_metadata = {
            "source": "reviewer_feedback",
            "invoice_no": invoice_no,
            "vendor_id": invoice_data.get("vendor_id"),
            "vendor_name": invoice_data.get("vendor_name"),
            "reviewer_id": reviewer_id,
            "decision": decision,
            "reviewed_at": reviewed_at,
            "indexed_at": datetime.now().isoformat(),
        }

        feedback_text = self._format_feedback_text(invoice_data, feedback)

        chunks = self.chunk_text(feedback_text, base_metadata)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        self.index.add(embeddings)

        for chunk, embedding in zip(chunks, embeddings):
            self.documents.append(
                {
                    **chunk,
                    "embedding": embedding.tolist(),
                    "doc_id": len(self.documents),
                }
            )

        self._sync_qdrant([
            {**chunk, "embedding": embedding.tolist()}
            for chunk, embedding in zip(chunks, embeddings)
        ])

        logger.info(f"Indexed {len(chunks)} feedback chunks for invoice {invoice_no}")
        return len(chunks)

    def _format_feedback_text(
        self, invoice_data: Dict[str, Any], feedback: Dict[str, Any]
    ) -> str:
        """Format reviewer feedback as searchable text"""
        invoice_no = invoice_data.get("invoice_no") or "N/A"
        vendor_name = invoice_data.get("vendor_name") or "N/A"
        total_amount = invoice_data.get("total_amount")
        total_amount = "N/A" if total_amount is None else total_amount
        currency = invoice_data.get("currency") or "USD"

        reviewer_notes = feedback.get("reviewer_notes") or "No reviewer notes recorded"
        decision = feedback.get("decision") or "N/A"
        reviewer_id = feedback.get("reviewer_id") or "unknown"
        reviewed_at = feedback.get("reviewed_at") or "N/A"
        data_edited = feedback.get("data_edited", False)

        text_parts = [
            f"Reviewer Feedback for Invoice {invoice_no}",
            f"Invoice Number: {invoice_no}",
            f"Vendor: {vendor_name}",
            f"Total Amount: {total_amount} {currency}",
            f"",
            f"Review Decision: {decision}",
            f"Reviewed By: {reviewer_id}",
            f"Review Date: {reviewed_at}",
            f"Data Edited: {'Yes' if data_edited else 'No'}",
            f"",
            f"Reviewer Notes: {reviewer_notes}",
        ]

        return "\n".join(text_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        return {
            "total_documents": len(self.documents),
            "index_dimension": self.embedding_dim,
            "embedding_model": self.embedding_model_name,
            "index_path": str(self.index_path),
            "invoices_indexed": len(
                set(
                    doc["metadata"].get("invoice_no")
                    for doc in self.documents
                    if doc["metadata"].get("invoice_no")
                )
            ),
            "reports_indexed": len(
                [
                    doc
                    for doc in self.documents
                    if doc["metadata"].get("source") == "report"
                ]
            ),
            "feedback_indexed": len(
                [
                    doc
                    for doc in self.documents
                    if doc["metadata"].get("source") == "reviewer_feedback"
                ]
            ),
        }

    def run(self, data: Dict[str, Any], data_type: str = "invoice") -> Dict[str, Any]:
        """
        Index data (invoice or report).

        Args:
            data: Validation result or report
            data_type: "invoice" or "report"

        Returns:
            Indexing result
        """
        logger.info(f"Starting indexing for {data_type}")

        if data_type == "invoice":
            chunks_added = self.index_invoice_data(data)
        elif data_type == "report":
            chunks_added = self.index_report(data)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        self.save_index()

        stats = self.get_stats()
        logger.info(f"Indexing complete. Total documents: {stats['total_documents']}")

        return {"chunks_added": chunks_added, "stats": stats, "success": True}


