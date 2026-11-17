"""
Integrated RAG Query Interface
===============================
Complete RAG pipeline for querying invoice information.

This module provides a simple interface to:
1. Index invoice data and reports
2. Query the indexed data
3. Get high-quality answers with evaluation metrics
"""

import logging
import re
import os
import sys
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.rag_agents.indexing_agent import IndexingAgent
from agents.rag_agents.retrieval_agent import RetrievalAgent
from agents.rag_agents.augmentation_agent import AugmentationAgent
from agents.rag_agents.generation_agent import GenerationAgent
from agents.rag_agents.reflection_agent import ReflectionAgent
from agents.rag_agents.query_router_agent import QueryRouterAgent
from config.feedback_manager import get_feedback_manager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAGInterface")


def emit(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


class InvoiceRAGSystem:
    """
    Complete RAG system for invoice querying.

    Combines Indexing, Retrieval, Generation, and Reflection agents
    into a single easy-to-use interface.
    """

    def __init__(self, model: str = None, vector_db_path: str = "./data/vector_db"):
        """
        Initialize the RAG system.

        Args:
            model: LiteLLM model string (e.g., 'bedrock/cohere.command-r-plus-v1:0')
                  Set to None to use template/heuristic modes
            vector_db_path: Path to store vector database
        """
        self.indexer = IndexingAgent(index_path=vector_db_path)
        self.retriever = RetrievalAgent(
            indexing_agent=self.indexer, similarity_threshold=0.3
        )
        self.augmenter = AugmentationAgent(llm={"model": model} if model else None)
        self.generator = GenerationAgent(model=model)
        self.reflector = ReflectionAgent(model=model)
        self.query_router = QueryRouterAgent()
        self.feedback_manager = get_feedback_manager()

        self.model = model

        logger.info("Invoice RAG System initialized")
        if not model:
            logger.warning("⚠️  Running without LLM - using template/heuristic modes")
        else:
            logger.info(f"📡 Using LiteLLM model: {model}")

    def reload_index(self):
        """
        Reload the index from disk to pick up new documents.
        Call this after indexing new documents to refresh the in-memory index.
        """
        logger.info("=== RELOAD_INDEX CALLED ===")
        logger.info(f"Documents before reload: {len(self.indexer.documents)}")
        logger.info("Reloading index from disk...")
        self.indexer.load_index()
        logger.info(f"Documents after reload: {len(self.indexer.documents)}")
        logger.info(f"Index reloaded with {len(self.indexer.documents)} documents")
        logger.info("=== RELOAD_INDEX COMPLETE ===")

    def index_invoice(self, invoice_data: Dict[str, Any]) -> bool:
        """
        Index an invoice for querying.

        Args:
            invoice_data: Invoice data dictionary

        Returns:
            Success status
        """
        try:
            self.indexer.index_invoice_data(invoice_data)
            logger.info(f"Indexed invoice {invoice_data.get('invoice_no', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to index invoice: {e}")
            return False

    def index_report(
        self, report_path: str, invoice_no: str, vendor_id: Optional[str] = None
    ) -> bool:
        """
        Index a validation report.

        Args:
            report_path: Path to JSON report file
            invoice_no: Invoice number
            vendor_id: Vendor ID (optional)

        Returns:
            Success status
        """
        try:
            self.indexer.index_report(report_path, invoice_no, vendor_id)
            logger.info(f"Indexed report for {invoice_no}")
            return True
        except Exception as e:
            logger.error(f"Failed to index report: {e}")
            return False

    def query(
        self,
        question: str,
        top_k: int = 5,
        include_followups: bool = False,
        evaluate: bool = True,
    ) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve
            include_followups: Generate follow-up questions
            evaluate: Run quality evaluation

        Returns:
            Complete query result with answer and metrics
        """
        logger.info("=" * 80)
        logger.info(f"RAG QUERY: {question}")
        logger.info("=" * 80)

        review_answer = self._answer_human_review_question(question)
        if review_answer:
            logger.info("Answered from human feedback history")
            return review_answer

        logger.info("Step 1: Retrieving context...")
        route = self.query_router.route(question)
        retrieval_result = self.retriever.run(
            question, filters=route["filters"] or None, top_k=top_k
        )

        if retrieval_result["num_results"] == 0:
            logger.warning("No relevant context found")
            return {
                "query": question,
                "answer": "I couldn't find any relevant information in the indexed documents to answer your question.",
                "has_answer": False,
                "sources": [],
                "evaluation": None,
                "success": True,
            }

        logger.info(f"Retrieved {retrieval_result['num_results']} relevant chunks")

        logger.info("Step 2: Reranking context...")
        augmented_result = self.augmenter.run(
            retrieval_result["query"], retrieval_result, top_k=top_k
        )
        logger.info(f"Reranked to {augmented_result['num_results']} chunks")

        logger.info("Step 3: Generating answer...")
        if include_followups:
            generation_result = self.generator.generate_with_followup(
                question, augmented_result["context"], augmented_result["results"]
            )
        else:
            generation_result = self.generator.generate_answer(
                question, augmented_result["context"], augmented_result["results"]
            )

        logger.info("Answer generated")

        evaluation_result = None
        if evaluate:
            logger.info("Step 4: Evaluating quality...")
            evaluation_result = self.reflector.run(
                query=question,
                context=augmented_result["context"],
                answer=generation_result["answer"],
                sources=augmented_result["results"],
            )
            logger.info(
                f"Quality: {evaluation_result['quality']} ({evaluation_result['overall_score']:.3f})"
            )

        result = {
            "query": question,
            "query_route": route["route"],
            "answer": generation_result["answer"],
            "has_answer": generation_result["has_answer"],
            "confidence": generation_result["confidence"],
            "sources": [
                {
                    "text_preview": (
                        s["text"][:200] + "..." if len(s["text"]) > 200 else s["text"]
                    ),
                    "metadata": s["metadata"],
                    "similarity_score": s["similarity_score"],
                }
                for s in generation_result["sources_used"]
            ],
            "num_sources": len(generation_result["sources_used"]),
            "evaluation": evaluation_result,
            "success": True,
        }

        if include_followups and "suggested_followups" in generation_result:
            result["suggested_followups"] = generation_result["suggested_followups"]

        logger.info("=" * 80)
        logger.info("QUERY COMPLETED")
        logger.info("=" * 80)

        return result

    def _answer_human_review_question(self, question: str) -> Optional[Dict[str, Any]]:
        """Answer invoice-specific human review questions from persisted feedback."""
        invoice_no = self._extract_invoice_no(question)
        if not invoice_no or not self._is_review_decision_question(question):
            return None

        history = self.feedback_manager.get_invoice_feedback_history(invoice_no)
        if not history:
            return None

        history = sorted(history, key=lambda item: item.get("timestamp", ""))
        latest = history[-1]
        previous = history[:-1]

        decision = (latest.get("decision") or latest.get("feedback_type") or "unknown").lower()
        feedback_type = latest.get("feedback_type", "review")
        reviewer_id = latest.get("reviewer_id", "unknown")
        timestamp = latest.get("timestamp", "N/A")
        notes = (latest.get("user_notes") or "").strip()
        reason_label = self._decision_reason_label(feedback_type, decision)
        changes = latest.get("changes_summary", {}) or {}
        has_corrections = bool(changes.get("changes_detected"))

        answer_parts = [
            f"{invoice_no} is currently human {decision}.",
            f"The latest review decision was {decision} by {reviewer_id} on {timestamp}.",
        ]
        if notes:
            answer_parts.append(f"{reason_label}: {self._sentence(notes)}")
        else:
            answer_parts.append(f"No {reason_label.lower()} was recorded for this decision.")

        if has_corrections:
            changed_fields = changes.get("changed_fields", [])
            if changed_fields:
                answer_parts.append(
                    "The reviewer approved it after correcting: "
                    + ", ".join(str(field) for field in changed_fields)
                    + "."
                )
            else:
                answer_parts.append("The reviewer made corrections before the final decision.")
        else:
            answer_parts.append("No data edits were recorded with the latest decision.")

        if previous:
            previous_summary = self._summarize_previous_feedback(previous)
            answer_parts.append(
                f"There are {len(previous)} earlier review record(s), but the latest review is treated as the current decision. "
                f"Earlier history: {previous_summary}."
            )

        source_text = self._format_feedback_source_text(invoice_no, latest, current=True)
        sources = [
            {
                "text_preview": source_text,
                "metadata": {
                    "source": "feedback_history",
                    "invoice_no": invoice_no,
                    "feedback_type": feedback_type,
                    "decision": decision,
                    "reviewer_id": reviewer_id,
                    "reviewed_at": timestamp,
                    "is_latest_decision": True,
                },
                "similarity_score": 1.0,
            }
        ]

        return {
            "query": question,
            "answer": " ".join(answer_parts),
            "has_answer": True,
            "confidence": "high",
            "sources": sources,
            "num_sources": len(sources),
            "evaluation": None,
            "success": True,
            "method": "feedback_history",
        }

    def _extract_invoice_no(self, question: str) -> Optional[str]:
        """Extract invoice IDs such as INV-1001, INV 1001, INV_L005, or invoice 1001."""
        match = re.search(r"\bINV[\s:_-]*([A-Z]*[\s:_-]?\d[\w-]*)\b", question, re.IGNORECASE)
        if match:
            suffix = re.sub(r"[\s:_]+", "-", match.group(1).strip().upper())
            return f"INV-{suffix}"

        match = re.search(r"\binvoice[\s#:_-]+([A-Z]*[\s:_-]?\d[\w-]*)\b", question, re.IGNORECASE)
        if match:
            suffix = re.sub(r"[\s:_]+", "-", match.group(1).strip().upper())
            return f"INV-{suffix}"

        return None

    def _is_review_decision_question(self, question: str) -> bool:
        query = question.lower()
        decision_terms = [
            "approve",
            "approved",
            "approval",
            "reject",
            "rejected",
            "rejection",
            "decision",
            "review",
            "reviewer",
            "human",
            "why was",
            "why is",
            "status",
        ]
        return any(term in query for term in decision_terms)

    def _summarize_previous_feedback(self, history: List[Dict[str, Any]]) -> str:
        counts = {}
        for entry in history:
            decision = entry.get("decision") or entry.get("feedback_type") or "unknown"
            counts[decision] = counts.get(decision, 0) + 1
        return ", ".join(f"{count} {decision}" for decision, count in sorted(counts.items()))

    def _format_feedback_source_text(
        self, invoice_no: str, feedback: Dict[str, Any], current: bool = False
    ) -> str:
        status = "Current Latest Review" if current else "Review"
        decision = feedback.get("decision", "unknown")
        feedback_type = feedback.get("feedback_type", "unknown")
        reason_label = self._decision_reason_label(feedback_type, decision)
        notes = feedback.get("user_notes") or "No reviewer notes recorded"
        return (
            f"{status} for Invoice {invoice_no}. "
            f"Decision: {decision}. "
            f"Feedback Type: {feedback_type}. "
            f"Reviewed By: {feedback.get('reviewer_id', 'unknown')}. "
            f"Review Date: {feedback.get('timestamp', 'N/A')}. "
            f"{reason_label}: {notes}"
        )

    def _sentence(self, value: str) -> str:
        value = value.strip()
        if value.endswith((".", "!", "?")):
            return value
        return f"{value}."

    def _decision_reason_label(self, feedback_type: str, decision: str) -> str:
        value = f"{feedback_type} {decision}".lower()
        if "approval" in value or "approved" in value:
            return "Approval reason"
        if "rejection" in value or "rejected" in value:
            return "Rejection reason"
        if "correction" in value or "corrected" in value:
            return "Correction reason"
        if "escalation" in value or "escalated" in value:
            return "Escalation reason"
        return "Review reason"

    def query_by_invoice(
        self, question: str, invoice_no: str, top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Query specific invoice.

        Args:
            question: Question about the invoice
            invoice_no: Invoice number to search
            top_k: Number of results

        Returns:
            Query result
        """
        logger.info(f"Querying invoice {invoice_no}...")

        retrieval_result = self.retriever.search_by_invoice(invoice_no, top_k=top_k)

        if retrieval_result["num_results"] == 0:
            return {
                "query": question,
                "answer": f"No information found for invoice {invoice_no}.",
                "has_answer": False,
                "sources": [],
                "success": True,
            }

        generation_result = self.generator.generate_answer(
            question, retrieval_result["context"], retrieval_result["results"]
        )

        return {
            "query": question,
            "invoice_no": invoice_no,
            "answer": generation_result["answer"],
            "has_answer": generation_result["has_answer"],
            "sources": generation_result["sources_used"],
            "success": True,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        indexer_stats = self.indexer.get_stats()
        return {
            "total_documents": indexer_stats.get("total_documents", 0),
            "total_chunks": indexer_stats.get(
                "total_documents", 0
            ),                             
            "vector_db_path": str(self.indexer.index_path),
            "has_llm": self.model is not None,
            "model": self.model,
            "invoices_indexed": indexer_stats.get("invoices_indexed", 0),
            "reports_indexed": indexer_stats.get("reports_indexed", 0),
        }




def create_rag_system(
    use_llm: bool = False,
    model: str = "bedrock/cohere.command-r-plus-v1:0",
    vector_db_path: str = "./data/vector_db",
) -> InvoiceRAGSystem:
    """
    Create a RAG system with optional LLM.

    Args:
        use_llm: Whether to use LLM for generation
        model: LiteLLM model string
        vector_db_path: Path to vector database

    Returns:
        Initialized RAG system
    """
    if use_llm:
        logger.info(f"Initializing RAG system with LiteLLM model: {model}")
        return InvoiceRAGSystem(model=model, vector_db_path=vector_db_path)
    else:
        logger.info("Initializing RAG system without LLM (template mode)")
        return InvoiceRAGSystem(model=None, vector_db_path=vector_db_path)




def interactive_query_mode():
    """Run interactive query mode"""
    emit("=" * 80)
    emit("INVOICE RAG SYSTEM - Interactive Query Mode")
    emit("=" * 80)

    emit("\nInitializing RAG system...")
    rag = create_rag_system(use_llm=False)

    stats = rag.get_stats()
    emit(f"\n📊 System Stats:")
    emit(f"   Documents indexed: {stats['total_documents']}")
    emit(f"   Total chunks: {stats['total_chunks']}")
    emit(f"   LLM enabled: {stats['has_llm']}")

    if stats["total_documents"] == 0:
        emit("\n⚠️  No documents indexed yet!")
        emit("   Run test_rag_pipeline.py to index sample data first.")
        return

    emit("\n" + "=" * 80)
    emit("Enter your questions (type 'quit' or 'exit' to stop)")
    emit("=" * 80)

    while True:
        emit("\n")
        question = input("❓ Your question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            emit("\n👋 Goodbye!")
            break

        if not question:
            continue

        result = rag.query(question, evaluate=True)

        emit("\n" + "-" * 80)
        emit(f"💬 Answer: {result['answer']}")
        emit(f"\n📊 Confidence: {result['confidence']}")
        emit(f"📚 Sources: {result['num_sources']}")

        if result["evaluation"]:
            eval_data = result["evaluation"]
            emit(
                f"\n✨ Quality: {eval_data['quality']} ({eval_data['overall_score']:.3f})"
            )
            emit(
                f"   • Context Relevance: {eval_data['metrics']['context_relevance']['score']:.3f}"
            )
            emit(
                f"   • Groundedness: {eval_data['metrics']['groundedness']['score']:.3f}"
            )
            emit(
                f"   • Answer Relevance: {eval_data['metrics']['answer_relevance']['score']:.3f}"
            )

        if result["sources"]:
            emit(f"\n📑 Top Sources:")
            for i, source in enumerate(result["sources"][:3], 1):
                metadata = source["metadata"]
                emit(
                    f"   [{i}] Invoice: {metadata.get('invoice_no', 'N/A')}, "
                    f"Score: {source['similarity_score']:.3f}"
                )

        emit("-" * 80)



if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_query_mode()
    else:
        emit("=" * 80)
        emit("INVOICE RAG SYSTEM")
        emit("=" * 80)
        emit("\nUsage:")
        emit(
            "  python workflows/rag_interface.py --interactive    # Start interactive query mode"
        )
        emit("\nProgrammatic usage:")
        emit(
            """
from workflows.rag_interface import create_rag_system

rag = create_rag_system(use_llm=False)

rag.index_invoice({
    "invoice_no": "INV-001",
    "vendor_name": "Acme Corp",
    "total_amount": 1000.0
})

result = rag.query("What is the total for INV-001?")
answer = result["answer"]
        """
        )
        emit("\n💡 Run 'python test_rag_pipeline.py' to test with sample data")
        emit("=" * 80)
