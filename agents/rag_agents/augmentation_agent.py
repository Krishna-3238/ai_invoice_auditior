"""
RAG Augmentation Agent
======================
Persona: Context Reranker
Role: Rerank retrieved context for optimal relevance

This agent:
1. Takes retrieved chunks from RetrievalAgent
2. Reranks them using cross-encoder or LLM
3. Filters low-quality results
4. Returns optimized context for generation

The agent improves RAG quality by reordering results based on
relevance to the specific query.
Prompts are loaded dynamically from config/persona_invoice_agent.yaml.
"""

import logging
import os
import sys
from typing import Dict, Any, List, Optional
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.persona_loader import AgentPersona, get_prompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AugmentationAgent")


class AugmentationAgent:
    """
    Augmentation Agent that reranks retrieved context for better relevance.
    Persona: Context Reranker

    Reranking improves RAG quality by:
    - Reordering results based on query-specific relevance
    - Filtering out marginally relevant chunks
    - Boosting results with key entities (invoice numbers, amounts)

    Prompts are loaded dynamically from persona configuration.
    """

    def __init__(self, llm=None, rerank_threshold: float = 0.3):
        """
        Initialize the Augmentation Agent.

        Args:
            llm: LLM for LLM-based reranking (via LiteLLM)
            rerank_threshold: Minimum score to keep a result (0-1)
        """
        self.llm = llm
        self.rerank_threshold = rerank_threshold

        self.persona = AgentPersona("augmentation_agent")

        if not llm:
            logger.warning("⚠️  No LLM provided. Using heuristic-based reranking.")
        else:
            logger.info("Augmentation Agent initialized with LLM")

    def rerank(
        self, query: str, results: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved results for better relevance.

        Args:
            query: User query
            results: List of retrieved chunks with similarity scores
            top_k: Number of top results to return after reranking

        Returns:
            Reranked list of results
        """
        if not results:
            logger.warning("No results to rerank")
            return []

        if self.llm:
            try:
                reranked = self._rerank_with_llm(query, results)
            except Exception as e:
                logger.warning(f"LLM reranking failed: {e}. Using heuristic.")
                reranked = self._rerank_heuristic(query, results)
        else:
            reranked = self._rerank_heuristic(query, results)

        filtered = [r for r in reranked if r["rerank_score"] >= self.rerank_threshold]

        if top_k:
            filtered = filtered[:top_k]

        logger.info(f"Reranked {len(results)} → {len(filtered)} results")

        return filtered

    def _rerank_heuristic(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Heuristic-based reranking using keyword matching and entity boosting.
        """
        query_lower = query.lower()

        invoice_pattern = r"inv[-_]?\d+|invoice\s+\d+|#\d+"
        po_pattern = r"po[-_]?\d+|purchase\s+order\s+\d+"
        amount_pattern = r"\$\d+|\d+\.\d+\s*(usd|eur|gbp)"

        query_has_invoice = bool(re.search(invoice_pattern, query_lower, re.IGNORECASE))
        query_has_po = bool(re.search(po_pattern, query_lower, re.IGNORECASE))
        query_has_amount = bool(re.search(amount_pattern, query_lower, re.IGNORECASE))

        stop_words = {
            "what",
            "is",
            "the",
            "for",
            "in",
            "on",
            "at",
            "to",
            "a",
            "an",
            "how",
            "when",
            "where",
            "who",
            "which",
            "show",
            "me",
            "tell",
            "get",
            "find",
            "search",
            "look",
        }
        query_words = set(re.findall(r"\w+", query_lower)) - stop_words

        reranked = []

        for result in results:
            text = result.get("text", "").lower()
            metadata = result.get("metadata", {})
            base_score = result.get("similarity_score", 0.5)

            rerank_score = base_score

            keyword_matches = sum(1 for word in query_words if word in text)
            keyword_boost = (
                (keyword_matches / len(query_words)) * 0.2 if query_words else 0
            )
            rerank_score += keyword_boost

            if query_has_invoice:
                text_has_invoice = bool(re.search(invoice_pattern, text, re.IGNORECASE))
                if text_has_invoice:
                    rerank_score += 0.15

            if query_has_po:
                text_has_po = bool(re.search(po_pattern, text, re.IGNORECASE))
                if text_has_po:
                    rerank_score += 0.15

            if query_has_amount:
                text_has_amount = bool(re.search(amount_pattern, text, re.IGNORECASE))
                if text_has_amount:
                    rerank_score += 0.1

            if "invoice_no" in metadata:
                invoice_in_query = any(
                    inv in query_lower
                    for inv in [
                        metadata["invoice_no"].lower(),
                        metadata["invoice_no"].replace("-", "").lower(),
                    ]
                )
                if invoice_in_query:
                    rerank_score += 0.25                                        

            rerank_score = min(1.0, rerank_score)

            reranked.append(
                {
                    **result,
                    "rerank_score": round(rerank_score, 3),
                    "original_score": base_score,
                    "rerank_method": "heuristic",
                }
            )

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked

    def _rerank_with_llm(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        LLM-based reranking using relevance scoring.
        """
        from litellm import completion

        reranked = []

        for i, result in enumerate(results):
            text = result.get("text", "")
            base_score = result.get("similarity_score", 0.5)

            prompt = f"""Rate how relevant this document chunk is to answering the query.

Query: {query}

Document Chunk:
{text[:500]}

Rate the relevance on a scale of 0.0 to 1.0:
- 1.0: Highly relevant, directly answers the query
- 0.5: Somewhat relevant, contains related information
- 0.0: Not relevant to the query

Provide only the numeric score (e.g., 0.85):"""

            try:
                response = completion(
                    model=(
                        self.llm.model
                        if hasattr(self.llm, "model")
                        else "bedrock/cohere.command-r-plus-v1:0"
                    ),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=10,
                )

                score_text = response.choices[0].message.content.strip()

                score_match = re.search(r"([0-9.]+)", score_text)
                llm_score = float(score_match.group(1)) if score_match else 0.5
                llm_score = max(0.0, min(1.0, llm_score))

                combined_score = (llm_score * 0.7) + (base_score * 0.3)

                reranked.append(
                    {
                        **result,
                        "rerank_score": round(combined_score, 3),
                        "llm_score": round(llm_score, 3),
                        "original_score": base_score,
                        "rerank_method": "llm",
                    }
                )

            except Exception as e:
                logger.warning(f"LLM scoring failed for chunk {i}: {e}")
                reranked.append(
                    {
                        **result,
                        "rerank_score": base_score,
                        "original_score": base_score,
                        "rerank_method": "fallback",
                    }
                )

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked

    def augment_context(
        self, query: str, retrieval_result: Dict[str, Any], top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Augment retrieval results with reranking.

        Args:
            query: User query
            retrieval_result: Output from RetrievalAgent
            top_k: Number of results to keep

        Returns:
            Augmented results with reranked context
        """
        results = retrieval_result.get("results", [])

        if not results:
            logger.warning("No results to augment")
            return retrieval_result

        reranked = self.rerank(query, results, top_k=top_k)

        new_context = "\n\n".join(
            [f"[Source {i+1}] {r['text']}" for i, r in enumerate(reranked)]
        )

        return {
            **retrieval_result,
            "results": reranked,
            "context": new_context,
            "num_results": len(reranked),
            "augmented": True,
            "rerank_method": reranked[0]["rerank_method"] if reranked else "none",
        }

    def run(
        self, query: str, retrieval_result: Dict[str, Any], top_k: Optional[int] = 5
    ) -> Dict[str, Any]:
        """
        Execute complete augmentation pipeline.

        Args:
            query: User query
            retrieval_result: Output from RetrievalAgent
            top_k: Number of top results to keep

        Returns:
            Augmented retrieval result
        """
        logger.info("=" * 60)
        logger.info("AUGMENTATION AGENT: Reranking context")
        logger.info("=" * 60)

        original_count = retrieval_result.get("num_results", 0)
        logger.info(f"Original results: {original_count}")

        augmented = self.augment_context(query, retrieval_result, top_k=top_k)

        new_count = augmented.get("num_results", 0)
        logger.info(f"After reranking: {new_count}")

        if new_count > 0:
            logger.info(
                f"Top reranked score: {augmented['results'][0]['rerank_score']:.3f}"
            )

        logger.info("=" * 60)
        logger.info("AUGMENTATION COMPLETE")
        logger.info("=" * 60)

        return augmented


