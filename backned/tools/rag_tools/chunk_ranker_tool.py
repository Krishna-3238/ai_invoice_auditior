"""Chunk Ranker Tool — re-ranks retrieved chunks using keyword boosting,
source-type weighting, and marginal relevance to improve context quality.

Exposes `chunk_ranker_tool` as a LangChain @tool for the Augmentation Agent.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Source-type priority weights (higher = more relevant to invoice queries)
_SOURCE_WEIGHTS: dict[str, float] = {
    "summary": 1.40,   # synthetic aggregate summary — highest for cross-invoice queries
    "report":  1.20,   # individual audit HTML reports
    "erp":     1.10,   # ERP master data
    "invoice": 1.05,   # raw invoice text
}

# Query intent keywords → source-type boosts
_INTENT_BOOSTS: list[tuple[list[str], str, float]] = [
    # Aggregate / counting queries → pull the summary chunk first
    (["how many", "count", "total", "all invoices", "list invoices", "summary",
      "overview", "auto approved", "auto-approved", "auto approve",
      "manual review", "flagged", "rejected", "across invoices"],
     "summary", 0.30),
    # Discrepancy / error detail → individual report chunks
    (["discrepanc", "mismatch", "price mismatch", "why", "reason", "flagged",
      "manual review", "missing field", "error", "fail", "issue", "problem",
      "missing", "invoice_no", "currency missing", "data validation",
      "business validation", "erp cross", "decision rationale"],
     "report", 0.20),
    # Vendor / ERP lookups
    (["vendor id", "vendor ids", "all vendors", "list vendors", "show vendors",
      "vend-", "supplier", "company"],
     "erp", 0.25),
    (["sku", "item code", "all sku", "list sku", "all items"],
     "erp", 0.20),
    (["po number", "purchase order", "all po", "list po"],
     "erp", 0.20),
    (["confidence", "score", "approve", "manual", "review", "audit"],
     "report", 0.10),
]


@tool
def chunk_ranker_tool(query: str, chunks: list[dict[str, Any]], top_n: int = 8) -> dict:
    """Re-rank retrieved chunks by combining similarity score, source priority,
    query-intent keyword boosts, and maximum marginal relevance (MMR).

    Args:
        query:   Original user query.
        chunks:  List of dicts from semantic_retriever_tool
                 (each has text, source, score, chunk_id).
        top_n:   Final number of chunks to keep for context.

    Returns:
        dict with keys: ranked_chunks (list), query (str), status (str).
    """
    if not chunks:
        return {"query": query, "ranked_chunks": [], "status": "no_chunks"}

    q_lower = query.lower()

    #  Step 1: apply source weights + keyword boosts 
    scored: list[dict] = []
    for c in chunks:
        src_type  = c["source"].split(":")[0]
        base      = c["score"]
        weight    = _SOURCE_WEIGHTS.get(src_type, 1.0)

        intent_boost = 0.0
        for keywords, target_src, boost in _INTENT_BOOSTS:
            if src_type == target_src and any(kw in q_lower for kw in keywords):
                intent_boost += boost

        # Keyword density bonus: count query terms in chunk text
        q_terms   = re.findall(r"\w+", q_lower)
        text_lower = c["text"].lower()
        kw_hit    = sum(1 for t in q_terms if len(t) > 3 and t in text_lower)
        kw_bonus  = min(kw_hit * 0.02, 0.10)

        final_score = base * weight + intent_boost + kw_bonus
        scored.append({**c, "final_score": final_score, "src_type": src_type})

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    #  Step 2: MMR — penalise near-duplicate chunks 
    selected: list[dict] = []
    seen_texts: list[str] = []

    for chunk in scored:
        if len(selected) >= top_n:
            break
        chunk_text = chunk["text"].lower()

        # Compute max overlap with already-selected chunks
        max_overlap = max(
            (_token_overlap(chunk_text, s) for s in seen_texts),
            default=0.0,
        )
        if max_overlap > 0.85:
            # Skip near-duplicate (threshold raised so related chunks from the
            # same report — which share header text — are not wrongly deduplicated)
            continue

        selected.append(chunk)
        seen_texts.append(chunk_text)

    logger.info(
        "Ranker: %d candidates -> %d selected | top score=%.3f",
        len(chunks), len(selected),
        selected[0]["final_score"] if selected else 0,
    )

    return {
        "query":         query,
        "ranked_chunks": selected,
        "status":        "ok",
    }


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two lowercase strings."""
    ta = set(re.findall(r"\w+", a))
    tb = set(re.findall(r"\w+", b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
