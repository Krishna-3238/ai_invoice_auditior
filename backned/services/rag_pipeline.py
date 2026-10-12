"""RAG Pipeline — orchestrates all 5 RAG agents in sequence with conversation memory.

Flow:  IndexingAgent → RetrievalAgent → AugmentationAgent → GenerationAgent → ReflectionAgent

Memory is maintained in-process as a simple list of {role, content} dicts.
The pipeline can be used as a singleton (with `get_rag_pipeline()`) so the
index is only loaded once per Streamlit session.

Observability
-------------
Every ``query()`` call emits a LangFuse **trace** named ``rag-pipeline-query``
with child spans for each of the 5 RAG stages and automatic generation-level
tracing for every Azure OpenAI chat completion.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from backned.agents.rag_agents.indexing_agent    import IndexingAgent
from backned.agents.rag_agents.retrieval_agent   import RetrievalAgent
from backned.agents.rag_agents.augmentation_agent import AugmentationAgent
from backned.agents.rag_agents.generation_agent  import GenerationAgent
from backned.agents.rag_agents.reflection_agent  import ReflectionAgent

logger = logging.getLogger(__name__)

#  Memory 

@dataclass
class ConversationMemory:
    """Simple sliding-window conversation memory."""
    history: list[dict[str, str]] = field(default_factory=list)
    max_turns: int = 20   # keep last 20 user+assistant turns (10 exchanges)

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        # Trim to max_turns
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_history(self) -> list[dict[str, str]]:
        return list(self.history)

    def clear(self) -> None:
        self.history.clear()


#  Pipeline result 

@dataclass
class RAGResult:
    query:          str
    answer:         str
    sources_used:   list[str]
    context_chunks: int
    reflection:     dict[str, Any]
    retry_count:    int
    latency_ms:     float
    index_status:   str


#  Pipeline 

class RAGPipeline:
    """Full 5-agent RAG pipeline with conversation memory."""

    def __init__(
        self,
        retrieval_top_k: int = 10,
        rerank_top_n:    int = 5,
    ) -> None:
        self._indexing_agent    = IndexingAgent()
        self._retrieval_agent   = RetrievalAgent()
        self._augmentation_agent = AugmentationAgent()
        self._generation_agent  = GenerationAgent()
        self._reflection_agent  = ReflectionAgent()

        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_n    = rerank_top_n

        self.memory           = ConversationMemory()
        self._index_built     = False
        self._index_result:   dict = {}

    #  Index management 

    def ensure_index(self, force_reindex: bool = False) -> dict[str, Any]:
        """Build or refresh the Qdrant index. Idempotent unless forced."""
        if self._index_built and not force_reindex:
            return self._index_result

        logger.info("RAG Pipeline: running Indexing Agent ...")
        self._index_result = self._indexing_agent.run(force_reindex=force_reindex)
        self._index_built  = True
        return self._index_result

    #  Query 

    def query(self, user_query: str, invoice_id: str | None = None) -> RAGResult:
        """Process a user query through all 5 RAG agents.

        Automatically builds the index if it has not been built yet.
        """
        t0 = time.perf_counter()

        # 1. Indexing (no-op if up-to-date)
        idx_result = self.ensure_index()

        # 2. Retrieval
        logger.info("[RAG] Step 2/5 - Retrieval")
        retrieval_result = self._retrieval_agent.run(
            query=user_query, top_k=self._retrieval_top_k, invoice_id=invoice_id
        )
        raw_chunks = retrieval_result.get("chunks", [])

        # 3. Augmentation (re-ranking)
        logger.info("[RAG] Step 3/5 - Augmentation")
        augment_result = self._augmentation_agent.run(
            query=user_query, chunks=raw_chunks, top_n=self._rerank_top_n,
        )
        ranked_chunks = augment_result.get("ranked_chunks", raw_chunks[:self._rerank_top_n])

        # 4. Generation
        logger.info("[RAG] Step 4/5 - Generation")
        gen_result = self._generation_agent.run(
            query=user_query,
            ranked_chunks=ranked_chunks,
            chat_history=self.memory.get_history(),
        )
        raw_answer   = gen_result.get("answer", "")
        sources_used = gen_result.get("sources_used", [])

        # 5. Reflection
        logger.info("[RAG] Step 5/5 - Reflection")
        ref_result = self._reflection_agent.run(
            query=user_query,
            answer=raw_answer,
            ranked_chunks=ranked_chunks,
            chat_history=self.memory.get_history(),
            sources_used=sources_used,
            invoice_id=invoice_id,
        )
        final_answer = ref_result["final_answer"]

        # Update memory
        self.memory.add("user",      user_query)
        self.memory.add("assistant", final_answer)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("DONE [RAG] Query complete in %.0f ms", latency_ms)

        return RAGResult(
            query          = user_query,
            answer         = final_answer,
            sources_used   = sources_used,
            context_chunks = gen_result.get("context_chunks", 0),
            reflection     = ref_result["reflection"],
            retry_count    = ref_result["retry_count"],
            latency_ms     = latency_ms,
            index_status   = idx_result.get("status", "unknown"),
        )

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory.clear()
        logger.info("RAG Pipeline: conversation memory cleared")


#  Singleton accessor 

_pipeline_instance: RAGPipeline | None = None

def get_rag_pipeline() -> RAGPipeline:
    """Return the singleton RAGPipeline (creates it on first call)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
