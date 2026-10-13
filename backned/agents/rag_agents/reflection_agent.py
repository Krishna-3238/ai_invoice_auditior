"""Reflection Agent — evaluates the generated answer for quality and completeness,
flags hallucination risk, and decides whether to accept or request regeneration.

LangGraph flow:  reflect_node → (accept | retry) → END
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any, Literal

from langgraph.graph import StateGraph, END
from backned.llm import invoke_bedrock_chat

from backned.config.logging_setup import setup_logging
from backned.config.settings import BEDROCK_RAG_MODEL

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3

_REFLECTION_SYSTEM = """\
You are a quality-control evaluator for an AI invoice support agent.

Given a user question, the context used, and the generated answer, evaluate:
1. RELEVANCE  — Does the answer directly address the question? (0-1)
2. GROUNDEDNESS  — Is the answer supported by the provided context? (0-1)
               Flag if the answer contains facts NOT present in the context.
3. CONTEXT RELEVANCE — Is the retrieved context relevant to the question? (0-1)

Respond ONLY with valid JSON in exactly this format:
{
  "relevance": <number 0-1>,
  "groundedness": <number 0-1>,
  "context_relevance": <number 0-1>,
  "issues": ["<issue 1>", ...],
  "verdict": "accept" | "retry",
  "feedback": "<one-sentence improvement suggestion if verdict=retry, else empty>"
}

The application calculates the average of the three scores. Accept only when it is greater than 0.7.
"""

#  State 

class ReflectionState(TypedDict):
    query:         str
    invoice_id:    str | None
    ranked_chunks: list[dict[str, Any]]
    chat_history:  list[dict[str, str]]
    answer:        str
    sources_used:  list[str]
    reflection:    dict[str, Any]
    retry_count:   int
    final_answer:  str
    error:         str


#  Nodes 

def reflect_node(state: ReflectionState) -> ReflectionState:
    """Evaluate the generated answer using LLM-as-judge."""
    logger.info("Reflection Agent: evaluating answer quality ...")

    context_summary = "\n".join(
        f"[{c.get('source','?')}]: {c.get('text','')[:200]}"
        for c in state["ranked_chunks"][:4]
    )

    prompt = (
        f"Question: {state['query']}\n\n"
        f"Context used:\n{context_summary}\n\n"
        f"Generated answer:\n{state['answer']}"
    )

    try:
        content = invoke_bedrock_chat(
            model_id=BEDROCK_RAG_MODEL,
            messages=[
                {"role": "system", "content": _REFLECTION_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        import json
        content = content or "{}"
        content = content.strip().lstrip("```json").rstrip("```").strip()
        reflection = json.loads(content)
        scores = [float(reflection.get(key, 0)) for key in ("relevance", "groundedness", "context_relevance")]
        reflection["overall"] = sum(scores) / 3
        reflection["verdict"] = "accept" if reflection["overall"] > 0.7 else "retry"
    except Exception as exc:
        logger.warning("Reflection parse error: %s", exc)
        reflection = {"verdict": "retry", "overall": 0, "relevance": 0, "groundedness": 0, "context_relevance": 0, "issues": ["Could not score response"], "feedback": "Recheck the question against retrieved evidence."}

    logger.info(
        "Reflection scores: overall=%s, verdict=%s",
        reflection.get("overall", "?"), reflection.get("verdict", "?"),
    )
    return {**state, "reflection": reflection}


def accept_node(state: ReflectionState) -> ReflectionState:
    """Accept grounded answers or withhold responses below the score threshold."""
    accepted = state["reflection"].get("overall", 0) > 0.7
    answer = state["answer"] if accepted else "I could not verify a sufficiently relevant and grounded answer from the invoice evidence. Please review the invoice report."
    return {**state, "final_answer": answer}


def retry_node(state: ReflectionState) -> ReflectionState:
    """Retrieve fresh chunks and regenerate using the reflection feedback."""
    feedback = state["reflection"].get("feedback", "")
    logger.info("Reflection: retrying generation (feedback: %s)", feedback)

    from backned.agents.rag_agents.generation_agent import GenerationAgent  # noqa: PLC0415
    from backned.agents.rag_agents.retrieval_agent import RetrievalAgent
    from backned.agents.rag_agents.augmentation_agent import AugmentationAgent
    retrieved = RetrievalAgent().run(state["query"], invoice_id=state.get("invoice_id"))
    reranked = AugmentationAgent().run(state["query"], retrieved.get("chunks", []))
    ranked_chunks = reranked.get("ranked_chunks", retrieved.get("chunks", []))
    gen = GenerationAgent()

    # Append feedback as a system hint via chat_history
    augmented_history = list(state["chat_history"]) + [{
        "role": "system",
        "content": f"[Quality feedback]: {feedback}  Improve the answer accordingly.",
    }]

    result = gen.run(
        query=state["query"],
        ranked_chunks=ranked_chunks,
        chat_history=augmented_history,
    )
    new_answer = result.get("answer", state["answer"])
    return {
        **state,
        "ranked_chunks": ranked_chunks,
        "answer":      new_answer,
        "retry_count": state["retry_count"] + 1,
        "final_answer": new_answer,
    }


def _route(state: ReflectionState) -> Literal["accept", "retry"]:
    if state["reflection"].get("verdict") == "retry" and state["retry_count"] < _MAX_RETRIES:
        return "retry"
    return "accept"


#  Graph 

def build_reflection_graph() -> StateGraph:
    g = StateGraph(ReflectionState)
    g.add_node("reflect", reflect_node)
    g.add_node("accept",  accept_node)
    g.add_node("retry",   retry_node)

    g.set_entry_point("reflect")
    g.add_conditional_edges("reflect", _route, {"accept": "accept", "retry": "retry"})
    g.add_edge("accept", END)
    g.add_edge("retry", "reflect")

    return g.compile()


class ReflectionAgent:
    """Reflection Agent — evaluates and optionally re-generates answers."""

    def __init__(self) -> None:
        self._graph = build_reflection_graph()

    def run(
        self,
        query: str,
        answer: str,
        ranked_chunks: list[dict],
        invoice_id: str | None = None,
        chat_history: list[dict] | None = None,
        sources_used: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate the answer and return the final (possibly improved) answer.

        Returns dict with final_answer (str), reflection (dict), retry_count (int).
        """
        state = ReflectionState(
            query=query,
            invoice_id=invoice_id,
            ranked_chunks=ranked_chunks,
            chat_history=chat_history or [],
            answer=answer,
            sources_used=sources_used or [],
            reflection={},
            retry_count=0,
            final_answer="",
            error="",
        )
        final = self._graph.invoke(state)
        return {
            "final_answer": final["final_answer"] or answer,
            "reflection":   final["reflection"],
            "retry_count":  final["retry_count"],
        }
