"""
RAG Reflection Agent
====================
Persona: Quality Evaluator
Role: Evaluate RAG output quality using RAG Triad metrics

This agent evaluates the RAG pipeline using three key metrics:
1. Context Relevance: How relevant is the retrieved context to the query?
2. Groundedness: Is the answer grounded in the retrieved context?
3. Answer Relevance: How relevant is the answer to the original query?

These metrics help ensure high-quality, trustworthy answers.
Prompts are loaded dynamically from config/persona_invoice_agent.yaml.
"""

import logging
import re
import os
import sys
from typing import Dict, Any, List, Optional
from litellm import completion

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.persona_loader import AgentPersona, get_prompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ReflectionAgent")


class ReflectionAgent:
    """
    Reflection Agent that evaluates RAG quality using the RAG Triad.
    Persona: Quality Evaluator

    RAG Triad Metrics:
    - Context Relevance: Retrieved context relevance to query (0.0 - 1.0)
    - Groundedness: Answer groundedness in context (0.0 - 1.0)
    - Answer Relevance: Answer relevance to query (0.0 - 1.0)

    Prompts are loaded dynamically from persona configuration.
    """

    def __init__(self, model: str = "bedrock/cohere.command-r-plus-v1:0"):
        """
        Initialize Reflection Agent.

        Args:
            model: LiteLLM model string for evaluation (set to None for heuristics)
        """
        self.model = model

        self.persona = AgentPersona("reflection_agent")

        if not model:
            logger.warning("⚠️  No model provided. Using heuristic-based evaluation.")
        else:
            logger.info(f"Reflection Agent initialized with model: {model}")

    def evaluate_context_relevance(
        self, query: str, context: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate how relevant the retrieved context is to the query.

        Args:
            query: User query
            context: Retrieved context
            sources: Source documents

        Returns:
            Context relevance score and explanation
        """
        if not context or not sources:
            return {
                "score": 0.0,
                "explanation": "No context was retrieved for this query.",
                "method": "heuristic",
            }

        if self.model:
            try:
                return self._evaluate_context_relevance_llm(query, context)
            except Exception as e:
                logger.warning(f"LLM evaluation failed: {e}. Using heuristic.")

        score = self._heuristic_context_relevance(query, context, sources)

        if score >= 0.7:
            explanation = "Retrieved context appears highly relevant to the query."
        elif score >= 0.4:
            explanation = "Retrieved context has moderate relevance to the query."
        else:
            explanation = "Retrieved context may not be very relevant to the query."

        return {"score": score, "explanation": explanation, "method": "heuristic"}

    def evaluate_groundedness(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate if the answer is grounded in the retrieved context.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Groundedness score and explanation
        """
        if not answer:
            return {
                "score": 0.0,
                "explanation": "No answer was generated.",
                "method": "heuristic",
            }

        if self.model:
            try:
                return self._evaluate_groundedness_llm(answer, context)
            except Exception as e:
                logger.warning(f"LLM evaluation failed: {e}. Using heuristic.")

        score = self._heuristic_groundedness(answer, context)

        if score >= 0.7:
            explanation = "Answer appears well-grounded in the provided context."
        elif score >= 0.4:
            explanation = "Answer is partially grounded in the context."
        else:
            explanation = "Answer may contain information not found in the context."

        return {"score": score, "explanation": explanation, "method": "heuristic"}

    def evaluate_answer_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Evaluate if the answer is relevant to the query.

        Args:
            query: User query
            answer: Generated answer

        Returns:
            Answer relevance score and explanation
        """
        if not answer:
            return {
                "score": 0.0,
                "explanation": "No answer was generated.",
                "method": "heuristic",
            }

        if self.model:
            try:
                return self._evaluate_answer_relevance_llm(query, answer)
            except Exception as e:
                logger.warning(f"LLM evaluation failed: {e}. Using heuristic.")

        score = self._heuristic_answer_relevance(query, answer)

        if score >= 0.7:
            explanation = "Answer directly addresses the query."
        elif score >= 0.4:
            explanation = "Answer partially addresses the query."
        else:
            explanation = "Answer may not fully address the query."

        return {"score": score, "explanation": explanation, "method": "heuristic"}

    def _heuristic_context_relevance(
        self, query: str, context: str, sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate context relevance using keyword overlap"""
        query_lower = query.lower()
        context_lower = context.lower()

        stop_words = {"what", "is", "the", "for", "in", "on", "at", "to", "a", "an"}
        query_words = set(re.findall(r"\w+", query_lower)) - stop_words

        if not query_words:
            return 0.5                                  

        matches = sum(1 for word in query_words if word in context_lower)
        overlap_score = matches / len(query_words)

        if sources:
            avg_similarity = sum(s.get("similarity_score", 0.5) for s in sources) / len(
                sources
            )
            combined_score = (overlap_score * 0.6) + (avg_similarity * 0.4)
            return min(1.0, combined_score)

        return overlap_score

    def _heuristic_groundedness(self, answer: str, context: str) -> float:
        """Calculate groundedness using fact extraction"""
        answer_lower = answer.lower()
        context_lower = context.lower()

        no_answer_phrases = [
            "not enough information",
            "cannot answer",
            "not found",
            "insufficient information",
            "don't have",
            "unable to",
        ]

        if any(phrase in answer_lower for phrase in no_answer_phrases):
            return 0.9                                                     

        answer_facts = set(
            re.findall(r"\b[A-Z][A-Za-z0-9-]+\b", answer)
        )                
        answer_facts.update(re.findall(r"\b\d+\.?\d*\b", answer))           

        if not answer_facts:
            return 0.6                                   

        grounded_facts = sum(
            1 for fact in answer_facts if fact.lower() in context_lower
        )

        return grounded_facts / len(answer_facts) if answer_facts else 0.6

    def _heuristic_answer_relevance(self, query: str, answer: str) -> float:
        """Calculate answer relevance using keyword matching"""
        query_lower = query.lower()
        answer_lower = answer.lower()

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
        }
        query_words = set(re.findall(r"\w+", query_lower)) - stop_words

        if not query_words:
            return 0.5

        matches = sum(1 for word in query_words if word in answer_lower)
        overlap_score = matches / len(query_words)

        no_answer_phrases = ["cannot answer", "not found", "don't have"]
        if any(phrase in answer_lower for phrase in no_answer_phrases):
            return overlap_score * 0.5

        return overlap_score

    def _evaluate_context_relevance_llm(
        self, query: str, context: str
    ) -> Dict[str, Any]:
        """Evaluate context relevance using LLM. Loads prompt from persona config."""
        prompt = self.persona.get_prompt(
            "evaluate_context_relevance", query=query, context=context[:2000]
        )

        if not prompt:
            prompt = f"""Evaluate how relevant the following context is to answering the query.

Query: {query}

Context:
{context[:2000]}

Rate the context relevance on a scale of 0.0 to 1.0:
- 1.0: Context directly contains information to answer the query
- 0.5: Context is somewhat relevant but may not fully answer the query
- 0.0: Context is not relevant to the query

Provide your evaluation in this format:
Score: [0.0-1.0]
Explanation: [Brief explanation]"""

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )

        text = response.choices[0].message.content

        score_match = re.search(r"Score:\s*([0-9.]+)", text)
        score = float(score_match.group(1)) if score_match else 0.5
        score = max(0.0, min(1.0, score))

        exp_match = re.search(r"Explanation:\s*(.+)", text, re.DOTALL)
        explanation = (
            exp_match.group(1).strip() if exp_match else "LLM evaluation completed."
        )

        return {"score": score, "explanation": explanation[:500], "method": "llm"}

    def _evaluate_groundedness_llm(self, answer: str, context: str) -> Dict[str, Any]:
        """Evaluate groundedness using LLM. Loads prompt from persona config."""
        prompt = self.persona.get_prompt(
            "evaluate_groundedness", context=context[:2000], answer=answer
        )

        if not prompt:
            prompt = f"""Evaluate if the answer is grounded in (supported by) the provided context.

Context:
{context[:2000]}

Answer:
{answer}

Rate the groundedness on a scale of 0.0 to 1.0:
- 1.0: All claims in the answer are directly supported by the context
- 0.5: Some claims are supported, but some may not be
- 0.0: Answer contains information not found in the context

Provide your evaluation in this format:
Score: [0.0-1.0]
Explanation: [Brief explanation]"""

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )

        text = response.choices[0].message.content

        score_match = re.search(r"Score:\s*([0-9.]+)", text)
        score = float(score_match.group(1)) if score_match else 0.5
        score = max(0.0, min(1.0, score))

        exp_match = re.search(r"Explanation:\s*(.+)", text, re.DOTALL)
        explanation = (
            exp_match.group(1).strip() if exp_match else "LLM evaluation completed."
        )

        return {"score": score, "explanation": explanation[:500], "method": "llm"}

    def _evaluate_answer_relevance_llm(self, query: str, answer: str) -> Dict[str, Any]:
        """Evaluate answer relevance using LLM. Loads prompt from persona config."""
        prompt = self.persona.get_prompt(
            "evaluate_answer_relevance", query=query, answer=answer
        )

        if not prompt:
            prompt = f"""Evaluate if the answer is relevant to and addresses the query.

Query: {query}

Answer:
{answer}

Rate the answer relevance on a scale of 0.0 to 1.0:
- 1.0: Answer directly and completely addresses the query
- 0.5: Answer partially addresses the query
- 0.0: Answer does not address the query

Provide your evaluation in this format:
Score: [0.0-1.0]
Explanation: [Brief explanation]"""

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )

        text = response.choices[0].message.content

        score_match = re.search(r"Score:\s*([0-9.]+)", text)
        score = float(score_match.group(1)) if score_match else 0.5
        score = max(0.0, min(1.0, score))

        exp_match = re.search(r"Explanation:\s*(.+)", text, re.DOTALL)
        explanation = (
            exp_match.group(1).strip() if exp_match else "LLM evaluation completed."
        )

        return {"score": score, "explanation": explanation[:500], "method": "llm"}

    def run(
        self, query: str, context: str, answer: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute complete RAG Triad evaluation.

        Args:
            query: Original user query
            context: Retrieved context
            answer: Generated answer
            sources: Source documents

        Returns:
            Complete evaluation with all three metrics
        """
        logger.info("=" * 60)
        logger.info("REFLECTION AGENT: Evaluating RAG quality")
        logger.info("=" * 60)

        context_relevance = self.evaluate_context_relevance(query, context, sources)
        groundedness = self.evaluate_groundedness(answer, context)
        answer_relevance = self.evaluate_answer_relevance(query, answer)

        overall_score = (
            context_relevance["score"]
            + groundedness["score"]
            + answer_relevance["score"]
        ) / 3.0

        if overall_score >= 0.8:
            quality = "Excellent"
        elif overall_score >= 0.6:
            quality = "Good"
        elif overall_score >= 0.4:
            quality = "Fair"
        else:
            quality = "Poor"

        result = {
            "query": query,
            "metrics": {
                "context_relevance": context_relevance,
                "groundedness": groundedness,
                "answer_relevance": answer_relevance,
            },
            "overall_score": round(overall_score, 3),
            "quality": quality,
            "evaluation_method": context_relevance["method"],
        }

        logger.info(f"Context Relevance: {context_relevance['score']:.3f}")
        logger.info(f"Groundedness: {groundedness['score']:.3f}")
        logger.info(f"Answer Relevance: {answer_relevance['score']:.3f}")
        logger.info(f"Overall Score: {overall_score:.3f} ({quality})")
        logger.info("=" * 60)

        return result


