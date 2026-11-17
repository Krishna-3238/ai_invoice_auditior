"""
RAG Generation Agent
====================
Persona: Answer Generator
Role: Generate accurate answers based on retrieved context

This agent:
1. Receives user query and retrieved context
2. Uses LLM to generate accurate answers
3. Cites sources from retrieved documents
4. Handles cases with insufficient context
5. Provides confidence scores

The agent uses AWS Bedrock Cohere Command R+ for generation.
Prompts are loaded dynamically from config/persona_invoice_agent.yaml.
"""

import logging
import os
import sys
from typing import Dict, Any, List, Optional
from litellm import completion

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.persona_loader import AgentPersona, get_prompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GenerationAgent")


class GenerationAgent:
    """
    Generation Agent that creates answers using LLM and retrieved context.
    Persona: Answer Generator

    Implements RAG (Retrieval-Augmented Generation) pattern.
    Prompts are loaded dynamically from persona configuration.
    """

    def __init__(self, model: str = "bedrock/cohere.command-r-plus-v1:0"):
        """
        Initialize the Generation Agent.

        Args:
            model: LiteLLM model string (e.g., 'bedrock/cohere.command-r-plus-v1:0')
                  Set to None to use template mode without LLM
        """
        self.model = model

        self.persona = AgentPersona("generation_agent")

        if not model:
            logger.warning(
                "⚠️  No model provided. Agent will return template responses."
            )
        else:
            logger.info(f"Generation Agent initialized with model: {model}")

    def generate_answer(
        self, query: str, context: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer based on query and retrieved context.
        Loads prompt from persona configuration.

        Args:
            query: User question
            context: Retrieved context from documents
            sources: List of source documents with metadata

        Returns:
            Generated answer with metadata
        """
        if not self.model:
            return self._generate_template_answer(query, context, sources)

        try:
            source_info = self._format_sources(sources)

            prompt = self.persona.get_prompt(
                "generate_answer", context=context, source_info=source_info, query=query
            )

            if not prompt:
                prompt = f"""You are an AI assistant helping users query invoice information. 
Answer the user's question based ONLY on the provided context from invoice documents.

Context from Invoice Documents:
{context}

Source Documents:
{source_info}

User Question: {query}

Instructions:
1. Answer the question accurately based ONLY on the provided context
2. If the context contains the answer, provide a clear and concise response
3. Cite specific invoice numbers and documents when possible
4. If the context does NOT contain enough information to answer, say so explicitly
5. Do not make up information or use knowledge outside the provided context
6. Use a professional and helpful tone

Answer:"""

            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )

            answer_text = response.choices[0].message.content.strip()

            has_answer = not any(
                phrase in answer_text.lower()
                for phrase in [
                    "not enough information",
                    "cannot answer",
                    "not found in the context",
                    "insufficient information",
                ]
            )

            logger.info("Generated answer using LiteLLM")

            return {
                "answer": answer_text,
                "has_answer": has_answer,
                "sources_used": sources,
                "confidence": "high" if has_answer else "low",
                "method": "llm",
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}. Using template response.")
            return self._generate_template_answer(query, context, sources)

    def _generate_template_answer(
        self, query: str, context: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate simple template-based answer (fallback)"""
        if not sources:
            answer = "I couldn't find any relevant information in the indexed invoices to answer your question."
            has_answer = False
        else:
            invoice_nos = list(
                set(
                    s["metadata"].get("invoice_no")
                    for s in sources
                    if s["metadata"].get("invoice_no")
                )
            )

            answer = f"Based on the available information from invoice(s) {', '.join(invoice_nos)}, "
            answer += "please review the retrieved context for relevant details. "
            answer += f"I found {len(sources)} relevant document sections."
            has_answer = True

        return {
            "answer": answer,
            "has_answer": has_answer,
            "sources_used": sources,
            "confidence": "medium" if has_answer else "low",
            "method": "template",
        }

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format source metadata for prompt"""
        formatted = []
        for i, source in enumerate(sources, 1):
            metadata = source["metadata"]
            formatted.append(
                f"[Source {i}] Invoice: {metadata.get('invoice_no', 'N/A')}, "
                f"Vendor: {metadata.get('vendor_id', 'N/A')}, "
                f"PO: {metadata.get('po_number', 'N/A')}"
            )
        return "\n".join(formatted)

    def generate_with_followup(
        self, query: str, context: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer with suggested follow-up questions.
        Loads prompt from persona configuration.

        Args:
            query: User question
            context: Retrieved context
            sources: Source documents

        Returns:
            Answer with follow-up suggestions
        """
        result = self.generate_answer(query, context, sources)

        if self.model and result["has_answer"]:
            try:
                followup_prompt = self.persona.get_prompt(
                    "generate_followups", query=query, answer=result["answer"]
                )

                if not followup_prompt:
                    followup_prompt = f"""Based on this query and answer about invoices, suggest 3 relevant follow-up questions a user might want to ask.

Original Query: {query}
Answer: {result['answer']}

Provide 3 short, specific follow-up questions (one per line, no numbering):"""

                response = completion(
                    model=self.model,
                    messages=[{"role": "user", "content": followup_prompt}],
                    temperature=0.7,
                    max_tokens=150,
                )

                followup_text = response.choices[0].message.content.strip()

                followups = [
                    line.strip().strip("123.-) ")
                    for line in followup_text.split("\n")
                    if line.strip()
                ]
                result["suggested_followups"] = followups[:3]

            except Exception as e:
                logger.warning(f"Could not generate follow-ups: {e}")
                result["suggested_followups"] = []
        else:
            result["suggested_followups"] = []

        return result

    def run(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        include_followups: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute generation pipeline.

        Args:
            query: User query
            retrieval_result: Output from RetrievalAgent
            include_followups: Whether to generate follow-up questions

        Returns:
            Complete generation result
        """
        logger.info("=" * 60)
        logger.info("GENERATION AGENT: Generating answer")
        logger.info("=" * 60)

        context = retrieval_result.get("context", "")
        sources = retrieval_result.get("results", [])

        if not sources:
            logger.warning("⚠️  No context provided for generation")
            return {
                "query": query,
                "answer": "I couldn't find any relevant information in the indexed documents to answer your question.",
                "has_answer": False,
                "sources_used": [],
                "confidence": "none",
                "success": False,
            }

        if include_followups:
            result = self.generate_with_followup(query, context, sources)
        else:
            result = self.generate_answer(query, context, sources)

        result["query"] = query
        result["success"] = True

        logger.info("=" * 60)
        logger.info(f"GENERATION COMPLETE: {result['confidence']} confidence")
        logger.info("=" * 60)

        return result


