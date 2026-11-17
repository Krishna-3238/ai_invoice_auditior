"""
MCP Tool Registration for LangGraph Agents
============================================
Registers the LangGraph-based agents' tools with MCP:
- Invoice-Watcher Tool (Monitor Agent)
- DataCompletenessChecker Tool (Data Validation Agent)
- Business Validation Tool (Business Validation Agent)

Also registers LangGraph agents with the A2A Protocol.

And registers enhanced RAG tools:
- Vector-Indexer Tool
- Semantic-Retriever Tool
- Chunk-Ranker Tool
- Response-Synthesizer Tool
- RAG-Evaluator Tool
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protocols.a2a_protocol import (
    AgentCard,
    AgentFramework,
    AgentStatus,
    get_registry,
)
from protocols.mcp_protocol import MCPToolDescriptor, ToolCategory, get_tool_registry

logger = logging.getLogger("ToolRegistration")


def register_langgraph_agents():
    """Register all LangGraph agents with A2A protocol."""
    registry = get_registry()

    registry.register(
        AgentCard(
            agent_id="lg_monitor_agent",
            name="Monitor Agent (LangGraph)",
            description="Watches for new invoices arriving in the incoming directory",
            framework=AgentFramework.LANGGRAPH,
            capabilities=["file_monitoring", "workflow_triggering"],
            tools=["invoice_watcher_tool"],
        )
    )

    registry.register(
        AgentCard(
            agent_id="lg_data_validation_agent",
            name="Data Validation Agent (LangGraph)",
            description="Validates invoice data completeness, types, and internal consistency",
            framework=AgentFramework.LANGGRAPH,
            capabilities=["field_extraction", "data_validation", "calculation_check"],
            tools=["data_completeness_checker_tool"],
        )
    )

    registry.register(
        AgentCard(
            agent_id="lg_business_validation_agent",
            name="Business Validation Agent (LangGraph)",
            description="Validates invoice data against ERP records (PO, Vendor, SKU)",
            framework=AgentFramework.LANGGRAPH,
            capabilities=[
                "erp_validation",
                "vendor_check",
                "po_matching",
                "price_check",
            ],
            tools=["business_validation_tool"],
        )
    )

    registry.register(
        AgentCard(
            agent_id="lg_indexing_agent",
            name="RAG Indexing Agent (LangGraph)",
            description="Indexes invoice data and reports into vector database",
            framework=AgentFramework.LANGGRAPH,
            capabilities=["document_chunking", "embedding", "vector_indexing"],
            tools=["vector_indexer_tool"],
        )
    )

    registry.register(
        AgentCard(
            agent_id="lg_retrieval_agent",
            name="RAG Retrieval Agent (LangGraph)",
            description="Retrieves relevant document chunks via semantic search",
            framework=AgentFramework.LANGGRAPH,
            capabilities=["semantic_search", "metadata_filtering"],
            tools=["semantic_retriever_tool"],
        )
    )

    registry.register(
        AgentCard(
            agent_id="lg_augmentation_agent",
            name="RAG Augmentation Agent (LangGraph)",
            description="Reranks retrieved context for optimal relevance",
            framework=AgentFramework.LANGGRAPH,
            capabilities=["reranking", "entity_boosting", "context_optimization"],
            tools=["chunk_ranker_tool"],
        )
    )

    registry.register(
        AgentCard(
            agent_id="lg_generation_agent",
            name="RAG Generation Agent (LangGraph)",
            description="Generates answers from retrieved context using LLM",
            framework=AgentFramework.LANGGRAPH,
            capabilities=["answer_generation", "source_citation"],
            tools=["response_synthesizer_tool"],
        )
    )

    registry.register(
        AgentCard(
            agent_id="lg_reflection_agent",
            name="RAG Reflection Agent (LangGraph)",
            description="Evaluates RAG output quality using RAG Triad metrics",
            framework=AgentFramework.LANGGRAPH,
            capabilities=["context_relevance", "groundedness", "answer_relevance"],
            tools=["rag_evaluator_tool"],
        )
    )

    logger.info(f"Registered {len(registry.list_all())} agents with A2A protocol")


def register_mcp_tools():
    """Register all MCP tools for LangGraph-based agents."""
    tool_registry = get_tool_registry()

    tool_registry.register_tool(
        MCPToolDescriptor(
            tool_id="invoice_watcher_tool",
            name="Invoice-Watcher Tool",
            description=(
                "Monitors the incoming directory for new invoice files. "
                "Detects PDF, DOCX, PNG, and JPG files and triggers the "
                "invoice auditing workflow. Uses watchdog for efficient "
                "file system event handling."
            ),
            category=ToolCategory.MONITORING,
            input_schema={
                "type": "object",
                "properties": {
                    "watch_directory": {"type": "string", "default": "./data/incoming"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "file_metadata": {"type": "object"},
                },
            },
        ),
        handler=_invoice_watcher_handler,
    )

    tool_registry.register_tool(
        MCPToolDescriptor(
            tool_id="data_completeness_checker_tool",
            name="DataCompletenessChecker Tool",
            description=(
                "Validates invoice data for completeness and internal consistency. "
                "Checks required fields, data types, currency validity, and "
                "calculation accuracy. Uses LLM for intelligent field extraction."
            ),
            category=ToolCategory.VALIDATION,
            input_schema={
                "type": "object",
                "properties": {
                    "translated_text": {"type": "string"},
                    "tables": {"type": "array"},
                    "metadata": {"type": "object"},
                },
                "required": ["translated_text"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "invoice_data": {"type": "object"},
                    "line_items": {"type": "array"},
                    "validation_issues": {"type": "array"},
                    "is_valid": {"type": "boolean"},
                },
            },
        ),
        handler=_data_completeness_handler,
    )

    tool_registry.register_tool(
        MCPToolDescriptor(
            tool_id="business_validation_tool",
            name="Business Validation Tool",
            description=(
                "Validates invoice data against ERP records including "
                "vendor verification, PO matching, price/quantity checks, "
                "and tolerance enforcement. Connects to mock ERP API."
            ),
            category=ToolCategory.VALIDATION,
            input_schema={
                "type": "object",
                "properties": {
                    "invoice_data": {"type": "object"},
                    "line_items": {"type": "array"},
                    "validation_issues": {"type": "array"},
                },
                "required": ["invoice_data"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "validation_issues": {"type": "array"},
                    "recommendation": {"type": "string"},
                    "is_valid": {"type": "boolean"},
                },
            },
        ),
        handler=_business_validation_handler,
    )

    tool_registry.register_tool(
        MCPToolDescriptor(
            tool_id="vector_indexer_tool",
            name="Vector-Indexer Tool",
            description=(
                "Chunks invoice documents and reports, generates embeddings "
                "using sentence transformers, and stores them in FAISS vector "
                "database for efficient retrieval."
            ),
            category=ToolCategory.RAG,
            input_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "object"},
                    "data_type": {"type": "string", "enum": ["invoice", "report"]},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "chunks_added": {"type": "integer"},
                    "stats": {"type": "object"},
                },
            },
        ),
        handler=_vector_indexer_handler,
    )

    tool_registry.register_tool(
        MCPToolDescriptor(
            tool_id="semantic_retriever_tool",
            name="Semantic-Retriever Tool",
            description=(
                "Performs semantic similarity search over indexed invoice documents. "
                "Converts queries to embeddings and retrieves top-k relevant chunks "
                "from FAISS vector database with similarity scoring."
            ),
            category=ToolCategory.RAG,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "filter_metadata": {"type": "object"},
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "num_results": {"type": "integer"},
                },
            },
        ),
        handler=_semantic_retriever_handler,
    )

    tool_registry.register_tool(
        MCPToolDescriptor(
            tool_id="chunk_ranker_tool",
            name="Chunk-Ranker Tool",
            description=(
                "Reranks retrieved document chunks for optimal relevance using "
                "keyword matching, entity boosting, and metadata matching. "
                "Filters low-quality results below threshold."
            ),
            category=ToolCategory.RAG,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "results": {"type": "array"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query", "results"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "context": {"type": "string"},
                },
            },
        ),
        handler=_chunk_ranker_handler,
    )

    tool_registry.register_tool(
        MCPToolDescriptor(
            tool_id="response_synthesizer_tool",
            name="Response-Synthesizer Tool",
            description=(
                "Generates accurate, grounded answers from retrieved context "
                "using LLM. Cites sources, provides confidence scores, and "
                "handles cases with insufficient context."
            ),
            category=ToolCategory.RAG,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "context": {"type": "string"},
                    "sources": {"type": "array"},
                },
                "required": ["query", "context"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "has_answer": {"type": "boolean"},
                    "confidence": {"type": "string"},
                },
            },
        ),
        handler=_response_synthesizer_handler,
    )

    tool_registry.register_tool(
        MCPToolDescriptor(
            tool_id="rag_evaluator_tool",
            name="RAG-Evaluator Tool",
            description=(
                "Evaluates RAG output quality using the RAG Triad metrics: "
                "Answer Relevance, Context Relevance, and Groundedness. "
                "Provides overall quality scores and explanations."
            ),
            category=ToolCategory.RAG,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "context": {"type": "string"},
                    "answer": {"type": "string"},
                    "sources": {"type": "array"},
                },
                "required": ["query", "context", "answer"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "metrics": {"type": "object"},
                    "overall_score": {"type": "number"},
                    "quality": {"type": "string"},
                },
            },
        ),
        handler=_rag_evaluator_handler,
    )

    logger.info(f"Registered {len(tool_registry.list_tools())} MCP tools")




def _invoice_watcher_handler(input_data: Dict) -> Dict:
    """Handler for Invoice-Watcher Tool."""
    from pathlib import Path

    watch_dir = input_data.get("watch_directory", "./data/incoming")
    path = Path(watch_dir)
    if not path.exists():
        return {"file_path": "", "file_metadata": {"error": "Directory not found"}}

    files = list(path.glob("*"))
    invoice_files = [
        f
        for f in files
        if f.suffix.lower() in [".pdf", ".docx", ".png", ".jpg", ".jpeg"]
    ]
    return {
        "files_found": len(invoice_files),
        "file_list": [str(f) for f in invoice_files],
        "watch_directory": str(path),
    }


def _data_completeness_handler(input_data: Dict) -> Dict:
    """Handler for DataCompletenessChecker Tool."""
    from agents.langgraph_agents.validation_agent import InvoiceDataValidator

    validator = InvoiceDataValidator(model="bedrock/amazon.nova-lite-v1:0")
    return validator.run(input_data)


def _business_validation_handler(input_data: Dict) -> Dict:
    """Handler for Business Validation Tool."""
    from agents.langgraph_agents.validation_agent import BusinessValidator

    validator = BusinessValidator(
        model="bedrock/amazon.nova-lite-v1:0",
        erp_api_url="http://localhost:8000",
    )
    return validator.run(input_data)


def _vector_indexer_handler(input_data: Dict) -> Dict:
    """Handler for Vector-Indexer Tool."""
    from agents.rag_agents.indexing_agent import IndexingAgent

    indexer = IndexingAgent(index_path="./data/vector_db")
    data_type = input_data.get("data_type", "invoice")
    data = input_data.get("data", input_data)
    return indexer.run(data, data_type=data_type)


def _semantic_retriever_handler(input_data: Dict) -> Dict:
    """Handler for Semantic-Retriever Tool."""
    from agents.rag_agents.indexing_agent import IndexingAgent
    from agents.rag_agents.retrieval_agent import RetrievalAgent

    indexer = IndexingAgent(index_path="./data/vector_db")
    retriever = RetrievalAgent(indexing_agent=indexer, similarity_threshold=0.3)
    return retriever.run(
        input_data.get("query", ""),
        top_k=input_data.get("top_k", 5),
    )


def _chunk_ranker_handler(input_data: Dict) -> Dict:
    """Handler for Chunk-Ranker Tool."""
    from agents.rag_agents.augmentation_agent import AugmentationAgent

    augmenter = AugmentationAgent()
    return augmenter.run(
        input_data.get("query", ""),
        {"results": input_data.get("results", [])},
        top_k=input_data.get("top_k", 5),
    )


def _response_synthesizer_handler(input_data: Dict) -> Dict:
    """Handler for Response-Synthesizer Tool."""
    from agents.rag_agents.generation_agent import GenerationAgent

    generator = GenerationAgent()
    return generator.generate_answer(
        input_data.get("query", ""),
        input_data.get("context", ""),
        input_data.get("sources", []),
    )


def _rag_evaluator_handler(input_data: Dict) -> Dict:
    """Handler for RAG-Evaluator Tool."""
    from agents.rag_agents.reflection_agent import ReflectionAgent

    reflector = ReflectionAgent()
    return reflector.run(
        query=input_data.get("query", ""),
        context=input_data.get("context", ""),
        answer=input_data.get("answer", ""),
        sources=input_data.get("sources", []),
    )


from typing import Dict


def initialize_all_registrations():
    """Initialize all A2A and MCP registrations."""
    logger.info("=" * 60)
    logger.info("Initializing A2A + MCP Protocol Registrations")
    logger.info("=" * 60)

    register_langgraph_agents()
    register_mcp_tools()

    from agents.adk_agents.extractor_agent import ADKExtractorAgent
    from agents.adk_agents.translation_agent import ADKTranslationAgent
    from agents.adk_agents.reporting_agent import ADKReportingAgent

    adk_extractor = None if os.getenv("A2A_EXTRACTOR_URL") else ADKExtractorAgent()
    adk_translator = None if os.getenv("A2A_TRANSLATOR_URL") else ADKTranslationAgent()
    adk_reporter = None if os.getenv("A2A_REPORTER_URL") else ADKReportingAgent()

    a2a_registry = get_registry()
    mcp_registry = get_tool_registry()

    logger.info("=" * 60)
    logger.info("Registration Complete:")
    logger.info(f"  A2A Agents: {len(a2a_registry.list_all())}")
    for card in a2a_registry.list_all():
        logger.info(f"    - {card.name} ({card.framework.value})")
    logger.info(f"  MCP Tools: {len(mcp_registry.list_tools())}")
    for tool in mcp_registry.list_tools():
        logger.info(f"    - {tool.name} ({tool.category.value})")
    logger.info("=" * 60)

    return {
        "adk_extractor": adk_extractor,
        "adk_translator": adk_translator,
        "adk_reporter": adk_reporter,
    }
