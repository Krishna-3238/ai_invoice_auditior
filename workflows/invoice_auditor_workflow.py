"""
Invoice Auditor Workflow - LangGraph + Google ADK Orchestration
================================================================
This workflow orchestrates all agents in the AI Invoice Auditor system
using a dual-framework architecture:

LangGraph Agents (orchestration, validation, RAG):
1. Monitor Agent (Invoice-Watcher Tool)
2. Validation Agent - Data Completeness (DataCompletenessChecker Tool)
3. Validation Agent - Business (Business Validation Tool)
4. RAG Agents (Vector-Indexer, Semantic-Retriever, Chunk-Ranker,
               Response-Synthesizer, RAG-Evaluator)

Google ADK Agents (extraction, translation, reporting):
5. Extractor Agent (Data Harvester Tool)
6. Translation Agent (Lang-Bridge Tool)
7. Reporting Agent (Insight Reporter Tool)

Protocols & Infrastructure:
- A2A Protocol: Standardized inter-framework communication
- MCP Protocol: Dynamic tool discovery and invocation
- RAI Guardrails: Responsible AI safety layers
- LangFuse: Real-time observability and tracing
"""

from typing import Dict, Any, List, TypedDict
from pathlib import Path
import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.langgraph_agents.validation_agent import (
    InvoiceDataValidator,
    BusinessValidator,
)
from agents.rag_agents.indexing_agent import IndexingAgent

from protocols.a2a_protocol import get_registry, get_bridge, AgentFramework
from protocols.mcp_protocol import get_tool_registry
from protocols.tool_registration import initialize_all_registrations
from guardrails import get_guardrails
from observability import get_observability

from config.logging_config import get_logger
from integrations.storage import PostgresInvoiceStore, persist_workflow_result

logger = get_logger("InvoiceWorkflow")

_initialized = False
_adk_agents = {}


def _ensure_initialized():
    """Ensure all protocols and agents are registered."""
    global _initialized, _adk_agents
    if not _initialized:
        _adk_agents = initialize_all_registrations()
        _initialized = True
        logger.info("Dual-framework infrastructure initialized")




class InvoiceWorkflowState(TypedDict):
    """
    State that flows through the workflow.
    Each agent adds its output to this state.
    """

    invoice_path: str
    invoice_filename: str

    file_metadata: Dict[str, Any]

    extracted_text: str
    extracted_tables: List[List[List[str]]]
    extraction_metadata: Dict[str, Any]

    translated_text: str
    translated_tables: List[List[List[str]]]
    original_language: str
    translation_confidence: float
    translation_method: str

    invoice_data: Dict[str, Any]
    data_validation_issues: List[Dict[str, Any]]
    business_validation_issues: List[Dict[str, Any]]
    validation_summary: Dict[str, Any]

    report_path: str
    html_report_path: str
    recommendation: str

    indexed: bool

    needs_translation: bool
    validation_passed: bool
    error: str | None

    workflow_id: str
    started_at: str
    completed_at: str | None



@observe(name="monitor_agent")
def monitor_node(state: InvoiceWorkflowState) -> InvoiceWorkflowState:
    """
    Monitor Agent (LangGraph): Extract file metadata.
    Tool: Invoice-Watcher Tool (via MCP)
    """
    logger.info("=" * 80)
    logger.info("NODE: Monitor Agent [LangGraph] — Invoice-Watcher Tool")
    logger.info("=" * 80)

    _ensure_initialized()
    observability = get_observability()
    trace_id = state.get("workflow_id", "unknown")

    span_id = observability.start_span(
        trace_id,
        "monitor_agent",
        input_data={
            "file": state["invoice_filename"],
            "invoice_path": state["invoice_path"],
            "tool": "Invoice-Watcher Tool",
            "framework": "langgraph",
        },
        metadata={"node": "monitor", "agent_type": "langgraph"},
    )

    try:
        fs_span = observability.start_span(
            trace_id,
            "monitor_filesystem_check",
            input_data={"path": state["invoice_path"]},
        )

        file_path = Path(state["invoice_path"])
        file_exists = file_path.exists()
        file_size = file_path.stat().st_size if file_exists else 0
        file_ext = file_path.suffix

        observability.end_span(
            fs_span,
            output={
                "exists": file_exists,
                "size_bytes": file_size,
                "extension": file_ext,
            },
        )

        meta_span = observability.start_span(
            trace_id,
            "monitor_metadata_assembly",
            input_data={"filename": state["invoice_filename"], "extension": file_ext},
        )

        metadata = {
            "filename": state["invoice_filename"],
            "size_bytes": file_size,
            "extension": file_ext,
            "exists": file_exists,
            "tool": "Invoice-Watcher Tool",
            "framework": "langgraph",
        }

        state["file_metadata"] = metadata
        logger.info(f"Invoice-Watcher: File metadata extracted: {metadata}")

        observability.end_span(meta_span, output={"metadata": metadata})

        monitor_score = 1.0 if file_exists and file_size > 0 else 0.0
        observability.record_score(
            trace_id,
            "monitor_file_valid",
            monitor_score,
            comment=f"File exists={file_exists}, size={file_size}B, ext={file_ext}",
        )

        observability.end_span(
            span_id,
            output={
                "metadata": metadata,
                "file_valid": file_exists and file_size > 0,
            },
        )

    except Exception as e:
        logger.error(f"Monitor failed: {e}")
        state["error"] = str(e)
        observability.end_span(
            span_id,
            output={"error": str(e), "error_type": type(e).__name__},
            level="ERROR",
        )
        observability.record_score(
            trace_id, "monitor_file_valid", 0.0, comment=f"Error: {e}"
        )

    return state

@observe(name="extraction_agent")
def extractor_node(state: InvoiceWorkflowState) -> InvoiceWorkflowState:
    """
    Extractor Agent (Google ADK): Extract text and tables from invoice.
    Tool: Data Harvester Tool (via A2A + MCP)
    """
    logger.info("=" * 80)
    logger.info("NODE: Extractor Agent [Google ADK] — Data Harvester Tool")
    logger.info("=" * 80)

    _ensure_initialized()
    observability = get_observability()
    bridge = get_bridge()
    guardrails = get_guardrails()
    trace_id = state.get("workflow_id", "unknown")

    span_id = observability.start_span(
        trace_id,
        "extractor_agent_adk",
        input_data={
            "file": state["invoice_path"],
            "file_extension": state.get("file_metadata", {}).get("extension", ""),
            "file_size_bytes": state.get("file_metadata", {}).get("size_bytes", 0),
            "tool": "Data Harvester Tool",
            "framework": "google_adk",
        },
        metadata={"node": "extract", "agent_type": "google_adk", "protocol": "A2A"},
    )

    try:
        a2a_span = observability.start_span(
            trace_id,
            "extractor_a2a_invocation",
            input_data={
                "source_agent": "lg_monitor_agent",
                "target_agent": "adk_extractor_agent",
                "protocol": "A2A",
            },
        )

        result = bridge.invoke_agent(
            source_agent_id="lg_monitor_agent",
            target_agent_id="adk_extractor_agent",
            payload={
                "file_path": state["invoice_path"],
                "_trace_id": trace_id,
            },
        )

        observability.end_span(
            a2a_span,
            output={
                "a2a_success": True,
                "result_keys": list(result.keys()) if result else [],
            },
        )

        state["extracted_text"] = result.get("raw_text", "")
        state["extracted_tables"] = result.get("tables", [])
        state["extraction_metadata"] = result.get("metadata", {})

        rai_span = observability.start_span(
            trace_id,
            "extractor_rai_check",
            input_data={
                "text_preview": (
                    state["extracted_text"][:200] if state["extracted_text"] else ""
                ),
                "text_length": len(state["extracted_text"]),
            },
        )

        rai_check = guardrails.check_output(
            state["extracted_text"], agent_id="adk_extractor_agent"
        )

        rai_violations = []
        if rai_check.violations:
            logger.warning(f"RAI violations in extraction: {len(rai_check.violations)}")
            rai_violations = [
                {"type": getattr(v, "type", str(v)), "detail": str(v)}
                for v in rai_check.violations
            ]

        observability.end_span(
            rai_span,
            output={
                "violations_count": (
                    len(rai_check.violations) if rai_check.violations else 0
                ),
                "violations": rai_violations,
                "passed": not bool(rai_check.violations),
            },
            level="WARNING" if rai_check.violations else "DEFAULT",
        )

        logger.info(
            f"Data Harvester: Extracted {len(state['extracted_text'])} chars, "
            f"{len(state['extracted_tables'])} tables"
        )

        extraction_score = (
            1.0
            if len(state["extracted_text"]) > 50
            else (0.5 if len(state["extracted_text"]) > 0 else 0.0)
        )
        observability.record_score(
            trace_id,
            "extraction_quality",
            extraction_score,
            comment=f"Extracted {len(state['extracted_text'])} chars, {len(state['extracted_tables'])} tables",
        )

        observability.end_span(
            span_id,
            output={
                "text_len": len(state["extracted_text"]),
                "text_preview": (
                    state["extracted_text"][:300] if state["extracted_text"] else ""
                ),
                "tables_count": len(state["extracted_tables"]),
                "extraction_metadata": state["extraction_metadata"],
                "rai_violations": len(rai_violations),
            },
        )

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        state["error"] = str(e)
        observability.record_score(
            trace_id, "extraction_quality", 0.0, comment=f"Error: {e}"
        )
        observability.end_span(
            span_id,
            output={
                "error": str(e),
                "error_type": type(e).__name__,
            },
            level="ERROR",
        )

    return state

@observe(name="translation_agent")
def translation_node(state: InvoiceWorkflowState) -> InvoiceWorkflowState:
    """
    Translation Agent (Google ADK): Translate to English if needed.
    Tool: Lang-Bridge Tool (via A2A + MCP)
    """
    logger.info("=" * 80)
    logger.info("NODE: Translation Agent [Google ADK] — Lang-Bridge Tool")
    logger.info("=" * 80)

    _ensure_initialized()
    observability = get_observability()
    bridge = get_bridge()
    guardrails = get_guardrails()
    trace_id = state.get("workflow_id", "unknown")

    span_id = observability.start_span(
        trace_id,
        "translation_agent_adk",
        input_data={
            "text_len": len(state.get("extracted_text", "")),
            "text_preview": state.get("extracted_text", "")[:200],
            "tables_count": len(state.get("extracted_tables", [])),
            "tool": "Lang-Bridge Tool",
            "framework": "google_adk",
        },
        metadata={"node": "translate", "agent_type": "google_adk", "protocol": "A2A"},
    )

    try:
        a2a_span = observability.start_span(
            trace_id,
            "translation_a2a_invocation",
            input_data={
                "source_agent": "lg_extractor_agent",
                "target_agent": "adk_translation_agent",
                "protocol": "A2A",
                "payload_text_len": len(state["extracted_text"]),
            },
        )

        result = bridge.invoke_agent(
            source_agent_id="lg_extractor_agent",
            target_agent_id="adk_translation_agent",
            payload={
                "raw_text": state["extracted_text"],
                "tables": state["extracted_tables"],
                "metadata": state["extraction_metadata"],
                "_trace_id": trace_id,
            },
        )

        observability.end_span(
            a2a_span,
            output={
                "a2a_success": True,
                "result_keys": list(result.keys()) if result else [],
            },
        )

        state["translated_text"] = result.get(
            "translated_text", state["extracted_text"]
        )
        state["translated_tables"] = result.get(
            "translated_tables", state["extracted_tables"]
        )
        state["original_language"] = result.get("original_language", "en")
        state["translation_confidence"] = result.get("translation_confidence", 1.0)
        state["translation_method"] = result.get("translation_method", "none")
        state["needs_translation"] = result.get("original_language", "en") != "en"

        if state["needs_translation"] and state["translation_method"] != "none":
            observability.log_generation(
                trace_id=trace_id,
                name="translation_llm_call",
                model="bedrock/cohere.command-r-plus-v1:0",
                prompt=f"Translate {state['original_language']}→en ({len(state['extracted_text'])} chars)",
                completion=(
                    state["translated_text"][:500] if state["translated_text"] else ""
                ),
                metadata={
                    "source_language": state["original_language"],
                    "method": state["translation_method"],
                    "confidence": state["translation_confidence"],
                    "input_length": len(state["extracted_text"]),
                    "output_length": len(state["translated_text"]),
                },
            )

        rai_span = observability.start_span(
            trace_id,
            "translation_rai_check",
            input_data={
                "text_preview": (
                    state["translated_text"][:200] if state["translated_text"] else ""
                ),
                "original_language": state["original_language"],
            },
        )

        rai_check = guardrails.check_output(
            state["translated_text"], agent_id="adk_translation_agent"
        )

        rai_violations = []
        if rai_check.violations:
            logger.warning(
                f"RAI violations in translation: {len(rai_check.violations)}"
            )
            rai_violations = [
                {"type": getattr(v, "type", str(v)), "detail": str(v)}
                for v in rai_check.violations
            ]

        observability.end_span(
            rai_span,
            output={
                "violations_count": len(rai_violations),
                "violations": rai_violations,
                "passed": not bool(rai_violations),
            },
            level="WARNING" if rai_violations else "DEFAULT",
        )

        logger.info(f"Lang-Bridge: {state['original_language']} → en")
        logger.info(f"   Method: {state['translation_method']}")
        logger.info(f"   Confidence: {state['translation_confidence']:.2f}")

        observability.record_score(
            trace_id,
            "translation_confidence",
            state["translation_confidence"],
            comment=f"Language: {state['original_language']}, method: {state['translation_method']}",
        )

        if state["needs_translation"]:
            observability.record_score(
                trace_id,
                "translation_needed",
                1.0,
                comment=f"Source language: {state['original_language']}",
            )
        else:
            observability.record_score(
                trace_id,
                "translation_needed",
                0.0,
                comment="Document already in English",
            )

        observability.end_span(
            span_id,
            output={
                "language": state["original_language"],
                "method": state["translation_method"],
                "confidence": state["translation_confidence"],
                "needs_translation": state["needs_translation"],
                "translated_text_preview": (
                    state["translated_text"][:300] if state["translated_text"] else ""
                ),
                "rai_violations": len(rai_violations),
            },
        )

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        state["error"] = str(e)
        state["translated_text"] = state["extracted_text"]
        state["translated_tables"] = state["extracted_tables"]
        state["original_language"] = "unknown"
        state["translation_confidence"] = 0.0
        state["translation_method"] = "none"
        state["needs_translation"] = False
        observability.record_score(
            trace_id, "translation_confidence", 0.0, comment=f"Error: {e}"
        )
        observability.end_span(
            span_id,
            output={
                "error": str(e),
                "error_type": type(e).__name__,
                "fallback": "used_extracted_text",
            },
            level="ERROR",
        )

    return state

@observe(name="validation_agent")
def validation_node(state: InvoiceWorkflowState) -> InvoiceWorkflowState:
    """
    Validation Agent (LangGraph): Validate data completeness and business rules.
    Tools: DataCompletenessChecker + Business Validation Tool (via MCP)
    """
    logger.info("=" * 80)
    logger.info(
        "NODE: Validation Agent [LangGraph] — DataCompletenessChecker + Business Validation Tool"
    )
    logger.info("=" * 80)

    _ensure_initialized()
    observability = get_observability()
    guardrails = get_guardrails()
    trace_id = state.get("workflow_id", "unknown")

    span_id = observability.start_span(
        trace_id,
        "validation_agent_langgraph",
        input_data={
            "translated_text_len": len(state.get("translated_text", "")),
            "translated_text_preview": state.get("translated_text", "")[:200],
            "tables_count": len(state.get("translated_tables", [])),
            "tool": "DataCompletenessChecker + Business Validation Tool",
            "framework": "langgraph",
        },
        metadata={"node": "validate", "agent_type": "langgraph"},
    )

    try:
        erp_api_url = os.getenv("ERP_API_URL", "http://localhost:8000")
        init_span = observability.start_span(
            trace_id,
            "validation_init",
            input_data={
                "model": "bedrock/amazon.nova-lite-v1:0",
                "erp_url": erp_api_url,
            },
        )

        data_validator = InvoiceDataValidator(model="bedrock/amazon.nova-lite-v1:0")
        business_validator = BusinessValidator(
            model="bedrock/amazon.nova-lite-v1:0", erp_api_url=erp_api_url
        )

        validation_input = {
            "translated_text": state["translated_text"],
            "tables": state["translated_tables"],
            "metadata": state["extraction_metadata"],
        }

        observability.end_span(init_span, output={"validators_initialized": True})

        rai_integrity_span = observability.start_span(
            trace_id,
            "validation_rai_integrity_check",
            input_data={
                "invoice_data_keys": list(state.get("invoice_data", {}).keys())
            },
        )

        integrity_check = guardrails.check_invoice_data_integrity(
            state.get("invoice_data", {})
        )

        integrity_violations = []
        if integrity_check.violations:
            logger.warning(
                f"RAI data integrity issues: {len(integrity_check.violations)}"
            )
            integrity_violations = [
                {"type": getattr(v, "type", str(v)), "detail": str(v)}
                for v in integrity_check.violations
            ]

        observability.end_span(
            rai_integrity_span,
            output={
                "violations_count": len(integrity_violations),
                "violations": integrity_violations,
                "passed": not bool(integrity_violations),
            },
            level="WARNING" if integrity_violations else "DEFAULT",
        )

        data_val_span = observability.start_span(
            trace_id,
            "data_completeness_checker",
            input_data={
                "tool": "DataCompletenessChecker Tool",
                "input_text_len": len(state["translated_text"]),
                "input_tables_count": len(state["translated_tables"]),
            },
        )

        logger.info("Step 1: DataCompletenessChecker Tool...")
        data_result = data_validator.run(validation_input)

        state["invoice_data"] = data_result.get("invoice_data", {})
        state["data_validation_issues"] = data_result.get("validation_issues", [])

        observability.log_generation(
            trace_id=trace_id,
            name="data_validator_field_extraction_llm",
            model="bedrock/amazon.nova-lite-v1:0",
            prompt=f"Extract invoice fields from text ({len(state['translated_text'])} chars)",
            completion=json.dumps(state["invoice_data"], default=str)[:2000],
            metadata={
                "fields_extracted": list(state["invoice_data"].keys()),
                "non_null_fields": [
                    k for k, v in state["invoice_data"].items() if v is not None
                ],
            },
        )

        data_errors = [
            i for i in state["data_validation_issues"] if i.get("severity") == "error"
        ]
        data_warnings = [
            i for i in state["data_validation_issues"] if i.get("severity") == "warning"
        ]

        logger.info(f"   Data issues found: {len(state['data_validation_issues'])}")

        observability.end_span(
            data_val_span,
            output={
                "total_data_issues": len(state["data_validation_issues"]),
                "data_errors": len(data_errors),
                "data_warnings": len(data_warnings),
                "invoice_data": state["invoice_data"],
                "issues": state["data_validation_issues"][
                    :10
                ],                            
            },
        )

        total_fields = len(state["invoice_data"])
        non_null_fields = len(
            [v for v in state["invoice_data"].values() if v is not None]
        )
        completeness = non_null_fields / max(total_fields, 1)
        observability.record_score(
            trace_id,
            "data_completeness",
            completeness,
            comment=f"{non_null_fields}/{total_fields} fields populated, {len(data_errors)} errors",
        )

        critical_errors = [
            i for i in state["data_validation_issues"] if i.get("severity") == "error"
        ]

        biz_val_span = observability.start_span(
            trace_id,
            "business_validation_tool",
            input_data={
                "tool": "Business Validation Tool",
                "critical_data_errors": len(critical_errors),
                "skip_business_validation": len(critical_errors) > 0,
                "invoice_no": state["invoice_data"].get("invoice_no", "N/A"),
                "po_number": state["invoice_data"].get("po_number", "N/A"),
            },
        )

        if len(critical_errors) == 0:
            logger.info("Step 2: Business Validation Tool...")
            business_result = business_validator.run(data_result)

            state["business_validation_issues"] = business_result.get(
                "validation_issues", []
            )

            biz_errors = [
                i
                for i in state["business_validation_issues"]
                if i.get("severity") == "error"
            ]
            biz_warnings = [
                i
                for i in state["business_validation_issues"]
                if i.get("severity") == "warning"
            ]

            logger.info(
                f"   Business issues found: {len(state['business_validation_issues'])}"
            )

            observability.end_span(
                biz_val_span,
                output={
                    "skipped": False,
                    "total_business_issues": len(state["business_validation_issues"]),
                    "business_errors": len(biz_errors),
                    "business_warnings": len(biz_warnings),
                    "issues": state["business_validation_issues"][:10],
                    "recommendation_from_biz": business_result.get(
                        "recommendation", "N/A"
                    ),
                },
            )

            biz_score = (
                1.0 if len(biz_errors) == 0 else max(0.0, 1.0 - len(biz_errors) * 0.25)
            )
            observability.record_score(
                trace_id,
                "business_validation",
                biz_score,
                comment=f"{len(biz_errors)} errors, {len(biz_warnings)} warnings",
            )
        else:
            logger.warning(
                "⚠️  Skipping Business Validation Tool due to critical data errors"
            )
            state["business_validation_issues"] = []

            observability.end_span(
                biz_val_span,
                output={
                    "skipped": True,
                    "reason": f"{len(critical_errors)} critical data errors",
                    "critical_errors": [
                        e.get("message", str(e)) for e in critical_errors[:5]
                    ],
                },
                level="WARNING",
            )

            observability.record_score(
                trace_id,
                "business_validation",
                0.0,
                comment=f"Skipped: {len(critical_errors)} critical data errors",
            )

        all_issues = (
            state["data_validation_issues"] + state["business_validation_issues"]
        )

        rai_bias_span = observability.start_span(
            trace_id,
            "validation_rai_bias_check",
            input_data={"total_issues": len(all_issues)},
        )

        validation_history = []
        try:
            validation_history = PostgresInvoiceStore().recent_validation_results()
        except Exception as history_error:
            logger.warning("Could not load validation history for RAI bias check: %s", history_error)
        validation_history.append({
            "invoice_data": state.get("invoice_data", {}),
            "recommendation": "MANUAL_REVIEW",
        })
        bias_check = guardrails.check_validation_bias(validation_history)

        bias_violations = []
        if bias_check.violations:
            logger.warning(f"RAI bias check warnings: {len(bias_check.violations)}")
            bias_violations = [
                {"type": getattr(v, "type", str(v)), "detail": str(v)}
                for v in bias_check.violations
            ]

        observability.end_span(
            rai_bias_span,
            output={
                "violations_count": len(bias_violations),
                "violations": bias_violations,
                "passed": not bool(bias_violations),
            },
            level="WARNING" if bias_violations else "DEFAULT",
        )

        total_errors = len([i for i in all_issues if i.get("severity") == "error"])
        total_warnings = len([i for i in all_issues if i.get("severity") == "warning"])

        state["validation_summary"] = {
            "total_issues": len(all_issues),
            "errors": total_errors,
            "warnings": total_warnings,
            "data_issues": len(state["data_validation_issues"]),
            "business_issues": len(state["business_validation_issues"]),
        }

        state["validation_passed"] = total_errors == 0

        logger.info(f"Validation complete:")
        logger.info(
            f"   Total issues: {len(all_issues)} ({total_errors} errors, {total_warnings} warnings)"
        )
        logger.info(f"   Validation passed: {state['validation_passed']}")

        validation_score = (
            1.0
            if total_errors == 0 and total_warnings == 0
            else (0.5 if total_errors == 0 else max(0.0, 1.0 - total_errors * 0.2))
        )
        observability.record_score(
            trace_id,
            "validation_overall",
            validation_score,
            comment=f"{total_errors} errors, {total_warnings} warnings, passed={state['validation_passed']}",
        )

        observability.end_span(
            span_id,
            output={
                "total_issues": len(all_issues),
                "errors": total_errors,
                "warnings": total_warnings,
                "data_issues": len(state["data_validation_issues"]),
                "business_issues": len(state["business_validation_issues"]),
                "passed": state["validation_passed"],
                "invoice_data": state["invoice_data"],
                "all_issues": all_issues[:15],                               
                "rai_integrity_violations": len(integrity_violations),
                "rai_bias_violations": len(bias_violations),
            },
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        state["error"] = str(e)
        state["invoice_data"] = {}
        state["data_validation_issues"] = []
        state["business_validation_issues"] = []
        state["validation_summary"] = {"total_issues": 1, "errors": 1, "warnings": 0}
        state["validation_passed"] = False
        observability.record_score(
            trace_id, "validation_overall", 0.0, comment=f"Error: {e}"
        )
        observability.end_span(
            span_id,
            output={
                "error": str(e),
                "error_type": type(e).__name__,
            },
            level="ERROR",
        )

    return state

@observe(name="reporting_agent")
def reporting_node(state: InvoiceWorkflowState) -> InvoiceWorkflowState:
    """
    Reporting Agent (Google ADK): Generate validation reports.
    Tool: Insight Reporter Tool (via A2A + MCP)
    """
    logger.info("=" * 80)
    logger.info("NODE: Reporting Agent [Google ADK] — Insight Reporter Tool")
    logger.info("=" * 80)

    _ensure_initialized()
    observability = get_observability()
    bridge = get_bridge()
    guardrails = get_guardrails()
    trace_id = state.get("workflow_id", "unknown")

    span_id = observability.start_span(
        trace_id,
        "reporting_agent_adk",
        input_data={
            "invoice_no": state.get("invoice_data", {}).get("invoice_no", "N/A"),
            "validation_errors": state["validation_summary"].get("errors", 0),
            "validation_warnings": state["validation_summary"].get("warnings", 0),
            "validation_passed": state.get("validation_passed", False),
            "tool": "Insight Reporter Tool",
            "framework": "google_adk",
        },
        metadata={"node": "report", "agent_type": "google_adk", "protocol": "A2A"},
    )

    try:
        rec_span = observability.start_span(
            trace_id,
            "reporting_recommendation_logic",
            input_data={
                "errors": state["validation_summary"]["errors"],
                "warnings": state["validation_summary"]["warnings"],
            },
        )

        if state["validation_summary"]["errors"] == 0:
            if state["validation_summary"]["warnings"] == 0:
                recommendation = "APPROVE"
            else:
                recommendation = "MANUAL_REVIEW"
        else:
            recommendation = "REJECT"

        observability.end_span(
            rec_span,
            output={
                "recommendation": recommendation,
                "logic": f"errors={state['validation_summary']['errors']}, warnings={state['validation_summary']['warnings']}",
            },
        )

        a2a_span = observability.start_span(
            trace_id,
            "reporting_a2a_invocation",
            input_data={
                "source_agent": "lg_validation_agent",
                "target_agent": "adk_reporting_agent",
                "protocol": "A2A",
                "recommendation": recommendation,
                "invoice_no": state["invoice_data"].get("invoice_no", "N/A"),
            },
        )

        result = bridge.invoke_agent(
            source_agent_id="lg_validation_agent",
            target_agent_id="adk_reporting_agent",
            payload={
                "invoice_data": state["invoice_data"],
                "validation_issues": state["data_validation_issues"]
                + state["business_validation_issues"],
                "validation_summary": state["validation_summary"],
                "translation_metadata": {
                    "original_language": state["original_language"],
                    "translation_confidence": state["translation_confidence"],
                    "translation_method": state["translation_method"],
                },
                "extraction_metadata": state["extraction_metadata"],
                "recommendation": recommendation,
                "save_format": "both",
                "_trace_id": trace_id,
            },
        )

        observability.end_span(
            a2a_span,
            output={
                "a2a_success": True,
                "result_keys": list(result.keys()) if result else [],
            },
        )

        observability.log_generation(
            trace_id=trace_id,
            name="reporting_summary_llm_call",
            model="bedrock/amazon.nova-lite-v1:0",
            prompt=f"Generate executive summary for invoice {state['invoice_data'].get('invoice_no', 'N/A')}, recommendation={recommendation}",
            completion=result.get("summary", "")[:1000] if result else "",
            metadata={
                "recommendation": recommendation,
                "total_issues": state["validation_summary"].get("total_issues", 0),
                "invoice_no": state["invoice_data"].get("invoice_no", "N/A"),
            },
        )

        files = result.get("files", {})
        state["report_path"] = files.get("json", "")
        state["html_report_path"] = files.get("html", "")
        state["recommendation"] = recommendation

        rai_span = observability.start_span(
            trace_id,
            "reporting_rai_compliance_check",
            input_data={
                "invoice_no": state["invoice_data"].get("invoice_no", "N/A"),
                "recommendation": recommendation,
            },
        )

        compliance_check = guardrails.check_compliance(state)

        compliance_violations = []
        if compliance_check.violations:
            logger.warning(f"RAI compliance issues: {len(compliance_check.violations)}")
            compliance_violations = [
                {"type": getattr(v, "type", str(v)), "detail": str(v)}
                for v in compliance_check.violations
            ]

        observability.end_span(
            rai_span,
            output={
                "violations_count": len(compliance_violations),
                "violations": compliance_violations,
                "passed": not bool(compliance_violations),
            },
            level="WARNING" if compliance_violations else "DEFAULT",
        )

        logger.info(f"Insight Reporter: Reports generated:")
        logger.info(f"   JSON: {state['report_path']}")
        logger.info(f"   HTML: {state['html_report_path']}")
        logger.info(f"   Recommendation: {recommendation}")

        report_generated = (
            1.0 if state["report_path"] or state["html_report_path"] else 0.0
        )
        observability.record_score(
            trace_id,
            "report_generated",
            report_generated,
            comment=f"JSON={'yes' if state['report_path'] else 'no'}, HTML={'yes' if state['html_report_path'] else 'no'}",
        )

        observability.end_span(
            span_id,
            output={
                "recommendation": recommendation,
                "json_report": state["report_path"],
                "html_report": state["html_report_path"],
                "invoice_no": state["invoice_data"].get("invoice_no", "N/A"),
                "rai_compliance_violations": len(compliance_violations),
            },
        )

    except Exception as e:
        logger.error(f"Reporting failed: {e}")
        state["error"] = str(e)
        state["report_path"] = ""
        state["html_report_path"] = ""
        state["recommendation"] = "ERROR"
        observability.record_score(
            trace_id, "report_generated", 0.0, comment=f"Error: {e}"
        )
        observability.end_span(
            span_id,
            output={
                "error": str(e),
                "error_type": type(e).__name__,
            },
            level="ERROR",
        )

    return state

@observe(name="indexing_agent")
def indexing_node(state: InvoiceWorkflowState) -> InvoiceWorkflowState:
    """
    RAG Indexing (LangGraph): Index invoice for future queries.
    Tool: Vector-Indexer Tool (via MCP)
    """
    logger.info("=" * 80)
    logger.info("NODE: RAG Indexing Agent [LangGraph] — Vector-Indexer Tool")
    logger.info("=" * 80)

    _ensure_initialized()
    observability = get_observability()
    trace_id = state.get("workflow_id", "unknown")

    invoice_no = state.get("invoice_data", {}).get("invoice_no", "UNKNOWN")
    span_id = observability.start_span(
        trace_id,
        "indexing_agent_langgraph",
        input_data={
            "invoice_no": invoice_no,
            "has_report": bool(state.get("report_path")),
            "invoice_data_keys": list(state.get("invoice_data", {}).keys()),
            "tool": "Vector-Indexer Tool",
            "framework": "langgraph",
        },
        metadata={"node": "index", "agent_type": "langgraph"},
    )

    try:
        init_span = observability.start_span(
            trace_id, "indexing_init", input_data={"index_path": "./data/vector_db"}
        )

        indexer = IndexingAgent(index_path="./data/vector_db")

        observability.end_span(init_span, output={"indexer_initialized": True})

        inv_idx_span = observability.start_span(
            trace_id,
            "indexing_invoice_data",
            input_data={
                "invoice_no": invoice_no,
                "invoice_data": state["invoice_data"],
                "has_line_items": bool(state.get("line_items")),
                "has_validation_issues": bool(state.get("validation_issues")),
            },
        )

        validation_result = {
            "invoice_data": state["invoice_data"],
            "line_items": state.get("line_items", []),
            "validation_issues": state.get("validation_issues", []),
            "validation_summary": state.get("validation_summary", {}),
        }
        indexer.index_invoice_data(validation_result)

        observability.end_span(
            inv_idx_span,
            output={
                "indexed_invoice": True,
                "invoice_no": invoice_no,
            },
        )

        report_indexed = False
        if state.get("report_path") and os.path.exists(state["report_path"]):
            report_idx_span = observability.start_span(
                trace_id,
                "indexing_report",
                input_data={"report_path": state["report_path"]},
            )

            try:
                with open(state["report_path"], "r") as f:
                    report_data = json.load(f)
                indexer.index_report(report_data)
                report_indexed = True

                observability.end_span(
                    report_idx_span,
                    output={
                        "indexed_report": True,
                        "report_path": state["report_path"],
                    },
                )
            except Exception as e:
                logger.warning(f"Could not load report for indexing: {e}")
                observability.end_span(
                    report_idx_span,
                    output={
                        "indexed_report": False,
                        "error": str(e),
                    },
                    level="WARNING",
                )

        save_span = observability.start_span(
            trace_id, "indexing_save", input_data={"index_path": "./data/vector_db"}
        )

        indexer.save_index()

        observability.end_span(save_span, output={"saved": True})

        state["indexed"] = True
        logger.info(f"Vector-Indexer: Indexed invoice {invoice_no}")

        observability.record_score(
            trace_id,
            "indexing_success",
            1.0,
            comment=f"Invoice {invoice_no} indexed, report_indexed={report_indexed}",
        )

        observability.end_span(
            span_id,
            output={
                "indexed": True,
                "invoice_no": invoice_no,
                "report_indexed": report_indexed,
            },
        )

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        state["indexed"] = False
        observability.record_score(
            trace_id, "indexing_success", 0.0, comment=f"Error: {e}"
        )
        observability.end_span(
            span_id,
            output={
                "error": str(e),
                "error_type": type(e).__name__,
                "indexed": False,
            },
            level="ERROR",
        )

    return state




def should_continue_after_extraction(state: InvoiceWorkflowState) -> str:
    """Route after extraction based on success."""
    observability = get_observability()
    trace_id = state.get("workflow_id", "unknown")

    decision = "translate"
    reason = "Extraction successful, proceeding to translation"

    if state.get("error"):
        decision = "end"
        reason = f"Extraction failed: {state['error']}"
        logger.error(" Extraction failed, ending workflow")

    route_span = observability.start_span(
        trace_id,
        "routing_after_extraction",
        input_data={
            "has_error": bool(state.get("error")),
            "extracted_text_len": len(state.get("extracted_text", "")),
        },
    )
    observability.end_span(
        route_span,
        output={
            "decision": decision,
            "reason": reason,
        },
        level="ERROR" if decision == "end" else "DEFAULT",
    )

    return decision


def should_continue_after_translation(state: InvoiceWorkflowState) -> str:
    """Route after translation."""
    observability = get_observability()
    trace_id = state.get("workflow_id", "unknown")

    decision = "validate"
    reason = "Translation successful, proceeding to validation"

    if state.get("error"):
        decision = "end"
        reason = f"Translation failed: {state['error']}"
        logger.error(" Translation failed, ending workflow")

    route_span = observability.start_span(
        trace_id,
        "routing_after_translation",
        input_data={
            "has_error": bool(state.get("error")),
            "needs_translation": state.get("needs_translation", False),
            "original_language": state.get("original_language", "unknown"),
        },
    )
    observability.end_span(
        route_span,
        output={
            "decision": decision,
            "reason": reason,
        },
        level="ERROR" if decision == "end" else "DEFAULT",
    )

    return decision


def should_continue_after_validation(state: InvoiceWorkflowState) -> str:
    """Route after validation."""
    observability = get_observability()
    trace_id = state.get("workflow_id", "unknown")

    decision = "report"
    reason = "Proceeding to reporting"

    if state.get("error"):
        reason = f"Validation had error ({state['error']}), but continuing to reporting"
        logger.error(" Validation failed, but continuing to reporting")

    route_span = observability.start_span(
        trace_id,
        "routing_after_validation",
        input_data={
            "has_error": bool(state.get("error")),
            "validation_passed": state.get("validation_passed", False),
            "total_errors": state.get("validation_summary", {}).get("errors", 0),
            "total_warnings": state.get("validation_summary", {}).get("warnings", 0),
        },
    )
    observability.end_span(
        route_span,
        output={
            "decision": decision,
            "reason": reason,
            "validation_passed": state.get("validation_passed", False),
        },
    )

    return decision


def should_continue_after_reporting(state: InvoiceWorkflowState) -> str:
    """Route after reporting."""
    observability = get_observability()
    trace_id = state.get("workflow_id", "unknown")

    route_span = observability.start_span(
        trace_id,
        "routing_after_reporting",
        input_data={
            "has_report": bool(state.get("report_path")),
            "recommendation": state.get("recommendation", "UNKNOWN"),
        },
    )
    observability.end_span(
        route_span,
        output={
            "decision": "index",
            "reason": "Proceeding to indexing",
        },
    )

    return "index"




def create_invoice_workflow() -> StateGraph:
    """
    Create the invoice auditor workflow graph.

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Building Invoice Auditor Workflow...")

    workflow = StateGraph(InvoiceWorkflowState)

    workflow.add_node("monitor", monitor_node)
    workflow.add_node("extract", extractor_node)
    workflow.add_node("translate", translation_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("report", reporting_node)
    workflow.add_node("index", indexing_node)

    workflow.set_entry_point("monitor")

    workflow.add_edge("monitor", "extract")

    workflow.add_conditional_edges(
        "extract",
        should_continue_after_extraction,
        {"translate": "translate", "end": END},
    )

    workflow.add_conditional_edges(
        "translate",
        should_continue_after_translation,
        {"validate": "validate", "end": END},
    )

    workflow.add_conditional_edges(
        "validate", should_continue_after_validation, {"report": "report", "end": END}
    )

    workflow.add_conditional_edges(
        "report", should_continue_after_reporting, {"index": "index", "end": END}
    )

    workflow.add_edge("index", END)

    checkpointer = MemorySaver()
    compiled_workflow = workflow.compile(checkpointer=checkpointer)

    logger.info("Workflow compiled successfully")

    return compiled_workflow



def run_workflow(invoice_path: str, workflow_id: str = None) -> Dict[str, Any]:
    """
    Execute the invoice auditor workflow for a single invoice.
    Dual-Framework: LangGraph orchestration + Google ADK agents via A2A Protocol.

    Args:
        invoice_path: Path to invoice file
        workflow_id: Optional workflow ID for tracking

    Returns:
        Final state after workflow completion
    """
    from datetime import datetime
    import uuid

    if not workflow_id:
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"

    _ensure_initialized()
    observability = get_observability()

    logger.info("=" * 80)
    logger.info(f"STARTING INVOICE AUDITOR WORKFLOW: {workflow_id}")
    logger.info(f"  Frameworks: LangGraph + Google ADK (A2A Protocol)")
    logger.info("  Invoice: %s", Path(invoice_path).name)
    logger.info("=" * 80)

    trace_id = observability.start_trace(
        name=f"invoice_audit_{workflow_id}",
        input_data={
            "invoice_filename": Path(invoice_path).name,
            "workflow_id": workflow_id,
            "file_extension": Path(invoice_path).suffix,
        },
        metadata={
            "framework": "dual_langgraph_adk",
            "protocol": "A2A + MCP",
            "guardrails": "RAI",
            "orchestrator": "LangGraph StateGraph",
            "adk_agents": ["extractor", "translation", "reporting"],
            "langgraph_agents": ["monitor", "validation", "indexing"],
            "models": [
                "bedrock/amazon.nova-lite-v1:0",
                "bedrock/cohere.command-r-plus-v1:0",
            ],
        },
        user_id="invoice_auditor_system",
        session_id=workflow_id,
    )

    initial_state: InvoiceWorkflowState = {
        "invoice_path": invoice_path,
        "invoice_filename": Path(invoice_path).name,
        "workflow_id": workflow_id,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "needs_translation": False,
        "validation_passed": False,
        "indexed": False,
        "file_metadata": {},
        "extracted_text": "",
        "extracted_tables": [],
        "extraction_metadata": {},
        "translated_text": "",
        "translated_tables": [],
        "original_language": "en",
        "translation_confidence": 1.0,
        "translation_method": "none",
        "invoice_data": {},
        "data_validation_issues": [],
        "business_validation_issues": [],
        "validation_summary": {},
        "report_path": "",
        "html_report_path": "",
        "recommendation": "UNKNOWN",
    }

    workflow = create_invoice_workflow()

    config = {"configurable": {"thread_id": workflow_id}}
    final_state = None

    try:
        for state in workflow.stream(initial_state, config):
            logger.debug(f"State update: {list(state.keys())}")
            final_state = state

        if final_state:
            final_state = (
                list(final_state.values())[-1] if final_state else initial_state
            )

        final_state["completed_at"] = datetime.now().isoformat()

        final_state["validation_issues"] = final_state.get(
            "data_validation_issues", []
        ) + final_state.get("business_validation_issues", [])

        final_state["report_paths"] = {
            "json": final_state.get("report_path", ""),
            "html": final_state.get("html_report_path", ""),
        }

        storage_status = persist_workflow_result(final_state)
        final_state["external_storage"] = storage_status
        logger.info("External storage status: %s", storage_status)

        observability.end_trace(
            trace_id,
            output={
                "recommendation": final_state.get("recommendation", "UNKNOWN"),
                "validation_passed": final_state.get("validation_passed", False),
                "indexed": final_state.get("indexed", False),
                "invoice_no": final_state.get("invoice_data", {}).get(
                    "invoice_no", "UNKNOWN"
                ),
                "original_language": final_state.get("original_language", "unknown"),
                "total_issues": final_state.get("validation_summary", {}).get(
                    "total_issues", 0
                ),
                "errors": final_state.get("validation_summary", {}).get("errors", 0),
                "warnings": final_state.get("validation_summary", {}).get(
                    "warnings", 0
                ),
                "json_report": final_state.get("report_path", ""),
                "html_report": final_state.get("html_report_path", ""),
            },
        )

        observability.record_score(
            trace_id,
            "workflow_success",
            1.0,
            comment="Workflow completed without fatal errors",
        )

        if final_state.get("recommendation") == "APPROVE":
            observability.record_score(
                trace_id,
                "invoice_approved",
                1.0,
                comment=f"Invoice {final_state.get('invoice_data', {}).get('invoice_no', 'N/A')} approved",
            )
        elif final_state.get("recommendation") == "REJECT":
            observability.record_score(
                trace_id,
                "invoice_rejected",
                1.0,
                comment=f"Invoice {final_state.get('invoice_data', {}).get('invoice_no', 'N/A')} rejected with {final_state.get('validation_summary', {}).get('errors', 0)} errors",
            )
        elif final_state.get("recommendation") == "MANUAL_REVIEW":
            observability.record_score(
                trace_id,
                "invoice_manual_review",
                1.0,
                comment=f"Invoice {final_state.get('invoice_data', {}).get('invoice_no', 'N/A')} sent for manual review with {final_state.get('validation_summary', {}).get('warnings', 0)} warnings",
            )

        pipeline_quality = 1.0
        if final_state.get("error"):
            pipeline_quality -= 0.3
        if not final_state.get("indexed", False):
            pipeline_quality -= 0.1
        if final_state.get("translation_confidence", 1.0) < 0.8:
            pipeline_quality -= 0.1
        if final_state.get("validation_summary", {}).get("errors", 0) > 0:
            pipeline_quality -= 0.2
        observability.record_score(
            trace_id,
            "pipeline_quality",
            max(0.0, pipeline_quality),
            comment=f"End-to-end quality: rec={final_state.get('recommendation', 'UNKNOWN')}, "
            f"indexed={final_state.get('indexed', False)}, "
            f"lang_conf={final_state.get('translation_confidence', 1.0):.2f}",
        )

        observability.flush()

        logger.info("=" * 80)
        logger.info(f"WORKFLOW COMPLETED: {workflow_id}")
        logger.info(f"  Recommendation: {final_state.get('recommendation', 'UNKNOWN')}")
        logger.info(
            f"  Validation Passed: {final_state.get('validation_passed', False)}"
        )
        logger.info(f"  Indexed: {final_state.get('indexed', False)}")
        logger.info("=" * 80)

        return final_state

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        initial_state["error"] = str(e)
        initial_state["error_type"] = type(e).__name__
        initial_state["completed_at"] = datetime.now().isoformat()

        try:
            persist_workflow_result(initial_state)
        except Exception as storage_error:
            logger.error("Could not persist failed workflow audit record: %s", storage_error)

        observability.end_trace(
            trace_id,
            output={
                "error": str(e),
                "error_type": type(e).__name__,
            },
            status="error",
        )
        observability.record_score(
            trace_id,
            "workflow_success",
            0.0,
            comment=f"Fatal error: {type(e).__name__}: {str(e)[:200]}",
        )
        observability.record_score(
            trace_id, "pipeline_quality", 0.0, comment=f"Workflow failed: {e}"
        )
        observability.flush()

        return initial_state




def run_batch_workflow(invoice_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Process multiple invoices through the workflow.

    Args:
        invoice_paths: List of invoice file paths

    Returns:
        List of final states for each invoice
    """
    logger.info(f"Starting batch processing for {len(invoice_paths)} invoices")

    results = []

    for i, invoice_path in enumerate(invoice_paths, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing invoice {i}/{len(invoice_paths)}")
        logger.info(f"{'='*80}\n")

        try:
            result = run_workflow(invoice_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {invoice_path}: {e}")
            results.append(
                {
                    "invoice_path": invoice_path,
                    "error": str(e),
                    "recommendation": "ERROR",
                }
            )

    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total processed: {len(results)}")
    logger.info(
        f"Approved: {sum(1 for r in results if r.get('recommendation') == 'APPROVE')}"
    )
    logger.info(
        f"Manual Review: {sum(1 for r in results if r.get('recommendation') == 'MANUAL_REVIEW')}"
    )
    logger.info(
        f"Rejected: {sum(1 for r in results if r.get('recommendation') == 'REJECT')}"
    )
    logger.info(
        f"Errors: {sum(1 for r in results if r.get('recommendation') == 'ERROR')}"
    )
    logger.info("=" * 80)

    return results



if __name__ == "__main__":
    """
    Test the workflow with sample invoices.
    """
    import sys

    if len(sys.argv) > 1:
        invoice_path = sys.argv[1]
        result = run_workflow(invoice_path)

        logger.info("Workflow result")
        logger.info("Invoice: %s", result["invoice_filename"])
        logger.info("Language: %s", result["original_language"])
        logger.info("Recommendation: %s", result["recommendation"])
        logger.info(
            "Validation Issues: %s",
            result["validation_summary"].get("total_issues", 0),
        )
        logger.info("Report: %s", result["html_report_path"])
    else:
        sample_invoices = [
            "./data/sample_data/INV_EN_001.pdf",
            "./data/sample_data/INV_ES_003.pdf",
            "./data/sample_data/INV_DE_004.pdf",
        ]

        existing_invoices = [p for p in sample_invoices if os.path.exists(p)]

        if existing_invoices:
            logger.info("Processing %s sample invoices", len(existing_invoices))
            results = run_batch_workflow(existing_invoices)
        else:
            logger.warning("No sample invoices found")
            logger.info("Usage: python workflows/invoice_auditor_workflow.py <invoice_path>")
            logger.info("Or place sample invoices in ./data/sample_data/")
