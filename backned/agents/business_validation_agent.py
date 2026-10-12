"""Business Validation Agent — cross-references invoice data against ERP records.

LangGraph: business_validate_all  summarize_business_validation  END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backned.models.translation import TranslationResult
from backned.models.validation import ValidationResult
from backned.models.business_validation import BusinessValidationResult

#  MCP client for tool access 
from backned.tool_dispatch import invoke_tool_sync

#  LangFuse observability 
from backned.config.observability import observe, langfuse_context

logger = logging.getLogger(__name__)


# 
# LangGraph — State Schema
# 
class BusinessValidationState(TypedDict):
    """State flowing through the Business Validation graph."""
    translation_results: list[dict]
    validation_results: list[dict]        # from data completeness checker
    business_validation_results: list[dict]
    errors: list[str]
    status: str


# 
# LangGraph — Node functions
# 
def business_validate_all_node(
    state: BusinessValidationState,
) -> dict[str, Any]:
    """Iterate over every translation result and invoke the Business
    Validation Tool to cross-reference against ERP data."""
    logger.info("Business Validation Agent - starting ERP validation ...")

    translations: list[dict] = state.get("translation_results", [])
    validations: list[dict] = state.get("validation_results", [])
    all_results: list[dict] = []
    errors: list[str] = []

    # Build a lookup from filename → validation_result for context
    val_by_file: dict[str, dict] = {}
    for vr in validations:
        val_by_file[vr.get("filename", "")] = vr

    for tr_dict in translations:
        filename = tr_dict.get("filename", "?")
        email_id = tr_dict.get("email_id", "unknown")

        inv = tr_dict.get("translated_invoice") or {}
        po_number = inv.get("po_number", "N/A")

        logger.info(
            "Business validating: %s (PO: %s, email: %s)",
            filename, po_number, email_id,
        )

        try:
            # Invoke MCP tool (async-only tools are wrapped by invoke_tool_sync)
            result = invoke_tool_sync("business_validation", {
                "translation_result": tr_dict,
                "validation_result": val_by_file.get(filename),
            })
            all_results.append(result)

            # Log summary
            status = result.get("validation_status", "?")
            n_errors = result.get("total_errors", 0)
            n_warnings = result.get("total_warnings", 0)
            matched = result.get("matched_line_items", 0)
            status_indicator = "[OK]" if status == "matched" else (
                "[WARN]" if n_errors == 0 else "[ERR]"
            )

            logger.info(
                "%s %s - %s | Matched: %d | "
                "Errors: %d | Warnings: %d",
                status_indicator, filename, status.upper(),
                matched, n_errors, n_warnings,
            )

        except Exception as exc:
            err_msg = f"Business validation failed for {filename}: {exc}"
            errors.append(err_msg)
            logger.error("ERROR: %s", err_msg)

    logger.info(
        "Business validation complete - %d file(s), %d error(s)",
        len(all_results), len(errors),
    )

    return {
        "business_validation_results": all_results,
        "errors": errors,
        "status": "validated",
    }


def summarize_business_validation_node(
    state: BusinessValidationState,
) -> dict[str, Any]:
    """Aggregate business validation results and log a summary."""
    results = state.get("business_validation_results", [])
    errors = list(state.get("errors", []))

    matched = 0
    discrepancies = 0
    po_not_found = 0
    skipped = 0

    for r in results:
        status = r.get("validation_status", "skipped")
        filename = r.get("filename", "?")

        if status == "matched":
            matched += 1
        elif status == "discrepancies_found":
            discrepancies += 1
            # Log the discrepancies
            for d in r.get("discrepancies", []):
                sev = "[ERR]" if d.get("severity") == "error" else "[WARN]"
                logger.warning(
                    "%s %s: %s", sev, filename, d.get("message", "?")
                )
        elif status == "po_not_found":
            po_not_found += 1
        else:
            skipped += 1

    logger.info(
        "Business validation summary: "
        "%d matched, %d with discrepancies, "
        "%d PO-not-found, %d skipped",
        matched, discrepancies, po_not_found, skipped,
    )

    return {"status": "completed", "errors": errors}


# 
# LangGraph — Graph builder
# 
def build_business_validation_graph():
    """Build and compile the Business Validation LangGraph."""
    workflow = StateGraph(BusinessValidationState)

    workflow.add_node(
        "business_validate_all", business_validate_all_node
    )
    workflow.add_node(
        "summarize_business_validation",
        summarize_business_validation_node,
    )

    workflow.set_entry_point("business_validate_all")
    workflow.add_edge(
        "business_validate_all", "summarize_business_validation"
    )
    workflow.add_edge("summarize_business_validation", END)

    return workflow.compile()


# 
# Public API — BusinessValidationAgent
# 
class BusinessValidationAgent:
    """Orchestrates the Business Validation LangGraph pipeline."""

    def __init__(self) -> None:
        self.graph = build_business_validation_graph()

    @observe(name="agent-business-validation")
    def run(
        self,
        translation_results: list[TranslationResult],
        validation_results: list[ValidationResult] | None = None,
        callbacks: list | None = None,
    ) -> list[BusinessValidationResult]:
        """Cross-reference all invoices against ERP; returns list[BusinessValidationResult]."""
        logger.info("Starting Business Validation Agent ...")

        if not translation_results:
            logger.info(
                "No translation results to validate - skipping."
            )
            return []

        initial_state: BusinessValidationState = {
            "translation_results": [
                tr.model_dump() for tr in translation_results
            ],
            "validation_results": [
                vr.model_dump() for vr in (validation_results or [])
            ],
            "business_validation_results": [],
            "errors": [],
            "status": "idle",
        }

        cfg = {"callbacks": callbacks} if callbacks else {}
        final_state = self.graph.invoke(initial_state, config=cfg)

        raw_results = final_state.get(
            "business_validation_results", []
        )
        return [BusinessValidationResult(**r) for r in raw_results]


# 
# CLI entry-point
# 
if __name__ == "__main__":
    from backned.config.logging_setup import setup_logging
    setup_logging()

    # 1) Monitor — scan inbox
    from backned.agents.invoice_monitor_agent import InvoiceMonitorAgent

    monitor = InvoiceMonitorAgent()
    monitor.reset_processed_state()
    monitor_result = monitor.run()

    # 2) Extract — harvest invoice data
    from backned.agents.extractor_agent import ExtractorAgent

    extractor = ExtractorAgent()
    extraction_results = extractor.run(monitor_result)

    # 3) Translate — convert to English
    from backned.agents.translation_agent import TranslationAgent

    translator = TranslationAgent()
    translation_results = translator.run(extraction_results)

    # 4) Data validation — completeness check
    from backned.agents.validation_agent import ValidationAgent

    validator = ValidationAgent()
    validation_results = validator.run(translation_results)

    # 5) Business validation — cross-reference with ERP
    biz_validator = BusinessValidationAgent()
    biz_results = biz_validator.run(
        translation_results, validation_results
    )

    #  Pretty-print results 
    print("\n"+ "="* 80)
    print("BUSINESS VALIDATION AGENT — RESULTS")
    print("="* 80)
    print(f"Total invoices validated: {len(biz_results)}")

    n_matched = sum(
        1 for v in biz_results if v.validation_status == "matched"
    )
    n_disc = sum(
        1 for v in biz_results
        if v.validation_status == "discrepancies_found"
    )
    n_no_po = sum(
        1 for v in biz_results if v.validation_status == "po_not_found"
    )
    n_skip = sum(
        1 for v in biz_results if v.validation_status == "skipped"
    )
    print(
        f"Matched: {n_matched}  |  "
        f"Discrepancies: {n_disc}  |  "
        f"PO-Not-Found: {n_no_po}  |  "
        f"Skipped: {n_skip}"
    )

    for idx, bv in enumerate(biz_results, 1):
        status_emoji = {
            "matched": "",
            "discrepancies_found": "",
            "po_not_found": "",
            "vendor_not_found": "",
            "skipped": "",
        }.get(bv.validation_status, "?")

        print(f"\n{'─' * 80}")
        print(
            f"{status_emoji} [{idx}] {bv.filename}  "
            f"(email: {bv.email_id})"
        )
        print(f"{'─' * 80}")
        print(f"  Status   : {bv.validation_status.upper()}")
        print(f"  Is Valid  : {'Yes'if bv.is_valid else 'No'}")
        print(f"  Errors   : {bv.total_errors}")
        print(f"  Warnings  : {bv.total_warnings}")
        print(
            f"     Matched Items  : {bv.matched_line_items}"
        )

        #  ERP Lookup Info 
        if bv.erp_lookup:
            lk = bv.erp_lookup
            print(f"\n  ERP LOOKUP:")
            print(
                f"       PO Number      : "
                f"{lk.po_number or 'N/A'} "
                f"({' found' if lk.po_found else ' not found'})"
            )
            print(
                f"       Vendor ID      : "
                f"{lk.vendor_id or 'N/A'} "
                f"({' found' if lk.vendor_found else ' not found'})"
            )
            if lk.erp_vendor_name:
                print(
                    f"       ERP Vendor     : {lk.erp_vendor_name}"
                )
            print(
                f"       ERP Currency   : {lk.erp_currency or 'N/A'}"
            )
            print(
                f"       ERP Line Items : {lk.erp_line_items_count}"
            )
            print(
                f"       Inv Line Items : "
                f"{lk.invoice_line_items_count}"
            )

        #  Discrepancies 
        if bv.discrepancies:
            print(
                f"\n      DISCREPANCIES ({len(bv.discrepancies)}):"
            )
            for d in bv.discrepancies:
                sev = "" if d.severity == "error" else ""
                row = (
                    f"[Row {d.line_index}]"
                    if d.line_index >= 0 else "[Header]"
                )
                print(
                    f"       {sev} {row} [{d.discrepancy_type}] "
                    f"{d.message}"
                )
                if d.invoice_value or d.erp_value:
                    print(
                        f"          Invoice: {d.invoice_value or '—'}  "
                        f"|  ERP: {d.erp_value or '—'}"
                    )
                if d.difference_pct is not None:
                    print(
                        f"          Difference: {d.difference_pct:.1f}%"
                    )
        else:
            print(
                f"\n      All line items match ERP records!"
            )

    print("\n"+ "="* 80)
    print("Business validation pipeline complete.")
    print("="* 80)
