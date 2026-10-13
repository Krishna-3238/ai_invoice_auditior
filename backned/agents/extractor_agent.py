"""Extractor Agent — extracts structured invoice data from PDF, DOCX, and image attachments.

LangGraph: extract_all  validate_extractions  END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backned.models import ExtractionResult, MonitorResult, MonitoredEmail

from backned.tool_dispatch import invoke_tool_sync


logger = logging.getLogger(__name__)


# 
# LangGraph — State Schema
# 
class ExtractorState(TypedDict):
    """State flowing through the Extractor graph."""
    monitored_emails: list[dict]       # list of MonitoredEmail dicts
    extraction_results: list[dict]     # list of ExtractionResult dicts
    errors: list[str]
    status: str  # idle | extracting | extracted | completed | error


# 
# LangGraph — Node functions
# 
def extract_all_node(state: ExtractorState) -> dict[str, Any]:
    """Iterate over every monitored email and invoke the Data Harvester
    Tool on each attachment."""
    logger.info("Extractor Agent - starting extraction ...")

    emails: list[dict] = state.get("monitored_emails", [])
    all_results: list[dict] = []
    errors: list[str] = []

    for email_dict in emails:
        email_id = email_dict.get("email_meta", {}).get("email_id", "unknown")
        invoice_files = email_dict.get("invoice_files", [])

        if not invoice_files:
            errors.append(f"No attachments for email {email_id}")
            continue

        for inv_file in invoice_files:
            file_path = inv_file.get("file_path", "")
            filename = inv_file.get("filename", "")

            logger.info(
                "Extracting: %s (email: %s)", filename, email_id
            )

            try:
                result = invoke_tool_sync("data_harvester", {
                    "file_path": file_path,
                    "email_id": email_id,
                })
                all_results.append(result)

                # Log summary
                inv = result.get("extracted_invoice")
                n_items = len(inv.get("line_items", [])) if inv else 0
                inv_no = inv.get("invoice_no", "N/A") if inv else "N/A"
                logger.info(
                    "OK: %s | Invoice#: %s | Items: %d",
                    filename, inv_no, n_items,
                )

                if result.get("extraction_errors"):
                    errors.extend(result["extraction_errors"])

            except Exception as exc:
                err_msg = f"Extraction failed for {filename}: {exc}"
                errors.append(err_msg)
                logger.error("ERROR: %s", err_msg)

    logger.info(
        "Extraction complete - %d file(s) processed, %d error(s)",
        len(all_results), len(errors),
    )

    return {
        "extraction_results": all_results,
        "errors": errors,
        "status": "extracted",
    }


def validate_extractions_node(state: ExtractorState) -> dict[str, Any]:
    """Quick sanity check on extraction results.  Logs warnings for
    extractions with low confidence or missing critical fields."""
    results = state.get("extraction_results", [])
    errors = list(state.get("errors", []))

    no_invoice_count = 0

    for r in results:
        filename = r.get("filename", "?")
        inv = r.get("extracted_invoice")

        if not inv:
            no_invoice_count += 1
            logger.warning("WARN: %s: no structured data extracted", filename)
            continue

        # Check for missing critical fields
        missing = []
        if not inv.get("invoice_no"):
            missing.append("invoice_no")
        if not inv.get("po_number"):
            missing.append("po_number")
        if not inv.get("total_amount") and inv.get("total_amount") != 0:
            missing.append("total_amount")
        if not inv.get("line_items"):
            missing.append("line_items")

        if missing:
            logger.warning(
                "WARN: %s: missing fields: %s", filename, missing
            )

    if no_invoice_count:
        logger.info(
            "Validation summary: %d no-data",
            no_invoice_count,
        )

    return {"status": "completed", "errors": errors}


# 
# LangGraph — Graph builder
# 
def build_extractor_graph():
    """Build and compile the Extractor LangGraph."""
    workflow = StateGraph(ExtractorState)

    workflow.add_node("extract_all", extract_all_node)
    workflow.add_node("validate_extractions", validate_extractions_node)

    workflow.set_entry_point("extract_all")
    workflow.add_edge("extract_all", "validate_extractions")
    workflow.add_edge("validate_extractions", END)

    from backned.checkpointing import compile_graph
    return compile_graph(workflow)


# 
# Public API — ExtractorAgent
# 
class ExtractorAgent:
    """Orchestrates the Extractor LangGraph pipeline."""

    def __init__(self) -> None:
        self.graph = build_extractor_graph()

    def run(self, monitor_result: MonitorResult) -> list[ExtractionResult]:
        """Run extraction for all new emails; returns list[ExtractionResult]."""
        logger.info("Starting Extractor Agent …")

        if not monitor_result.new_emails:
            logger.info("No emails to extract — skipping.")
            return []

        initial_state: ExtractorState = {
            "monitored_emails": [
                em.model_dump() for em in monitor_result.new_emails
            ],
            "extraction_results": [],
            "errors": [],
            "status": "idle",
        }

        from backned.checkpointing import invoke_config
        final_state = self.graph.invoke(initial_state, config=invoke_config())

        raw_results = final_state.get("extraction_results", [])
        return [ExtractionResult(**r) for r in raw_results]


# 
# CLI entry-point
# 
if __name__ == "__main__":
    from backned.config.logging_setup import setup_logging
    setup_logging()

    # Run monitor first to get new emails
    from backned.agents.invoice_monitor_agent import InvoiceMonitorAgent

    monitor = InvoiceMonitorAgent()
    monitor.reset_processed_state()
    monitor_result = monitor.run()

    # Then run extractor
    extractor = ExtractorAgent()
    results = extractor.run(monitor_result)

    print("\n"+ "="* 80)
    print("EXTRACTOR AGENT — RESULTS")
    print("="* 80)
    print(f"Total files extracted: {len(results)}")

    for idx, r in enumerate(results, 1):
        print(f"\n{'─' * 80}")
        print(f"[{idx}] {r.filename} (email: {r.email_id})")
        print(f"{'─' * 80}")
        print(f"  Format  : {r.file_format}")
        print(f"  File Path : {r.file_path}")

        #  Raw Extracted Text 
        print(f"\n  {'─' * 50}")
        print(f"  RAW EXTRACTED TEXT ({len(r.raw_text)} chars):")
        print(f"  {'─' * 50}")
        if r.raw_text.strip():
            for line in r.raw_text.strip().splitlines():
                print(f"  {line}")
        else:
            print("  (no raw text extracted)")

        #  Structured Invoice Data 
        if r.extracted_invoice:
            inv = r.extracted_invoice
            print(f"\n  {'─' * 50}")
            print(f"  STRUCTURED INVOICE DATA:")
            print(f"  {'─' * 50}")
            print(f"  Invoice #  : {inv.invoice_no or 'N/A'}")
            print(f"  Invoice Date : {inv.invoice_date or 'N/A'}")
            print(f"  PO Number  : {inv.po_number or 'N/A'}")
            print(f"  Vendor ID  : {inv.vendor_id or 'N/A'}")
            print(f"  Vendor Name  : {inv.vendor_name or 'N/A'}")
            print(f"  Currency  : {inv.currency or 'N/A'}")
            print(f"  Subtotal  : {inv.subtotal}")
            print(f"  Tax Amount  : {inv.tax_amount}")
            print(f"  Total Amount : {inv.total_amount}")
            print(f"  Language  : {inv.detected_language or 'N/A'}")

            #  Line Items 
            print(f"\n LINE ITEMS ({len(inv.line_items)}):")
            if inv.line_items:
                print(f"  {'Code':<14} {'Description':<35} {'Qty':>6} {'Price':>10} {'Total':>10}")
                print(f"  {'─' * 14} {'─' * 35} {'─' * 6} {'─' * 10} {'─' * 10}")
                for li in inv.line_items:
                    code = li.item_code or "?"
                    desc = (li.description or "")[:35]
                    qty = f"{li.qty}" if li.qty is not None else "—"
                    price = f"{li.unit_price}" if li.unit_price is not None else "—"
                    total = f"{li.total}" if li.total is not None else "—"
                    print(f"  {code:<14} {desc:<35} {qty:>6} {price:>10} {total:>10}")
            else:
                print("  (no line items found)")

            #  Additional Fields 
            if inv.additional_fields:
                print(f"\n  ADDITIONAL FIELDS:")
                for k, v in inv.additional_fields.items():
                    print(f"  {k}: {v}")
        else:
            print(f"\n  No structured invoice data extracted.")

        #  Extraction Errors 
        if r.extraction_errors:
            print(f"\n  EXTRACTION ERRORS:")
            for err in r.extraction_errors:
                print(f"  • {err}")

    print("\n"+ "="* 80)
    print("Extraction pipeline complete.")
    print("="* 80)
