"""Reporting Agent — generates HTML audit reports for every processed invoice.

LangGraph: generate_reports  summarize_reports  END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backned.models.translation import TranslationResult
from backned.models.validation import ValidationResult
from backned.models.business_validation import BusinessValidationResult
from backned.models.reporting import InvoiceReport, ReportingSummary

#  MCP client for tool access 
from backned.tool_dispatch import invoke_tool_sync

#  LangFuse observability 
from backned.config.observability import observe, langfuse_context

logger = logging.getLogger(__name__)


# 
# LangGraph — State Schema
# 
class ReportingState(TypedDict):
    """State flowing through the Reporting graph."""
    translation_results: list[dict]
    validation_results: list[dict]
    business_validation_results: list[dict]
    reports: list[dict]
    errors: list[str]
    status: str


# 
# LangGraph — Node functions
# 
def generate_reports_node(state: ReportingState) -> dict[str, Any]:
    """Generate an HTML report for each invoice by correlating all
    pipeline stage results."""
    logger.info("Reporting Agent - generating reports ...")

    translations: list[dict] = state.get("translation_results", [])
    validations: list[dict] = state.get("validation_results", [])
    biz_validations: list[dict] = state.get(
        "business_validation_results", []
    )

    # Build lookup maps by filename
    val_by_file: dict[str, dict] = {}
    for vr in validations:
        val_by_file[vr.get("filename", "")] = vr

    bv_by_file: dict[str, dict] = {}
    for bvr in biz_validations:
        bv_by_file[bvr.get("filename", "")] = bvr

    all_reports: list[dict] = []
    errors: list[str] = []

    for tr_dict in translations:
        filename = tr_dict.get("filename", "?")
        email_id = tr_dict.get("email_id", "unknown")

        logger.info("Generating report: %s (email: %s)", filename, email_id)

        try:
            # Invoke MCP tool (async-only tools are wrapped by invoke_tool_sync)
            report_dict = invoke_tool_sync("insight_reporter", {
                "translation_result": tr_dict,
                "validation_result": val_by_file.get(filename),
                "business_validation_result": bv_by_file.get(filename),
            })
            all_reports.append(report_dict)

            rec = report_dict.get("recommendation", "?")
            html_path = report_dict.get("report_html_path", "?")
            status_indicator = {
                "auto_approve": "[APPROVE]",
                "manual_review": "[REVIEW]",
                "reject": "[REJECT]",
            }.get(rec, "[?]")

            logger.info(
                "%s %s -> %s | Report: %s",
                status_indicator, filename, rec.upper(), html_path,
            )

        except Exception as exc:
            err_msg = f"Report generation failed for {filename}: {exc}"
            errors.append(err_msg)
            logger.error("ERROR: %s", err_msg)

    logger.info(
        "Report generation complete - %d report(s), %d error(s)",
        len(all_reports), len(errors),
    )

    return {
        "reports": all_reports,
        "errors": errors,
        "status": "generated",
    }


def summarize_reports_node(state: ReportingState) -> dict[str, Any]:
    """Log an aggregate summary of all reports."""
    reports = state.get("reports", [])
    errors = list(state.get("errors", []))

    auto = sum(1 for r in reports if r.get("recommendation") == "auto_approve")
    review = sum(1 for r in reports if r.get("recommendation") == "manual_review")
    reject = sum(1 for r in reports if r.get("recommendation") == "reject")

    logger.info(
        "Reporting summary: %d total - %d auto-approve, "
        "%d manual-review, %d reject",
        len(reports), auto, review, reject,
    )

    return {"status": "completed", "errors": errors}


# 
# LangGraph — Graph builder
# 
def build_reporting_graph():
    """Build and compile the Reporting LangGraph."""
    workflow = StateGraph(ReportingState)

    workflow.add_node("generate_reports", generate_reports_node)
    workflow.add_node("summarize_reports", summarize_reports_node)

    workflow.set_entry_point("generate_reports")
    workflow.add_edge("generate_reports", "summarize_reports")
    workflow.add_edge("summarize_reports", END)

    return workflow.compile()


# 
# Public API — ReportingAgent
# 
class ReportingAgent:
    """Orchestrates the Reporting LangGraph pipeline."""

    def __init__(self) -> None:
        self.graph = build_reporting_graph()

    @observe(name="agent-reporting")
    def run(
        self,
        translation_results: list[TranslationResult],
        validation_results: list[ValidationResult],
        business_validation_results: list[BusinessValidationResult],
        callbacks: list | None = None,
    ) -> ReportingSummary:
        """Generate HTML reports for all invoices; returns ReportingSummary."""
        logger.info("Starting Reporting Agent ...")

        if not translation_results:
            logger.info("No results to report - skipping.")
            return ReportingSummary()

        initial_state: ReportingState = {
            "translation_results": [
                tr.model_dump() for tr in translation_results
            ],
            "validation_results": [
                vr.model_dump() for vr in validation_results
            ],
            "business_validation_results": [
                bvr.model_dump() for bvr in business_validation_results
            ],
            "reports": [],
            "errors": [],
            "status": "idle",
        }

        cfg = {"callbacks": callbacks} if callbacks else {}
        final_state = self.graph.invoke(initial_state, config=cfg)

        raw_reports = final_state.get("reports", [])
        reports = [InvoiceReport(**r) for r in raw_reports]

        auto = sum(1 for r in reports if r.recommendation == "auto_approve")
        review = sum(1 for r in reports if r.recommendation == "manual_review")
        reject = sum(1 for r in reports if r.recommendation == "reject")

        # Determine output dir from first report
        output_dir = ""
        if reports and reports[0].report_html_path:
            from pathlib import Path
            output_dir = str(Path(reports[0].report_html_path).parent)

        return ReportingSummary(
            total_invoices=len(reports),
            auto_approved=auto,
            manual_review=review,
            rejected=reject,
            reports=reports,
            output_dir=output_dir,
        )


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

    # 4) Data Validation — check completeness
    from backned.agents.validation_agent import ValidationAgent

    validator = ValidationAgent()
    validation_results = validator.run(translation_results)

    # 5) Business Validation — cross-reference ERP
    from backned.agents.business_validation_agent import BusinessValidationAgent

    biz_validator = BusinessValidationAgent()
    business_results = biz_validator.run(translation_results, validation_results)

    # 6) Reporting — generate HTML reports
    reporter = ReportingAgent()
    summary = reporter.run(
        translation_results, validation_results, business_results
    )

    #  Pretty-print summary 
    print("\n"+ "="* 80)
    print("REPORTING AGENT — SUMMARY")
    print("="* 80)
    print(f"Total Invoices : {summary.total_invoices}")
    print(f"Auto-Approve : {summary.auto_approved}")
    print(f"Manual Review: {summary.manual_review}")
    print(f"Rejected  : {summary.rejected}")
    print(f"Output Dir  : {summary.output_dir}")

    for idx, r in enumerate(summary.reports, 1):
        rec_emoji = {
            "auto_approve": "",
            "manual_review": "",
            "reject": "",
        }.get(r.recommendation, "")

        print(f"\n{'─' * 80}")
        print(f"{rec_emoji} [{idx}] {r.filename} (email: {r.email_id})")
        print(f"{'─' * 80}")
        print(f"  Invoice #  : {r.invoice_no or 'N/A'}")
        print(f"  PO #   : {r.po_number or 'N/A'}")
        print(f"  Vendor   : {r.vendor_id or 'N/A'} / {r.vendor_name or 'N/A'}")
        print(f"  Total   : {r.currency or ''} {r.total_amount}")
        print(f"  Language  : {r.source_language or 'en'} "
              f"({'translated' if r.was_translated else 'original'})")
        print(f"  Translation Confidence : {r.translation_confidence:.0%}")
        print(f"  Data Validation : {r.data_validation_status or 'N/A'} "
              f"(E:{r.data_validation_errors} W:{r.data_validation_warnings})")
        print(f"  Biz Validation : {r.business_validation_status or 'N/A'} "
              f"(E:{r.business_validation_errors} W:{r.business_validation_warnings} "
              f"Matched:{r.business_validation_matched})")
        print(f"  Recommendation : {r.recommendation.replace('_', ' ').upper()}")
        if r.recommendation_reasons:
            for reason in r.recommendation_reasons:
                print(f"  → {reason}")
        print(f"  Report File  : {r.report_html_path or 'N/A'}")

    print("\n"+ "="* 80)
    print("Reporting pipeline complete — HTML reports saved.")
    print("="* 80)
