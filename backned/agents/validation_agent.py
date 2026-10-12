"""Invoice Data Validation Agent — validates invoice completeness against rules.yaml.

LangGraph: validate_all  summarize_validation  END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backned.models.translation import TranslationResult
from backned.models.validation import ValidationResult

#  MCP client for tool access 
from backned.tool_dispatch import invoke_tool_sync

#  LangFuse observability 
from backned.config.observability import observe, langfuse_context

logger = logging.getLogger(__name__)


# 
# LangGraph — State Schema
# 
class ValidationState(TypedDict):
    """State flowing through the Validation graph."""
    translation_results: list[dict]    # list of TranslationResult dicts
    validation_results: list[dict]     # list of ValidationResult dicts
    errors: list[str]
    status: str  # idle | validating | validated | completed | error


# 
# LangGraph — Node functions
# 
def validate_all_node(state: ValidationState) -> dict[str, Any]:
    """Iterate over every translation result and invoke the Data
    Completeness Checker Tool on each."""
    logger.info("Validation Agent - starting data validation ...")

    translations: list[dict] = state.get("translation_results", [])
    all_results: list[dict] = []
    errors: list[str] = []

    for tr_dict in translations:
        filename = tr_dict.get("filename", "?")
        email_id = tr_dict.get("email_id", "unknown")

        logger.info(
            "Validating: %s (email: %s)", filename, email_id
        )

        try:
            # Invoke MCP tool (async-only tools are wrapped by invoke_tool_sync)
            result = invoke_tool_sync("data_completeness_checker", {
                "translation_result": tr_dict,
            })
            all_results.append(result)

            # Log summary
            status = result.get("validation_status", "?")
            n_errors = result.get("total_errors", 0)
            n_warnings = result.get("total_warnings", 0)
            status_indicator = "[OK]" if status == "passed" else (
                "[WARN]" if status == "passed_with_warnings" else "[ERR]"
            )

            logger.info(
                "%s %s - %s | Errors: %d | Warnings: %d",
                status_indicator, filename, status.upper(), n_errors, n_warnings,
            )

        except Exception as exc:
            err_msg = f"Validation failed for {filename}: {exc}"
            errors.append(err_msg)
            logger.error("ERROR: %s", err_msg)

    logger.info(
        "Validation complete - %d file(s) validated, %d error(s)",
        len(all_results), len(errors),
    )

    return {
        "validation_results": all_results,
        "errors": errors,
        "status": "validated",
    }


def summarize_validation_node(state: ValidationState) -> dict[str, Any]:
    """Aggregate validation results and log a summary.  Flag invoices
    that need manual review."""
    results = state.get("validation_results", [])
    errors = list(state.get("errors", []))

    passed = 0
    passed_with_warnings = 0
    failed = 0

    for r in results:
        filename = r.get("filename", "?")
        status = r.get("validation_status", "failed")

        if status == "passed":
            passed += 1
        elif status == "passed_with_warnings":
            passed_with_warnings += 1
            # Log the warnings for visibility
            for issue in r.get("header_issues", []):
                if issue.get("severity") == "warning":
                    logger.warning(
                        "WARN: %s: %s", filename, issue.get("message")
                    )
            for issue in r.get("line_item_issues", []):
                if issue.get("severity") == "warning":
                    logger.warning(
                        "WARN: %s: %s", filename, issue.get("message")
                    )
        else:
            failed += 1
            # Log the errors for visibility
            for issue in r.get("header_issues", []):
                if issue.get("severity") == "error":
                    logger.error(
                        "ERROR: %s: %s", filename, issue.get("message")
                    )
            for issue in r.get("line_item_issues", []):
                if issue.get("severity") == "error":
                    logger.error(
                        "ERROR: %s: %s", filename, issue.get("message")
                    )

    logger.info(
        "Validation summary: %d passed, %d with warnings, %d failed "
        "(total: %d)",
        passed, passed_with_warnings, failed, len(results),
    )

    return {"status": "completed", "errors": errors}


# 
# LangGraph — Graph builder
# 
def build_validation_graph():
    """Build and compile the Invoice Data Validation LangGraph."""
    workflow = StateGraph(ValidationState)

    workflow.add_node("validate_all", validate_all_node)
    workflow.add_node("summarize_validation", summarize_validation_node)

    workflow.set_entry_point("validate_all")
    workflow.add_edge("validate_all", "summarize_validation")
    workflow.add_edge("summarize_validation", END)

    return workflow.compile()


# 
# Public API — ValidationAgent
# 
class ValidationAgent:
    """Orchestrates the Invoice Data Validation LangGraph pipeline."""

    def __init__(self) -> None:
        self.graph = build_validation_graph()

    @observe(name="agent-validation")
    def run(
        self, translation_results: list[TranslationResult], callbacks: list | None = None
    ) -> list[ValidationResult]:
        """Validate all translation results; returns list[ValidationResult]."""
        logger.info("Starting Validation Agent ...")

        if not translation_results:
            logger.info("No translation results to validate - skipping.")
            return []

        initial_state: ValidationState = {
            "translation_results": [
                tr.model_dump() for tr in translation_results
            ],
            "validation_results": [],
            "errors": [],
            "status": "idle",
        }

        cfg = {"callbacks": callbacks} if callbacks else {}
        final_state = self.graph.invoke(initial_state, config=cfg)

        raw_results = final_state.get("validation_results", [])
        return [ValidationResult(**r) for r in raw_results]


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

    # 4) Validate — check completeness & accuracy
    validator = ValidationAgent()
    validation_results = validator.run(translation_results)

    #  Pretty-print results 
    print("\n"+ "="* 80)
    print("VALIDATION AGENT — RESULTS")
    print("="* 80)
    print(f"Total invoices validated: {len(validation_results)}")

    # Quick summary counters
    n_passed = sum(1 for v in validation_results if v.validation_status == "passed")
    n_warn = sum(1 for v in validation_results if v.validation_status == "passed_with_warnings")
    n_fail = sum(1 for v in validation_results if v.validation_status == "failed")
    print(f"Passed: {n_passed} |    Warnings: {n_warn} |   Failed: {n_fail}")

    for idx, v in enumerate(validation_results, 1):
        status_emoji = {
            "passed": "",
            "passed_with_warnings": "",
            "failed": "",
        }.get(v.validation_status, "?")

        print(f"\n{'─' * 80}")
        print(f"{status_emoji} [{idx}] {v.filename} (email: {v.email_id})")
        print(f"{'─' * 80}")
        print(f"  Status  : {v.validation_status.upper()}")
        print(f"  Is Valid  : {'Yes'if v.is_valid else 'No'}")
        print(f"  Errors  : {v.total_errors}")
        print(f"  Warnings  : {v.total_warnings}")
        print(f"  Language  : {v.source_language or 'N/A'}")
        print(f"  Rules Applied : {', '.join(v.rules_applied)}")

        #  Header Issues 
        if v.header_issues:
            print(f"\n  HEADER ISSUES ({len(v.header_issues)}):")
            for issue in v.header_issues:
                sev = "" if issue.severity == "error" else ""
                print(
                    f"       {sev} [{issue.issue_type}] {issue.field_name}: "
                    f"{issue.message}"
                )
                if issue.expected or issue.actual:
                    print(
                        f"          Expected: {issue.expected or '—'}  |  "
                        f"Actual: {issue.actual or '—'}"
                    )

        #  Line Item Issues 
        if v.line_item_issues:
            print(f"\n  LINE ITEM ISSUES ({len(v.line_item_issues)}):")
            for issue in v.line_item_issues:
                sev = "" if issue.severity == "error" else ""
                print(
                    f"       {sev} [Row {issue.line_index}] "
                    f"[{issue.issue_type}] {issue.field_name}: "
                    f"{issue.message}"
                )
                if issue.expected or issue.actual:
                    print(
                        f"          Expected: {issue.expected or '—'}  |  "
                        f"Actual: {issue.actual or '—'}"
                    )

        #  No issues 
        if not v.header_issues and not v.line_item_issues:
            print(f"\n  No validation issues — all checks passed!")

    print("\n"+ "="* 80)
    print("Validation pipeline complete.")
    print("="* 80)
