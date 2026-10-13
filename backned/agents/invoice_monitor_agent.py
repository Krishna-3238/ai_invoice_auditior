"""Invoice Monitor Agent — scans the email inbox for new, unprocessed invoices.

LangGraph: scan_inbox  evaluate_results  mark_processed  END
                                               END
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backned.models import MonitorResult


from backned.tool_dispatch import invoke_tool_sync
from backned.checkpointing import compile_graph, invoke_config

#  Helper functions & constants from tool module 
from backned.tools.invoice_watcher_tool import (
    EMAILS_DIR,
)

logger = logging.getLogger(__name__)


# 
# LangGraph — State Schema
# 
class MonitorState(TypedDict):
    """Typed state flowing through the Invoice Monitor graph."""
    scan_requested: bool
    monitor_result: dict | None
    new_email_ids: list[str]
    errors: list[str]
    status: str  # idle | scanning | scanned | completed* | error


# 
# LangGraph — Node functions
# 
def scan_inbox_node(state: MonitorState) -> dict[str, Any]:
    """Invoke the Invoice-Watcher Tool to scan for new emails."""
    logger.info("Invoice Monitor Agent - scanning inbox ...")
    try:
        result: dict = invoke_tool_sync(
            "invoice_watcher", {"scan_directory": str(EMAILS_DIR)}
        )
        new_ids = [
            e["email_meta"]["email_id"] for e in result.get("new_emails", [])
        ]
        logger.info(
            "Scan complete - %d scanned, %d already processed, %d new",
            result["total_scanned"],
            result["already_processed"],
            result["newly_found"],
        )
        return {
            "monitor_result": result,
            "new_email_ids": new_ids,
            "errors": result.get("errors", []),
            "status": "scanned",
        }
    except Exception as exc:
        logger.error("Scan failed: %s", exc)
        return {
            "monitor_result": None,
            "new_email_ids": [],
            "errors": [str(exc)],
            "status": "error",
        }


def evaluate_results_node(state: MonitorState) -> dict[str, Any]:
    """Log what was found and decide the next step."""
    result = state.get("monitor_result")

    if not result or result.get("newly_found", 0) == 0:
        logger.info("DONE: No new emails to process.")
        return {"status": "completed_no_new"}

    logger.info("Found %d new email(s):", result["newly_found"])
    for email in result["new_emails"]:
        meta = email["email_meta"]
        files = [f["filename"] for f in email["invoice_files"]]
        logger.info(
            "-> %s | From: %s | Lang: %s | Files: %s",
            meta["email_id"],
            meta["sender"],
            meta["language"],
            files,
        )
    return {"status": "completed_with_new"}


def mark_processed_node(state: MonitorState) -> dict[str, Any]:
    """Finish scanning; PostgreSQL content checksums track processed files."""
    return {"status": "completed"}


# 
# LangGraph — Routing
# 
def _should_mark_processed(state: MonitorState) -> str:
    """Conditional edge after evaluate_results."""
    if state.get("status") == "completed_with_new":
        return "mark_processed"
    return "end"


# 
# LangGraph — Graph builder
# 
def build_monitor_graph():
    """Build and compile the Invoice Monitor LangGraph."""
    workflow = StateGraph(MonitorState)

    # Nodes
    workflow.add_node("scan_inbox", scan_inbox_node)
    workflow.add_node("evaluate_results", evaluate_results_node)
    workflow.add_node("mark_processed", mark_processed_node)

    # Edges
    workflow.set_entry_point("scan_inbox")
    workflow.add_edge("scan_inbox", "evaluate_results")
    workflow.add_conditional_edges(
        "evaluate_results",
        _should_mark_processed,
        {"mark_processed": "mark_processed", "end": END},
    )
    workflow.add_edge("mark_processed", END)

    return compile_graph(workflow)


# 
# Public API — InvoiceMonitorAgent
# 
class InvoiceMonitorAgent:
    """Orchestrates the Invoice Monitor LangGraph pipeline."""

    def __init__(self) -> None:
        self.graph = build_monitor_graph()

    #  Run the full pipeline 
    def run(self) -> MonitorResult:
        """Execute the monitoring pipeline and return a MonitorResult."""
        logger.info("Starting Invoice Monitor Agent …")

        initial_state: MonitorState = {
            "scan_requested": True,
            "monitor_result": None,
            "new_email_ids": [],
            "errors": [],
            "status": "idle",
        }

        final_state = self.graph.invoke(initial_state, config=invoke_config())

        raw = final_state.get("monitor_result")
        if raw:
            return MonitorResult(**raw)

        return MonitorResult(errors=final_state.get("errors", []))

    #  Helpers 
    def reset_processed_state(self) -> None:
        """Delete the processed-state file; all emails will be re-processed next run."""
        if False:
            PROCESSED_STATE_FILE.unlink()
            logger.info("Processed state reset — all emails will be re-scanned.")

    @staticmethod
    def get_processed_ids() -> set[str]:
        """Compatibility method; PostgreSQL checksums own deduplication."""
        return set()


# 
# CLI entry-point
# 
if __name__ == "__main__":
    from backned.config.logging_setup import setup_logging
    setup_logging()

    agent = InvoiceMonitorAgent()
    result = agent.run()

    print("\n"+ "="* 60)
    print("INVOICE MONITOR AGENT — RESULTS")
    print("="* 60)
    print(f"Total emails scanned : {result.total_scanned}")
    print(f"Already processed : {result.already_processed}")
    print(f"Newly found   : {result.newly_found}")

    if result.new_emails:
        print("\n New Emails:")
        for em in result.new_emails:
            m = em.email_meta
            print(f"• {m.email_id}")
            print(f"  From : {m.sender}")
            print(f"  Subject : {m.subject}")
            print(f"  Language: {m.language}")
            print(f"  Files : {[f.filename for f in em.invoice_files]}")

    if result.errors:
        print(f"\n Errors: {result.errors}")

    print("="* 60)
