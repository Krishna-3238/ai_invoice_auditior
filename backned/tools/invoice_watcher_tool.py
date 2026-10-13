"""Invoice-Watcher Tool — scans the email inbox for new, unprocessed invoice attachments.

Exposes a LangChain @tool for use by the Invoice Monitor Agent.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import tool

from backned.models import (
    InvoiceFile,
    MonitoredEmail,
    MonitorResult,
)
from backned.db import repository

logger = logging.getLogger(__name__)

# 
# 
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # → backned/
from backned.config.settings import INCOMING_DIR
INVOICES_DIR = INCOMING_DIR
EMAILS_DIR = INCOMING_DIR
PROCESSED_STATE_FILE = BASE_DIR / "data" / "processed_emails.json"


# 
# Invoice-Watcher Tool  (LangChain @tool)
# 
@tool
def invoice_watcher_tool(scan_directory: str | None = None) -> dict:
    """Invoice-Watcher Tool — Monitors the incoming-emails directory for
    new, unprocessed invoice emails.

    Scans supported invoice files and skips checksums already audited in PostgreSQL.

    Args:
        scan_directory: Optional path override for the incoming invoices directory.

    Returns:
        dict serialised from MonitorResult (new_emails, counts, errors).
    """
    emails_dir = Path(scan_directory) if scan_directory else INVOICES_DIR

    #  Guard: directory must exist 
    if not emails_dir.exists():
        return MonitorResult(
            errors=[f"Incoming directory not found: {emails_dir}"],
        ).model_dump()

    invoice_paths = [p for p in sorted(emails_dir.iterdir()) if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".png"}]
    total_scanned = len(invoice_paths)

    new_emails: list[MonitoredEmail] = []
    errors: list[str] = []
    for path in invoice_paths:
        # Derive a stable ID from the filename:
        #   INV_EN_001.meta.json  →  INV_EN_001
        try:
            checksum = repository.checksum_file(path)
            if repository.is_audited(checksum):
                continue
            repository.record_discovered(path)
            invoice_file = InvoiceFile(filename=path.name, file_path=str(path.resolve()), file_extension=path.suffix.lower(), file_size_bytes=path.stat().st_size)
            new_emails.append(MonitoredEmail(email_meta={"email_id": f"file-{checksum[:16]}", "sender": "folder-monitor", "subject": path.name, "received_timestamp": path.stat().st_mtime, "language": "unknown", "attachments": [path.name]}, meta_file_path="", invoice_files=[invoice_file], status="new"))

        except Exception as exc:
            errors.append(f"Error processing {path.name}: {exc}")
            logger.error("Failed to process %s: %s", path, exc)


    result = MonitorResult(
        new_emails=new_emails,
        total_scanned=total_scanned,
        already_processed=total_scanned - len(new_emails) - len(
            [e for e in errors if "Error processing" in e]
        ),
        newly_found=len(new_emails),
        errors=errors,
    )
    return result.model_dump()


# 
# Processed-State Tracker helpers
# 
