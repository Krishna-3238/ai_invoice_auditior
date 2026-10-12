"""Invoice-Watcher Tool — scans the email inbox for new, unprocessed invoice attachments.

Exposes a LangChain @tool for use by the Invoice Monitor Agent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from backned.models import (
    EmailMeta,
    InvoiceFile,
    MonitoredEmail,
    MonitorResult,
)
from backned.db import repository

logger = logging.getLogger(__name__)

# 
# Path constants  (relative to *this* file → backned/mcp/tools/)
# 
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # → backned/
INCOMING_DIR = BASE_DIR / "data" / "incoming"
EMAILS_DIR = INCOMING_DIR / "emails"
INVOICES_DIR = INCOMING_DIR / "invoices"
PROCESSED_STATE_FILE = BASE_DIR / "data" / "processed_emails.json"


# 
# Invoice-Watcher Tool  (LangChain @tool)
# 
@tool
def invoice_watcher_tool(scan_directory: str | None = None) -> dict:
    """Invoice-Watcher Tool — Monitors the incoming-emails directory for
    new, unprocessed invoice emails.

    Reads every `*.meta.json` in the emails folder, checks which ones
    have **not** been processed yet, verifies that the referenced
    attachment files exist in the invoices folder, and returns only the
    new emails.

    Args:
        scan_directory: Optional path override for the emails directory.

    Returns:
        dict serialised from MonitorResult (new_emails, counts, errors).
    """
    emails_dir = Path(scan_directory) if scan_directory else EMAILS_DIR

    #  Guard: directory must exist 
    if not emails_dir.exists():
        return MonitorResult(
            errors=[f"Emails directory not found: {emails_dir}"],
        ).model_dump()

    processed_ids = load_processed_emails()

    #  Scan for *.meta.json 
    meta_files = sorted(emails_dir.glob("*.meta.json"))
    total_scanned = len(meta_files)

    new_emails: list[MonitoredEmail] = []
    errors: list[str] = []
    referenced_files: set[Path] = set()

    for meta_file in meta_files:
        # Derive a stable ID from the filename:
        #   INV_EN_001.meta.json  →  INV_EN_001
        email_id = meta_file.name.replace(".meta.json", "")

        if email_id in processed_ids:
            continue  # already handled

        try:
            raw = json.loads(meta_file.read_text(encoding="utf-8"))

            email_meta = EmailMeta(
                email_id=email_id,
                sender=raw["sender"],
                subject=raw["subject"],
                received_timestamp=raw["received_timestamp"],
                language=raw.get("language", "unknown"),
                attachments=raw.get("attachments", []),
            )

            #  Resolve attachment files 
            invoice_files: list[InvoiceFile] = []
            for att_name in email_meta.attachments:
                att_path = INVOICES_DIR / att_name

                # Fallback: meta may reference a different extension
                # (e.g. .docx) than what's on disk (.pdf).
                if not att_path.exists():
                    base = Path(att_name).stem
                    candidates = list(INVOICES_DIR.glob(f"{base}.*"))
                    if candidates:
                        att_path = candidates[0]
                        att_name = att_path.name
                    else:
                        errors.append(
                            f"Attachment not found for {email_id}: {att_name}"
                        )
                        continue

                invoice_files.append(
                    InvoiceFile(
                        filename=att_name,
                        file_path=str(att_path.resolve()),
                        file_extension=att_path.suffix.lower(),
                        file_size_bytes=att_path.stat().st_size,
                    )
                )
                referenced_files.add(att_path.resolve())

                # Content checksum is authoritative: renaming an invoice or
                # receiving it in another email must not trigger a second audit.
                if repository.is_audited(repository.checksum_file(att_path)):
                    logger.info("Skipping previously audited content: %s", att_name)
                    invoice_files.pop()
                    continue
                repository.record_discovered(att_path)

            new_emails.append(
                MonitoredEmail(
                    email_meta=email_meta,
                    meta_file_path=str(meta_file.resolve()),
                    invoice_files=invoice_files,
                    status="new",
                )
            )

        except Exception as exc:
            errors.append(f"Error processing {meta_file.name}: {exc}")
            logger.error("Failed to process %s: %s", meta_file, exc)

    # Also watch invoice files placed directly in the mounted directory.
    # Sidecar-referenced files above are not duplicated here.
    for path in sorted(INVOICES_DIR.iterdir()) if INVOICES_DIR.exists() else []:
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".docx", ".png"}:
            continue
        resolved = path.resolve()
        if resolved in referenced_files:
            continue
        try:
            checksum = repository.checksum_file(resolved)
            if repository.is_audited(checksum):
                continue
            repository.record_discovered(resolved)
            email_id = f"file-{checksum[:16]}"
            meta = EmailMeta(
                email_id=email_id, sender="folder-monitor",
                subject=f"Invoice file {path.name}",
                received_timestamp=datetime.now(timezone.utc),
                language="unknown", attachments=[path.name],
            )
            new_emails.append(MonitoredEmail(
                email_meta=meta, meta_file_path="",
                invoice_files=[InvoiceFile(
                    filename=path.name, file_path=str(resolved),
                    file_extension=path.suffix.lower(), file_size_bytes=path.stat().st_size,
                )], status="new",
            ))
        except Exception as exc:
            errors.append(f"Could not record incoming file {path.name}: {exc}")

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
def load_processed_emails() -> set[str]:
    """Return the set of email IDs that have already been processed."""
    if PROCESSED_STATE_FILE.exists():
        with open(PROCESSED_STATE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return set(data.get("processed_ids", []))
    return set()


def save_processed_emails(processed_ids: set[str]) -> None:
    """Persist the set of processed email IDs to disk."""
    PROCESSED_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "processed_ids": sorted(processed_ids),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
            fh,
            indent=2,
        )


def mark_as_processed(email_ids: list[str]) -> None:
    """Add *email_ids* to the processed-state file."""
    current = load_processed_emails()
    current.update(email_ids)
    save_processed_emails(current)
