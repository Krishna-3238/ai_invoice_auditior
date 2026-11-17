"""
Monitor Agent for AI Invoice Auditor
=====================================
Persona: System Watchdog
Role: Watch for new invoices arriving in the incoming directory

This agent monitors the ./data/incoming/ folder for new invoice files
(PDF, DOCX, PNG, JPG) and triggers the LangGraph workflow when detected.
Uses the Watchdog library for efficient file system event handling.
"""

import time
import sys
import os
import imaplib
import email
import re
import threading
from email.header import decode_header, make_header
from pathlib import Path
from email.message import Message
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.logging_config import get_logger, log_workflow_event
from workflows.invoice_auditor_workflow import run_workflow
from integrations.storage import RedisCache

logger = get_logger("MonitorAgent")

WATCH_DIRECTORY = "./data/incoming"
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class EmailInvoiceMonitor:
    """Poll an IMAP inbox, persist invoice attachments, and submit them to the workflow."""

    def __init__(self):
        self.host = os.getenv("IMAP_HOST", "")
        self.username = os.getenv("IMAP_USERNAME", "")
        self.password = os.getenv("IMAP_PASSWORD", "")
        self.mailbox = os.getenv("IMAP_MAILBOX", "INBOX")
        self.interval = int(os.getenv("IMAP_POLL_SECONDS", "60"))
        self.cache = RedisCache()
        self.uid_key = f"invoice-auditor:imap:last-uid:{self.username}:{self.mailbox}"

    @property
    def configured(self):
        return bool(self.host and self.username and self.password)

    @staticmethod
    def _decoded_filename(part: Message) -> str:
        value = part.get_filename() or "invoice-attachment"
        return str(make_header(decode_header(value)))

    def poll_once(self) -> int:
        if not self.configured:
            return 0

        saved = 0
        with imaplib.IMAP4_SSL(
            self.host, int(os.getenv("IMAP_PORT", "993")),
            timeout=int(os.getenv("IMAP_TIMEOUT_SECONDS", "30")),
        ) as client:
            client.login(self.username, self.password)
            status, _ = client.select(self.mailbox, readonly=False)
            if status != "OK":
                raise RuntimeError(f"Could not select IMAP mailbox {self.mailbox}")
            status, data = client.uid("search", None, "UNSEEN")
            if status != "OK":
                raise RuntimeError("IMAP UID search failed")

            for uid_bytes in (data[0] or b"").split():
                uid = int(uid_bytes)
                status, message_data = client.uid("fetch", str(uid), "(RFC822)")
                if status != "OK":
                    logger.warning("Could not fetch email UID %s", uid)
                    continue
                raw_message = next(
                    (part[1] for part in message_data if isinstance(part, tuple)), None
                )
                if not raw_message:
                    continue
                message = email.message_from_bytes(raw_message)
                message_id = message.get("Message-ID", f"uid-{uid}")
                message_key = re.sub(r"[^A-Za-z0-9_-]", "_", message_id)[:100]
                message_dir = Path(WATCH_DIRECTORY) / "email" / message_key
                processed_any = False
                attachments_found = 0
                all_attachments_succeeded = True

                for part in message.walk():
                    if part.is_multipart():
                        continue
                    filename = self._decoded_filename(part)
                    extension = Path(filename).suffix.lower()
                    if extension not in SUPPORTED_EXTENSIONS:
                        continue
                    attachments_found += 1
                    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
                    message_dir.mkdir(parents=True, exist_ok=True)
                    target = message_dir / safe_name
                    if not target.exists():
                        payload = part.get_payload(decode=True)
                        if not payload:
                            continue
                        target.write_bytes(payload)
                    logger.info("Invoice attachment received from email UID %s: %s", uid, target)
                    result = InvoiceHandler().trigger_workflow(str(target), source="email")
                    if result and not result.get("error"):
                        processed_any = True
                        saved += 1
                    else:
                        all_attachments_succeeded = False
                        logger.warning("Invoice attachment workflow failed for %s", target)

                # Advance over emails without supported attachments, or fully processed mail.
                # Failed messages remain eligible for retry on the next poll.
                if all_attachments_succeeded and (processed_any or attachments_found == 0):
                    client.uid("store", str(uid), "+FLAGS", "(\\Seen)")
                else:
                    # Leave the message unread so it can be retried on the next poll.
                    continue

        return saved

    def run_forever(self):
        while True:
            try:
                count = self.poll_once()
                if count:
                    logger.info("Processed %s invoice attachment(s) from email", count)
            except Exception as exc:
                logger.exception("Email invoice polling failed: %s", exc)
            time.sleep(self.interval)


class InvoiceHandler(FileSystemEventHandler):
    """Handles file system events for new invoices."""

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        file_name = os.path.basename(file_path)

        if file_name.lower().endswith(SUPPORTED_EXTENSIONS):
            logger.info(f"New invoice detected: {file_name}")
            self.trigger_workflow(file_path)
        else:
            logger.warning(f"Unsupported file format detected: {file_name}")

    def trigger_workflow(self, file_path, source="folder_watch"):
        """
        Passes the file path to the LangGraph workflow.
        Invokes the full invoice auditor pipeline.
        """
        logger.info(f"Initiating extraction workflow for: {file_path}")
        log_workflow_event(
            "monitor_detected",
            os.path.basename(file_path),
            {"source": source, "path": file_path},
            logger,
        )

        try:
            result = run_workflow(file_path)

            invoice_id = result.get("invoice_data", {}).get("invoice_no", "UNKNOWN")
            recommendation = result.get("recommendation", "UNKNOWN")

            logger.info(f"Workflow completed for {file_path}")
            logger.info(f"   Invoice: {invoice_id}")
            logger.info(f"   Recommendation: {recommendation}")

            log_workflow_event(
                "monitor_completed",
                invoice_id,
                {
                    "recommendation": recommendation,
                    "issues": result.get("validation_summary", {}).get(
                        "total_issues", 0
                    ),
                },
                logger,
            )

            return result

        except Exception as e:
            logger.error(f"Workflow failed for {file_path}: {str(e)}")
            log_workflow_event(
                "monitor_error", os.path.basename(file_path), {"error": str(e)}, logger
            )
            return None


if __name__ == "__main__":
    if not os.path.exists(WATCH_DIRECTORY):
        os.makedirs(WATCH_DIRECTORY)
        logger.info(f"Created directory: {WATCH_DIRECTORY}")

    event_handler = InvoiceHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)

    email_monitor = EmailInvoiceMonitor()
    if email_monitor.configured:
        threading.Thread(target=email_monitor.run_forever, daemon=True).start()
        logger.info("IMAP inbox monitoring enabled for %s", email_monitor.mailbox)
    else:
        logger.info("IMAP inbox monitoring disabled; configure IMAP_HOST, IMAP_USERNAME, and IMAP_PASSWORD")

    logger.info(f"Monitoring folder: {WATCH_DIRECTORY}")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Monitor stopped by user.")
    observer.join()
