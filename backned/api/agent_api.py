"""HTTP service for invoice processing, audit review, corrections, and RAG."""
from __future__ import annotations

from typing import Any
from dataclasses import asdict, is_dataclass
import re
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backned.db import repository
from backned.config.settings import REDIS_URL, QDRANT_URL, CHECKPOINT_DATABASE_URL, WATCH_INTERVAL_SECONDS

app = FastAPI(title="Invoice Auditor Agent", version="1.0.0")
_pipeline_lock = threading.Lock()


def _plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_plain(x) for x in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if is_dataclass(value):
        return _plain(asdict(value))
    return value


@app.on_event("startup")
def startup():
    repository.initialize_schema()
    watcher = threading.Thread(target=_watch_incoming, name="invoice-folder-watcher", daemon=True)
    watcher.start()


def _watch_incoming():
    while True:
        time.sleep(WATCH_INTERVAL_SECONDS)
        try:
            run_pipeline()
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Folder scan failed")


@app.get("/health")
def health():
    try:
        with repository.connect() as conn:
            conn.execute("SELECT 1")
        import redis
        redis.Redis.from_url(REDIS_URL).ping()
        from qdrant_client import QdrantClient
        QdrantClient(url=QDRANT_URL).get_collections()
        return {"status": "healthy"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dependency not ready: {exc}") from exc


@app.post("/api/pipeline/run")
def run_pipeline():
    if not _pipeline_lock.acquire(blocking=False):
        return {"results": {}, "reports": [], "status": "already_running"}
    from fronted.pipeline_runner import PipelineRunner
    from backned.checkpointing import set_checkpoint_context, reset_checkpoint_context
    from langgraph.checkpoint.postgres import PostgresSaver
    thread_id = str(uuid.uuid4())
    try:
        with PostgresSaver.from_conn_string(CHECKPOINT_DATABASE_URL) as saver:
            saver.setup()
            tokens = set_checkpoint_context(saver, thread_id)
            try:
                result = PipelineRunner().run_full_pipeline()
            finally:
                reset_checkpoint_context(tokens)
    finally:
        _pipeline_lock.release()
    extractions = {x.filename: _plain(x) for x in result.get("extraction_results", [])}
    translations = {x.filename: _plain(x) for x in result.get("translation_results", [])}
    reports = result.get("reporting_summary")
    report_items = _plain(getattr(reports, "reports", [])) if reports else []
    completed_files = {item.get("filename") for item in report_items}
    if result.get("errors"):
        monitor = result.get("monitor_result")
        for item in getattr(monitor, "new_emails", []):
            for invoice_file in item.invoice_files:
                if invoice_file.filename not in completed_files:
                    repository.mark_failed(invoice_file.file_path, " | ".join(result["errors"]))
    for report in report_items:
        extraction = extractions.get(report.get("filename"), {})
        translation = translations.get(report.get("filename"), {})
        translated = translation.get("translated_invoice") or {}
        raw_invoice = extraction.get("extracted_invoice") or {}
        repository.save_audit(
            file_path=extraction.get("file_path") or report.get("file_path"),
            raw_text=extraction.get("raw_text", ""),
            translated_text=_translated_text(translated),
            source_language=translation.get("source_language"),
            extracted=extraction,
            report=report,
            extraction_method=extraction.get("file_format", "unknown"),
            thread_id=thread_id,
        )
    from backned.services.qdrant_store import index_all
    index_all()
    return {"results": _plain(result), "reports": report_items}


def _translated_text(invoice: dict) -> str:
    lines = [str(invoice.get("translated_vendor_name") or "")]
    lines.extend(str(row.get("translated_description") or "") for row in invoice.get("line_items", []))
    return "\n".join(v for v in lines if v)


@app.get("/api/invoices")
def invoices():
    return [_plain(row) for row in repository.list_audits()]


@app.get("/api/invoices/{invoice_id}")
def invoice(invoice_id: str):
    row = repository.get_audit(invoice_id)
    if not row:
        raise HTTPException(404, "Invoice not found")
    return _plain(row)


class Feedback(BaseModel):
    field_name: str
    corrected_value: str
    corrected_by: str = "auditor"


@app.post("/api/invoices/{invoice_id}/feedback")
def feedback(invoice_id: str, data: Feedback):
    try:
        feedback_id = repository.save_feedback(invoice_id, data.field_name, data.corrected_value, data.corrected_by)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"feedback_id": feedback_id, "stored": True, "revalidated": False}


@app.post("/api/invoices/{invoice_id}/revalidate")
def revalidate(invoice_id: str):
    from backned.models import ExtractionResult
    from backned.agents.translation_agent import TranslationAgent
    from backned.agents.validation_agent import ValidationAgent
    from backned.agents.business_validation_agent import BusinessValidationAgent
    from backned.agents.reporting_agent import ReportingAgent

    try:
        source, feedback_rows = repository.apply_feedback(invoice_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not feedback_rows:
        raise HTTPException(400, "No pending corrections")
    extraction = ExtractionResult(**source)
    translated = TranslationAgent().run([extraction])
    validation = ValidationAgent().run(translated)
    business = BusinessValidationAgent().run(translated, validation)
    reports = ReportingAgent().run(translated, validation, business)
    report = _plain(reports.reports[0]) if reports.reports else {}
    repository.update_audit_report(invoice_id, report)
    repository.mark_feedback_revalidated([x["id"] for x in feedback_rows], report)
    return {"invoice_id": invoice_id, "report": report, "revalidated": True}


@app.get("/api/query")
def query(q: str, invoice_id: str | None = None):
    # Structured lookups use PostgreSQL; open-ended questions use Qdrant RAG.
    structured_intent = any(term in q.lower() for term in ("report", "details", "status", "recommendation", "total amount", "invoice date"))
    if invoice_id and structured_intent:
        row = repository.get_audit(invoice_id)
        if not row:
            raise HTTPException(404, "Invoice not found")
        return {"answer": row["report_json"], "sources": [invoice_id], "mode": "postgres"}
    if any(term in q.lower() for term in ("how many", "count", "number of invoices")):
        with repository.connect() as conn:
            counts = conn.execute("SELECT recommendation,count(*) AS count FROM audit.invoice_audit WHERE recommendation <> 'pending' GROUP BY recommendation ORDER BY recommendation").fetchall()
            total = sum(int(row["count"]) for row in counts)
        return {"answer": {"total_processed": total, "by_recommendation": counts}, "sources": ["audit.invoice_audit"], "mode": "postgres"}
    match = re.search(r"\b(INV[-_A-Z0-9]+)\b", q, re.IGNORECASE)
    if match:
        with repository.connect() as conn:
            row = conn.execute("SELECT invoice_id,report_json FROM audit.invoice_audit WHERE upper(invoice_number)=upper(%s) OR upper(file_path) LIKE %s ORDER BY processed_at DESC LIMIT 1",
                               (match.group(1), f"%{match.group(1)}%" )).fetchone()
        if row and structured_intent:
            return {"answer": row["report_json"], "sources": [str(row["invoice_id"])], "mode": "postgres"}
        if row and not invoice_id:
            invoice_id = str(row["invoice_id"])
    import redis
    cache = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    cache_key = "llm:" + __import__("hashlib").sha256(f"{q}|{invoice_id or ''}".encode()).hexdigest()
    cached = cache.get(cache_key)
    if cached:
        import json
        return json.loads(cached)
    from backned.services.rag_pipeline import get_rag_pipeline
    result = get_rag_pipeline().query(q, invoice_id=invoice_id)
    import json
    response = {"answer": result.answer, "sources": result.sources_used, "scores": result.reflection, "mode": "qdrant-rag"}
    cache.setex(cache_key, 3600, json.dumps(response))
    return response
