"""Durable PostgreSQL system of record for invoice audits and feedback."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from backned.config.settings import DATABASE_URL


def connect():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def initialize_schema() -> None:
    sql = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")
    with connect() as conn:
        conn.execute(sql)


def mark_failed(file_path: str | Path, error: str) -> None:
    path = Path(file_path)
    checksum = checksum_file(path)
    record_discovered(path)
    with connect() as conn:
        conn.execute("UPDATE audit.invoice_audit SET validation_status='failed', recommendation='manual_review', processing_error=%s, processed_at=now() WHERE file_checksum=%s", (error[:4000], checksum))


def checksum_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_by_checksum(checksum: str) -> dict | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM audit.invoice_audit WHERE file_checksum=%s", (checksum,)).fetchone()


def is_audited(checksum: str) -> bool:
    """Discovered or failed files are retried until processing succeeds."""
    with connect() as conn:
        row = conn.execute(
            "SELECT validation_status FROM audit.invoice_audit WHERE file_checksum=%s",
            (checksum,),
        ).fetchone()
    return bool(row and row["validation_status"] not in {"discovered", "failed"})


def record_discovered(path: str | Path) -> str:
    path = Path(path)
    checksum = checksum_file(path)
    existing = find_by_checksum(checksum)
    if existing:
        return str(existing["invoice_id"])
    invoice_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute("INSERT INTO audit.invoice_audit(invoice_id,file_path,file_checksum,extraction_method,validation_status,recommendation) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(file_checksum) DO NOTHING",
                     (invoice_id,str(path),checksum,path.suffix.lower().lstrip("."),"discovered","pending"))
        row = conn.execute("SELECT invoice_id FROM audit.invoice_audit WHERE file_checksum=%s", (checksum,)).fetchone()
    return str(row["invoice_id"])


def save_audit(*, file_path: str, raw_text: str, translated_text: str,
               source_language: str | None, extracted: dict, report: dict,
               extraction_method: str, thread_id: str | None = None) -> str:
    invoice_id = str(uuid.uuid4())
    checksum = checksum_file(file_path)
    inv = (extracted or {}).get("extracted_invoice", extracted or {})
    recommendation = str(report.get("recommendation", "manual_review"))
    validation_status = str(report.get("data_validation_status") or "partial")
    invoice_date = _parse_date(report.get("invoice_date") or inv.get("invoice_date"))
    report_html_path = report.get("report_html_path")
    report_html = Path(report_html_path).read_text(encoding="utf-8") if report_html_path and Path(report_html_path).exists() else None
    with connect() as conn:
        conn.execute("""INSERT INTO audit.invoice_audit
          (invoice_id,file_path,file_checksum,extraction_method,raw_extracted_text,translated_text,
           source_language,translation_confidence,vendor_name,po_number,invoice_number,invoice_date,
           currency,total_amount,validation_status,recommendation,extracted_raw,report_json,report_html,thread_id)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
          ON CONFLICT(file_checksum) DO UPDATE SET
            raw_extracted_text=excluded.raw_extracted_text, translated_text=excluded.translated_text,
            source_language=excluded.source_language, translation_confidence=excluded.translation_confidence,
            vendor_name=excluded.vendor_name, po_number=excluded.po_number, invoice_number=excluded.invoice_number,
            invoice_date=excluded.invoice_date, currency=excluded.currency, total_amount=excluded.total_amount,
            validation_status=excluded.validation_status, recommendation=excluded.recommendation,
            extracted_raw=excluded.extracted_raw, report_json=excluded.report_json, report_html=excluded.report_html,
            processing_error=NULL, processed_at=now()""",
          (invoice_id,file_path,checksum,extraction_method,raw_text,translated_text,source_language,
           report.get("translation_confidence"),report.get("vendor_name") or inv.get("vendor_name"),
           report.get("po_number") or inv.get("po_number"),report.get("invoice_no") or inv.get("invoice_no"),
           invoice_date,report.get("currency") or inv.get("currency"),report.get("total_amount") or inv.get("total_amount"),
           validation_status,recommendation,json.dumps(extracted,default=str),json.dumps(report,default=str),report_html,thread_id))
        row = conn.execute("SELECT invoice_id FROM audit.invoice_audit WHERE file_checksum=%s", (checksum,)).fetchone()
        invoice_id = str(row["invoice_id"])
        conn.execute("DELETE FROM audit.invoice_line_items WHERE invoice_id=%s", (invoice_id,))
        conn.execute("DELETE FROM audit.discrepancies WHERE invoice_id=%s", (invoice_id,))
        for i, item in enumerate(inv.get("line_items", []), 1):
            conn.execute("""INSERT INTO audit.invoice_line_items
              (invoice_id,line_number,item_code,description,quantity,unit_price,line_total)
              VALUES (%s,%s,%s,%s,%s,%s,%s)""",
              (invoice_id,i,item.get("item_code"),item.get("description") or item.get("translated_description"),
               item.get("qty") or item.get("quantity"),item.get("unit_price"),item.get("total") or item.get("line_total")))
        for issue in report.get("business_discrepancies", []):
            conn.execute("""INSERT INTO audit.discrepancies
              (invoice_id,field_name,invoice_value,erp_value,deviation_pct,severity)
              VALUES (%s,%s,%s,%s,%s,%s)""",
              (invoice_id,issue.get("field_name"),str(issue.get("invoice_value", "")),str(issue.get("erp_value", "")),
               issue.get("difference_pct") or issue.get("deviation_pct"),issue.get("severity", "error")))
    return invoice_id


def _parse_date(value: Any):
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def list_audits() -> list[dict]:
    with connect() as conn:
        return conn.execute("SELECT invoice_id,file_path,file_checksum,raw_extracted_text,report_html,source_language,vendor_name,po_number,invoice_number,invoice_date,currency,total_amount,validation_status,recommendation,report_json,processed_at FROM audit.invoice_audit ORDER BY processed_at DESC").fetchall()


def get_audit(invoice_id: str) -> dict | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM audit.invoice_audit WHERE invoice_id=%s", (invoice_id,)).fetchone()


def save_feedback(invoice_id: str, field_name: str, corrected_value: str, corrected_by: str = "auditor") -> int:
    with connect() as conn:
        row = conn.execute("SELECT extracted_raw FROM audit.invoice_audit WHERE invoice_id=%s", (invoice_id,)).fetchone()
        if not row:
            raise LookupError("Invoice not found")
        data = row["extracted_raw"]
        current: Any = data
        for part in field_name.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                current = None
                break
        result = conn.execute("""INSERT INTO audit.human_feedback
          (invoice_id,field_name,original_value,corrected_value,corrected_by)
          VALUES (%s,%s,%s,%s,%s) RETURNING id""",
          (invoice_id,field_name,None if current is None else str(current),corrected_value,corrected_by)).fetchone()
        return int(result["id"])


def apply_feedback(invoice_id: str) -> tuple[dict, list[dict]]:
    with connect() as conn:
        audit = conn.execute("SELECT extracted_raw FROM audit.invoice_audit WHERE invoice_id=%s", (invoice_id,)).fetchone()
        if not audit:
            raise LookupError("Invoice not found")
        feedback = conn.execute("SELECT * FROM audit.human_feedback WHERE invoice_id=%s AND revalidated=FALSE ORDER BY corrected_at", (invoice_id,)).fetchall()
    data = audit["extracted_raw"]
    for item in feedback:
        parts = item["field_name"].split(".")
        target = data
        for part in parts[:-1]:
            if part.isdigit() and isinstance(target, list):
                target = target[int(part)]
            else:
                target = target.setdefault(part, {})
        value: Any = item["corrected_value"]
        try:
            value = float(value)
        except (ValueError, TypeError):
            pass
        if isinstance(target, list) and parts[-1].isdigit():
            target[int(parts[-1])] = value
        else:
            target[parts[-1]] = value
    return data, feedback


def mark_feedback_revalidated(ids: list[int], result: dict) -> None:
    with connect() as conn:
        conn.execute("UPDATE audit.human_feedback SET revalidated=TRUE,revalidation_result=%s WHERE id=ANY(%s)", (json.dumps(result,default=str),ids))


def update_audit_report(invoice_id: str, report: dict) -> None:
    report_html_path = report.get("report_html_path")
    report_html = Path(report_html_path).read_text(encoding="utf-8") if report_html_path and Path(report_html_path).exists() else None
    with connect() as conn:
        conn.execute("UPDATE audit.invoice_audit SET report_json=%s,report_html=COALESCE(%s,report_html),recommendation=%s,validation_status=%s,processed_at=now() WHERE invoice_id=%s",
                     (json.dumps(report,default=str),report_html,report.get("recommendation"),report.get("data_validation_status"),invoice_id))
        conn.execute("DELETE FROM audit.discrepancies WHERE invoice_id=%s", (invoice_id,))
        for issue in report.get("business_discrepancies", []):
            conn.execute("INSERT INTO audit.discrepancies(invoice_id,field_name,invoice_value,erp_value,deviation_pct,severity) VALUES (%s,%s,%s,%s,%s,%s)",
                         (invoice_id,issue.get("field_name"),str(issue.get("invoice_value", "")),str(issue.get("erp_value", "")),issue.get("difference_pct") or issue.get("deviation_pct"),issue.get("severity", "error")))
