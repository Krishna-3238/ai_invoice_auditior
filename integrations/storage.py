"""PostgreSQL, Qdrant, and Redis adapters used by the workflow and RAG stack.

PostgreSQL is required for durable audit and feedback storage. Qdrant and Redis
are configured through environment variables; FAISS remains a local vector fallback.
"""

import hashlib
import json
import logging
import os
from dateutil import parser as date_parser
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

logger = logging.getLogger("StorageIntegrations")


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class PostgresInvoiceStore:
    """Durable invoice audit and report storage using PostgreSQL JSONB."""

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv("DATABASE_URL", "")

    def _connect(self):
        if not self.dsn:
            raise RuntimeError("DATABASE_URL must be configured for durable audit storage")
        import psycopg
        return psycopg.connect(self.dsn)

    def ensure_schema(self):
        with self._connect() as conn:
            conn.execute("CREATE SCHEMA IF NOT EXISTS audit")
            conn.execute("""CREATE TABLE IF NOT EXISTS audit.invoice_audit (
                workflow_id TEXT PRIMARY KEY,
                invoice_no TEXT,
                vendor_id TEXT,
                invoice_date DATE,
                total_amount NUMERIC,
                currency TEXT,
                processing_status TEXT NOT NULL DEFAULT 'completed',
                error_type TEXT,
                error_message TEXT,
                invoice_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                report_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS audit.human_feedback (
                feedback_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                invoice_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                user_notes TEXT,
                original_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                corrected_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                changes_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS human_feedback_invoice_created_idx "
                         "ON audit.human_feedback (invoice_id, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS invoice_audit_created_idx "
                         "ON audit.invoice_audit (created_at DESC)")

    @property
    def enabled(self) -> bool:
        return bool(self.dsn)

    def save(self, workflow_id: str, invoice_data: Dict[str, Any], report: Dict[str, Any]):
        if not self.enabled:
            raise RuntimeError("DATABASE_URL must be configured for durable audit storage")
        from psycopg.types.json import Jsonb

        self.ensure_schema()
        raw_date = invoice_data.get("invoice_date")
        try:
            invoice_date = date_parser.parse(str(raw_date)).date().isoformat() if raw_date else None
        except (TypeError, ValueError, OverflowError):
            invoice_date = None
        raw_amount = invoice_data.get("total_amount")
        try:
            total_amount = Decimal(str(raw_amount).replace(",", "")) if raw_amount is not None else None
        except (InvalidOperation, ValueError):
            total_amount = None
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO audit.invoice_audit
                    (workflow_id, invoice_no, vendor_id, invoice_date, total_amount,
                     currency, processing_status, error_type, error_message,
                     invoice_payload, report_payload)
                    VALUES (%s, %s, %s, %s::date, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (workflow_id) DO UPDATE SET
                    invoice_no=EXCLUDED.invoice_no, vendor_id=EXCLUDED.vendor_id,
                    invoice_date=EXCLUDED.invoice_date, total_amount=EXCLUDED.total_amount,
                    currency=EXCLUDED.currency, processing_status=EXCLUDED.processing_status,
                    error_type=EXCLUDED.error_type, error_message=EXCLUDED.error_message,
                    invoice_payload=EXCLUDED.invoice_payload, report_payload=EXCLUDED.report_payload,
                    updated_at=NOW()""",
                (
                    workflow_id,
                    invoice_data.get("invoice_no"),
                    invoice_data.get("vendor_id"),
                    invoice_date,
                    total_amount,
                    invoice_data.get("currency"),
                    report.get("processing_status", "completed"),
                    report.get("error_type"),
                    report.get("error_message"),
                    Jsonb(invoice_data, dumps=lambda v: json.dumps(v, default=_json_default)),
                    Jsonb(report, dumps=lambda v: json.dumps(v, default=_json_default)),
                ),
            )
        return True

    def list_reports(self, limit: int = 500) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        self.ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT workflow_id, invoice_no, vendor_id, processing_status, report_payload, created_at "
                "FROM audit.invoice_audit ORDER BY created_at DESC LIMIT %s", (limit,)
            ).fetchall()
        return [{"workflow_id": r[0], "invoice_no": r[1], "vendor_id": r[2],
                 "processing_status": r[3], "report": r[4], "created_at": r[5]}
                for r in rows]

    def count_reports(self) -> int:
        if not self.enabled:
            return 0
        self.ensure_schema()
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM audit.invoice_audit").fetchone()[0]

    def recent_validation_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read recent vendor outcomes for the RAI disparate-outcome check."""
        if not self.enabled:
            return []
        self.ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT invoice_payload, report_payload->'validation_summary'->>'recommendation'
                   FROM audit.invoice_audit ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            ).fetchall()
        return [
            {"invoice_data": row[0], "recommendation": row[1] or "UNKNOWN"}
            for row in rows
        ]


class RedisCache:
    """Small JSON cache for workflow results and mailbox UID checkpoints."""

    def __init__(self, url: Optional[str] = None, ttl_seconds: int = 86400):
        self.url = url or os.getenv("REDIS_URL", "")
        self.ttl_seconds = ttl_seconds
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def _get_client(self):
        if not self.enabled:
            return None
        if self._client is None:
            import redis
            self._client = redis.Redis.from_url(self.url, decode_responses=True)
        return self._client

    def get_json(self, key: str):
        client = self._get_client()
        value = client.get(key) if client else None
        return json.loads(value) if value else None

    def set_json(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        client = self._get_client()
        if client:
            client.set(key, json.dumps(value, default=_json_default), ex=ttl_seconds or self.ttl_seconds)

    def get(self, key: str):
        client = self._get_client()
        return client.get(key) if client else None

    def set(self, key: str, value: str):
        client = self._get_client()
        if client:
            client.set(key, value)

    def delete(self, key: str):
        client = self._get_client()
        if client:
            client.delete(key)


class QdrantVectorStore:
    """Qdrant vector index with invoice and source metadata in point payloads."""

    def __init__(self, url: Optional[str] = None, collection: Optional[str] = None):
        self.url = url or os.getenv("QDRANT_URL", "")
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "invoice_chunks")
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def _get_client(self):
        if not self.enabled:
            return None
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self.url, timeout=5)
        return self._client

    def upsert(self, chunks: List[Dict[str, Any]], dimension: int):
        if not chunks or not self.enabled:
            return 0
        from qdrant_client.http import models

        client = self._get_client()
        if not client.collection_exists(self.collection):
            client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        points = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            key = "|".join(str(metadata.get(k, "")) for k in ("source", "invoice_no", "chunk_index", "indexed_at"))
            digest = hashlib.md5(key.encode("utf-8")).hexdigest()
            point_id = "-".join((digest[:8], digest[8:12], digest[12:16], digest[16:20], digest[20:]))
            points.append(models.PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload={"text": chunk["text"], "metadata": metadata},
            ))
        client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search(self, vector: List[float], limit: int, filters: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return None
        from qdrant_client.http import models

        conditions = [models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
                      for key, value in (filters or {}).items()]
        query_filter = models.Filter(must=conditions) if conditions else None
        results = self._get_client().query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points
        return [
            {"text": point.payload.get("text", ""),
             "metadata": point.payload.get("metadata", {}),
             "similarity_score": float(point.score), "rank": rank}
            for rank, point in enumerate(results, 1)
        ]


def persist_workflow_result(state: Dict[str, Any]) -> Dict[str, bool]:
    """Persist invoice, report, errors, and cache snapshot durably."""
    invoice = state.get("invoice_data", {})
    report = {}
    report_path = state.get("report_path")
    if report_path and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    html_path = state.get("html_report_path")
    if html_path and os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as handle:
            report["html_report"] = handle.read()
    report.setdefault("processing_status", "failed" if state.get("error") else "completed")
    report.setdefault("error_message", state.get("error"))
    report.setdefault("error_type", state.get("error_type"))
    saved = PostgresInvoiceStore().save(state.get("workflow_id", "unknown"), invoice, report)
    cache = RedisCache()
    redis_saved = False
    if cache.enabled:
        try:
            cache.set_json(f"invoice-auditor:workflow:{state.get('workflow_id', 'unknown')}", {
                "invoice_data": invoice,
                "recommendation": state.get("recommendation"),
                "validation_summary": state.get("validation_summary", {}),
                "report_path": report_path,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            })
            redis_saved = True
        except Exception:
            logger.warning("Redis workflow cache write failed; PostgreSQL audit record was saved", exc_info=True)
    return {"postgres": saved, "redis": redis_saved}
