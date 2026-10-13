"""Qdrant indexing and filtered retrieval backed by PostgreSQL audit rows."""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from backned.llm import embed_bedrock_documents

from backned.config.settings import QDRANT_URL, BEDROCK_EMBEDDING_MODEL
from backned.db.repository import list_audits

COLLECTION = "invoice_audits"
CHUNK_SIZE = 1200
OVERLAP = 180


def _client():
    return QdrantClient(url=QDRANT_URL)


def _chunks(text: str):
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        part = text[start:end].strip()
        if part:
            yield part
        if end == len(text):
            break
        start = end - OVERLAP


def _vectors(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return embed_bedrock_documents(BEDROCK_EMBEDDING_MODEL, texts)


def index_all() -> dict[str, Any]:
    client = _client()
    audits = list_audits()
    if not client.collection_exists(COLLECTION):
        client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))
    points: list[models.PointStruct] = []
    for row in audits:
        invoice_id = str(row["invoice_id"])
        client.delete(collection_name=COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key="invoice_id", match=models.MatchValue(value=invoice_id))])), wait=True)
        report = row.get("report_json") or {}
        sources = [
            ("invoice", row.get("raw_extracted_text") or ""),
            ("report", row.get("report_html") or str(report)),
        ]
        for source_type, body in sources:
            chunks = list(_chunks(body))
            vectors = _vectors(chunks)
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{invoice_id}:{source_type}:{idx}"))
                points.append(models.PointStruct(id=point_id, vector=vector, payload={
                    "invoice_id": invoice_id, "chunk_text": chunk,
                    "source_type": source_type, "vendor": row.get("vendor_name"),
                    "language": row.get("source_language"),
                }))
    if points:
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
    return {"indexed_invoices": len(audits), "chunks": len(points), "collection": COLLECTION}


def retrieve(query: str, limit: int = 8, invoice_id: str | None = None) -> list[dict]:
    client = _client()
    if not client.collection_exists(COLLECTION):
        return []
    vector = _vectors([query])[0]
    query_filter = None
    if invoice_id:
        query_filter = models.Filter(must=[models.FieldCondition(key="invoice_id", match=models.MatchValue(value=invoice_id))])
    result = client.query_points(collection_name=COLLECTION, query=vector, query_filter=query_filter, limit=limit, with_payload=True)
    return [{"text": point.payload.get("chunk_text", ""), "source": f"{point.payload.get('source_type')}:{point.payload.get('invoice_id')}", "score": point.score, "invoice_id": point.payload.get("invoice_id"), "vendor": point.payload.get("vendor")} for point in result.points]
