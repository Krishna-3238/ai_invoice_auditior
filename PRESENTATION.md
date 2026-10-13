# Invoice Auditor — 5 slides

## Slide 1 — Goal

Containerized multilingual invoice auditing for PDF, DOCX, and scanned PNG files, from folder ingestion through human review.

## Slide 2 — Processing flow

Folder monitor and SHA-256 deduplication → text extraction/OCR → language detection and translation → completeness and arithmetic checks → ERP comparison → JSON/HTML report.

## Slide 3 — Services and data

Six Compose services: agent, Streamlit dashboard, mock ERP, PostgreSQL, Qdrant, and Redis. PostgreSQL stores audit history, discrepancies, feedback, ERP fixtures, and LangGraph checkpoints.

## Slide 4 — Human review and questions

Auditors inspect discrepancies and confidence, correct fields, and request revalidation. RAG answers use Qdrant retrieval and the RAG Triad threshold with a three-retry limit.

## Slide 5 — Run and verify

Set local database and AWS Bedrock credentials in `.env`, run `docker compose up --build -d`, add invoices to the mounted folder, and open `http://localhost:8501`. Verify persistence, duplicate skipping, ERP service DNS, and restart behavior.
