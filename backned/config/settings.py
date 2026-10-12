"""Runtime configuration for the containerized invoice auditor."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://auditor:auditor@postgres:5432/invoice_auditor")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
ERP_API_BASE_URL = os.getenv("ERP_API_BASE_URL", "http://mock-erp:8000")
INCOMING_DIR = Path(os.getenv("INCOMING_DIR", str(PROJECT_ROOT / "backned" / "data" / "incoming" / "invoices")))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(PROJECT_ROOT / "outputs" / "reports")))

LITELLM_REASONING_MODEL = os.getenv("LITELLM_REASONING_MODEL", "bedrock/amazon.nova-lite-v1:0")
LITELLM_RAG_MODEL = os.getenv("LITELLM_RAG_MODEL", "bedrock/cohere.command-r-plus-v1:0")
LITELLM_TRANSLATION_MODEL = os.getenv("LITELLM_TRANSLATION_MODEL", "bedrock/cohere.command-r-plus-v1:0")
LITELLM_EMBEDDING_MODEL = os.getenv("LITELLM_EMBEDDING_MODEL", "bedrock/amazon.titan-embed-text-v2:0")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

