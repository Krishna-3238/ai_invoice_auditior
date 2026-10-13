"""Runtime configuration for the containerized invoice auditor."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.environ["DATABASE_URL"]
CHECKPOINT_DATABASE_URL = os.getenv("CHECKPOINT_DATABASE_URL", DATABASE_URL + "?options=-c%20search_path%3Dlanggraph%2Cpublic")
INCOMING_DIR = Path(os.getenv("INCOMING_DIR", str(PROJECT_ROOT / "backned" / "data" / "incoming" / "invoices")))
WATCH_INTERVAL_SECONDS = float(os.getenv("WATCH_INTERVAL_SECONDS", "10"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
ERP_API_BASE_URL = os.getenv("ERP_API_BASE_URL", "http://mock-erp:8000")
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(PROJECT_ROOT / "outputs" / "reports")))

BEDROCK_REASONING_MODEL = os.getenv("BEDROCK_REASONING_MODEL", "amazon.nova-lite-v1:0")
BEDROCK_RAG_MODEL = os.getenv("BEDROCK_RAG_MODEL", "cohere.command-r-plus-v1:0")
BEDROCK_TRANSLATION_MODEL = os.getenv("BEDROCK_TRANSLATION_MODEL", "cohere.command-r-plus-v1:0")
BEDROCK_EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

