"""
AI Invoice Auditor - Streamlit Dashboard
======================================================
Human-in-the-loop interface for invoice processing using LangGraph workflow.

Navigation (horizontal nav bar - project2 style):
- Dashboard: Overview & metrics
- Upload & Process: Input invoices
- Review & Approve: Combined review + decision workflow
- Query Invoices: RAG search
- Analytics: Trends & insights

Styling: Project 2 (modern horizontal nav, gradient header, right sidebar)
Logic:   Project 1 (LangGraph workflow, RAG system, feedback manager)
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workflows.invoice_auditor_workflow import run_workflow
from workflows.rag_interface import create_rag_system
from agents.rag_agents.indexing_agent import IndexingAgent
from config.logging_config import init_logging, get_logger, log_workflow_event
from config.feedback_manager import get_feedback_manager

init_logging()
logger = get_logger("StreamlitUI")
feedback_manager = get_feedback_manager()

st.set_page_config(
    page_title="AI Invoice Auditor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_custom_styles():
    """Inject a dark animated command-center UI."""
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #05070d;
            --bg-1: #0b1020;
            --panel: rgba(15, 23, 42, 0.78);
            --panel-strong: rgba(17, 24, 39, 0.94);
            --line: rgba(148, 163, 184, 0.18);
            --line-strong: rgba(34, 211, 238, 0.32);
            --text: #e5edf8;
            --muted: #94a3b8;
            --cyan: #22d3ee;
            --blue: #60a5fa;
            --green: #34d399;
            --amber: #fbbf24;
            --red: #fb7185;
            --violet: #a78bfa;
            --shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
        }

        @keyframes auroraFlow {
            0% { background-position: 0% 50%, 100% 0%, 50% 100%; }
            50% { background-position: 100% 50%, 0% 100%, 20% 0%; }
            100% { background-position: 0% 50%, 100% 0%, 50% 100%; }
        }

        @keyframes scanline {
            0% { transform: translateY(-12vh); opacity: 0; }
            12% { opacity: 0.28; }
            55% { opacity: 0.08; }
            100% { transform: translateY(115vh); opacity: 0; }
        }

        @keyframes panelIn {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes glowPulse {
            0%, 100% { box-shadow: 0 0 0 rgba(34, 211, 238, 0); }
            50% { box-shadow: 0 0 34px rgba(34, 211, 238, 0.22); }
        }

        section[data-testid="stSidebar"] { display: none; }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 16% 8%, rgba(34, 211, 238, 0.18), transparent 30%),
                radial-gradient(circle at 78% 4%, rgba(167, 139, 250, 0.14), transparent 28%),
                radial-gradient(circle at 52% 92%, rgba(52, 211, 153, 0.10), transparent 34%),
                linear-gradient(135deg, var(--bg-0) 0%, var(--bg-1) 54%, #020617 100%);
            background-size: 140% 140%, 120% 120%, 130% 130%, 100% 100%;
            animation: auroraFlow 18s ease-in-out infinite;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(148, 163, 184, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148, 163, 184, 0.035) 1px, transparent 1px);
            background-size: 44px 44px;
            mask-image: linear-gradient(to bottom, black, transparent 85%);
            z-index: 0;
        }

        .stApp::after {
            content: "";
            position: fixed;
            left: 0;
            right: 0;
            top: 0;
            height: 18vh;
            pointer-events: none;
            background: linear-gradient(to bottom, transparent, rgba(34, 211, 238, 0.08), transparent);
            animation: scanline 7s linear infinite;
            z-index: 0;
        }

        [data-testid="stAppViewContainer"] > .main {
            position: relative;
            z-index: 1;
            background: transparent;
        }

        .block-container {
            max-width: 1440px;
            padding: 5.5rem 2rem 3rem !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            letter-spacing: 0;
        }

        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stText, .stCaption, label, p {
            color: var(--text);
        }

        .header-container {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(8, 13, 27, 0.76)),
                linear-gradient(90deg, rgba(34, 211, 238, 0.14), rgba(96, 165, 250, 0.08), rgba(52, 211, 153, 0.08));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            margin: 0.75rem 0 1rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
            animation: panelIn 420ms ease-out both;
        }

        .header-container::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.12), transparent);
            transform: translateX(-100%);
            animation: shimmer 6s ease-in-out infinite;
        }

        @keyframes shimmer {
            0%, 45% { transform: translateX(-100%); }
            70%, 100% { transform: translateX(100%); }
        }

        .hero-kicker {
            color: var(--cyan);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            color: #f8fafc;
            font-size: clamp(2rem, 4vw, 4.2rem);
            line-height: 0.95;
            font-weight: 900;
            margin: 0;
        }

        .hero-subtitle {
            color: var(--muted);
            margin: 0.75rem 0 0;
            max-width: 760px;
            font-size: 1.02rem;
        }

        .hero-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .hero-chip {
            color: #dbeafe;
            background: rgba(15, 23, 42, 0.76);
            border: 1px solid rgba(96, 165, 250, 0.28);
            border-radius: 999px;
            padding: 0.42rem 0.72rem;
            font-size: 0.78rem;
            font-weight: 700;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }

        [data-testid="column"] {
            animation: panelIn 520ms ease-out both;
        }

        .stButton > button {
            color: #e0f2fe !important;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(30, 41, 59, 0.86)) !important;
            border: 1px solid rgba(34, 211, 238, 0.25) !important;
            border-radius: 12px !important;
            padding: 0.66rem 0.95rem !important;
            font-weight: 800 !important;
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.24);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            border-color: rgba(34, 211, 238, 0.72) !important;
            box-shadow: 0 18px 36px rgba(34, 211, 238, 0.16);
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.48), rgba(30, 41, 59, 0.92)) !important;
        }

        .stButton > button[kind="primary"],
        .stDownloadButton > button {
            background: linear-gradient(135deg, #0891b2, #2563eb 55%, #10b981) !important;
            color: #f8fafc !important;
            border-color: rgba(125, 211, 252, 0.6) !important;
            animation: glowPulse 3s ease-in-out infinite;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(160deg, rgba(15, 23, 42, 0.88), rgba(8, 13, 27, 0.82));
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem;
            box-shadow: var(--shadow);
            min-height: 112px;
            position: relative;
            overflow: hidden;
        }

        div[data-testid="stMetric"]::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--cyan), var(--green), transparent);
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-size: 1.45rem !important;
            font-weight: 900 !important;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted) !important;
        }

        .stExpander {
            background: rgba(15, 23, 42, 0.74) !important;
            border: 1px solid var(--line) !important;
            border-radius: 14px !important;
            box-shadow: var(--shadow);
            overflow: hidden;
        }

        div[data-testid="stFileUploader"] {
            background: rgba(15, 23, 42, 0.72);
            border-radius: 16px;
            padding: 1.2rem;
            border: 1px dashed rgba(34, 211, 238, 0.48);
            box-shadow: inset 0 0 28px rgba(34, 211, 238, 0.06);
        }

        .stChatMessage {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            box-shadow: var(--shadow);
        }

        .status-badge {
            display: inline-block;
            padding: 0.48rem 0.7rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.78rem;
            border: 1px solid currentColor;
            background: rgba(15, 23, 42, 0.76);
        }

        .status-success {
            color: var(--green);
        }

        .status-warning {
            color: var(--amber);
        }

        .status-error {
            color: var(--red);
        }

        .status-approve {
            background: rgba(52, 211, 153, 0.13);
            color: #86efac;
            border: 1px solid rgba(52, 211, 153, 0.42);
            padding: 0.55rem 0.75rem;
            border-radius: 999px;
            font-weight: bold;
            display: inline-block;
            margin: 0.35rem 0;
        }

        .status-reject {
            background: rgba(251, 113, 133, 0.13);
            color: #fda4af;
            border: 1px solid rgba(251, 113, 133, 0.42);
            padding: 0.55rem 0.75rem;
            border-radius: 999px;
            font-weight: bold;
            display: inline-block;
            margin: 0.35rem 0;
        }

        .status-review {
            background: rgba(251, 191, 36, 0.13);
            color: #fde68a;
            border: 1px solid rgba(251, 191, 36, 0.42);
            padding: 0.55rem 0.75rem;
            border-radius: 999px;
            font-weight: bold;
            display: inline-block;
            margin: 0.35rem 0;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
        }

        [data-testid="stTable"],
        [data-testid="stJson"],
        [data-testid="stVegaLiteChart"],
        [data-testid="stPlotlyChart"] {
            background: rgba(15, 23, 42, 0.76);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.6rem;
            box-shadow: var(--shadow);
        }

        .sidebar-metric {
            background: linear-gradient(160deg, rgba(15, 23, 42, 0.92), rgba(8, 13, 27, 0.82));
            border: 1px solid var(--line);
            padding: 1rem;
            border-radius: 14px;
            margin-bottom: 0.8rem;
            box-shadow: var(--shadow);
        }

        .sidebar-metric-label {
            font-size: 0.85rem;
            color: var(--muted);
            font-weight: 700;
        }

        .sidebar-metric-value {
            font-size: 1.8rem;
            color: #f8fafc;
            font-weight: 900;
            margin-top: 0.2rem;
        }

        .main-header {
            font-size: clamp(1.75rem, 3vw, 2.8rem);
            font-weight: 900;
            color: #f8fafc;
            margin-bottom: 1rem;
            padding-bottom: 0.7rem;
            border-bottom: 1px solid var(--line);
            position: relative;
        }

        .main-header::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: -1px;
            width: 160px;
            height: 2px;
            background: linear-gradient(90deg, var(--cyan), var(--green));
        }

        .metric-card {
            background: rgba(15, 23, 42, 0.76);
            border: 1px solid var(--line);
            padding: 15px;
            border-radius: 14px;
            margin: 10px 0;
            box-shadow: var(--shadow);
        }

        [data-testid="stAlert"] {
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid var(--line);
            border-radius: 14px;
            color: var(--text);
        }

        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.32), transparent);
            margin: 1.35rem 0;
        }

        input, textarea, select,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] textarea {
            color: var(--text) !important;
            background: rgba(2, 6, 23, 0.74) !important;
            border-color: rgba(148, 163, 184, 0.22) !important;
        }

        [data-baseweb="radio"] label,
        [data-baseweb="checkbox"] label,
        [data-baseweb="tab-list"] button {
            color: var(--text) !important;
        }

        [data-baseweb="tab-list"] {
            gap: 0.45rem;
            border-bottom: 1px solid var(--line);
        }

        [data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.68);
            border: 1px solid var(--line);
            border-radius: 999px 999px 0 0;
            padding: 0.45rem 0.9rem;
        }

        [data-baseweb="tab"][aria-selected="true"] {
            color: #f8fafc !important;
            border-color: rgba(34, 211, 238, 0.55);
            background: rgba(14, 116, 144, 0.35);
        }

        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--cyan), var(--green)) !important;
        }

        code {
            color: #bae6fd !important;
            background: rgba(14, 116, 144, 0.22) !important;
            border-radius: 6px;
        }

        @media (max-width: 900px) {
            .block-container {
                padding: 4.75rem 1rem 2rem !important;
            }
            .header-container {
                border-radius: 14px;
                padding: 1.1rem;
            }
            .hero-title {
                font-size: 2.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_styles()

def get_total_processed_reports():
    from integrations.storage import PostgresInvoiceStore
    return PostgresInvoiceStore().count_reports()

if "processed_invoices" not in st.session_state:
    st.session_state.processed_invoices = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"


st.markdown(
    """
    <div class="header-container">
        <div class="hero-kicker">Autonomous audit operations</div>
        <h1 class="hero-title">AI Invoice Auditor</h1>
        <p class="hero-subtitle">Dark-mode control surface for extraction, validation, review, search, and audit analytics.</p>
        <div class="hero-strip">
            <span class="hero-chip">LangGraph workflow</span>
            <span class="hero-chip">Human review queue</span>
            <span class="hero-chip">RAG invoice search</span>
            <span class="hero-chip">Live audit trail</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)

with nav_col1:
    if st.button("Command", key="nav_dashboard", use_container_width=True):
        st.session_state.current_page = "Dashboard"
        st.rerun()
with nav_col2:
    if st.button("Ingest", key="nav_upload", use_container_width=True):
        st.session_state.current_page = "Upload & Process"
        st.rerun()
with nav_col3:
    if st.button("Review", key="nav_review", use_container_width=True):
        st.session_state.current_page = "Review & Approve"
        st.rerun()
with nav_col4:
    if st.button("Search", key="nav_query", use_container_width=True):
        st.session_state.current_page = "Query Invoices"
        st.rerun()
with nav_col5:
    if st.button("Signals", key="nav_analytics", use_container_width=True):
        st.session_state.current_page = "Analytics"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

page = st.session_state.current_page

st.markdown(
    f"""
    <div class="hero-strip" style="margin: 0.2rem 0 1.2rem;">
        <span class="hero-chip">Active workspace: {page}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

main_col, sidebar_col = st.columns([3, 1])

with sidebar_col:
    st.markdown("### Runtime")

    try:
        import requests
        erp_status = requests.get(
            f"{os.getenv('ERP_API_URL', 'http://localhost:8000').rstrip('/')}/health",
            timeout=2,
        )
        if erp_status.status_code == 200:
            st.markdown('<div class="status-badge status-success">ERP API Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-error">ERP API Offline</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="status-badge status-error">ERP API Offline</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    vector_db_path = Path("./data/vector_db")
    if vector_db_path.exists():
        st.markdown('<div class="status-badge status-success">Vector DB Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-warning">Vector DB Pending</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Queue")

    total_inv = get_total_processed_reports()
    needs_review_count = len(
        [i for i in st.session_state.processed_invoices
         if i.get("recommendation") in ["MANUAL_REVIEW", "REJECT"]]
    )

    st.markdown(
        f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Total Processed</div>
            <div class="sidebar-metric-value">{total_inv}</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Needs Review</div>
            <div class="sidebar-metric-value">{needs_review_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Workflow health, review pressure, and indexed invoice state update as invoices move through the system.")



def get_all_validation_issues(inv):
    issues = []
    if inv.get("validation_issues"):
        issues.extend(inv.get("validation_issues", []))
    if inv.get("data_validation_issues"):
        issues.extend(inv.get("data_validation_issues", []))
    if inv.get("business_validation_issues"):
        issues.extend(inv.get("business_validation_issues", []))
    seen = set()
    unique_issues = []
    for issue in issues:
        msg = (
            issue.get("message", str(issue)) if isinstance(issue, dict) else str(issue)
        )
        if msg not in seen:
            seen.add(msg)
            unique_issues.append(issue)
    return unique_issues


def get_rejection_reason(inv):
    recommendation = inv.get("recommendation", "")
    if recommendation not in ["REJECT", "MANUAL_REVIEW"]:
        return None, None
    if inv.get("reviewer_notes"):
        return inv.get("reviewer_notes"), "reviewer"
    issues = get_all_validation_issues(inv)
    if issues:
        return issues, "validation"
    summary = inv.get("validation_summary", {})
    if summary.get("total_issues", 0) > 0 or summary.get("errors", 0) > 0:
        errors = summary.get("errors", summary.get("error_count", 0))
        warnings = summary.get("warnings", summary.get("warning_count", 0))
        return f"{errors} error(s), {warnings} warning(s)", "summary"
    return "No specific reason recorded", "unknown"


def get_status_badge(recommendation):
    if recommendation == "APPROVE":
        return '<div class="status-approve">APPROVED</div>'
    elif recommendation == "REJECT":
        return '<div class="status-reject">REJECTED</div>'
    elif recommendation == "MANUAL_REVIEW":
        return '<div class="status-review">NEEDS REVIEW</div>'
    else:
        return f'<div class="status-review">{recommendation}</div>'


def format_currency(amount, currency="USD"):
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}
    symbol = symbols.get(currency, currency)
    try:
        return f"{symbol}{float(amount):,.2f}"
    except Exception:
        return f"{symbol}{amount}"


def safe_number(value, default=0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_invoice_no_from_query(query: str):
    match = re.search(r"\bINV[\s:_-]*([A-Z]*[\s:_-]?\d[\w-]*)\b", query, re.IGNORECASE)
    if match:
        suffix = re.sub(r"[\s:_]+", "-", match.group(1).strip().upper())
        return f"INV-{suffix}"

    match = re.search(r"\binvoice[\s#:_-]+([A-Z]*[\s:_-]?\d[\w-]*)\b", query, re.IGNORECASE)
    if match:
        suffix = re.sub(r"[\s:_]+", "-", match.group(1).strip().upper())
        return f"INV-{suffix}"

    return None


def is_review_decision_query(query: str):
    query_lower = query.lower()
    decision_terms = [
        "approve", "approved", "approval",
        "reject", "rejected", "rejection",
        "decision", "review", "reviewer", "human",
        "status", "why",
    ]
    return any(term in query_lower for term in decision_terms)


def decision_reason_label(feedback_type: str, decision: str):
    value = f"{feedback_type} {decision}".lower()
    if "approval" in value or "approved" in value:
        return "Approval reason"
    if "rejection" in value or "rejected" in value:
        return "Rejection reason"
    if "correction" in value or "corrected" in value:
        return "Correction reason"
    if "escalation" in value or "escalated" in value:
        return "Escalation reason"
    return "Review reason"


def sentence_text(value: str):
    value = value.strip()
    if value.endswith((".", "!", "?")):
        return value
    return f"{value}."


def summarize_feedback_history(history):
    counts = {}
    for entry in history:
        decision = entry.get("decision") or entry.get("feedback_type") or "unknown"
        counts[decision] = counts.get(decision, 0) + 1
    return ", ".join(f"{count} {decision}" for decision, count in sorted(counts.items()))


def get_human_review_search_answer(query: str):
    invoice_no = extract_invoice_no_from_query(query)
    if not invoice_no or not is_review_decision_query(query):
        return None

    history = feedback_manager.get_invoice_feedback_history(invoice_no)
    if not history:
        return None

    history = sorted(history, key=lambda item: item.get("timestamp", ""))
    latest = history[-1]
    previous = history[:-1]

    decision = (latest.get("decision") or latest.get("feedback_type") or "unknown").lower()
    feedback_type = latest.get("feedback_type", "review")
    reviewer_id = latest.get("reviewer_id", "unknown")
    timestamp = latest.get("timestamp", "N/A")
    notes = (latest.get("user_notes") or "").strip()
    reason_label = decision_reason_label(feedback_type, decision)
    changes = latest.get("changes_summary", {}) or {}

    answer_parts = [
        f"{invoice_no} is currently human {decision}.",
        f"The latest review decision was {decision} by {reviewer_id} on {timestamp}.",
    ]
    if notes:
        answer_parts.append(f"{reason_label}: {sentence_text(notes)}")
    else:
        answer_parts.append(f"No {reason_label.lower()} was recorded for this decision.")

    if changes.get("changes_detected"):
        changed_fields = changes.get("changed_fields", [])
        if changed_fields:
            answer_parts.append(
                "The reviewer approved it after correcting: "
                + ", ".join(str(field) for field in changed_fields)
                + "."
            )
        else:
            answer_parts.append("The reviewer made corrections before the final decision.")
    else:
        answer_parts.append("No data edits were recorded with the latest decision.")

    if previous:
        answer_parts.append(
            f"There are {len(previous)} earlier review record(s), but the latest review is treated as the current decision. "
            f"Earlier history: {summarize_feedback_history(previous)}."
        )

    source_text = (
        f"Current Latest Review for Invoice {invoice_no}. "
        f"Decision: {decision}. Feedback Type: {feedback_type}. "
        f"Reviewed By: {reviewer_id}. Review Date: {timestamp}. "
        f"{reason_label}: {notes or 'No reviewer notes recorded'}"
    )

    return {
        "query": query,
        "answer": " ".join(answer_parts),
        "has_answer": True,
        "confidence": "high",
        "sources": [{
            "text_preview": source_text,
            "metadata": {
                "source": "feedback_history",
                "invoice_no": invoice_no,
                "feedback_type": feedback_type,
                "decision": decision,
                "reviewer_id": reviewer_id,
                "reviewed_at": timestamp,
                "is_latest_decision": True,
            },
            "similarity_score": 1.0,
        }],
        "num_sources": 1,
        "evaluation": None,
        "success": True,
        "method": "feedback_history",
    }


def display_invoice_summary(result):
    invoice_data = result.get("invoice_data", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Invoice #", invoice_data.get("invoice_no", "N/A"))
    with col2:
        st.metric("Vendor", invoice_data.get("vendor_name", "N/A"))
    with col3:
        total = invoice_data.get("total_amount", 0)
        currency = invoice_data.get("currency", "USD")
        st.metric("Total", format_currency(total, currency))
    with col4:
        st.metric("Date", invoice_data.get("invoice_date", "N/A"))


def display_validation_results(result):
    validation_summary = result.get("validation_summary", {})
    issues = get_all_validation_issues(result)
    st.markdown("### Validation Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Issues", validation_summary.get("total_issues", 0))
    with col2:
        errors = validation_summary.get("errors", validation_summary.get("error_count", 0))
        st.metric("Errors", errors)
    with col3:
        warnings = validation_summary.get("warnings", validation_summary.get("warning_count", 0))
        st.metric("Warnings", warnings)
    if issues:
        st.markdown("#### Issues Found")
        issues_df = pd.DataFrame(issues)
        st.dataframe(issues_df, use_container_width=True)
    else:
        st.success("No validation issues found!")


def index_processed_invoice(result: dict):
    try:
        indexer = IndexingAgent(index_path="./data/vector_db")
        chunks_added = indexer.index_invoice_data(result)
        if "report_json" in result:
            report_chunks = indexer.index_report(result["report_json"])
            chunks_added += report_chunks
        indexer.save_index()
        invoice_no = result.get("invoice_data", {}).get("invoice_no", "UNKNOWN")
        if chunks_added > 0:
            logger.info(f"Automatically indexed {chunks_added} chunks for invoice {invoice_no}")
        if "rag_system" in st.session_state and st.session_state.rag_system is not None:
            try:
                st.session_state.rag_system.reload_index()
            except Exception as e:
                logger.warning(f"Failed to reload RAG index: {e}")
                st.session_state.rag_system = None
        return chunks_added
    except Exception as e:
        logger.error(f"Failed to auto-index invoice: {str(e)}")
        return 0


def index_reviewer_feedback(invoice_data: dict, feedback: dict):
    try:
        indexer = IndexingAgent(index_path="./data/vector_db")
        chunks_added = indexer.index_reviewer_feedback(invoice_data, feedback)
        indexer.save_index()
        if "rag_system" in st.session_state and st.session_state.rag_system is not None:
            try:
                st.session_state.rag_system.reload_index()
            except Exception as e:
                logger.warning(f"Failed to reload RAG index: {e}")
                st.session_state.rag_system = None
        return chunks_added
    except Exception as e:
        logger.error(f"Failed to index reviewer feedback: {str(e)}")
        return 0


def generate_dynamic_queries(invoices):
    if not invoices:
        return [
            "What is the total amount for invoice INV-1001?",
            "List all invoices from a specific vendor",
            "Show me invoices that were rejected",
            "Which invoices have validation errors?",
            "What is the average invoice amount?",
        ]
    vendors = set(
        [inv.get("invoice_data", {}).get("vendor_name")
         for inv in invoices if inv.get("invoice_data", {}).get("vendor_name")]
    )
    invoice_numbers = [
        inv.get("invoice_data", {}).get("invoice_no")
        for inv in invoices[:3]
        if inv.get("invoice_data", {}).get("invoice_no")
    ]
    statuses = set([inv.get("recommendation") for inv in invoices])
    has_errors = any(
        [inv.get("validation_summary", {}).get("total_issues", 0) > 0 for inv in invoices]
    )
    try:
        from litellm import completion
        context = f"""Generate 5 natural language query suggestions for an invoice RAG system.
IMPORTANT: All suggestions must be in ENGLISH regardless of original invoice language.
Available data:
- Total invoices: {len(invoices)}
- Vendors: {', '.join(list(vendors)[:5])}
- Sample invoice numbers: {', '.join(invoice_numbers)}
- Statuses: {', '.join(statuses)}
- Has validation errors: {has_errors}
Generate diverse English queries. Return ONLY a JSON array of 5 English query strings."""
        response = completion(
            model="bedrock/cohere.command-r-plus-v1:0",
            messages=[{"role": "user", "content": context}],
            temperature=0.7,
            max_tokens=300,
        )
        queries_text = response.choices[0].message.content.strip()
        json_match = re.search(r"\[.*\]", queries_text, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group(0))
            if isinstance(queries, list) and len(queries) >= 5:
                return queries[:5]
    except Exception:
        pass
    queries = []
    if invoice_numbers:
        queries.append(f"What is the total amount for invoice {invoice_numbers[0]}?")
    if vendors:
        vendor = list(vendors)[0]
        queries.append(f"List all invoices from {vendor}")
    if "REJECT" in statuses or "MANUAL_REVIEW" in statuses:
        queries.append("Show me invoices that need manual review or were rejected")
    if has_errors:
        queries.append("Which invoices have validation errors?")
    queries.append(f"What is the total amount across all {len(invoices)} invoices?")
    while len(queries) < 5:
        queries.append("Summarize the invoice processing results")
    return queries[:5]



with main_col:

    if page == "Dashboard":
        st.markdown('<div class="main-header">Command Center</div>', unsafe_allow_html=True)
        st.markdown(
            "Monitor invoice throughput, validation outcomes, and review pressure from one operational surface."
        )

        st.markdown("### Live Overview")
        total = get_total_processed_reports()
        approved = len([i for i in st.session_state.processed_invoices if i.get("recommendation") == "APPROVE"])
        rejected = len([i for i in st.session_state.processed_invoices if i.get("recommendation") == "REJECT"])
        needs_review = len([i for i in st.session_state.processed_invoices if i.get("recommendation") == "MANUAL_REVIEW"])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Invoices", total)
        with col2:
            st.metric("Approved", approved)
        with col3:
            st.metric("Rejected", rejected)
        with col4:
            st.metric("Needs Review", needs_review,
                      delta=None if needs_review == 0 else f"{needs_review} pending")

        st.markdown("---")
        st.markdown("### Recent Activity")

        if st.session_state.processed_invoices:
            recent = st.session_state.processed_invoices[-10:][::-1]
            for inv in recent:
                invoice_data = inv.get("invoice_data", {})
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                with col1:
                    st.write(f"**{invoice_data.get('invoice_no', 'N/A')}**")
                with col2:
                    st.write(invoice_data.get("vendor_name", "N/A"))
                with col3:
                    total_amt = invoice_data.get("total_amount", 0)
                    currency = invoice_data.get("currency", "USD")
                    st.write(format_currency(total_amt, currency))
                with col4:
                    st.markdown(get_status_badge(inv.get("recommendation", "UNKNOWN")), unsafe_allow_html=True)

                reason, reason_type = get_rejection_reason(inv)
                if reason:
                    if reason_type == "reviewer":
                        st.error(f"**Reason:** {reason}")
                    elif reason_type == "validation":
                        if isinstance(reason, list):
                            issue_msgs = []
                            for issue in reason[:3]:
                                msg = (issue.get("message", str(issue)) if isinstance(issue, dict) else str(issue))
                                issue_msgs.append(f"- {msg}")
                            if len(reason) > 3:
                                issue_msgs.append(f"- ... and {len(reason) - 3} more issues")
                            st.warning(f"**Validation Issues:**\n" + "\n".join(issue_msgs))
                        else:
                            st.warning(f"**Validation Issues:** {reason}")
                    elif reason_type == "summary":
                        st.warning(f"**Validation:** {reason}")
                    else:
                        st.info(f"**Note:** {reason}")

                st.markdown("---")
        else:
            st.info("No invoices have entered the audit pipeline yet. Start from Ingest to process the first file.")

    elif page == "Upload & Process":
        st.markdown('<div class="main-header">Invoice Ingest</div>', unsafe_allow_html=True)
        st.markdown("Feed invoices into the LangGraph audit workflow from uploads, samples, or the incoming folder.")

        upload_method = st.radio(
            "Input channel",
            ["Upload File", "Select from Samples", "Process Incoming Folder"],
            horizontal=True,
        )

        invoice_path = None

        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload invoice (PDF, DOCX, PNG, JPG)",
                type=["pdf", "docx", "png", "jpg", "jpeg"],
            )
            if uploaded_file:
                upload_dir = Path("./data/incoming")
                upload_dir.mkdir(parents=True, exist_ok=True)
                invoice_path = upload_dir / uploaded_file.name
                with open(invoice_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File uploaded: {uploaded_file.name}")

        elif upload_method == "Select from Samples":
            sample_dir = Path("./data/sample_data")
            if sample_dir.exists():
                sample_files = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.png"))
                sample_names = [f.name for f in sample_files]
                if sample_names:
                    selected_sample = st.selectbox("Select sample invoice:", sample_names)
                    invoice_path = sample_dir / selected_sample
                    st.info(f"Selected: {selected_sample}")
                else:
                    st.warning("No sample files found in ./data/sample_data/")
            else:
                st.warning("Sample data directory not found")

        elif upload_method == "Process Incoming Folder":
            incoming_dir = Path("./data/incoming")
            incoming_dir.mkdir(parents=True, exist_ok=True)
            supported_extensions = (".pdf", ".docx", ".png", ".jpg", ".jpeg")
            incoming_files = [
                f for f in incoming_dir.iterdir()
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]
            st.markdown("### Incoming Watch Folder")
            st.info(f"**Folder:** `{incoming_dir.absolute()}`")
            if incoming_files:
                st.success(f"Found **{len(incoming_files)}** invoice(s) ready to process:")
                for f in incoming_files:
                    st.write(f"  • {f.name}")
                st.markdown("---")
                if st.button("Process All Incoming Invoices", type="primary", use_container_width=True):
                    progress_bar = st.progress(0, text="Starting batch processing...")
                    results_summary = {"success": 0, "failed": 0, "errors": []}
                    for idx, file_path in enumerate(incoming_files):
                        progress = (idx + 1) / len(incoming_files)
                        progress_bar.progress(progress, text=f"Processing {file_path.name} ({idx+1}/{len(incoming_files)})")
                        try:
                            log_workflow_event("batch_processing_started", file_path.name,
                                               {"source": "incoming_folder", "batch_index": idx + 1,
                                                "batch_total": len(incoming_files)}, logger)
                            result = run_workflow(str(file_path))
                            index_processed_invoice(result)
                            result["invoice_filename"] = file_path.name
                            result["batch_processed"] = True
                            st.session_state.processed_invoices.append(result)
                            results_summary["success"] += 1
                            invoice_id = result.get("invoice_data", {}).get("invoice_no", file_path.name)
                            log_workflow_event("batch_processing_completed", invoice_id,
                                               {"recommendation": result.get("recommendation")}, logger)
                        except Exception as e:
                            results_summary["failed"] += 1
                            results_summary["errors"].append({"file": file_path.name, "error": str(e)})
                            logger.error(f"Failed to process {file_path.name}: {str(e)}")
                    progress_bar.progress(1.0, text="Batch processing complete!")
                    st.markdown("### Batch Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Successful", results_summary["success"])
                    with col2:
                        st.metric("Failed", results_summary["failed"])
                    if results_summary["errors"]:
                        with st.expander("View Errors"):
                            for err in results_summary["errors"]:
                                st.error(f"**{err['file']}:** {err['error']}")
                    if results_summary["success"] > 0:
                        st.success(f"Successfully processed {results_summary['success']} invoice(s)!")
                        st.info("Go to **Dashboard** or **Review & Approve** to see the results.")
            else:
                st.warning("No invoice files found in the incoming folder.")
                st.info(f"Tip: Drop PDF, DOCX, PNG, or JPG files into:\n`{incoming_dir.absolute()}`")

        st.markdown("---")
        if (
            upload_method != "Process Incoming Folder"
            and invoice_path
            and st.button("Process Invoice", type="primary", use_container_width=True)
        ):
            with st.spinner("Processing invoice through LangGraph workflow..."):
                progress_bar = st.progress(0, text="Starting workflow...")
                try:
                    logger.info(f"Starting invoice processing for: {invoice_path.name}")
                    log_workflow_event("processing_started", invoice_path.name, {"source": "streamlit_ui"}, logger)
                    progress_bar.progress(0.2, text="Extracting content...")
                    result = run_workflow(str(invoice_path))
                    progress_bar.progress(0.9, text="Indexing for search...")
                    index_processed_invoice(result)
                    progress_bar.progress(1.0, text="Complete!")
                    result["invoice_filename"] = invoice_path.name
                    st.session_state.processed_invoices.append(result)
                    st.session_state.current_result = result
                    invoice_id = result.get("invoice_data", {}).get("invoice_no", invoice_path.name)
                    log_workflow_event("processing_completed", invoice_id,
                                      {"recommendation": result.get("recommendation"),
                                       "validation_issues": result.get("validation_summary", {}).get("total_issues", 0)},
                                      logger)
                    st.success("Invoice processed successfully!")
                    st.markdown("---")
                    st.markdown("### Processing Output")
                    display_invoice_summary(result)
                    st.markdown("---")
                    recommendation = result.get("recommendation", "UNKNOWN")
                    st.markdown(get_status_badge(recommendation), unsafe_allow_html=True)
                    tab1, tab2, tab3 = st.tabs(["Validation", "Reports", "Translation"])
                    with tab1:
                        display_validation_results(result)
                    with tab2:
                        st.markdown("### Generated Reports")
                        report_paths = result.get("report_paths", {})
                        col1, col2 = st.columns(2)
                        with col1:
                            html_report = report_paths.get("html")
                            if html_report and Path(html_report).exists():
                                st.markdown("**HTML Report:**")
                                with open(html_report, "r") as f:
                                    html_content = f.read()
                                st.components.v1.html(html_content, height=400, scrolling=True)
                            else:
                                st.warning("HTML report not generated")
                        with col2:
                            json_report = report_paths.get("json")
                            if json_report and Path(json_report).exists():
                                st.markdown("**JSON Report:**")
                                with open(json_report, "r") as f:
                                    json_content = json.load(f)
                                st.json(json_content)
                                st.download_button(
                                    "Download JSON",
                                    data=json.dumps(json_content, indent=2),
                                    file_name=f"invoice_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                )
                            else:
                                st.warning("JSON report not generated")
                    with tab3:
                        st.markdown("### Translation Info")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Language", result.get("original_language", "N/A"))
                            st.metric("Translation Method", result.get("translation_method", "N/A"))
                        with col2:
                            st.metric("Translation Confidence", f"{result.get('translation_confidence', 0):.2f}")
                            st.metric("Needs Translation", "Yes" if result.get("needs_translation") else "No")
                    st.markdown("---")
                    st.markdown("### Decision Route")
                    if recommendation == "APPROVE":
                        st.success("Invoice automatically approved!")
                    else:
                        st.warning("This invoice needs manual review. Go to **Review & Approve** to make a decision.")
                        if st.button("Go to Review & Approve", use_container_width=True):
                            st.session_state.current_page = "Review & Approve"
                            st.rerun()
                except Exception as e:
                    progress_bar.empty()
                    logger.error(f"Error processing invoice {invoice_path.name}: {str(e)}", exc_info=True)
                    log_workflow_event("processing_error", invoice_path.name, {"error": str(e)}, logger)
                    st.error(f"Error processing invoice: {str(e)}")
                    st.exception(e)

    elif page == "Review & Approve":
        st.markdown('<div class="main-header">Review Queue</div>', unsafe_allow_html=True)

        if not st.session_state.processed_invoices:
            st.info("The review queue is empty. Process invoices from Ingest to populate this workspace.")
            if st.button("Go to Upload & Process"):
                st.session_state.current_page = "Upload & Process"
                st.rerun()
        else:
            st.markdown("Inspect extracted data, validation issues, reviewer notes, and final approval actions in one queue.")
            tab1, tab2 = st.tabs(["Needs Decision", "Invoice Ledger"])

            with tab1:
                st.markdown("### Decision Queue")
                review_invoices = [
                    (idx, inv) for idx, inv in enumerate(st.session_state.processed_invoices)
                    if inv.get("recommendation") not in ["APPROVE"]
                ]
                if not review_invoices:
                    st.success("No invoices are waiting for a manual decision.")
                else:
                    st.warning(f"**{len(review_invoices)} invoice(s)** need your review")
                    for list_idx, (orig_idx, invoice) in enumerate(review_invoices):
                        invoice_no = invoice.get("invoice_data", {}).get("invoice_no", "N/A")
                        vendor = invoice.get("invoice_data", {}).get("vendor_name", "Unknown")
                        status = invoice.get("recommendation", "UNKNOWN")
                        with st.expander(f"{invoice_no} | {vendor} | {status}", expanded=(list_idx == 0)):
                            col1, col2 = st.columns([3, 2])
                            with col1:
                                st.markdown("#### Invoice Summary")
                                display_invoice_summary(invoice)
                                st.markdown("---")
                                display_validation_results(invoice)
                                with st.expander("View Extracted Text"):
                                    extracted = invoice.get("translated_text", invoice.get("extracted_text", "N/A"))
                                    st.text(extracted[:1500] + "..." if len(extracted) > 1500 else extracted)
                            with col2:
                                st.markdown("### Decision Console")
                                recommendation = invoice.get("recommendation", "UNKNOWN")
                                st.markdown(get_status_badge(recommendation), unsafe_allow_html=True)
                                st.markdown("---")
                                decision = st.radio(
                                    "Choose action:",
                                    ["Approve", "Reject", "Edit & Approve", "View Full Details"],
                                    key=f"decision_{orig_idx}",
                                )
                                notes = st.text_area(
                                    "Reviewer notes:",
                                    placeholder="Add notes about your decision (required for reject)...",
                                    key=f"notes_{orig_idx}",
                                    height=100,
                                )
                                if decision == "Edit & Approve":
                                    st.markdown("#### Edit Invoice Data")
                                    st.caption("Modify the JSON below to correct any errors")
                                    edited_data = st.text_area(
                                        "Edit JSON:",
                                        value=json.dumps(invoice.get("invoice_data", {}), indent=2),
                                        height=250,
                                        key=f"edit_{orig_idx}",
                                    )
                                if decision == "View Full Details":
                                    st.info("Tip: Switch to the **'All Invoices'** tab to view complete details")
                                can_submit = True
                                if decision == "Reject" and not notes.strip():
                                    st.warning("Notes are required when rejecting an invoice")
                                    can_submit = False
                                if st.button("Submit Decision", key=f"submit_{orig_idx}", type="primary",
                                             use_container_width=True, disabled=not can_submit):
                                    invoice_id = invoice.get("invoice_data", {}).get("invoice_no", f"INV-{orig_idx}")
                                    if decision == "Approve":
                                        invoice["recommendation"] = "APPROVE"
                                        invoice["manual_override"] = True
                                        invoice["reviewer_notes"] = notes
                                        invoice["reviewed_at"] = datetime.now().isoformat()
                                        feedback_id = feedback_manager.record_approval(
                                            invoice_id=invoice_id,
                                            validated_data=invoice.get("invoice_data", {}),
                                            user_notes=notes,
                                            reviewer_id="streamlit_user",
                                        )
                                        logger.info(f"Invoice {invoice_id} approved | Feedback ID: {feedback_id}")
                                        index_reviewer_feedback(invoice.get("invoice_data", {}), {
                                            "reviewer_notes": notes, "decision": "APPROVE",
                                            "reviewer_id": "streamlit_user",
                                            "reviewed_at": invoice["reviewed_at"], "data_edited": False,
                                        })
                                        st.success("Invoice approved!")
                                    elif decision == "Reject":
                                        invoice["recommendation"] = "REJECT"
                                        invoice["manual_override"] = True
                                        invoice["reviewer_notes"] = notes
                                        invoice["reviewed_at"] = datetime.now().isoformat()
                                        feedback_id = feedback_manager.record_rejection(
                                            invoice_id=invoice_id,
                                            rejected_data=invoice.get("invoice_data", {}),
                                            rejection_reason=notes,
                                            reviewer_id="streamlit_user",
                                        )
                                        logger.info(f"Invoice {invoice_id} rejected | Feedback ID: {feedback_id}")
                                        index_reviewer_feedback(invoice.get("invoice_data", {}), {
                                            "reviewer_notes": notes, "decision": "REJECT",
                                            "reviewer_id": "streamlit_user",
                                            "reviewed_at": invoice["reviewed_at"], "data_edited": False,
                                        })
                                        st.error("Invoice rejected!")
                                    elif decision == "Edit & Approve":
                                        try:
                                            new_data = json.loads(edited_data)
                                            original_data = invoice.get("invoice_data", {}).copy()
                                            invoice["invoice_data"] = new_data
                                            invoice["recommendation"] = "APPROVE"
                                            invoice["manual_override"] = True
                                            invoice["data_edited"] = True
                                            invoice["reviewer_notes"] = notes
                                            invoice["reviewed_at"] = datetime.now().isoformat()
                                            feedback_id = feedback_manager.record_correction(
                                                invoice_id=invoice_id,
                                                original_data=original_data,
                                                corrected_data=new_data,
                                                user_notes=notes,
                                                reviewer_id="streamlit_user",
                                            )
                                            logger.info(f"Invoice {invoice_id} edited and approved | Feedback ID: {feedback_id}")
                                            index_reviewer_feedback(new_data, {
                                                "reviewer_notes": notes, "decision": "APPROVE (EDITED)",
                                                "reviewer_id": "streamlit_user",
                                                "reviewed_at": invoice["reviewed_at"], "data_edited": True,
                                            })
                                            st.success("Invoice edited and approved!")
                                            st.balloons()
                                        except json.JSONDecodeError as e:
                                            st.error(f"Invalid JSON format: {str(e)}")
                                            can_submit = False
                                    if can_submit:
                                        st.rerun()

            with tab2:
                st.markdown("### Invoice Ledger")
                invoice_options = [
                    f"{inv.get('invoice_data', {}).get('invoice_no', 'N/A')} | "
                    f"{inv.get('invoice_data', {}).get('vendor_name', 'Unknown')} | "
                    f"{inv.get('recommendation', 'UNKNOWN')}"
                    for inv in st.session_state.processed_invoices
                ]
                selected_idx = st.selectbox(
                    "Select invoice to view:",
                    range(len(invoice_options)),
                    format_func=lambda x: invoice_options[x],
                )
                result = st.session_state.processed_invoices[selected_idx]
                st.markdown("---")
                recommendation = result.get("recommendation", "UNKNOWN")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(get_status_badge(recommendation), unsafe_allow_html=True)
                with col2:
                    if result.get("manual_override"):
                        st.success("Manually Reviewed")
                reason, reason_type = get_rejection_reason(result)
                if recommendation == "REJECT":
                    st.error("**REJECTION REASON:**")
                    if reason_type == "reviewer":
                        st.error(f"{reason}")
                    elif reason_type == "validation":
                        issues = reason if isinstance(reason, list) else [reason]
                        for issue in issues:
                            if isinstance(issue, dict):
                                st.error(f"- {issue.get('message', issue)}")
                            else:
                                st.error(f"- {issue}")
                    elif reason_type == "summary":
                        st.error(f"Validation failed: {reason}")
                    else:
                        st.error(f"{reason}")
                elif recommendation == "MANUAL_REVIEW":
                    st.warning("**REVIEW REQUIRED:**")
                    if reason_type == "reviewer":
                        st.warning(f"{reason}")
                    elif reason_type == "validation":
                        issues = reason if isinstance(reason, list) else [reason]
                        for issue in issues[:5]:
                            if isinstance(issue, dict):
                                st.warning(f"- {issue.get('message', issue)}")
                            else:
                                st.warning(f"- {issue}")
                        if len(issues) > 5:
                            st.warning(f"... and {len(issues) - 5} more issues")
                    elif reason_type == "summary":
                        st.warning(f"Validation issues: {reason}")
                    else:
                        st.warning(f"{reason}")
                elif result.get("reviewer_notes"):
                    st.info(f"**Reviewer Notes:** {result.get('reviewer_notes')}")
                if result.get("data_edited"):
                    st.warning("**Data was edited during review**")
                st.markdown("---")
                detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs(["Summary", "Validation", "Data", "Metadata"])
                with detail_tab1:
                    display_invoice_summary(result)
                    st.markdown("---")
                    st.markdown("### Extracted Text")
                    with st.expander("View full extracted text"):
                        st.text(result.get("translated_text", result.get("extracted_text", "N/A")))
                with detail_tab2:
                    display_validation_results(result)
                with detail_tab3:
                    st.markdown("### Invoice Data")
                    st.json(result.get("invoice_data", {}))
                    line_items = result.get("invoice_data", {}).get("line_items", [])
                    if line_items:
                        st.markdown("---")
                        st.markdown("#### Line Items")
                        items_df = pd.DataFrame(line_items)
                        st.dataframe(items_df, use_container_width=True)
                with detail_tab4:
                    st.markdown("### Processing Metadata")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**File Metadata:**")
                        st.json(result.get("file_metadata", {}))
                        st.markdown("**Extraction Metadata:**")
                        st.json(result.get("extraction_metadata", {}))
                    with col2:
                        st.markdown("**Translation Metadata:**")
                        st.json({
                            "original_language": result.get("original_language"),
                            "translation_method": result.get("translation_method"),
                            "translation_confidence": result.get("translation_confidence"),
                            "needs_translation": result.get("needs_translation"),
                        })
                        st.markdown("**Workflow Metadata:**")
                        st.json({
                            "workflow_id": result.get("workflow_id"),
                            "started_at": result.get("started_at"),
                            "completed_at": result.get("completed_at"),
                            "indexed": result.get("indexed"),
                            "reviewed_at": result.get("reviewed_at"),
                            "manual_override": result.get("manual_override", False),
                        })

    elif page == "Query Invoices":
        st.markdown('<div class="main-header">Invoice Search</div>', unsafe_allow_html=True)
        st.markdown("Ask natural-language questions against indexed invoice data and generated audit reports.")

        if st.session_state.rag_system is None:
            with st.spinner("Initializing RAG system..."):
                try:
                    st.session_state.rag_system = create_rag_system(use_llm=True)
                    st.success("RAG system initialized!")
                except Exception as e:
                    st.error(f"Failed to initialize RAG system: {str(e)}")
                    logger.error(f"Failed to create RAG system: {str(e)}", exc_info=True)
                    st.stop()

        rag_stats = st.session_state.rag_system.get_stats()
        if rag_stats.get("total_documents", 0) == 0:
            st.warning("**No invoices indexed yet.** Process some invoices first to enable querying.")
            st.info("Tip: Go to **Upload & Process** to add invoices to the system.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "pending_query" not in st.session_state:
            st.session_state.pending_query = None

        if st.session_state.pending_query:
            query_to_process = st.session_state.pending_query
            st.session_state.pending_query = None
            st.session_state.chat_history.append({"role": "user", "content": query_to_process})
            with st.spinner("Searching through invoices..."):
                try:
                    result = get_human_review_search_answer(query_to_process)
                    if result is None:
                        result = st.session_state.rag_system.query(query_to_process, top_k=3, evaluate=True)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.get("answer", "No answer generated"),
                        "sources": result.get("sources", []),
                        "evaluation": result.get("evaluation"),
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": f"Query failed: {str(e)}", "sources": [],
                    })

        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    if message.get("sources"):
                        with st.expander("Sources"):
                            for idx, source in enumerate(message["sources"], 1):
                                score = source.get("similarity_score", 0)
                                st.caption(f"Source {idx} (score: {score:.4f}): {source.get('text_preview', '')[:200]}")
                    if message.get("evaluation"):
                        evaluation = message["evaluation"]
                        metrics = evaluation.get("metrics", {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Context Relevance", f"{metrics.get('context_relevance', {}).get('score', 0):.2f}")
                        with col2:
                            st.metric("Groundedness", f"{metrics.get('groundedness', {}).get('score', 0):.2f}")
                        with col3:
                            st.metric("Overall Score", f"{evaluation.get('overall_score', 0):.2f}")

        query = st.text_area(
            "Enter your question:",
            placeholder="e.g., What is the total amount for invoice INV-1001?\nWhich invoices are from ABC Corp?",
            height=80,
            key="manual_query_input",
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of results:", min_value=1, max_value=10, value=3)
        with col2:
            use_evaluation = st.checkbox("Evaluate quality", value=True)

        if st.button("Search", type="primary", use_container_width=True):
            if query.strip():
                st.session_state.chat_history.append({"role": "user", "content": query})
                with st.spinner("Searching invoices..."):
                    try:
                        result = get_human_review_search_answer(query)
                        if result is None:
                            result = st.session_state.rag_system.query(query, top_k=top_k, evaluate=use_evaluation)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result.get("answer", "No answer generated"),
                            "sources": result.get("sources", []),
                            "evaluation": result.get("evaluation") if use_evaluation else None,
                        })
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": f"Query failed: {str(e)}", "sources": [],
                        })
                st.rerun()
            else:
                st.warning("Please enter a query")

        st.markdown("---")
        st.markdown("### Suggested Prompts")

        if "dynamic_queries" not in st.session_state:
            with st.spinner("Generating suggestions..."):
                st.session_state.dynamic_queries = generate_dynamic_queries(st.session_state.processed_invoices)

        sample_queries = st.session_state.dynamic_queries
        col1, col2 = st.columns(2)
        for idx, sample in enumerate(sample_queries):
            with col1 if idx % 2 == 0 else col2:
                if st.button(sample, key=f"sample_{idx}", use_container_width=True):
                    st.session_state.pending_query = sample
                    st.rerun()

        col_regen, col_clear = st.columns(2)
        with col_regen:
            if st.button("Regenerate Suggestions"):
                with st.spinner("Generating new suggestions..."):
                    st.session_state.dynamic_queries = generate_dynamic_queries(st.session_state.processed_invoices)
                st.rerun()
        with col_clear:
            if st.session_state.chat_history and st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.pending_query = None
                st.rerun()

    elif page == "Analytics":
        st.markdown('<div class="main-header">Audit Signals</div>', unsafe_allow_html=True)

        if not st.session_state.processed_invoices:
            st.info("No analytics are available until invoices have been processed.")
        else:
            st.markdown("Track processing volume, approval patterns, vendor distribution, and review audit history.")
            st.markdown("### Signal Summary")

            total = get_total_processed_reports()
            total_amount = sum(
                safe_number(inv.get("invoice_data", {}).get("total_amount"))
                for inv in st.session_state.processed_invoices
            )
            avg_issues = (
                sum(
                    safe_number(inv.get("validation_summary", {}).get("total_issues"))
                    for inv in st.session_state.processed_invoices
                ) / total
                if total > 0 else 0
            )
            approval_rate = (
                len([i for i in st.session_state.processed_invoices if i.get("recommendation") == "APPROVE"]) / total * 100
                if total > 0 else 0
            )
            languages = set([inv.get("original_language", "Unknown") for inv in st.session_state.processed_invoices])
            vendors = set([inv.get("invoice_data", {}).get("vendor_name", "Unknown") for inv in st.session_state.processed_invoices])

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Total Invoices", total)
            with col2:
                st.metric("Total Amount", format_currency(total_amount))
            with col3:
                st.metric("Avg Issues", f"{avg_issues:.1f}")
            with col4:
                st.metric("Approval Rate", f"{approval_rate:.1f}%")
            with col5:
                st.metric("Languages", len(languages))
            with col6:
                st.metric("Vendors", len(vendors))

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Recommendation Mix")
                recommendations = {}
                for inv in st.session_state.processed_invoices:
                    rec = inv.get("recommendation", "UNKNOWN")
                    recommendations[rec] = recommendations.get(rec, 0) + 1
                if recommendations:
                    rec_df = pd.DataFrame(list(recommendations.items()), columns=["Recommendation", "Count"])
                    st.bar_chart(rec_df.set_index("Recommendation"))
                else:
                    st.info("No data")
            with col2:
                st.markdown("### Top Vendors")
                vendor_counts = {}
                for inv in st.session_state.processed_invoices:
                    vendor = inv.get("invoice_data", {}).get("vendor_name", "Unknown")
                    vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
                if vendor_counts:
                    sorted_vendors = sorted(vendor_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    vendor_df = pd.DataFrame(sorted_vendors, columns=["Vendor", "Count"])
                    st.bar_chart(vendor_df.set_index("Vendor"))
                else:
                    st.info("No data")

            st.markdown("---")
            st.markdown("### All Invoices")

            def get_reason_text(inv):
                reason, reason_type = get_rejection_reason(inv)
                if not reason:
                    return ""
                if reason_type == "reviewer":
                    return reason
                if reason_type == "validation":
                    if isinstance(reason, list) and reason:
                        first_issue = reason[0]
                        if isinstance(first_issue, dict):
                            return first_issue.get("message", str(first_issue))
                        return str(first_issue)
                    return str(reason)
                if reason_type == "summary":
                    return reason
                return "See details"

            data = []
            for inv in st.session_state.processed_invoices:
                invoice_data = inv.get("invoice_data", {})
                data.append({
                    "Invoice #": invoice_data.get("invoice_no", "N/A"),
                    "Vendor": invoice_data.get("vendor_name", "N/A"),
                    "Date": invoice_data.get("invoice_date", "N/A"),
                    "Amount": safe_number(invoice_data.get("total_amount")),
                    "Currency": invoice_data.get("currency", "USD"),
                    "Status": inv.get("recommendation", "UNKNOWN"),
                    "Reason": get_reason_text(inv),
                    "Issues": safe_number(inv.get("validation_summary", {}).get("total_issues")),
                    "Language": inv.get("original_language", "Unknown"),
                    "Manual Review": "Yes" if inv.get("manual_override") else "No",
                })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"invoice_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("---")
            st.markdown("### Audit Trail")
            try:
                audit_stats = feedback_manager.get_audit_statistics()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Reviews", audit_stats.get("total_reviews", 0))
                with col2:
                    correction_rate = audit_stats.get("correction_rate", 0) * 100
                    st.metric("Correction Rate", f"{correction_rate:.1f}%")
                with col3:
                    by_type = audit_stats.get("by_type", {})
                    st.metric("Approvals", by_type.get("approval", 0))
                with col4:
                    st.metric("Rejections", by_type.get("rejection", 0))

                with st.expander("View Recent Audit Trail Entries"):
                    audit_entries = feedback_manager.get_audit_trail()[-20:]
                    if audit_entries:
                        audit_df = pd.DataFrame(audit_entries)
                        st.dataframe(audit_df, use_container_width=True)
                        audit_json = json.dumps(audit_entries, indent=2, default=str)
                        st.download_button(
                            "Download Audit Trail (JSON)",
                            data=audit_json,
                            file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                        )
                    else:
                        st.info("No audit entries yet. Make review decisions to build the audit trail.")
            except Exception as e:
                logger.error(f"Error loading audit trail: {str(e)}")
                st.warning("Could not load audit trail data.")
