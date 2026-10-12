"""
Invoice Auditor — Agents Package
"""

from backned.models import (
    EmailMeta,
    InvoiceFile,
    MonitoredEmail,
    MonitorResult,
    ExtractedLineItem,
    ExtractedInvoice,
    ExtractionResult,
    TranslatedLineItem,
    TranslatedInvoice,
    TranslationResult,
    FieldIssue,
    LineItemIssue,
    ValidationResult,
    LineItemDiscrepancy,
    ERPLookupResult,
    BusinessValidationResult,
    InvoiceReport,
    ReportingSummary,
)
from .invoice_monitor_agent import InvoiceMonitorAgent
from .extractor_agent import ExtractorAgent
from .translation_agent import TranslationAgent
from .validation_agent import ValidationAgent
from .business_validation_agent import BusinessValidationAgent
from .reporting_agent import ReportingAgent

# Tools are now accessed via MCP client
# Use: from backned.mcp.clients import get_tool_by_name, get_tools_sync

__all__ = [
    "EmailMeta",
    "InvoiceFile",
    "MonitoredEmail",
    "MonitorResult",
    "ExtractedLineItem",
    "ExtractedInvoice",
    "ExtractionResult",
    "TranslatedLineItem",
    "TranslatedInvoice",
    "TranslationResult",
    "FieldIssue",
    "LineItemIssue",
    "ValidationResult",
    "LineItemDiscrepancy",
    "ERPLookupResult",
    "BusinessValidationResult",
    "InvoiceReport",
    "ReportingSummary",
    "InvoiceMonitorAgent",
    "ExtractorAgent",
    "TranslationAgent",
    "ValidationAgent",
    "BusinessValidationAgent",
    "ReportingAgent",
]
