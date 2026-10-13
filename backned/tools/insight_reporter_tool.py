"""Insight Reporter Tool — generates HTML audit reports summarising all pipeline stage results.

Exposes a LangChain @tool for use by the Reporting Agent.
"""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import tool

from backned.models.reporting import InvoiceReport

logger = logging.getLogger(__name__)

# 
# Path constants
# 
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # → backned/
PROJECT_ROOT = BASE_DIR.parent  # → Invoice_Auditor/
RULES_PATH = BASE_DIR / "config" / "rules.yaml"

#  Cached rules 
_rules: dict | None = None


# 
# Insight Reporter Tool  (LangChain @tool)
# 
@tool
def insight_reporter_tool(
    translation_result: dict,
    validation_result: dict | None = None,
    business_validation_result: dict | None = None,
) -> dict:
    """Insight Reporter Tool — Generates a comprehensive HTML audit
    report for a single invoice.

    Aggregates data from translation, data-validation, and business-
    validation stages into a single ``InvoiceReport`` and writes an
    HTML report file.

    Args:
        translation_result: TranslationResult dict.
        validation_result: ValidationResult dict (optional).
        business_validation_result: BusinessValidationResult dict (optional).

    Returns:
        dict serialised from InvoiceReport (includes report_html_path).
    """
    rules = _load_rules()
    report_cfg = rules.get("reporting", {})
    output_dir = report_cfg.get("output_dir", "./outputs/reports")

    # Resolve output dir relative to project root
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    #  Build the InvoiceReport 
    report = _build_report(
        translation_result,
        validation_result,
        business_validation_result,
        rules,
    )

    #  Generate HTML 
    html = _render_html(report, rules)

    #  Write HTML file 
    safe_name = report.filename.replace(" ", "_").rsplit(".", 1)[0]
    html_filename = f"{report.email_id}_{safe_name}_report.html"
    html_path = out_path / html_filename
    html_path.write_text(html, encoding="utf-8")

    report.report_html_path = str(html_path.resolve())
    json_path = html_path.with_suffix(".json")
    report.report_json_path = str(json_path.resolve())
    json_path.write_text(json.dumps(report.model_dump(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    logger.info(
        "Report generated: %s → %s [%s]",
        report.filename,
        report.recommendation.upper(),
        html_path.name,
    )

    return report.model_dump()


# 
#  HELPER FUNCTIONS  (called by the tool above)
# 


# 
# Rules loader
# 

def _load_rules() -> dict:
    global _rules
    if _rules is None:
        if RULES_PATH.exists():
            with open(RULES_PATH, "r", encoding="utf-8") as fh:
                _rules = yaml.safe_load(fh)
        else:
            _rules = {}
    return _rules


# 
# Build InvoiceReport from pipeline dicts
# 

def _build_report(
    tr: dict,
    vr: dict | None,
    bvr: dict | None,
    rules: dict,
) -> InvoiceReport:
    """Aggregate all pipeline stage outputs into a single InvoiceReport."""

    inv = tr.get("translated_invoice") or {}
    line_items_raw = inv.get("line_items", [])

    # Convert line items to dicts with both original and translated
    line_items = []
    for li in line_items_raw:
        line_items.append({
            "item_code": li.get("item_code"),
            "original_description": li.get("original_description"),
            "translated_description": li.get("translated_description"),
            "qty": li.get("qty"),
            "unit_price": li.get("unit_price"),
            "total": li.get("total"),
        })

    #  Data validation issues 
    dv_issues: list[dict] = []
    if vr:
        for issue in vr.get("header_issues", []):
            dv_issues.append(issue if isinstance(issue, dict) else issue)
        for issue in vr.get("line_item_issues", []):
            dv_issues.append(issue if isinstance(issue, dict) else issue)

    #  Business validation discrepancies 
    bv_discs: list[dict] = []
    if bvr:
        for disc in bvr.get("discrepancies", []):
            bv_discs.append(disc if isinstance(disc, dict) else disc)

    erp_lookup = bvr.get("erp_lookup", {}) if bvr else {}

    #  Recommendation 
    recommendation, reasons = _compute_recommendation(
        tr, vr, bvr, rules
    )

    return InvoiceReport(
        email_id=tr.get("email_id", "unknown"),
        filename=tr.get("filename", "unknown"),
        file_path=tr.get("file_path", ""),
        # Header
        invoice_no=inv.get("invoice_no"),
        invoice_date=inv.get("invoice_date"),
        po_number=inv.get("po_number"),
        vendor_id=inv.get("vendor_id"),
        vendor_name=inv.get("translated_vendor_name"),
        currency=inv.get("currency"),
        subtotal=inv.get("subtotal"),
        tax_amount=inv.get("tax_amount"),
        total_amount=inv.get("total_amount"),
        # Translation
        source_language=tr.get("source_language"),
        was_translated=tr.get("was_translated", False),
        translation_confidence=tr.get("translation_confidence", 1.0),
        # Data validation
        data_validation_status=vr.get("validation_status") if vr else None,
        data_validation_errors=vr.get("total_errors", 0) if vr else 0,
        data_validation_warnings=vr.get("total_warnings", 0) if vr else 0,
        data_validation_issues=dv_issues,
        # Business validation
        business_validation_status=(
            bvr.get("validation_status") if bvr else None
        ),
        business_validation_errors=bvr.get("total_errors", 0) if bvr else 0,
        business_validation_warnings=(
            bvr.get("total_warnings", 0) if bvr else 0
        ),
        business_validation_matched=(
            bvr.get("matched_line_items", 0) if bvr else 0
        ),
        erp_po_found=erp_lookup.get("po_found", False),
        erp_vendor_found=erp_lookup.get("vendor_found", False),
        business_discrepancies=bv_discs,
        # Line items
        line_items=line_items,
        # Recommendation
        recommendation=recommendation,
        recommendation_reasons=reasons,
    )


# 
# Compute auto-approve confidence
# 

def _compute_recommendation(
    tr: dict,
    vr: dict | None,
    bvr: dict | None,
    rules: dict,
) -> tuple[str, list[str]]:
    """Recommend an outcome from configured validation findings."""
    reasons: list[str] = []
    inv = tr.get("translated_invoice") or {}
    dv_errors = vr.get("total_errors", 0) if vr else 0
    bv_errors = bvr.get("total_errors", 0) if bvr else 0
    discrepancies = bvr.get("discrepancies", []) if bvr else []
    policies = rules.get("validation_policies", {})
    if bvr and bvr.get("validation_status") == "po_not_found":
        reasons.append("Purchase order was not found in the ERP service")
        return "reject", reasons
    if dv_errors or bv_errors or discrepancies:
        reasons.append(f"Data validation found {dv_errors} issue(s)")
        reasons.append(f"ERP validation found {bv_errors} issue(s)")
        reasons.extend(str(item.get("message", "Discrepancy detected")) for item in discrepancies)
        return "manual_review", reasons
    reasons.append("Required fields and configured invoice checks passed")
    reasons.append("ERP comparisons found no discrepancies")
    return "approve", reasons

# HTML report renderer
# 

# Accent colour per recommendation
_REC_PALETTE = {
    "approve": {"bg": "#0f5132", "light": "#d1e7dd", "text": "#0f5132", "bar": "#198754"},
    "manual_review": {"bg": "#664d03", "light": "#fff3cd", "text": "#664d03", "bar": "#ffc107"},
    "reject":        {"bg": "#842029", "light": "#f8d7da", "text": "#842029", "bar": "#dc3545"},
}
_REC_ICONS = {"approve": "✓", "manual_review": "!", "reject": "✗"}


def _render_html(report: InvoiceReport, rules: dict) -> str:
    """Assemble all HTML sections into a complete report page."""
    include_confidence = rules.get("reporting", {}).get("include_translation_confidence", True)
    include_discrepancy = rules.get("reporting", {}).get("include_discrepancy_summary", True)

    pal = _REC_PALETTE.get(report.recommendation, _REC_PALETTE["manual_review"])
    rec_label = report.recommendation.replace("_", " ").title()
    generated_at = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    parts = [
        _html_skeleton_open(report, pal, rec_label, generated_at),
        _html_invoice_header(report),
    ]
    if include_confidence:
        parts.append(_html_confidence_row(report, pal))
    parts.append(_html_data_validation_section(report))
    if include_discrepancy:
        parts.append(_html_business_validation_section(report))
    parts.append(_html_line_items_section(report))
    parts.append(_html_recommendation_reasons(report, pal))
    parts.append(_html_skeleton_close(generated_at))
    return "\n".join(parts)


#  HTML skeleton 

def _html_skeleton_open(
    report: InvoiceReport, pal: dict, rec_label: str, generated_at: str,
) -> str:
    icon = _REC_ICONS.get(report.recommendation, "?")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit Report · {_esc(report.filename)}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px; line-height: 1.55; background: #f0f2f5; color: #1e2330;
  }}
  a {{ color: inherit; text-decoration: none; }}

  /*  Page wrapper  */
  .page {{ max-width: 980px; margin: 24px auto; }}

  /*  Top bar  */
  .topbar {{
    background: #1e2330; color: #fff; border-radius: 10px 10px 0 0;
    padding: 18px 28px; display: flex; align-items: center; justify-content: space-between;
  }}
  .topbar-left {{ display: flex; align-items: center; gap: 14px; }}
  .topbar-icon {{
    width: 38px; height: 38px; border-radius: 8px;
    background: {pal["bar"]}; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; font-weight: 700; flex-shrink: 0;
  }}
  .topbar-title {{ font-size: 14px; font-weight: 600; }}
  .topbar-sub {{ font-size: 11px; color: #8892a4; margin-top: 1px; }}
  .topbar-date {{ font-size: 11px; color: #8892a4; text-align: right; }}

  /*  Verdict strip  */
  .verdict {{
    background: {pal["light"]}; border-left: 4px solid {pal["bar"]};
    padding: 10px 28px; display: flex; align-items: center; gap: 10px;
    font-size: 12px; color: {pal["text"]};
  }}
  .verdict-icon {{
    width: 22px; height: 22px; border-radius: 50%;
    background: {pal["bar"]}; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; flex-shrink: 0;
  }}
  .verdict-label {{ font-weight: 700; font-size: 12px; }}
  .verdict-sub {{ color: #555; font-size: 11px; }}

  /*  Card shell  */
  .card {{
    background: #fff; border-radius: 0; border-bottom: 1px solid #e5e8ed;
    padding: 20px 28px;
  }}
  .card:last-of-type {{ border-radius: 0 0 10px 10px; border-bottom: none; }}
  .card-title {{
    font-size: 11px; font-weight: 700; letter-spacing: .6px;
    text-transform: uppercase; color: #8892a4; margin-bottom: 14px;
    display: flex; align-items: center; gap: 7px;
  }}
  .card-title::before {{
    content: ""; display: block; width: 3px; height: 13px;
    border-radius: 2px; background: {pal["bar"]};
  }}

  /*  Field grid  */
  .fgrid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px 16px; }}
  .fgrid-3 {{ grid-template-columns: repeat(3, 1fr); }}
  .fgrid-2 {{ grid-template-columns: repeat(2, 1fr); }}
  .fl {{ font-size: 10px; color: #8892a4; text-transform: uppercase;
         letter-spacing: .4px; margin-bottom: 2px; }}
  .fv {{ font-size: 12.5px; font-weight: 500; color: #1e2330; }}
  .fv-lg {{ font-size: 15px; font-weight: 700; color: #1e2330; }}
  .fv-mono {{ font-family: "SF Mono", "Cascadia Code", Consolas, monospace;
              font-size: 11.5px; }}
  .fv-muted {{ font-size: 11px; color: #8892a4; word-break: break-all; }}

  /*  Scorecard row  */
  .scorecard {{ display: flex; gap: 12px; }}
  .sc {{
    flex: 1; background: #f8f9fb; border: 1px solid #e5e8ed;
    border-radius: 8px; padding: 12px 14px; text-align: center;
  }}
  .sc-val {{ font-size: 20px; font-weight: 700; line-height: 1.2; }}
  .sc-lbl {{ font-size: 10px; color: #8892a4; margin-top: 2px; letter-spacing: .3px; }}
  .sc-green .sc-val {{ color: #198754; }}
  .sc-amber .sc-val {{ color: #d97706; }}
  .sc-red   .sc-val {{ color: #dc3545; }}
  .sc-blue  .sc-val {{ color: #0d6efd; }}
  .sc-dark  .sc-val {{ color: #1e2330; }}

  /*  Confidence bar  */
  .conf-wrap {{ margin-top: 8px; }}
  .conf-track {{
    height: 6px; background: #e5e8ed; border-radius: 3px; overflow: hidden;
    margin-top: 5px;
  }}
  .conf-fill {{ height: 100%; border-radius: 3px; }}
  .conf-labels {{
    display: flex; justify-content: space-between;
    font-size: 10px; color: #8892a4; margin-top: 3px;
  }}

  /*  Pill badges  */
  .pill {{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 20px; font-size: 10.5px; font-weight: 600;
  }}
  .pill-green  {{ background: #d1e7dd; color: #0f5132; }}
  .pill-amber  {{ background: #fff3cd; color: #664d03; }}
  .pill-red    {{ background: #f8d7da; color: #842029; }}
  .pill-blue   {{ background: #cfe2ff; color: #084298; }}
  .pill-grey   {{ background: #e9ecef; color: #495057; }}

  /*  Tables  */
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }}
  thead tr {{ background: #f8f9fb; }}
  th {{
    padding: 8px 10px; text-align: left; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .4px; color: #8892a4;
    border-bottom: 1px solid #e5e8ed; white-space: nowrap;
  }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #f0f2f5; vertical-align: top; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover td {{ background: #f8f9fb; }}
  .tr-err td  {{ background: #fff5f5; }}
  .tr-warn td {{ background: #fffdf0; }}
  td.num {{ text-align: right; font-family: "SF Mono", Consolas, monospace; font-size: 11.5px; }}
  th.num {{ text-align: right; }}

  /*  ERP lookup row  */
  .erp-row {{ display: flex; gap: 24px; margin-top: 12px; margin-bottom: 4px; }}
  .erp-item {{ display: flex; align-items: center; gap: 7px; font-size: 12px; }}
  .erp-label {{ color: #8892a4; font-size: 11px; }}

  /*  Reasons list  */
  .reasons {{ list-style: none; }}
  .reasons li {{
    padding: 6px 0 6px 16px; border-bottom: 1px solid #f0f2f5;
    position: relative; font-size: 12px; color: #3a4258;
  }}
  .reasons li:last-child {{ border-bottom: none; }}
  .reasons li::before {{
    content: ""; position: absolute; left: 0; top: 12px;
    width: 6px; height: 6px; border-radius: 50%; background: {pal["bar"]};
  }}

  /*  Pass banner  */
  .pass-banner {{
    display: flex; align-items: center; gap: 8px; padding: 8px 12px;
    background: #d1e7dd; border-radius: 6px; color: #0f5132;
    font-size: 12px; font-weight: 500; margin-top: 10px;
  }}

  /*  Footer  */
  .footer {{
    text-align: center; font-size: 10.5px; color: #8892a4;
    padding: 14px 28px; background: #f8f9fb;
    border-radius: 0 0 10px 10px; border-top: 1px solid #e5e8ed;
  }}
</style>
</head>
<body>
<div class="page">

<!-- Top bar -->
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-icon">{icon}</div>
    <div>
      <div class="topbar-title">Invoice Audit Report</div>
      <div class="topbar-sub">{_esc(report.filename)}&ensp;·&ensp;{_esc(report.email_id)}</div>
    </div>
  </div>
  <div class="topbar-date">Generated<br>{generated_at}</div>
</div>

<!-- Verdict strip -->
<div class="verdict">
  <div class="verdict-icon">{icon}</div>
  <div>
    <span class="verdict-label">{rec_label}</span>
    <span class="verdict-sub">&ensp;·&ensp;Translation confidence: <strong>{report.translation_confidence:.0%}</strong></span>
  </div>
</div>"""


def _html_skeleton_close(generated_at: str) -> str:
    return f"""
<div class="footer">
  AI Invoice Auditor &nbsp;·&nbsp; {generated_at}
</div>

</div><!-- /page -->
</body>
</html>"""


#  Scorecard 

def _html_invoice_header(report: InvoiceReport) -> str:
    return f"""
<div class="card">
  <div class="card-title">Invoice Details</div>
  <div class="fgrid">
    <div>
      <div class="fl">Invoice No.</div>
      <div class="fv fv-mono">{_esc(report.invoice_no or '—')}</div>
    </div>
    <div>
      <div class="fl">Invoice Date</div>
      <div class="fv">{_esc(report.invoice_date or '—')}</div>
    </div>
    <div>
      <div class="fl">PO Number</div>
      <div class="fv fv-mono">{_esc(report.po_number or '—')}</div>
    </div>
    <div>
      <div class="fl">Currency</div>
      <div class="fv">{_esc(report.currency or '—')}</div>
    </div>
    <div>
      <div class="fl">Vendor ID</div>
      <div class="fv fv-mono">{_esc(report.vendor_id or '—')}</div>
    </div>
    <div>
      <div class="fl">Vendor Name</div>
      <div class="fv">{_esc(report.vendor_name or '—')}</div>
    </div>
    <div>
      <div class="fl">Subtotal</div>
      <div class="fv fv-mono">{_fmt_amount(report.subtotal, report.currency)}</div>
    </div>
    <div>
      <div class="fl">Tax</div>
      <div class="fv fv-mono">{_fmt_amount(report.tax_amount, report.currency)}</div>
    </div>
  </div>
  <div style="margin-top:14px; padding-top:12px; border-top:1px solid #f0f2f5;
              display:flex; align-items:baseline; gap:10px;">
    <span style="font-size:10px; color:#8892a4; text-transform:uppercase;
                 letter-spacing:.4px;">Total Amount</span>
    <span class="fv-lg">{_fmt_amount(report.total_amount, report.currency)}</span>
  </div>
  <div style="margin-top:8px;">
    <div class="fl">Source File</div>
    <div class="fv-muted">{_esc(report.file_path)}</div>
  </div>
</div>"""


#  Confidence row 

def _html_confidence_row(report: InvoiceReport, pal: dict) -> str:
    aa_pct = report.translation_confidence * 100
    fill_color = pal["bar"]
    lang_pill = (
        f'<span class="pill pill-blue">{_esc(report.source_language or "?")} → en &nbsp;Translated</span>'
        if report.was_translated
        else '<span class="pill pill-green">Original English</span>'
    )
    segments = [
        ("Fields",     "", ""),
        ("Vendor",     "", ""),
        ("Item Codes", "", ""),
        ("Data Val.",  "", ""),
        ("ERP Val.",   "", ""),
        ("Translation","", ""),
    ]
    segment_html = " ".join(
        f'<span style="font-size:10px; color:#8892a4;">{s[0]}</span>'
        for s in segments
    )
    return f"""
<div class="card">
  <div class="card-title">Confidence &amp; Language</div>
  <div style="display:flex; gap:32px; align-items:flex-start; flex-wrap:wrap;">
    <div style="flex:1; min-width:240px;">
      <div class="fl">Source Language</div>
      <div class="fv" style="margin-top:3px;">{lang_pill}</div>
    </div>
    <div style="flex:2; min-width:280px;">
      <div style="display:flex; justify-content:space-between; align-items:baseline;">
      <div class="fl">Translation Confidence</div>
        <div style="font-size:16px; font-weight:700; color:{fill_color};">{aa_pct:.0f}%</div>
      </div>
      <div class="conf-track" style="margin-top:6px; height:8px;">
        <div class="conf-fill" style="width:{aa_pct:.0f}%; background:{fill_color};"></div>
      </div>
      <div class="conf-labels">
        <span>0%</span>
        <span style="color:{fill_color}; font-weight:600;">{aa_pct:.0f}%</span>
        <span>100%</span>
      </div>
      <div style="font-size:10px; color:#8892a4; margin-top:4px;">
        Confidence reported by the translation stage
      </div>
    </div>
  </div>
</div>"""


#  Data validation 

def _html_data_validation_section(report: InvoiceReport) -> str:
    status = report.data_validation_status or "N/A"
    status_pill = _pill(status)
    dv_w = report.data_validation_warnings

    if report.data_validation_issues:
        rows_html = ""
        for issue in report.data_validation_issues:
            sev = issue.get("severity", "error")
            tr_cls = "tr-err" if sev == "error" else "tr-warn"
            sev_pill = (
                '<span class="pill pill-red">Error</span>'
                if sev == "error"
                else '<span class="pill pill-amber">Warning</span>'
            )
            line_idx = issue.get("line_index")
            scope = f"Row&nbsp;{line_idx}" if line_idx is not None else "Header"
            field = _esc(issue.get("field_name", "—"))
            itype = _esc(issue.get("issue_type", "—"))
            msg   = _esc(issue.get("message", "—"))
            rows_html += f"""
      <tr class="{tr_cls}">
        <td>{sev_pill}</td><td>{scope}</td>
        <td class="fv-mono" style="font-size:11.5px;">{field}</td>
        <td>{itype}</td><td>{msg}</td>
      </tr>"""

        body = f"""
  <table>
    <thead><tr>
      <th>Sev.</th><th>Scope</th><th>Field</th><th>Check</th><th>Detail</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>"""
    else:
        body = '<div class="pass-banner">&#10003;&ensp;All data completeness checks passed — no issues found.</div>'

    return f"""
<div class="card">
  <div class="card-title">Data Validation</div>
  <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:4px;">
    <div>{status_pill}</div>
    <div style="font-size:11px; color:#8892a4;">
      <span style="color:#dc3545; font-weight:600;">{report.data_validation_errors}</span> error(s)
      &ensp;·&ensp;
      <span style="color:#d97706; font-weight:600;">{dv_w}</span> warning(s)
    </div>
  </div>
  {body}
</div>"""


#  Business validation 

def _html_business_validation_section(report: InvoiceReport) -> str:
    status = report.business_validation_status or "N/A"
    status_pill = _pill(status)

    po_pill = (
        '<span class="pill pill-green">&#10003;&ensp;Found</span>'
        if report.erp_po_found
        else '<span class="pill pill-red">&#10007;&ensp;Not Found</span>'
    )
    vendor_pill = (
        '<span class="pill pill-green">&#10003;&ensp;Found</span>'
        if report.erp_vendor_found
        else '<span class="pill pill-red">&#10007;&ensp;Not Found</span>'
    )

    if report.business_discrepancies:
        rows_html = ""
        for disc in report.business_discrepancies:
            sev = disc.get("severity", "error")
            tr_cls = "tr-err" if sev == "error" else "tr-warn"
            sev_pill = (
                '<span class="pill pill-red">Error</span>'
                if sev == "error"
                else '<span class="pill pill-amber">Warning</span>'
            )
            line_idx = disc.get("line_index")
            scope = f"Row&nbsp;{line_idx}" if line_idx is not None else "Header"
            dtype  = _esc(disc.get("discrepancy_type", "—"))
            msg    = _esc(disc.get("message", "—"))
            inv_v  = _esc(str(disc.get("invoice_value", "—")))
            erp_v  = _esc(str(disc.get("erp_value", "—")))
            diff   = disc.get("difference_pct")
            diff_s = f"{diff:+.1f}%" if diff is not None else "—"
            rows_html += f"""
      <tr class="{tr_cls}">
        <td>{sev_pill}</td><td>{scope}</td><td>{dtype}</td>
        <td>{msg}</td>
        <td class="num">{inv_v}</td>
        <td class="num">{erp_v}</td>
        <td class="num" style="color:#d97706;">{diff_s}</td>
      </tr>"""

        body = f"""
  <table>
    <thead><tr>
      <th>Sev.</th><th>Scope</th><th>Type</th><th>Detail</th>
      <th class="num">Invoice</th><th class="num">ERP</th><th class="num">Diff</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>"""
    else:
        body = '<div class="pass-banner">&#10003;&ensp;All line items matched ERP records — no discrepancies.</div>'

    return f"""
<div class="card">
  <div class="card-title">Business Validation · ERP Cross-Reference</div>
  <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:4px;">
    <div>{status_pill}</div>
    <div style="font-size:11px; color:#8892a4;">
      <span style="color:#198754; font-weight:600;">{report.business_validation_matched}</span> matched
      &ensp;·&ensp;
      <span style="color:#dc3545; font-weight:600;">{report.business_validation_errors}</span> error(s)
      &ensp;·&ensp;
      <span style="color:#d97706; font-weight:600;">{report.business_validation_warnings}</span> warning(s)
    </div>
  </div>
  <div class="erp-row">
    <div class="erp-item">
      <span class="erp-label">PO</span>
      <span class="fv-mono" style="font-size:11.5px;">{_esc(report.po_number or '—')}</span>
      {po_pill}
    </div>
    <div class="erp-item">
      <span class="erp-label">Vendor</span>
      <span class="fv-mono" style="font-size:11.5px;">{_esc(report.vendor_id or '—')}</span>
      {vendor_pill}
    </div>
  </div>
  {body}
</div>"""


#  Line items 

def _html_line_items_section(report: InvoiceReport) -> str:
    if not report.line_items:
        return """
<div class="card">
  <div class="card-title">Line Items</div>
  <p style="color:#8892a4; font-size:12px;">No line items extracted.</p>
</div>"""

    rows_html = ""
    for i, li in enumerate(report.line_items):
        code  = _esc(str(li.get("item_code") or "—"))
        orig  = str(li.get("original_description") or "—")
        trans = str(li.get("translated_description") or "—")
        qty   = li.get("qty")
        price = li.get("unit_price")
        total = li.get("total")

        if report.was_translated and orig != trans and orig != "—":
            desc_html = (
                f'{_esc(trans)}'
                f'<br><span style="font-size:10px;color:#8892a4;">{_esc(orig)}</span>'
            )
        else:
            desc_html = _esc(trans)

        rows_html += f"""
    <tr>
      <td style="color:#8892a4; font-size:11px; width:28px;">{i+1}</td>
      <td class="fv-mono" style="font-size:11.5px; white-space:nowrap;">{code}</td>
      <td>{desc_html}</td>
      <td class="num">{_fmt_num(qty)}</td>
      <td class="num">{_fmt_num(price)}</td>
      <td class="num" style="font-weight:600;">{_fmt_num(total)}</td>
    </tr>"""

    return f"""
<div class="card">
  <div class="card-title">Line Items <span style="font-weight:400; color:#8892a4;">({len(report.line_items)})</span></div>
  <table>
    <thead><tr>
      <th style="width:28px;">#</th>
      <th>Item Code</th>
      <th>Description</th>
      <th class="num">Qty</th>
      <th class="num">Unit Price</th>
      <th class="num">Total</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


#  Recommendation reasons 

def _html_recommendation_reasons(report: InvoiceReport, pal: dict) -> str:
    if not report.recommendation_reasons:
        return ""
    items_html = "".join(
        f"<li>{_esc(r)}</li>" for r in report.recommendation_reasons
    )
    return f"""
<div class="card">
  <div class="card-title">Decision Rationale</div>
  <ul class="reasons">{items_html}</ul>
</div>"""


#  Utility helpers 

def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_amount(val: float | None, currency: str | None) -> str:
    if val is None:
        return "N/A"
    c = currency or ""
    return f"{c} {val:,.2f}" if c else f"{val:,.2f}"


def _fmt_num(val: float | None) -> str:
    if val is None:
        return "—"
    if val == int(val):
        return f"{int(val)}"
    return f"{val:,.2f}"


def _status_badge(status: str) -> str:
    """Return a coloured pill for a validation status string (legacy compat)."""
    return _pill(status)


def _pill(status: str) -> str:
    """Return a styled pill badge for a validation status string."""
    s = status.lower().replace("_", " ")
    if status in ("passed", "matched"):
        cls = "pill-green"
    elif status in ("passed_with_warnings", "discrepancies_found", "vendor_not_found"):
        cls = "pill-amber"
    elif status in ("failed", "po_not_found", "skipped"):
        cls = "pill-red"
    else:
        cls = "pill-grey"
    return f'<span class="pill {cls}">{s.title()}</span>'
