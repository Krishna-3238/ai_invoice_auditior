"""Business Validation Tool — cross-verifies invoice data against the mock ERP (PO, vendor, SKU).

Exposes a LangChain @tool for use by the Business Validation Agent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import tool

from backned.models.business_validation import (
    BusinessValidationResult,
    ERPLookupResult,
    LineItemDiscrepancy,
)
from backned.config.settings import ERP_API_BASE_URL

logger = logging.getLogger(__name__)

# 
# Constants
# 
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # → backned/
RULES_PATH = BASE_DIR / "config" / "rules.yaml"
ERP_DATA_DIR = BASE_DIR / "data" / "ERP_mockdata"


_rules: dict | None = None


# 
# Business Validation Tool  (LangChain @tool)
# 
@tool
def business_validation_tool(
    translation_result: dict,
    validation_result: dict | None = None,
) -> dict:
    """Business Validation Tool — Cross-verifies invoice data against
    the enterprise system (ERP) to identify mismatches at the line-item
    level.

    Calls the Mock ERP API to fetch PO records, then compares every
    invoice line item against the ERP data for qty, unit_price, vendor,
    and currency discrepancies.

    Args:
        translation_result: A TranslationResult dict (translated invoice).
        validation_result: Optional ValidationResult dict from the
            data completeness checker (used for pass-through context).

    Returns:
        dict serialised from BusinessValidationResult.
    """
    rules = _load_rules()

    email_id = translation_result.get("email_id", "unknown")
    filename = translation_result.get("filename", "?")
    file_path = translation_result.get("file_path", "")
    source_language = translation_result.get("source_language")

    inv = translation_result.get("translated_invoice") or {}

    po_number = inv.get("po_number")
    vendor_id = inv.get("vendor_id")
    invoice_currency = inv.get("currency")
    invoice_line_items = inv.get("line_items", [])

    #  Guard: no PO number → can't cross-reference 
    if not po_number:
        logger.warning(
            "%s: no PO number found — skipping business validation",
            filename,
        )
        return BusinessValidationResult(
            email_id=email_id,
            filename=filename,
            file_path=file_path,
            is_valid=False,
            validation_status="skipped",
            erp_lookup=ERPLookupResult(
                invoice_line_items_count=len(invoice_line_items),
            ),
            discrepancies=[LineItemDiscrepancy(
                line_index=-1,
                field_name="po_number",
                discrepancy_type="item_not_in_po",
                severity="error",
                message="No PO number on invoice — cannot cross-reference with ERP",
            )],
            total_errors=1,
            source_language=source_language,

        ).model_dump()

    #  Fetch ERP data 
    po_data = _fetch_po(po_number)
    vendor_data = _fetch_vendor(vendor_id) if vendor_id else None
    sku_data = _fetch_all_skus()

    discrepancies: list[LineItemDiscrepancy] = []

    #  PO not found 
    if po_data is None:
        logger.warning("[PO NOT FOUND] %s: PO '%s'not found in ERP", filename, po_number)
        return BusinessValidationResult(
            email_id=email_id,
            filename=filename,
            file_path=file_path,
            is_valid=False,
            validation_status="po_not_found",
            erp_lookup=ERPLookupResult(
                po_number=po_number,
                po_found=False,
                vendor_id=vendor_id,
                vendor_found=vendor_data is not None,
                invoice_line_items_count=len(invoice_line_items),
            ),
            discrepancies=[LineItemDiscrepancy(
                line_index=-1,
                field_name="po_number",
                discrepancy_type="item_not_in_po",
                severity="error",
                message=f"PO '{po_number}' not found in ERP system",
                invoice_value=po_number,
            )],
            total_errors=1,
            source_language=source_language,
        ).model_dump()

    erp_line_items = po_data.get("line_items", [])
    erp_vendor_id = po_data.get("vendor_id")

    # Build ERP line-item index by item_code
    erp_item_index: dict[str, dict] = {}
    for erp_li in erp_line_items:
        code = erp_li.get("item_code", "")
        if code:
            erp_item_index[code.upper()] = erp_li

    # Build ERP description index for fallback matching
    erp_desc_index: dict[str, dict] = {}
    for erp_li in erp_line_items:
        desc = (erp_li.get("description") or "").strip().lower()
        if desc:
            erp_desc_index[desc] = erp_li

    #  1. Vendor verification 
    if vendor_id and erp_vendor_id:
        if vendor_id.upper() != erp_vendor_id.upper():
            discrepancies.append(LineItemDiscrepancy(
                line_index=-1,
                field_name="vendor_id",
                discrepancy_type="vendor_mismatch",
                severity="error",
                message=(
                    f"Invoice vendor '{vendor_id}' does not match "
                    f"PO vendor '{erp_vendor_id}'"
                ),
                invoice_value=vendor_id,
                erp_value=erp_vendor_id,
            ))

    #  2. Vendor exists in master 
    if vendor_id and vendor_data is None:
        discrepancies.append(LineItemDiscrepancy(
            line_index=-1,
            field_name="vendor_id",
            discrepancy_type="vendor_mismatch",
            severity="warning",
            message=f"Vendor '{vendor_id}' not found in ERP Vendor Master",
            invoice_value=vendor_id,
        ))

    #  3. Currency match 
    erp_currency = None
    if erp_line_items:
        erp_currency = erp_line_items[0].get("currency")
    if invoice_currency and erp_currency:
        if invoice_currency.upper() != erp_currency.upper():
            discrepancies.append(LineItemDiscrepancy(
                line_index=-1,
                field_name="currency",
                discrepancy_type="currency_mismatch",
                severity="error",
                message=(
                    f"Invoice currency '{invoice_currency}' does not "
                    f"match ERP PO currency '{erp_currency}'"
                ),
                invoice_value=invoice_currency,
                erp_value=erp_currency,
            ))

    #  4. Line-item cross-reference 
    tolerances = rules.get("tolerances", {})
    price_tol = tolerances.get("price_difference_percent", 5)
    qty_tol = tolerances.get("quantity_difference_percent", 0)

    matched_items = 0
    matched_invoice_codes: set[str] = set()

    for idx, inv_li in enumerate(invoice_line_items):
        inv_code = (inv_li.get("item_code") or "").upper()
        inv_desc = (
            inv_li.get("translated_description")
            or inv_li.get("description")
            or inv_li.get("original_description")
            or ""
        ).strip().lower()

        # 4a. Try matching by item_code first
        erp_li_match: dict | None = None
        matched_by = ""

        if inv_code and inv_code in erp_item_index:
            erp_li_match = erp_item_index[inv_code]
            matched_by = "item_code"

        # 4b. Fallback: match by description when item_code is missing
        elif inv_desc:
            erp_li_match = _match_by_description(
                inv_desc, erp_desc_index
            )
            if erp_li_match:
                matched_by = "description"
                logger.info(
                    "     Line [%d]: item_code missing — matched by "
                    "description '%s' → ERP item '%s'",
                    idx, inv_desc,
                    erp_li_match.get("item_code", "?"),
                )

        if erp_li_match is None:
            discrepancies.append(LineItemDiscrepancy(
                line_index=idx,
                item_code=inv_li.get("item_code"),
                field_name="item_code",
                discrepancy_type="item_not_in_po",
                severity="error",
                message=(
                    f"Line [{idx}]: item '{inv_li.get('item_code') or '(no code)'}' "
                    f"(desc: '{inv_desc or '?'}') "
                    f"not found in ERP PO '{po_number}'"
                ),
                invoice_value=inv_li.get("item_code") or inv_desc,
            ))

            # 4c. Check SKU Master
            if inv_code and inv_code not in {
                s.upper() for s in sku_data
            }:
                discrepancies.append(LineItemDiscrepancy(
                    line_index=idx,
                    item_code=inv_li.get("item_code"),
                    field_name="item_code",
                    discrepancy_type="item_not_in_sku_master",
                    severity="warning",
                    message=(
                        f"Line [{idx}]: item '{inv_li.get('item_code')}' "
                        f"not found in ERP SKU Master"
                    ),
                    invoice_value=inv_li.get("item_code"),
                ))
            continue

        # We have a match — track the ERP item code for the
        # "missing items" check later
        erp_match_code = (
            erp_li_match.get("item_code", "")
        ).upper()
        matched_invoice_codes.add(erp_match_code)
        item_match = True

        # 4d. Quantity comparison
        inv_qty = inv_li.get("qty")
        erp_qty = erp_li_match.get("qty")
        if inv_qty is not None and erp_qty is not None:
            try:
                inv_q = float(inv_qty)
                erp_q = float(erp_qty)
                if erp_q != 0:
                    qty_diff_pct = abs((inv_q - erp_q) / erp_q) * 100
                elif inv_q != 0:
                    qty_diff_pct = 100.0
                else:
                    qty_diff_pct = 0.0

                if qty_diff_pct > qty_tol:
                    item_match = False
                    item_label = inv_code or erp_match_code or inv_desc
                    discrepancies.append(LineItemDiscrepancy(
                        line_index=idx,
                        item_code=inv_li.get("item_code") or erp_li_match.get("item_code"),
                        field_name="qty",
                        discrepancy_type="quantity_mismatch",
                        severity="error",
                        message=(
                            f"Line [{idx}] {item_label}: qty mismatch — "
                            f"invoice={inv_q}, ERP={erp_q} "
                            f"(diff: {qty_diff_pct:.1f}%, "
                            f"tolerance: {qty_tol}%) "
                            f"[matched by {matched_by}]"
                        ),
                        invoice_value=str(inv_q),
                        erp_value=str(erp_q),
                        difference_pct=round(qty_diff_pct, 2),
                    ))
            except (ValueError, TypeError):
                pass

        # 4e. Unit price comparison
        inv_price = inv_li.get("unit_price")
        erp_price = erp_li_match.get("unit_price")
        if inv_price is not None and erp_price is not None:
            try:
                inv_p = float(inv_price)
                erp_p = float(erp_price)
                if erp_p != 0:
                    price_diff_pct = abs((inv_p - erp_p) / erp_p) * 100
                elif inv_p != 0:
                    price_diff_pct = 100.0
                else:
                    price_diff_pct = 0.0

                if price_diff_pct > price_tol:
                    item_match = False
                    item_label = inv_code or erp_match_code or inv_desc
                    discrepancies.append(LineItemDiscrepancy(
                        line_index=idx,
                        item_code=inv_li.get("item_code") or erp_li_match.get("item_code"),
                        field_name="unit_price",
                        discrepancy_type="price_mismatch",
                        severity="error",
                        message=(
                            f"Line [{idx}] {item_label}: price mismatch — "
                            f"invoice={inv_p}, ERP={erp_p} "
                            f"(diff: {price_diff_pct:.1f}%, "
                            f"tolerance: {price_tol}%) "
                            f"[matched by {matched_by}]"
                        ),
                        invoice_value=str(inv_p),
                        erp_value=str(erp_p),
                        difference_pct=round(price_diff_pct, 2),
                    ))
            except (ValueError, TypeError):
                pass

        if item_match:
            matched_items += 1

    #  5. Missing items (in ERP PO but not on invoice) 
    for erp_code, erp_li in erp_item_index.items():
        if erp_code not in matched_invoice_codes:
            discrepancies.append(LineItemDiscrepancy(
                line_index=-1,
                item_code=erp_li.get("item_code"),
                field_name="item_code",
                discrepancy_type="missing_item_on_invoice",
                severity="warning",
                message=(
                    f"ERP PO item '{erp_li.get('item_code')}' "
                    f"({erp_li.get('description', '?')}) is in the PO "
                    f"but missing from the invoice"
                ),
                erp_value=erp_li.get("item_code"),
            ))

    #  Aggregate 
    total_errors = sum(
        1 for d in discrepancies if d.severity == "error"
    )
    total_warnings = sum(
        1 for d in discrepancies if d.severity == "warning"
    )
    is_valid = total_errors == 0

    if not discrepancies:
        validation_status = "matched"
    else:
        validation_status = "discrepancies_found"

    erp_lookup = ERPLookupResult(
        po_number=po_number,
        po_found=True,
        vendor_id=vendor_id,
        vendor_found=vendor_data is not None,
        erp_line_items_count=len(erp_line_items),
        invoice_line_items_count=len(invoice_line_items),
        erp_currency=erp_currency,
        erp_vendor_name=(
            vendor_data.get("vendor_name") if vendor_data else None
        ),
    )

    result = BusinessValidationResult(
        email_id=email_id,
        filename=filename,
        file_path=file_path,
        is_valid=is_valid,
        validation_status=validation_status,
        erp_lookup=erp_lookup,
        discrepancies=discrepancies,
        total_errors=total_errors,
        total_warnings=total_warnings,
        matched_line_items=matched_items,
        source_language=source_language,
    )

    logger.info(
        "%s — %s | matched: %d/%d | errors: %d | warnings: %d",
        filename, validation_status.upper(),
        matched_items, len(invoice_line_items),
        total_errors, total_warnings,
    )

    return result.model_dump()


# 
#  HELPER FUNCTIONS  (called by the tool above)
# 


# 
# Description-based fuzzy matching
# 

def _match_by_description(
    inv_desc: str, erp_desc_index: dict[str, dict]
) -> dict | None:
    """Try to match an invoice line item to an ERP line item by
    description when item_code is missing.

    Strategy:
      1. Exact match (case-insensitive).
      2. Substring containment (either direction).
      3. Word-overlap ratio ≥ 60%.
    """
    inv_lower = inv_desc.strip().lower()
    if not inv_lower:
        return None

    # 1. Exact match
    if inv_lower in erp_desc_index:
        return erp_desc_index[inv_lower]

    # 2. Substring containment
    for erp_desc, erp_li in erp_desc_index.items():
        if inv_lower in erp_desc or erp_desc in inv_lower:
            return erp_li

    # 3. Word-overlap ratio
    inv_words = set(inv_lower.split())
    best_match: dict | None = None
    best_ratio = 0.0

    for erp_desc, erp_li in erp_desc_index.items():
        erp_words = set(erp_desc.split())
        if not erp_words or not inv_words:
            continue
        overlap = len(inv_words & erp_words)
        ratio = overlap / max(len(inv_words), len(erp_words))
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = erp_li

    if best_ratio >= 0.6:
        return best_match

    return None


# 
# Rules loader
# 

def _load_rules() -> dict:
    """Load and cache the rules.yaml configuration."""
    global _rules
    if _rules is None:
        if RULES_PATH.exists():
            with open(RULES_PATH, "r", encoding="utf-8") as fh:
                _rules = yaml.safe_load(fh)
        else:
            _rules = {
                "tolerances": {
                    "price_difference_percent": 5,
                    "quantity_difference_percent": 0,
                }
            }
    return _rules


# 
# ERP data fetchers (API-first, fallback to disk)
# 

def _fetch_po(po_number: str) -> dict | None:
    """Fetch a Purchase Order from the Mock ERP API.
    Falls back to disk if the API is unavailable."""
    # Try API first
    try:
        import httpx
        resp = httpx.get(
            f"{ERP_API_BASE_URL}/po/{po_number}", timeout=5.0
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
    except Exception as exc:
        raise RuntimeError(f"ERP API unavailable for PO {po_number}: {exc}") from exc


def _fetch_vendor(vendor_id: str) -> dict | None:
    """Fetch a vendor record from the Mock ERP API.
    Falls back to disk if the API is unavailable."""
    try:
        import httpx
        resp = httpx.get(
            f"{ERP_API_BASE_URL}/vendor/{vendor_id}", timeout=5.0
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
    except Exception as exc:
        raise RuntimeError(f"ERP API unavailable for vendor {vendor_id}: {exc}") from exc


def _fetch_all_skus() -> dict[str, dict]:
    """Fetch all SKU records as a {item_code: record} dict.
    Falls back to disk if the API is unavailable."""
    try:
        import httpx
        resp = httpx.get(f"{ERP_API_BASE_URL}/skus", timeout=5.0)
        if resp.status_code == 200:
            skus = resp.json()
            return {
                s["item_code"].upper(): s for s in skus
            }
    except Exception as exc:
        raise RuntimeError(f"ERP API unavailable for SKU master: {exc}") from exc


# 
