"""Data Completeness Checker Tool — validates invoice fields against business rules in rules.yaml.

Exposes a LangChain @tool for use by the Invoice Data Validation Agent.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import tool

from backned.models.validation import (
    FieldIssue,
    LineItemIssue,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# 
# Path constants
# 
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # → backned/
RULES_PATH = BASE_DIR / "config" / "rules.yaml"

#  Cached rules (loaded once) 
_rules: dict | None = None


# 
# Data Completeness Checker Tool  (LangChain @tool)
# 
@tool
def data_completeness_checker_tool(
    translation_result: dict,
) -> dict:
    """Data Completeness Checker Tool — Validates a translated invoice
    for completeness and accuracy against business rules defined in
    ``config/rules.yaml``.

    Checks required fields, data types, line-item math, invoice-level
    math, currency validity, and date parseability.

    Args:
        translation_result: A single TranslationResult dict (from the
            Translation Agent).

    Returns:
        dict serialised from ValidationResult.
    """
    rules = _load_rules()

    email_id = translation_result.get("email_id", "unknown")
    filename = translation_result.get("filename", "?")
    file_path = translation_result.get("file_path", "")
    source_language = translation_result.get("source_language")

    # The translated invoice is nested inside translation_result
    inv = translation_result.get("translated_invoice") or {}

    header_issues: list[FieldIssue] = []
    line_item_issues: list[LineItemIssue] = []
    rules_applied: list[str] = []

    #  1. Required header fields 
    header_issues.extend(_check_required_header_fields(inv, rules))
    rules_applied.append("required_header_fields")

    #  2. Data-type validation 
    header_issues.extend(_check_header_data_types(inv, rules))
    rules_applied.append("header_data_types")

    #  3. Date format validation 
    header_issues.extend(_check_date_format(inv))
    rules_applied.append("date_format")

    #  4. Currency validation 
    header_issues.extend(_check_currency(inv, rules))
    rules_applied.append("currency_validation")

    #  5. Required line-item fields 
    line_items = inv.get("line_items", [])
    line_item_issues.extend(
        _check_required_line_item_fields(line_items, rules)
    )
    rules_applied.append("required_line_item_fields")

    #  6. Line-item math (qty × price ≈ total) 
    line_item_issues.extend(
        _check_line_item_math(line_items, rules)
    )
    rules_applied.append("line_item_math")

    #  7. Invoice-level math 
    header_issues.extend(
        _check_invoice_math(inv, line_items, rules)
    )
    rules_applied.append("invoice_math")

    #  Aggregate 
    total_errors = sum(
        1 for i in header_issues if i.severity == "error"
    ) + sum(
        1 for i in line_item_issues if i.severity == "error"
    )
    total_warnings = sum(
        1 for i in header_issues if i.severity == "warning"
    ) + sum(
        1 for i in line_item_issues if i.severity == "warning"
    )

    is_valid = total_errors == 0
    if is_valid and total_warnings > 0:
        validation_status = "passed_with_warnings"
    elif is_valid:
        validation_status = "passed"
    else:
        validation_status = "failed"

    result = ValidationResult(
        email_id=email_id,
        filename=filename,
        file_path=file_path,
        is_valid=is_valid,
        validation_status=validation_status,
        header_issues=header_issues,
        line_item_issues=line_item_issues,
        total_errors=total_errors,
        total_warnings=total_warnings,
        rules_applied=rules_applied,
        source_language=source_language,
    )

    logger.info(
        "%s — %s | errors: %d | warnings: %d",
        filename, validation_status.upper(), total_errors, total_warnings,
    )

    return result.model_dump()


# 
#  HELPER FUNCTIONS  (called by the tool above)
# 


# 
# Rules loader
# 

def _load_rules() -> dict:
    """Load and cache the rules.yaml configuration."""
    global _rules
    if _rules is None:
        if not RULES_PATH.exists():
            logger.warning(
                "Rules file not found at %s - using defaults", RULES_PATH
            )
            _rules = _default_rules()
        else:
            with open(RULES_PATH, "r", encoding="utf-8") as fh:
                _rules = yaml.safe_load(fh)
            logger.info("RULES: Loaded validation rules from %s", RULES_PATH)
    return _rules


def _default_rules() -> dict:
    """Fallback rules when rules.yaml is missing."""
    return {
        "required_fields": {
            "header": [
                "invoice_no", "invoice_date", "vendor_id",
                "currency", "total_amount",
            ],
            "line_item": [
                "item_code", "description", "qty",
                "unit_price", "total",
            ],
        },
        "data_types": {
            "invoice_no": "str",
            "invoice_date": "date",
            "vendor_id": "str",
            "currency": "str",
            "total_amount": "float",
            "qty": "float",
            "unit_price": "float",
            "total": "float",
        },
        "accepted_currencies": ["USD", "EUR", "INR", "GBP"],
        "tolerances": {
            "price_difference_percent": 5,
            "quantity_difference_percent": 0,
            "tax_difference_percent": 2,
        },
    }


# 
# Field-name mapping (TranslatedInvoice → rules names)
# 
# TranslatedInvoice uses slightly different field names for some
# header fields. This map converts rules.yaml names → the actual
# key in the translated_invoice dict.
_HEADER_FIELD_MAP = {
    "invoice_no": "invoice_no",
    "invoice_date": "invoice_date",
    "vendor_id": "vendor_id",
    "vendor_name": "translated_vendor_name",
    "currency": "currency",
    "total_amount": "total_amount",
    "tax_amount": "tax_amount",
    "subtotal": "subtotal",
    "po_number": "po_number",
}

# For line items the TranslatedLineItem uses
# "translated_description" instead of "description".
_LINE_ITEM_FIELD_MAP = {
    "item_code": "item_code",
    "description": "translated_description",
    "qty": "qty",
    "unit_price": "unit_price",
    "total": "total",
}


# 
# Check 1: Required header fields
# 

def _check_required_header_fields(
    inv: dict, rules: dict
) -> list[FieldIssue]:
    """Flag missing mandatory header fields."""
    issues: list[FieldIssue] = []
    required = rules.get("required_fields", {}).get("header", [])
    action = (
        rules.get("validation_policies", {})
        .get("missing_field_action", "flag")
    )
    severity = "error" if action == "reject" else "error"

    for field in required:
        actual_key = _HEADER_FIELD_MAP.get(field, field)
        value = inv.get(actual_key)

        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append(FieldIssue(
                field_name=field,
                issue_type="missing",
                severity=severity,
                message=f"Required header field '{field}' is missing or empty",
                expected="non-empty value",
                actual=str(value) if value is not None else "null",
            ))
    return issues


# 
# Check 2: Header data-type validation
# 

def _check_header_data_types(
    inv: dict, rules: dict
) -> list[FieldIssue]:
    """Ensure header field values match the expected data types."""
    issues: list[FieldIssue] = []
    type_rules = rules.get("data_types", {})

    for field, expected_type in type_rules.items():
        # Only check header-level fields here (not line item ones)
        if field in ("qty", "unit_price", "total"):
            continue

        actual_key = _HEADER_FIELD_MAP.get(field, field)
        value = inv.get(actual_key)

        # Skip if value is None — already caught by required check
        if value is None:
            continue

        if expected_type == "float":
            if not isinstance(value, (int, float)):
                issues.append(FieldIssue(
                    field_name=field,
                    issue_type="wrong_type",
                    severity="error",
                    message=(
                        f"Field '{field}' expected numeric but got "
                        f"{type(value).__name__}: {value}"
                    ),
                    expected="float",
                    actual=type(value).__name__,
                ))

        elif expected_type == "str":
            if not isinstance(value, str):
                issues.append(FieldIssue(
                    field_name=field,
                    issue_type="wrong_type",
                    severity="warning",
                    message=(
                        f"Field '{field}' expected string but got "
                        f"{type(value).__name__}"
                    ),
                    expected="str",
                    actual=type(value).__name__,
                ))

        elif expected_type == "date":
            # Already checked in _check_date_format, skip here
            pass

    return issues


# 
# Check 3: Date format validation
# 

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d-%b-%Y",       # 25-May-2025
    "%b %d, %Y",      # May 25, 2025
    "%B %d, %Y",      # January 15, 2025
    "%d %B %Y",       # 25 January 2025
    "%d %b %Y",       # 25 Jan 2025
    "%B %d %Y",       # January 15 2025 (no comma)
    "%b %d %Y",       # Jan 15 2025 (no comma)
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%m.%d.%Y",
    "%Y.%m.%d",
    "%d %B, %Y",      # 25 January, 2025
]


def _check_date_format(inv: dict) -> list[FieldIssue]:
    """Ensure invoice_date is a parseable date string."""
    issues: list[FieldIssue] = []
    date_val = inv.get("invoice_date")

    if date_val is None:
        return issues  # already caught by required-fields check

    if not isinstance(date_val, str):
        issues.append(FieldIssue(
            field_name="invoice_date",
            issue_type="wrong_type",
            severity="error",
            message=f"invoice_date is not a string: {date_val}",
            expected="date string",
            actual=str(type(date_val).__name__),
        ))
        return issues

    # Try common date formats
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(date_val.strip(), fmt)
            return issues  # parsed successfully
        except ValueError:
            continue

    # None matched — it's a warning (GPT might return unusual formats)
    issues.append(FieldIssue(
        field_name="invoice_date",
        issue_type="invalid_value",
        severity="warning",
        message=(
            f"invoice_date '{date_val}' could not be parsed with "
            f"common date formats — may need manual review"
        ),
        expected="parseable date string",
        actual=date_val,
    ))
    return issues


# 
# Check 4: Currency validation
# 

def _check_currency(inv: dict, rules: dict) -> list[FieldIssue]:
    """Ensure currency is in the accepted list."""
    issues: list[FieldIssue] = []
    currency = inv.get("currency")

    if currency is None:
        return issues  # already caught by required-fields check

    accepted = rules.get("accepted_currencies", [])
    action = (
        rules.get("validation_policies", {})
        .get("invalid_currency_action", "reject")
    )
    severity = "error" if action == "reject" else "warning"

    if currency.upper() not in [c.upper() for c in accepted]:
        issues.append(FieldIssue(
            field_name="currency",
            issue_type="invalid_value",
            severity=severity,
            message=(
                f"Currency '{currency}' is not in the accepted list: "
                f"{accepted}"
            ),
            expected=f"one of {accepted}",
            actual=currency,
        ))
    return issues


# 
# Check 5: Required line-item fields
# 

def _check_required_line_item_fields(
    line_items: list[dict], rules: dict
) -> list[LineItemIssue]:
    """Flag missing mandatory fields in each line item."""
    issues: list[LineItemIssue] = []
    required = rules.get("required_fields", {}).get("line_item", [])

    for idx, li in enumerate(line_items):
        for field in required:
            actual_key = _LINE_ITEM_FIELD_MAP.get(field, field)
            value = li.get(actual_key)

            if value is None or (isinstance(value, str) and not value.strip()):
                issues.append(LineItemIssue(
                    line_index=idx,
                    field_name=field,
                    issue_type="missing",
                    severity="error",
                    message=(
                        f"Line item [{idx}]: required field "
                        f"'{field}' is missing or empty"
                    ),
                    expected="non-empty value",
                    actual=str(value) if value is not None else "null",
                ))
    return issues


# 
# Check 6: Line-item math (qty × unit_price ≈ total)
# 

def _check_line_item_math(
    line_items: list[dict], rules: dict
) -> list[LineItemIssue]:
    """Verify qty × unit_price ≈ line total for each row."""
    issues: list[LineItemIssue] = []
    tolerance_pct = (
        rules.get("tolerances", {}).get("price_difference_percent", 5)
    )

    for idx, li in enumerate(line_items):
        qty = li.get("qty")
        unit_price = li.get("unit_price")
        total = li.get("total")

        # Can only check if all three are present
        if qty is None or unit_price is None or total is None:
            continue

        try:
            expected_total = float(qty) * float(unit_price)
            actual_total = float(total)

            if actual_total == 0 and expected_total == 0:
                continue

            if expected_total == 0:
                pct_diff = 100.0
            else:
                pct_diff = abs(
                    (actual_total - expected_total) / expected_total
                ) * 100

            if pct_diff > tolerance_pct:
                issues.append(LineItemIssue(
                    line_index=idx,
                    field_name="total",
                    issue_type="math_error",
                    severity="error",
                    message=(
                        f"Line item [{idx}]: qty({qty}) × "
                        f"unit_price({unit_price}) = "
                        f"{expected_total:.2f}, but total = "
                        f"{actual_total:.2f} "
                        f"(diff: {pct_diff:.1f}%, "
                        f"tolerance: {tolerance_pct}%)"
                    ),
                    expected=f"{expected_total:.2f}",
                    actual=f"{actual_total:.2f}",
                ))
        except (ValueError, TypeError):
            pass  # non-numeric — already caught by type check

    return issues


# 
# Check 7: Invoice-level math
# 

def _check_invoice_math(
    inv: dict, line_items: list[dict], rules: dict
) -> list[FieldIssue]:
    """Verify invoice-level math consistency:
    - sum(line totals) ≈ subtotal
    - subtotal + tax ≈ total_amount
    """
    issues: list[FieldIssue] = []
    tolerance_pct = (
        rules.get("tolerances", {}).get("price_difference_percent", 5)
    )
    tax_tolerance_pct = (
        rules.get("tolerances", {}).get("tax_difference_percent", 2)
    )

    #  Sum of line totals vs subtotal 
    line_sum = 0.0
    computable = True
    for li in line_items:
        lt = li.get("total")
        if lt is not None:
            try:
                line_sum += float(lt)
            except (ValueError, TypeError):
                computable = False
        else:
            computable = False

    subtotal = inv.get("subtotal")
    if computable and subtotal is not None and len(line_items) > 0:
        try:
            sub_val = float(subtotal)
            if sub_val != 0:
                pct_diff = abs((line_sum - sub_val) / sub_val) * 100
                if pct_diff > tolerance_pct:
                    issues.append(FieldIssue(
                        field_name="subtotal",
                        issue_type="math_error",
                        severity="warning",
                        message=(
                            f"Sum of line totals ({line_sum:.2f}) ≠ "
                            f"subtotal ({sub_val:.2f}) — "
                            f"diff: {pct_diff:.1f}%"
                        ),
                        expected=f"{line_sum:.2f}",
                        actual=f"{sub_val:.2f}",
                    ))
        except (ValueError, TypeError):
            pass

    #  Subtotal + tax ≈ total_amount 
    total_amount = inv.get("total_amount")
    tax_amount = inv.get("tax_amount")

    if subtotal is not None and total_amount is not None:
        try:
            sub_val = float(subtotal)
            total_val = float(total_amount)
            tax_val = float(tax_amount) if tax_amount is not None else 0.0

            expected_total = sub_val + tax_val
            if total_val != 0:
                pct_diff = abs(
                    (expected_total - total_val) / total_val
                ) * 100
                if pct_diff > tax_tolerance_pct:
                    issues.append(FieldIssue(
                        field_name="total_amount",
                        issue_type="math_error",
                        severity="warning",
                        message=(
                            f"subtotal ({sub_val:.2f}) + "
                            f"tax ({tax_val:.2f}) = "
                            f"{expected_total:.2f}, but "
                            f"total_amount = {total_val:.2f} — "
                            f"diff: {pct_diff:.1f}%"
                        ),
                        expected=f"{expected_total:.2f}",
                        actual=f"{total_val:.2f}",
                    ))
        except (ValueError, TypeError):
            pass

    return issues
