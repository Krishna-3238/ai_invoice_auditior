"""
Validation Agent for AI Invoice Auditor
========================================
Persona: Data Auditor & ERP Auditor
Role: Validate invoice data for completeness, accuracy, and ERP compliance

This agent performs two types of validation:

1. Invoice Data Validation:
   - Check for missing mandatory fields
   - Verify data types and formats
   - Validate totals and calculations
   - Check currency validity

2. Business Validation:
   - Compare invoice data against ERP records
   - Validate vendor information
   - Match PO line items
   - Check price and quantity discrepancies
   - Apply tolerance thresholds

The agent uses LLM (AWS Bedrock Nova Lite) for intelligent field extraction
and semantic matching between invoice descriptions and ERP data.
Prompts are loaded dynamically from config/persona_invoice_agent.yaml.
"""

import logging
import re
import json
import yaml
import os
import sys
from typing import Dict, Any, List, Optional
from dateutil import parser as date_parser
from litellm import completion
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.persona_loader import AgentPersona

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ValidationAgent")


class InvoiceDataValidator:
    """
    Invoice Data Validation Agent
    Persona: Data Auditor

    Validates invoice structure, completeness, and internal consistency.
    Prompts are loaded dynamically from persona configuration.
    """

    def __init__(
        self,
        model: str = "bedrock/amazon.nova-lite-v1:0",
        config_path: str = "./config/rules.yaml",
    ):
        """
        Initialize the Invoice Data Validator.

        Args:
            model: LiteLLM model string for intelligent field extraction
                   Set to None to disable LLM-based extraction
            config_path: Path to rules.yaml configuration
        """
        self.model = model
        self.config = self._load_config(config_path)
        self.rules = self.config

        self.persona = AgentPersona("validation_agent")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load validation rules from rules.yaml"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"Validation rules loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def extract_invoice_fields(self, translated_text: str) -> Dict[str, Any]:
        """
        Extract structured invoice fields from translated text using LLM.
        Loads prompt from persona configuration.

        Args:
            translated_text: English text from translation agent

        Returns:
            Dictionary with extracted invoice fields
        """
        if not self.model:
            return self._extract_fields_regex(translated_text)

        try:
            prompt = self.persona.get_prompt(
                "extract_fields", translated_text=translated_text
            )

            if not prompt:
                prompt = f"""You are an invoice data extraction expert. Extract the following fields from the invoice text below.

Required fields:
- invoice_no: Invoice number/ID
- invoice_date: Invoice date (format: YYYY-MM-DD)
- vendor_id: Vendor ID or code
- vendor_name: Vendor name
- currency: Currency code (USD, EUR, GBP, INR, etc.)
- total_amount: Total invoice amount (numeric value only)
- po_number: Purchase Order number (if mentioned)
- tax_amount: Tax/VAT amount (if mentioned)

Invoice Text:
---
{translated_text}
---

Return ONLY a valid JSON object with the extracted fields. Use null for missing fields.
Example format:
{{
  "invoice_no": "INV-2025-001",
  "invoice_date": "2025-01-15",
  "vendor_id": "VEND-001",
  "vendor_name": "Global Logistics Ltd",
  "currency": "USD",
  "total_amount": 1500.00,
  "po_number": "PO-1001",
  "tax_amount": 150.00
}}"""

            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            response_text = response.choices[0].message.content.strip()

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                invoice_data = json.loads(json_match.group())
                logger.info("Successfully extracted invoice fields using LLM")
                return invoice_data
            else:
                logger.warning("LLM response doesn't contain valid JSON")
                return self._extract_fields_regex(translated_text)

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}. Falling back to regex.")
            return self._extract_fields_regex(translated_text)

    def _extract_fields_regex(self, text: str) -> Dict[str, Any]:
        """Fallback regex-based field extraction"""
        invoice_data = {
            "invoice_no": None,
            "invoice_date": None,
            "vendor_id": None,
            "vendor_name": None,
            "currency": None,
            "total_amount": None,
            "po_number": None,
            "tax_amount": None,
        }

        inv_patterns = [
            r"Invoice\s*(?:Number|No|#)?[:\s]+([A-Z0-9\-]+)",
            r"Invoice[:\s]+([A-Z0-9\-]+)",
            r"INV[:\-\s]+([A-Z0-9\-]+)",
        ]
        for pattern in inv_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                invoice_data["invoice_no"] = match.group(1).strip()
                break

        date_patterns = [
            r"Date[:\s]+(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})",
            r"Date[:\s]+(\d{4}[\-/]\d{1,2}[\-/]\d{1,2})",
            r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    parsed_date = date_parser.parse(match.group(1))
                    invoice_data["invoice_date"] = parsed_date.strftime("%Y-%m-%d")
                    break
                except:
                    continue

        vendor_id_match = re.search(
            r"Vendor\s*(?:ID|Code)?[:\s]+([A-Z0-9\-]+)", text, re.IGNORECASE
        )
        if vendor_id_match:
            invoice_data["vendor_id"] = vendor_id_match.group(1).strip()

        po_match = re.search(
            r"P[O|.O]\s*(?:Number)?[:\-\s]+([A-Z0-9\-]+)", text, re.IGNORECASE
        )
        if po_match:
            invoice_data["po_number"] = po_match.group(1).strip()

        currency_match = re.search(r"\b(USD|EUR|GBP|INR)\b", text)
        if currency_match:
            invoice_data["currency"] = currency_match.group(1)

        total_patterns = [
            r"Total[:\s]+(?:[A-Z]{3}\s*)?([0-9,]+\.?\d*)",
            r"Total\s*Amount[:\s]+(?:[A-Z]{3}\s*)?([0-9,]+\.?\d*)",
            r"Grand\s*Total[:\s]+(?:[A-Z]{3}\s*)?([0-9,]+\.?\d*)",
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(",", "")
                invoice_data["total_amount"] = float(amount_str)
                break

        return invoice_data

    def extract_line_items(self, tables: List[List[List[str]]]) -> List[Dict[str, Any]]:
        """
        Extract line items from invoice tables.

        Args:
            tables: List of tables from translation agent

        Returns:
            List of line item dictionaries
        """
        line_items = []

        if not tables:
            return line_items

        for table in tables:
            if not table or len(table) < 2:
                continue

            header = [str(cell).lower().strip() for cell in table[0]]

            desc_idx = self._find_column(
                header, ["description", "item", "product", "service"]
            )
            qty_idx = self._find_column(header, ["quantity", "qty", "amount"])
            price_idx = self._find_column(
                header, ["price", "unit price", "rate", "unit_price"]
            )
            total_idx = self._find_column(header, ["total", "amount", "line total"])
            code_idx = self._find_column(
                header, ["code", "sku", "item code", "item_code"]
            )

            for row in table[1:]:
                if not row or len(row) < 2:
                    continue

                item = {
                    "item_code": (
                        row[code_idx].strip()
                        if code_idx is not None and code_idx < len(row)
                        else None
                    ),
                    "description": (
                        row[desc_idx].strip()
                        if desc_idx is not None and desc_idx < len(row)
                        else None
                    ),
                    "qty": (
                        self._parse_number(row[qty_idx])
                        if qty_idx is not None and qty_idx < len(row)
                        else None
                    ),
                    "unit_price": (
                        self._parse_number(row[price_idx])
                        if price_idx is not None and price_idx < len(row)
                        else None
                    ),
                    "total": (
                        self._parse_number(row[total_idx])
                        if total_idx is not None and total_idx < len(row)
                        else None
                    ),
                }

                if item["description"] or item["item_code"]:
                    line_items.append(item)

        return line_items

    def _find_column(self, header: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index by matching keywords"""
        for i, col in enumerate(header):
            for keyword in keywords:
                if keyword in col:
                    return i
        return None

    def _parse_number(self, value: str) -> Optional[float]:
        """Parse numeric value from string"""
        if not value:
            return None
        try:
            clean_value = re.sub(r"[^\d.\-]", "", str(value))
            return float(clean_value) if clean_value else None
        except:
            return None

    def validate_required_fields(
        self, invoice_data: Dict[str, Any], line_items: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Check for missing required fields.

        Returns:
            List of validation issues
        """
        issues = []

        required_header = self.rules.get("required_fields", {}).get("header", [])
        for field in required_header:
            if not invoice_data.get(field):
                issues.append(
                    {
                        "type": "missing_field",
                        "severity": "error",
                        "field": field,
                        "message": f"Required header field '{field}' is missing",
                        "action": self.rules.get("validation_policies", {}).get(
                            "missing_field_action", "flag"
                        ),
                    }
                )

        required_line_item = self.rules.get("required_fields", {}).get("line_item", [])
        for idx, item in enumerate(line_items):
            for field in required_line_item:
                if not item.get(field):
                    issues.append(
                        {
                            "type": "missing_field",
                            "severity": "warning",
                            "field": field,
                            "line_item_index": idx,
                            "message": f"Required line item field '{field}' is missing in item {idx + 1}",
                            "action": "flag",
                        }
                    )

        return issues

    def validate_data_types(self, invoice_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate field data types"""
        issues = []
        data_types = self.rules.get("data_types", {})

        for field, expected_type in data_types.items():
            value = invoice_data.get(field)
            if value is None:
                continue

            try:
                if expected_type == "float":
                    float(value)
                elif expected_type == "date":
                    date_parser.parse(str(value))
                elif expected_type == "str":
                    str(value)
            except:
                issues.append(
                    {
                        "type": "invalid_data_type",
                        "severity": "error",
                        "field": field,
                        "expected": expected_type,
                        "received": type(value).__name__,
                        "message": f"Field '{field}' has invalid data type. Expected {expected_type}.",
                    }
                )

        return issues

    def validate_currency(self, invoice_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate currency code"""
        issues = []
        currency = invoice_data.get("currency")
        accepted_currencies = self.rules.get("accepted_currencies", [])

        if currency and currency not in accepted_currencies:
            action = self.rules.get("validation_policies", {}).get(
                "invalid_currency_action", "reject"
            )
            issues.append(
                {
                    "type": "invalid_currency",
                    "severity": "error" if action == "reject" else "warning",
                    "field": "currency",
                    "value": currency,
                    "message": f"Currency '{currency}' is not in accepted list: {accepted_currencies}",
                    "action": action,
                }
            )

        return issues

    def validate_calculations(
        self, invoice_data: Dict[str, Any], line_items: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Validate invoice calculations and totals"""
        issues = []

        for idx, item in enumerate(line_items):
            qty = item.get("qty")
            unit_price = item.get("unit_price")
            total = item.get("total")

            if qty and unit_price and total:
                calculated_total = qty * unit_price
                difference = abs(calculated_total - total)
                if difference > 0.01:                               
                    issues.append(
                        {
                            "type": "calculation_error",
                            "severity": "error",
                            "line_item_index": idx,
                            "message": f"Line item {idx + 1}: Calculated total {calculated_total:.2f} doesn't match stated total {total:.2f}",
                            "expected": calculated_total,
                            "actual": total,
                            "difference": difference,
                        }
                    )

        invoice_total = invoice_data.get("total_amount")
        if invoice_total and line_items:
            line_totals = [
                item.get("total", 0) for item in line_items if item.get("total")
            ]
            if line_totals:
                calculated_subtotal = sum(line_totals)
                tax = invoice_data.get("tax_amount", 0) or 0
                expected_total = calculated_subtotal + tax

                difference = abs(expected_total - invoice_total)
                if difference > 0.01:
                    issues.append(
                        {
                            "type": "total_mismatch",
                            "severity": "warning",
                            "field": "total_amount",
                            "message": f"Invoice total {invoice_total:.2f} doesn't match calculated total {expected_total:.2f}",
                            "expected": expected_total,
                            "actual": invoice_total,
                            "difference": difference,
                            "action": self.rules.get("validation_policies", {}).get(
                                "total_mismatch_action", "manual_review"
                            ),
                        }
                    )

        return issues

    def run(self, translation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute invoice data validation.

        Args:
            translation_data: Output from TranslationAgent

        Returns:
            Validation results with extracted data and issues
        """
        logger.info("Starting invoice data validation...")

        translated_text = translation_data.get("translated_text", "")
        translated_tables = translation_data.get("translated_tables", [])

        invoice_data = self.extract_invoice_fields(translated_text)
        line_items = self.extract_line_items(translated_tables)

        logger.info(f"Extracted {len(line_items)} line items")

        issues = []
        issues.extend(self.validate_required_fields(invoice_data, line_items))
        issues.extend(self.validate_data_types(invoice_data))
        issues.extend(self.validate_currency(invoice_data))
        issues.extend(self.validate_calculations(invoice_data, line_items))

        errors = [i for i in issues if i.get("severity") == "error"]
        warnings = [i for i in issues if i.get("severity") == "warning"]

        logger.info(
            f"Invoice data validation completed: {len(errors)} errors, {len(warnings)} warnings"
        )

        return {
            "invoice_data": invoice_data,
            "line_items": line_items,
            "validation_issues": issues,
            "errors_count": len(errors),
            "warnings_count": len(warnings),
            "is_valid": len(errors) == 0,
            "metadata": translation_data.get("metadata", {}),
        }


class BusinessValidator:
    """
    Business Validation Agent
    Persona: ERP Auditor

    Validates invoice data against ERP records (PO, Vendor, SKU).
    Uses FastAPI-based mock ERP system via HTTP requests.
    Prompts are loaded dynamically from persona configuration.
    """

    def __init__(
        self,
        model: str = "bedrock/amazon.nova-lite-v1:0",
        config_path: str = "./config/rules.yaml",
        erp_api_url: str = None,
    ):
        """
        Initialize the Business Validator.

        Args:
            model: LiteLLM model string for semantic matching
                   Set to None to disable LLM-based matching
            config_path: Path to rules.yaml
            erp_api_url: Base URL for the mock ERP FastAPI service
        """
        self.model = model
        self.config = self._load_config(config_path)
        self.erp_api_url = (erp_api_url or os.getenv("ERP_API_URL", "http://localhost:8000")).rstrip("/")

        self.persona = AgentPersona("business_validation_agent")

        self._check_erp_connectivity()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load validation rules"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _check_erp_connectivity(self):
        """Check if ERP API is accessible"""
        try:
            response = requests.get(f"{self.erp_api_url}/health", timeout=2)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Connected to ERP API at {self.erp_api_url}")
                logger.info(f"   - Vendors: {health_data.get('vendors_loaded', 0)}")
                logger.info(
                    f"   - PO Records: {health_data.get('po_records_loaded', 0)}"
                )
                logger.info(
                    f"   - SKU Records: {health_data.get('sku_records_loaded', 0)}"
                )
            else:
                logger.warning(f"⚠️  ERP API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to ERP API at {self.erp_api_url}")
            logger.error(f"   Error: {e}")
            logger.info(f"💡 Please start the mock ERP server:")
            logger.info(f"   cd mock_erp && python app.py")

    def _call_erp_api(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """
        Make a request to the ERP API.

        Args:
            endpoint: API endpoint (e.g., '/vendor/VEND-001')

        Returns:
            API response data or None if request fails
        """
        try:
            url = f"{self.erp_api_url}{endpoint}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                return response.json().get("data")
            elif response.status_code == 404:
                logger.warning(f"ERP API: Resource not found - {endpoint}")
                return None
            else:
                logger.error(f"ERP API error: {response.status_code} - {endpoint}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"ERP API request failed: {e}")
            return None

    def get_vendor(self, vendor_id: str) -> Optional[Dict[str, Any]]:
        """Get vendor information from ERP API"""
        return self._call_erp_api(f"/vendor/{vendor_id}")

    def get_po(self, po_number: str) -> Optional[Dict[str, Any]]:
        """Get Purchase Order from ERP API"""
        return self._call_erp_api(f"/po/{po_number}")

    def get_sku(self, item_code: str) -> Optional[Dict[str, Any]]:
        """Get SKU information from ERP API"""
        return self._call_erp_api(f"/sku/{item_code}")

    def validate_vendor(
        self, invoice_data: Dict[str, Any], po_record: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Validate vendor information against ERP API.
        Gets vendor_id from PO record and validates against vendor database.

        Args:
            invoice_data: Extracted invoice data
            po_record: PO record from ERP (contains vendor_id)

        Returns:
            List of validation issues
        """
        issues = []

        if not po_record:
            return issues

        vendor_id = po_record.get("vendor_id")
        if not vendor_id:
            issues.append(
                {
                    "type": "vendor_id_missing_in_po",
                    "severity": "error",
                    "field": "vendor_id",
                    "message": "PO record does not contain vendor_id",
                }
            )
            return issues

        vendor = self.get_vendor(vendor_id)

        if not vendor:
            issues.append(
                {
                    "type": "vendor_not_found",
                    "severity": "error",
                    "field": "vendor_id",
                    "value": vendor_id,
                    "message": f"Vendor '{vendor_id}' from PO not found in ERP system",
                }
            )
            return issues

        invoice_data["vendor_id"] = vendor_id
        invoice_data["vendor_name_from_erp"] = vendor.get("vendor_name")

        invoice_currency = invoice_data.get("currency")
        if invoice_currency and invoice_currency != vendor["currency"]:
            issues.append(
                {
                    "type": "currency_mismatch",
                    "severity": "warning",
                    "field": "currency",
                    "message": f"Invoice currency '{invoice_currency}' differs from vendor's default '{vendor['currency']}'",
                    "erp_value": vendor["currency"],
                    "invoice_value": invoice_currency,
                }
            )

        logger.info(f"Vendor validation: {vendor_id} - {vendor.get('vendor_name')}")
        return issues

    def validate_po_match(
        self, invoice_data: Dict[str, Any], line_items: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Validate invoice against PO records from ERP API"""
        issues = []
        po_number = invoice_data.get("po_number")

        if not po_number:
            issues.append(
                {
                    "type": "po_number_missing",
                    "severity": "warning",
                    "message": "No PO number found in invoice",
                }
            )
            return issues

        po_record = self.get_po(po_number)

        if not po_record:
            issues.append(
                {
                    "type": "po_not_found",
                    "severity": "error",
                    "field": "po_number",
                    "value": po_number,
                    "message": f"PO number '{po_number}' not found in ERP system",
                }
            )
            return issues

        logger.info(
            f"Found PO {po_number} with {len(po_record.get('line_items', []))} line items"
        )


        po_items = po_record.get("line_items", [])
        tolerances = self.config.get("tolerances", {})
        price_tolerance = tolerances.get("price_difference_percent", 5) / 100.0
        qty_tolerance = tolerances.get("quantity_difference_percent", 0) / 100.0

        for inv_item in line_items:
            item_code = inv_item.get("item_code")
            description = inv_item.get("description")

            po_item = None
            if item_code:
                po_item = next(
                    (item for item in po_items if item["item_code"] == item_code), None
                )

            if not po_item and description and self.model:
                po_item = self._semantic_match_item(description, po_items)

            if not po_item:
                issues.append(
                    {
                        "type": "item_not_in_po",
                        "severity": "warning",
                        "message": f"Item '{item_code or description}' not found in PO {po_number}",
                        "item": inv_item,
                    }
                )
                continue

            inv_qty = inv_item.get("qty")
            po_qty = po_item.get("qty")
            if inv_qty and po_qty:
                qty_diff = abs(inv_qty - po_qty) / po_qty
                if qty_diff > qty_tolerance:
                    issues.append(
                        {
                            "type": "quantity_mismatch",
                            "severity": "error",
                            "item_code": item_code,
                            "message": f"Quantity mismatch for '{item_code}': Invoice={inv_qty}, PO={po_qty}",
                            "invoice_value": inv_qty,
                            "erp_value": po_qty,
                            "difference_percent": qty_diff * 100,
                        }
                    )
                else:
                    logger.info(f"   Quantity match for {item_code}: {inv_qty}")

            inv_price = inv_item.get("unit_price")
            po_price = po_item.get("unit_price")
            if inv_price and po_price:
                price_diff = abs(inv_price - po_price) / po_price
                if price_diff > price_tolerance:
                    issues.append(
                        {
                            "type": "price_mismatch",
                            "severity": "warning",
                            "item_code": item_code,
                            "message": f"Price mismatch for '{item_code}': Invoice={inv_price}, PO={po_price} (tolerance: {price_tolerance*100}%)",
                            "invoice_value": inv_price,
                            "erp_value": po_price,
                            "difference_percent": price_diff * 100,
                            "within_tolerance": False,
                        }
                    )
                else:
                    logger.info(
                        f"   Price within tolerance for {item_code}: {inv_price} (PO: {po_price})"
                    )

        return issues

    def _semantic_match_item(
        self, description: str, po_items: List[Dict]
    ) -> Optional[Dict]:
        """Use LLM to semantically match invoice description to PO items"""
        if not self.model or not po_items:
            return None

        try:
            po_descriptions = [
                f"{item['item_code']}: {item['description']}" for item in po_items
            ]

            prompt = f"""Match the invoice line item description to the most similar PO item.

Invoice description: "{description}"

PO items:
{chr(10).join(f"{i+1}. {desc}" for i, desc in enumerate(po_descriptions))}

Return ONLY the number (1-{len(po_descriptions)}) of the best match, or 0 if no match."""

            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
            )

            response_text = response.choices[0].message.content.strip()

            match_num = int(re.search(r"\d+", response_text).group())
            if 1 <= match_num <= len(po_items):
                return po_items[match_num - 1]
        except:
            pass

        return None

    def run(self, data_validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute business validation against ERP.

        Args:
            data_validation_result: Output from InvoiceDataValidator

        Returns:
            Complete validation results
        """
        logger.info("Starting business validation against ERP...")

        invoice_data = data_validation_result.get("invoice_data", {})
        line_items = data_validation_result.get("line_items", [])

        po_number = invoice_data.get("po_number")
        po_record = None
        if po_number:
            po_record = self.get_po(po_number)
            logger.info(
                f"Fetched PO record for {po_number}: {'Found' if po_record else 'Not found'}"
            )

        erp_issues = []
        erp_issues.extend(self.validate_vendor(invoice_data, po_record))
        erp_issues.extend(self.validate_po_match(invoice_data, line_items))

        all_issues = data_validation_result.get("validation_issues", []) + erp_issues

        errors = [i for i in all_issues if i.get("severity") == "error"]
        warnings = [i for i in all_issues if i.get("severity") == "warning"]

        if len(errors) > 0:
            recommendation = "REJECT"
        elif len(warnings) > 3:
            recommendation = "MANUAL_REVIEW"
        else:
            recommendation = "APPROVE"

        logger.info(
            f"Business validation completed: {len(errors)} errors, {len(warnings)} warnings"
        )
        logger.info(f"Recommendation: {recommendation}")

        return {
            "invoice_data": invoice_data,
            "line_items": line_items,
            "validation_issues": all_issues,
            "erp_issues": erp_issues,
            "errors_count": len(errors),
            "warnings_count": len(warnings),
            "is_valid": len(errors) == 0,
            "recommendation": recommendation,
            "metadata": data_validation_result.get("metadata", {}),
        }


class ValidationAgent:
    """
    Combined Validation Agent
    Orchestrates both Invoice Data Validation and Business Validation.
    """

    def __init__(
        self,
        llm=None,
        config_path: str = "./config/rules.yaml",
        erp_api_url: str = None,
    ):
        """
        Initialize the complete Validation Agent.

        Args:
            llm: LLM instance (AWS Bedrock Nova Lite recommended)
            config_path: Path to rules.yaml
            erp_api_url: Base URL for mock ERP FastAPI service
        """
        self.data_validator = InvoiceDataValidator(llm=llm, config_path=config_path)
        self.business_validator = BusinessValidator(
            llm=llm, config_path=config_path, erp_api_url=erp_api_url
        )

    def run(self, translation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete validation pipeline.

        Args:
            translation_data: Output from TranslationAgent

        Returns:
            Complete validation results
        """
        logger.info("=" * 60)
        logger.info("VALIDATION AGENT: Starting validation pipeline")
        logger.info("=" * 60)

        data_result = self.data_validator.run(translation_data)

        if data_result["is_valid"] or data_result["errors_count"] < 3:
            final_result = self.business_validator.run(data_result)
        else:
            logger.warning(
                "Too many data validation errors. Skipping business validation."
            )
            final_result = data_result
            final_result["recommendation"] = "REJECT"

        logger.info("=" * 60)
        logger.info(f"VALIDATION COMPLETE: {final_result['recommendation']}")
        logger.info(
            f"Errors: {final_result['errors_count']}, Warnings: {final_result['warnings_count']}"
        )
        logger.info("=" * 60)

        return final_result


