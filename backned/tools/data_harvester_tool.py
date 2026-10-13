"""Data Harvester Tool — extracts invoice data using Amazon Bedrock.

Exposes a LangChain @tool for use by the Extractor Agent.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from backned.llm import invoke_bedrock_chat

from backned.models import (
    ExtractedInvoice,
    ExtractedLineItem,
    ExtractionResult,
)
from backned.config.settings import BEDROCK_REASONING_MODEL

logger = logging.getLogger(__name__)

#  Supported extensions 
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
IMAGE_EXTS = {".png"}


# 
# Data Harvester Tool  (LangChain @tool)
# 
@tool
def data_harvester_tool(
    file_path: str,
    email_id: str = "unknown",
) -> dict:
    """Data Harvester Tool — Extracts structured invoice data from
    PDF, DOCX, and image files using Amazon Bedrock.

    1. Detects file format from extension.
    2. For PDF/DOCX: extracts raw text offline, then sends to the configured Bedrock model
       for structured JSON parsing.
    3. For images: OCRs text and sends it to the configured Bedrock model
       extraction and structured parsing in one call.
    4. Returns an ExtractionResult dict with full structured data.

    Args:
        file_path: Absolute path to the invoice file.
        email_id:  Email ID this invoice belongs to.

    Returns:
        dict serialised from ExtractionResult.
    """
    fp = Path(file_path)

    #  Guard: file must exist 
    if not fp.exists():
        return ExtractionResult(
            email_id=email_id,
            filename=fp.name,
            file_path=str(fp),
            file_format="unknown",
            extraction_errors=[f"File not found: {fp}"],
        ).model_dump()

    ext = fp.suffix.lower()
    file_format = _detect_format(ext)

    raw_text = ""
    parsed_data: dict = {}
    errors: list[str] = []

    try:
        if file_format == "pdf":
            logger.info("Extracting text from PDF: %s", fp.name)
            raw_text = _extract_text_from_pdf(fp)
            if raw_text.strip():
                logger.info("Parsing with Bedrock (%d chars) ...", len(raw_text))
                parsed_data = _parse_with_bedrock(raw_text)
            else:
                errors.append("No text extracted from PDF")

        elif file_format == "docx":
            logger.info("Extracting text from DOCX: %s", fp.name)
            raw_text = _extract_text_from_docx(fp)
            if raw_text.strip():
                logger.info("Parsing with Bedrock (%d chars) ...", len(raw_text))
                parsed_data = _parse_with_bedrock(raw_text)
            else:
                errors.append("No text extracted from DOCX")

        elif file_format == "image":
            logger.info("OCR and Bedrock parsing: %s", fp.name)
            raw_text, parsed_data = _parse_image_with_bedrock(fp)

        else:
            errors.append(f"Unsupported format: {ext}")

    except json.JSONDecodeError as exc:
        errors.append(f"Bedrock returned invalid JSON: {exc}")
        logger.error("JSON parse error for %s: %s", fp.name, exc)
    except Exception as exc:
        errors.append(f"Extraction failed for {fp.name}: {exc}")
        logger.error("Extraction error for %s: %s", fp.name, exc)

    #  Build structured invoice 
    invoice: ExtractedInvoice | None = None
    if parsed_data:
        try:
            invoice = _dict_to_extracted_invoice(parsed_data)
        except Exception as exc:
            errors.append(f"Failed to build invoice model: {exc}")
            logger.error("Model build error: %s", exc)

    result = ExtractionResult(
        email_id=email_id,
        filename=fp.name,
        file_path=str(fp),
        file_format=file_format,
        raw_text=raw_text,
        extracted_invoice=invoice,
        extraction_errors=errors,
    )

    if invoice:
        n_items = len(invoice.line_items)
        logger.info(
            "%s → Invoice#: %s | PO: %s | Vendor: %s | Items: %d | Total: %s %s",
            fp.name,
            invoice.invoice_no or "N/A",
            invoice.po_number or "N/A",
            invoice.vendor_id or invoice.vendor_name or "N/A",
            n_items,
            invoice.currency or "?",
            invoice.total_amount,
        )

    return result.model_dump()


# 
#  HELPER FUNCTIONS  (called by the tool above)
# 


# 
# Format detection
# 

def _detect_format(ext: str) -> str:
    ext = ext.lower()
    if ext in PDF_EXTS:
        return "pdf"
    if ext in DOCX_EXTS:
        return "docx"
    if ext in IMAGE_EXTS:
        return "image"
    return "unknown"


# 
# Bedrock chat model
# 

# 
# Shared Bedrock extraction prompt
# 
_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert invoice data extractor.  Your job is to extract
EVERY piece of information from the invoice and return it as JSON.

Return ONLY valid JSON — no markdown fences, no commentary.

The JSON schema you MUST follow:
{
  "invoice_no": "<string or null>",
  "invoice_date": "<string in any date format found, or null>",
  "po_number": "<Purchase Order reference like PO-1001, or null>",
  "vendor_id": "<Vendor ID like VEND-001, or null>",
  "vendor_name": "<string or null>",
  "currency": "<ISO 4217 code like USD, EUR, INR, GBP — or null>",
  "total_amount": <number or null>,
  "tax_amount": <number or null>,
  "subtotal": <number or null>,
  "line_items": [
    {
      "item_code": "<SKU code like SKU-001, or null>",
      "description": "<item description in ORIGINAL language — do NOT translate>",
      "qty": <number or null>,
      "unit_price": <number or null>,
      "total": <number or null>
    }
  ],
  "detected_language": "<ISO 639-1 code of the document language, e.g. en, de, es>",
  "additional_fields": { "<any other key-value pairs you find>" }
}

Rules:
- Extract ALL line items, even if the table has many rows.
- Preserve the original language of descriptions — do NOT translate.
- For currency, convert symbols to ISO codes ($ → USD, € → EUR, ₹ → INR, £ → GBP).
  CRITICAL: Set currency to null if NO currency symbol or ISO code is EXPLICITLY AND VISIBLY
  present in the document. NEVER guess, infer, or assume a currency from the country,
  company name, or amount magnitude. Only set it if you can directly see it in the text.
- If a field is not present in the document, set it to null.
- Numbers must be JSON numbers, not strings.
- Do NOT wrap the JSON in ```json``` markers.
"""


# 
# Raw text extractors (fast, offline)
# 

def _extract_text_from_pdf(file_path: Path) -> str:
    """Read a PDF text layer first and OCR scanned pages with Tesseract."""
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            parts.append(text)

            # Also stringify any tables
            tables = page.extract_tables() or []
            for tbl in tables:
                for row in tbl:
                    if row:
                        cells = [c if c else "" for c in row]
                        parts.append(" | ".join(cells))
    text = "\n".join(parts).strip()
    if text:
        return text

    # Scanned PDF fallback required by the dataset contract. Poppler renders
    # pages; Tesseract performs OCR using the installed language packs.
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(str(file_path), dpi=250)
    language = os.getenv("TESSERACT_LANG", "eng+spa+deu")
    return "\n".join(pytesseract.image_to_string(image, lang=language) for image in images)


def _extract_text_from_docx(file_path: Path) -> str:
    """Extract raw text from a DOCX file."""
    from docx import Document

    doc = Document(str(file_path))
    parts: list[str] = []

    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cells))

    return "\n".join(parts)


# 
# Bedrock parsing: text → structured JSON
# 

def _parse_with_bedrock(raw_text: str) -> dict:
    """Send extracted text to Bedrock and get structured invoice JSON."""
    content = invoke_bedrock_chat(
        model_id=BEDROCK_REASONING_MODEL,
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract all invoice data from the following text and "
                    "return the JSON:\n\n"
                    f"{raw_text}"
                ),
            },
        ],
        max_tokens=4096,
        temperature=0.0,
    )

    content = content or "{}"
    # Strip markdown fences if the model adds them anyway
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]  # remove first line
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    return json.loads(content)


# 
# OCR and Bedrock parsing: image → structured JSON
# 

def _parse_image_with_bedrock(file_path: Path) -> tuple[str, dict]:
    """OCR a PNG scan with Tesseract and parse its text with Bedrock."""
    import pytesseract
    from PIL import Image

    raw_text = pytesseract.image_to_string(
        Image.open(file_path), lang=os.getenv("TESSERACT_LANG", "eng+spa+deu")
    )
    if not raw_text.strip():
        return "", {}
    return raw_text, _parse_with_bedrock(raw_text)

# JSON dict → Pydantic models
# 

def _dict_to_extracted_invoice(data: dict) -> ExtractedInvoice:
    """Convert the model-returned dict into an ExtractedInvoice model."""
    line_items = []
    for li in data.get("line_items", []):
        line_items.append(ExtractedLineItem(
            item_code=li.get("item_code"),
            description=li.get("description"),
            qty=_safe_float(li.get("qty")),
            unit_price=_safe_float(li.get("unit_price")),
            total=_safe_float(li.get("total")),
        ))

    return ExtractedInvoice(
        invoice_no=data.get("invoice_no"),
        invoice_date=data.get("invoice_date"),
        po_number=data.get("po_number"),
        vendor_id=data.get("vendor_id"),
        vendor_name=data.get("vendor_name"),
        currency=data.get("currency"),
        total_amount=_safe_float(data.get("total_amount")),
        tax_amount=_safe_float(data.get("tax_amount")),
        subtotal=_safe_float(data.get("subtotal")),
        line_items=line_items,
        detected_language=data.get("detected_language"),
        additional_fields=data.get("additional_fields", {}),
    )


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


