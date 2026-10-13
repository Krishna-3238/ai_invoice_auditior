"""Lang-Bridge Tool — translates invoice content into English using Amazon Bedrock.

Passes through English invoices unchanged. Exposes a LangChain @tool for use by the Translation Agent.
"""

from __future__ import annotations

import json
import hashlib
import logging
from typing import Any

from langchain_core.tools import tool
from backned.llm import invoke_bedrock_chat

from backned.models.extraction import ExtractedInvoice, ExtractionResult
from backned.models.translation import (
    TranslatedInvoice,
    TranslatedLineItem,
    TranslationResult,
)
from backned.config.settings import BEDROCK_TRANSLATION_MODEL
from backned.config.settings import REDIS_URL

logger = logging.getLogger(__name__)


# 
# Lang-Bridge Tool  (LangChain @tool)
# 
@tool
def lang_bridge_tool(extraction_result: dict) -> dict:
    """Lang-Bridge Tool — Translates extracted invoice content from
    any language into English using Amazon Bedrock.

    If the source language is already English, the data is passed
    through unchanged (no API call).  For non-English invoices, every
    free-text field is translated while numeric / code fields are
    preserved as-is.

    Args:
        extraction_result: A dict serialised from ExtractionResult
            (output of the Data Harvester Tool).

    Returns:
        dict serialised from TranslationResult.
    """
    #  Rehydrate the ExtractionResult 
    try:
        ext = ExtractionResult(**extraction_result)
    except Exception as exc:
        return TranslationResult(
            email_id=extraction_result.get("email_id", "unknown"),
            filename=extraction_result.get("filename", "unknown"),
            file_path=extraction_result.get("file_path", ""),
            translation_errors=[f"Invalid extraction result: {exc}"],
        ).model_dump()

    #  No invoice data → nothing to translate 
    if not ext.extracted_invoice:
        return TranslationResult(
            email_id=ext.email_id,
            filename=ext.filename,
            file_path=ext.file_path,
            source_language=None,
            was_translated=False,
            translation_confidence=0.0,
            translation_errors=["No extracted invoice data to translate"],
        ).model_dump()

    inv = ext.extracted_invoice
    src_lang = (inv.detected_language or "en").lower().strip()

    #  Already English → pass-through 
    if src_lang == "en":
        logger.info(
            "%s — already English, skipping translation", ext.filename
        )
        translated = _passthrough_invoice(inv)
        return TranslationResult(
            email_id=ext.email_id,
            filename=ext.filename,
            file_path=ext.file_path,
            source_language=src_lang,
            was_translated=False,
            translated_invoice=translated,
        ).model_dump()

    # Translate with the configured Bedrock model.
    logger.info(
        "%s — translating from '%s' to English …",
        ext.filename, src_lang,
    )

    errors: list[str] = []
    try:
        cache_key = "translation:" + hashlib.sha256(
            json.dumps({"language": src_lang, "invoice": inv.model_dump()}, sort_keys=True, default=str).encode()
        ).hexdigest()
        import redis
        cache = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        cached = cache.get(cache_key)
        if cached:
            translated_data = json.loads(cached)
        else:
            translated_data = _translate_with_bedrock(inv, src_lang)
            cache.setex(cache_key, 86400, json.dumps(translated_data, ensure_ascii=False))
        translated_invoice = _build_translated_invoice(
            inv, translated_data, src_lang
        )
    except Exception as exc:
        errors.append(f"Translation failed: {exc}")
        logger.error("Translation error for %s: %s", ext.filename, exc)
        # Fall back to passthrough so pipeline isn't broken
        translated_invoice = _passthrough_invoice(inv)

    return TranslationResult(
        email_id=ext.email_id,
        filename=ext.filename,
        file_path=ext.file_path,
        source_language=src_lang,
        was_translated=len(errors) == 0,
        translation_confidence=(float(translated_data.get("translation_confidence", 0.9)) if not errors else 0.0),
        translated_invoice=translated_invoice,
        translation_errors=errors,
    ).model_dump()


# 
#  HELPER FUNCTIONS  (called by the tool above)
# 


# 
# Bedrock chat model
# 

# 
# Bedrock translation prompt and call
# 

_TRANSLATION_SYSTEM_PROMPT = """\
You are an expert multilingual translator specialising in business and
financial documents.  Your task is to translate invoice content from the
source language into **English**.

IMPORTANT RULES:
1. Translate ALL free-text / natural-language fields INCLUDING dates.
2. Convert dates to English format (e.g. "10 de abril de 2025" → "April 10, 2025",
   "25. Mai 2025" → "May 25, 2025"). Always output dates as "%B %d, %Y" format
   (e.g. "January 15, 2025").
3. Do NOT translate or modify:
   - Invoice numbers, PO numbers, vendor IDs, SKU/item codes
   - Numeric values (amounts, quantities, prices)
   - Currency codes
4. Preserve the EXACT meaning of every term — do not paraphrase.
5. Return ONLY valid JSON — no markdown fences, no commentary.

Return JSON in this schema:
{
  "translation_confidence": <number between 0.0 and 1.0>,
  "invoice_date": "<date translated to English format, e.g. 'April 10, 2025'>",
  "vendor_name": "<translated vendor name, or original if already English>",
  "line_items": [
    {
      "item_code": "<unchanged>",
      "description": "<translated description>"
    }
  ],
  "additional_fields": {
    "<key>": "<translated value>"
  }
}

Rules for the response:
- line_items must have the SAME number of entries and SAME order as the
  input — match them by position.
- If a value is already in English, return it unchanged.
- Do NOT add extra keys.
"""


def _translate_with_bedrock(
    invoice: ExtractedInvoice, source_language: str
) -> dict:
    """Send the translatable fields to Bedrock and return the
    translated JSON dict."""
    # Build a compact representation of what needs translating
    translatable = {
        "source_language": source_language,
        "invoice_date": invoice.invoice_date,
        "vendor_name": invoice.vendor_name,
        "line_items": [
            {
                "item_code": li.item_code,
                "description": li.description,
            }
            for li in invoice.line_items
        ],
        "additional_fields": invoice.additional_fields or {},
    }

    user_msg = (
        f"Translate the following invoice content from "
        f"'{source_language}' to English.\n\n"
        f"{json.dumps(translatable, ensure_ascii=False, indent=2)}"
    )

    content = invoke_bedrock_chat(
        model_id=BEDROCK_TRANSLATION_MODEL,
        messages=[
            {"role": "system", "content": _TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=4096,
        temperature=0.0,
    )

    content = content or "{}"
    content = content.strip()
    # Strip markdown fences if the model adds them
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    return json.loads(content)


# 
# Build TranslatedInvoice from the model response and original
# 

def _build_translated_invoice(
    original: ExtractedInvoice,
    translated_data: dict,
    source_language: str,
) -> TranslatedInvoice:
    """Merge the original numeric/code fields with the translated
    text fields into a TranslatedInvoice model."""

    #  Translated line items 
    translated_items_raw = translated_data.get("line_items", [])
    line_items: list[TranslatedLineItem] = []

    for i, orig_li in enumerate(original.line_items):
        # Match by position
        trans_li = (
            translated_items_raw[i]
            if i < len(translated_items_raw)
            else {}
        )
        line_items.append(TranslatedLineItem(
            item_code=orig_li.item_code,
            original_description=orig_li.description,
            translated_description=trans_li.get(
                "description", orig_li.description
            ),
            qty=orig_li.qty,
            unit_price=orig_li.unit_price,
            total=orig_li.total,
        ))

    return TranslatedInvoice(
        # Codes & numbers — unchanged
        invoice_no=original.invoice_no,
        invoice_date=translated_data.get(
            "invoice_date", original.invoice_date
        ),
        po_number=original.po_number,
        vendor_id=original.vendor_id,
        currency=original.currency,
        total_amount=original.total_amount,
        tax_amount=original.tax_amount,
        subtotal=original.subtotal,
        # Translated text fields
        original_vendor_name=original.vendor_name,
        translated_vendor_name=translated_data.get(
            "vendor_name", original.vendor_name
        ),
        line_items=line_items,
        # Language info
        source_language=source_language,
        target_language="en",
        was_translated=True,
        # Additional fields
        original_additional_fields=original.additional_fields or {},
        translated_additional_fields=translated_data.get(
            "additional_fields", original.additional_fields or {}
        ),
    )


# 
# Pass-through: already English (no API call)
# 

def _passthrough_invoice(invoice: ExtractedInvoice) -> TranslatedInvoice:
    """Wrap an already-English invoice into TranslatedInvoice without
    calling the LLM.  original == translated for every field."""
    line_items = [
        TranslatedLineItem(
            item_code=li.item_code,
            original_description=li.description,
            translated_description=li.description,  # same
            qty=li.qty,
            unit_price=li.unit_price,
            total=li.total,
        )
        for li in invoice.line_items
    ]

    return TranslatedInvoice(
        invoice_no=invoice.invoice_no,
        invoice_date=invoice.invoice_date,
        po_number=invoice.po_number,
        vendor_id=invoice.vendor_id,
        original_vendor_name=invoice.vendor_name,
        translated_vendor_name=invoice.vendor_name,  # same
        currency=invoice.currency,
        total_amount=invoice.total_amount,
        tax_amount=invoice.tax_amount,
        subtotal=invoice.subtotal,
        line_items=line_items,
        source_language=invoice.detected_language or "en",
        target_language="en",
        was_translated=False,
        original_additional_fields=invoice.additional_fields or {},
        translated_additional_fields=invoice.additional_fields or {},  # same
    )
