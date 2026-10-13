"""Translation Agent — translates extracted invoice content from any language into English.

LangGraph: translate_all  verify_translations  END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backned.models.extraction import ExtractionResult
from backned.models.translation import TranslationResult

from backned.tool_dispatch import invoke_tool_sync


logger = logging.getLogger(__name__)


# 
# LangGraph — State Schema
# 
class TranslationState(TypedDict):
    """State flowing through the Translation graph."""
    extraction_results: list[dict]     # list of ExtractionResult dicts
    translation_results: list[dict]    # list of TranslationResult dicts
    errors: list[str]
    status: str  # idle | translating | translated | completed | error


# 
# LangGraph — Node functions
# 
def translate_all_node(state: TranslationState) -> dict[str, Any]:
    """Iterate over every extraction result and invoke the Lang-Bridge
    Tool to translate non-English content."""
    logger.info("Translation Agent - starting translations ...")

    extractions: list[dict] = state.get("extraction_results", [])
    all_results: list[dict] = []
    errors: list[str] = []

    for ext_dict in extractions:
        filename = ext_dict.get("filename", "?")
        email_id = ext_dict.get("email_id", "unknown")

        # Detect source language from the extracted invoice
        inv = ext_dict.get("extracted_invoice")
        src_lang = "?"
        if inv:
            src_lang = inv.get("detected_language", "?") or "?"

        logger.info(
            "Translating: %s (email: %s, lang: %s)",
            filename, email_id, src_lang,
        )

        try:
            result = invoke_tool_sync("lang_bridge", {
                "extraction_result": ext_dict,
            })
            all_results.append(result)

            # Log summary
            was_translated = result.get("was_translated", False)
            trans_inv = result.get("translated_invoice")
            vendor = (
                trans_inv.get("translated_vendor_name", "N/A")
                if trans_inv else "N/A"
            )
            n_items = (
                len(trans_inv.get("line_items", []))
                if trans_inv else 0
            )

            if was_translated:
                logger.info(
                    "OK: %s | %s -> en | Vendor: %s | Items: %d",
                    filename, src_lang, vendor, n_items,
                )
            else:
                logger.info(
                    "PASSED: %s | already English - passed through | "
                    "Vendor: %s | Items: %d",
                    filename, vendor, n_items,
                )

            if result.get("translation_errors"):
                errors.extend(result["translation_errors"])

        except Exception as exc:
            err_msg = f"Translation failed for {filename}: {exc}"
            errors.append(err_msg)
            logger.error("ERROR: %s", err_msg)

    logger.info(
        "Translation complete - %d file(s) processed, %d error(s)",
        len(all_results), len(errors),
    )

    return {
        "translation_results": all_results,
        "errors": errors,
        "status": "translated",
    }


def verify_translations_node(state: TranslationState) -> dict[str, Any]:
    """Quick sanity check on translation results.  Logs warnings for
    failed translations or unexpected outcomes."""
    results = state.get("translation_results", [])
    errors = list(state.get("errors", []))

    failed_count = 0
    translated_count = 0
    passthrough_count = 0

    for r in results:
        filename = r.get("filename", "?")
        was_translated = r.get("was_translated", False)
        trans_inv = r.get("translated_invoice")

        if not trans_inv:
            failed_count += 1
            logger.warning(
                "WARN: %s: no translated invoice produced", filename
            )
            continue

        if was_translated:
            translated_count += 1

            # Verify line-item count matches
            n_items = len(trans_inv.get("line_items", []))
            if n_items == 0:
                logger.warning(
                    "WARN: %s: translation produced 0 line items",
                    filename,
                )
        else:
            passthrough_count += 1

        # Check for translation errors
        if r.get("translation_errors"):
            logger.warning(
                "%s: translation errors: %s",
                filename, r["translation_errors"],
            )

    logger.info(
        "Translation summary: %d translated, %d passed-through, "
        "%d failed",
        translated_count, passthrough_count, failed_count,
    )

    return {"status": "completed", "errors": errors}


# 
# LangGraph — Graph builder
# 
def build_translation_graph():
    """Build and compile the Translation LangGraph."""
    workflow = StateGraph(TranslationState)

    workflow.add_node("translate_all", translate_all_node)
    workflow.add_node("verify_translations", verify_translations_node)

    workflow.set_entry_point("translate_all")
    workflow.add_edge("translate_all", "verify_translations")
    workflow.add_edge("verify_translations", END)

    from backned.checkpointing import compile_graph
    return compile_graph(workflow)


# 
# Public API — TranslationAgent
# 
class TranslationAgent:
    """Orchestrates the Translation LangGraph pipeline."""

    def __init__(self) -> None:
        self.graph = build_translation_graph()

    def run(
        self, extraction_results: list[ExtractionResult]
    ) -> list[TranslationResult]:
        """Translate all extraction results to English; returns list[TranslationResult]."""
        logger.info("Starting Translation Agent ...")

        if not extraction_results:
            logger.info("No extraction results to translate - skipping.")
            return []

        initial_state: TranslationState = {
            "extraction_results": [
                er.model_dump() for er in extraction_results
            ],
            "translation_results": [],
            "errors": [],
            "status": "idle",
        }

        from backned.checkpointing import invoke_config
        final_state = self.graph.invoke(initial_state, config=invoke_config())

        raw_results = final_state.get("translation_results", [])
        return [TranslationResult(**r) for r in raw_results]


# 
# CLI entry-point
# 
if __name__ == "__main__":
    from backned.config.logging_setup import setup_logging
    setup_logging()

    # 1) Monitor — scan inbox
    from backned.agents.invoice_monitor_agent import InvoiceMonitorAgent

    monitor = InvoiceMonitorAgent()
    monitor.reset_processed_state()
    monitor_result = monitor.run()

    # 2) Extract — harvest invoice data
    from backned.agents.extractor_agent import ExtractorAgent

    extractor = ExtractorAgent()
    extraction_results = extractor.run(monitor_result)

    # 3) Translate — convert to English
    translator = TranslationAgent()
    translation_results = translator.run(extraction_results)

    #  Pretty-print results 
    print("\n"+ "="* 80)
    print("TRANSLATION AGENT — RESULTS")
    print("="* 80)
    print(f"Total files translated: {len(translation_results)}")

    for idx, r in enumerate(translation_results, 1):
        print(f"\n{'─' * 80}")
        print(f"[{idx}] {r.filename} (email: {r.email_id})")
        print(f"{'─' * 80}")
        print(f"  Source Language : {r.source_language or 'N/A'}")
        print(f"  Was Translated : {'Yes'if r.was_translated else 'No (already English)'}")

        if r.translated_invoice:
            inv = r.translated_invoice
            print(f"\n  {'─' * 50}")
            print(f"  TRANSLATED INVOICE DATA:")
            print(f"  {'─' * 50}")
            print(f"  Invoice #  : {inv.invoice_no or 'N/A'}")
            print(f"  Invoice Date : {inv.invoice_date or 'N/A'}")
            print(f"  PO Number  : {inv.po_number or 'N/A'}")
            print(f"  Vendor ID  : {inv.vendor_id or 'N/A'}")

            if inv.was_translated and inv.original_vendor_name != inv.translated_vendor_name:
                print(f"  Vendor (orig) : {inv.original_vendor_name or 'N/A'}")
                print(f"  Vendor (en)  : {inv.translated_vendor_name or 'N/A'}")
            else:
                print(f"  Vendor Name  : {inv.translated_vendor_name or 'N/A'}")

            print(f"  Currency  : {inv.currency or 'N/A'}")
            print(f"  Subtotal  : {inv.subtotal}")
            print(f"  Tax Amount  : {inv.tax_amount}")
            print(f"  Total Amount : {inv.total_amount}")

            #  Line Items 
            print(f"\n  LINE ITEMS ({len(inv.line_items)}):")
            if inv.line_items:
                if inv.was_translated:
                    # Show original → translated side by side
                    print(f"  {'Code':<14} {'Original Description':<30} {'→ English':<30} {'Qty':>6} {'Price':>10} {'Total':>10}")
                    print(f"  {'─' * 14} {'─' * 30} {'─' * 30} {'─' * 6} {'─' * 10} {'─' * 10}")
                    for li in inv.line_items:
                        code = li.item_code or "?"
                        orig = (li.original_description or "")[:30]
                        trans = (li.translated_description or "")[:30]
                        qty = f"{li.qty}" if li.qty is not None else "—"
                        price = f"{li.unit_price}" if li.unit_price is not None else "—"
                        total = f"{li.total}" if li.total is not None else "—"
                        print(f"  {code:<14} {orig:<30} {trans:<30} {qty:>6} {price:>10} {total:>10}")
                else:
                    # English — single description column
                    print(f"  {'Code':<14} {'Description':<35} {'Qty':>6} {'Price':>10} {'Total':>10}")
                    print(f"  {'─' * 14} {'─' * 35} {'─' * 6} {'─' * 10} {'─' * 10}")
                    for li in inv.line_items:
                        code = li.item_code or "?"
                        desc = (li.translated_description or "")[:35]
                        qty = f"{li.qty}" if li.qty is not None else "—"
                        price = f"{li.unit_price}" if li.unit_price is not None else "—"
                        total = f"{li.total}" if li.total is not None else "—"
                        print(f"  {code:<14} {desc:<35} {qty:>6} {price:>10} {total:>10}")
            else:
                print("  (no line items found)")

            #  Additional Fields 
            if inv.translated_additional_fields:
                print(f"\n  ADDITIONAL FIELDS (English):")
                for k, v in inv.translated_additional_fields.items():
                    print(f"  {k}: {v}")

                if inv.was_translated and inv.original_additional_fields != inv.translated_additional_fields:
                    print(f"\n  ADDITIONAL FIELDS (Original):")
                    for k, v in inv.original_additional_fields.items():
                        print(f"  {k}: {v}")
        else:
            print(f"\n  No translated invoice data produced.")

        #  Translation Errors 
        if r.translation_errors:
            print(f"\n  TRANSLATION ERRORS:")
            for err in r.translation_errors:
                print(f"  • {err}")

    print("\n"+ "="* 80)
    print("Translation pipeline complete.")
    print("="* 80)
