"""
ADK Translation Agent – Lang-Bridge Tool
==========================================
Google ADK implementation of the Translation Agent.
Persona: Multilingual Translator
Tool Name: Lang-Bridge Tool

Translates non-English invoices to English while preserving
structure and accuracy, using MarianMT for offline and LLM for
high-quality translation.

Registered with:
- A2A Protocol as 'adk_translation_agent'
- MCP as 'lang_bridge_tool'
"""

import os
import sys
import re
import logging
from typing import Dict, Any, List, Optional

from langdetect import detect, DetectorFactory
from litellm import completion
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from agents.adk_agents.base_agent import ADKBaseAgent
from protocols.mcp_protocol import MCPToolDescriptor, ToolCategory
from config.persona_loader import AgentPersona
from guardrails import get_guardrails

DetectorFactory.seed = 0

logger = logging.getLogger("ADK.TranslationAgent")


class ADKTranslationAgent(ADKBaseAgent):
    """
    Google ADK Translation Agent with Lang-Bridge Tool.

    Converts multilingual invoices to English using both
    offline (MarianMT) and online (LLM via Bedrock) translation.
    """

    def __init__(
        self,
        model: str = "bedrock/cohere.command-r-plus-v1:0",
        use_llm: bool = True,
        config_path: str = "./config/rules.yaml",
    ):
        super().__init__(
            agent_id="adk_translation_agent",
            name="Translation Agent (ADK)",
            description="Translates non-English invoices to English while preserving structure",
            capabilities=[
                "language_detection",
                "text_translation",
                "table_translation",
                "currency_normalization",
            ],
            tools=["lang_bridge_tool"],
        )
        self.model = model
        self.use_llm = use_llm and (model is not None)
        self.config = self._load_config(config_path)
        self.persona = AgentPersona("translation_agent")

        self.marian_models = {}
        self.marian_tokenizers = {}

        self.supported_languages = {
            "es": "Helsinki-NLP/opus-mt-es-en",
            "de": "Helsinki-NLP/opus-mt-de-en",
            "fr": "Helsinki-NLP/opus-mt-fr-en",
            "it": "Helsinki-NLP/opus-mt-it-en",
            "pt": "Helsinki-NLP/opus-mt-pt-en",
            "zh": "Helsinki-NLP/opus-mt-zh-en",
            "ja": "Helsinki-NLP/opus-mt-ja-en",
            "ko": "Helsinki-NLP/opus-mt-ko-en",
            "ru": "Helsinki-NLP/opus-mt-ru-en",
            "ar": "Helsinki-NLP/opus-mt-ar-en",
        }

        self._register_lang_bridge_tool()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}

    def _register_lang_bridge_tool(self):
        """Register the Lang-Bridge Tool with MCP."""
        descriptor = MCPToolDescriptor(
            tool_id="lang_bridge_tool",
            name="Lang-Bridge Tool",
            description=(
                "Detects language of invoice text and translates to English. "
                "Supports 10+ languages via MarianMT (offline) and LLM (online). "
                "Preserves numbers, dates, codes, and document structure. "
                "Also normalizes currency symbols to ISO codes."
            ),
            category=ToolCategory.TRANSLATION,
            input_schema={
                "type": "object",
                "properties": {
                    "raw_text": {"type": "string"},
                    "tables": {"type": "array"},
                    "metadata": {"type": "object"},
                },
                "required": ["raw_text"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "translated_text": {"type": "string"},
                    "translated_tables": {"type": "array"},
                    "original_language": {"type": "string"},
                    "translation_confidence": {"type": "number"},
                    "translation_method": {"type": "string"},
                },
            },
        )
        self.register_mcp_tool(descriptor, self._lang_bridge_handler)

    def _lang_bridge_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """MCP handler for the Lang-Bridge Tool."""
        return self.run(input_data)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main translation workflow (A2A handler).

        Args:
            input_data: Dict with raw_text, tables, metadata

        Returns:
            Translation results
        """
        raw_text = input_data.get("raw_text", "")
        tables = input_data.get("tables", [])
        metadata = input_data.get("metadata", {})

        logger.info("Lang-Bridge: Starting translation workflow...")

        lang_code = self.detect_language(raw_text)
        if lang_code == "unknown":
            logger.warning("Could not detect language. Assuming English.")
            lang_code = "en"

        if lang_code == "en":
            logger.info("Text is already in English. Skipping translation.")
            return {
                "translated_text": raw_text,
                "translated_tables": tables,
                "original_language": lang_code,
                "translation_confidence": 1.0,
                "translation_method": "none",
                "metadata": metadata,
            }

        try:
            if self.use_llm and self.model:
                translation_result = self.translate_with_llm(raw_text, lang_code)
            elif lang_code in self.supported_languages:
                translation_result = self.translate_with_marian(raw_text, lang_code)
            else:
                return {
                    "translated_text": raw_text,
                    "translated_tables": tables,
                    "original_language": lang_code,
                    "translation_confidence": 0.0,
                    "translation_method": "unsupported",
                    "metadata": metadata,
                    "error": f"Language {lang_code} not supported",
                }

            translated_text = translation_result["translated_text"]
            confidence = translation_result["confidence"]
            method = translation_result["translation_method"]

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                "translated_text": raw_text,
                "translated_tables": tables,
                "original_language": lang_code,
                "translation_confidence": 0.0,
                "translation_method": "failed",
                "metadata": metadata,
                "error": str(e),
            }

        try:
            translated_tables = self.translate_tables(tables, lang_code)
        except Exception as e:
            logger.warning(f"Table translation failed: {e}")
            translated_tables = tables

        translated_text = self.normalize_currency(translated_text)

        rai_check = self.guardrails.check_output(
            translated_text, agent_id=self.agent_id
        )

        logger.info(
            f"Lang-Bridge: Translation complete ({lang_code} → en, confidence={confidence:.2f})"
        )

        return {
            "translated_text": translated_text,
            "translated_tables": translated_tables,
            "original_language": lang_code,
            "original_text": raw_text,
            "translation_confidence": confidence,
            "translation_method": method,
            "metadata": {
                **metadata,
                "language_detected": lang_code,
                "translation_applied": True,
                "tool": "Lang-Bridge Tool",
                "framework": "google_adk",
            },
        }

    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            clean_text = re.sub(r"[0-9$€£¥₹,.\-:/()\[\]{}]", " ", text)
            clean_text = " ".join(clean_text.split())
            if len(clean_text.strip()) < 10:
                return "unknown"
            return detect(clean_text)
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "unknown"

    def _load_marian_model(self, lang_code: str):
        """Lazy load MarianMT model."""
        if lang_code in self.marian_models:
            return

        from transformers import MarianMTModel, MarianTokenizer

        model_name = self.supported_languages.get(lang_code)
        if not model_name:
            raise ValueError(f"Language {lang_code} not supported by MarianMT")

        logger.info(f"Loading MarianMT model: {model_name}")
        self.marian_tokenizers[lang_code] = MarianTokenizer.from_pretrained(model_name)
        self.marian_models[lang_code] = MarianMTModel.from_pretrained(model_name)

    def translate_with_marian(self, text: str, lang_code: str) -> Dict[str, Any]:
        """Translate using MarianMT."""
        self._load_marian_model(lang_code)
        tokenizer = self.marian_tokenizers[lang_code]
        model = self.marian_models[lang_code]

        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        return {
            "translated_text": translated_text,
            "confidence": 0.85,
            "translation_method": "MarianMT",
        }

    def translate_with_llm(self, text: str, source_lang: str) -> Dict[str, Any]:
        """Translate using LLM via LiteLLM."""
        prompt = self.persona.get_prompt(
            "translate_text", source_lang=source_lang, text=text
        )
        if not prompt:
            prompt = f"""You are an expert multilingual translator specializing in business documents and invoices.

Task: Translate the following invoice text from {source_lang} to English.

Requirements:
1. Maintain the structure and formatting
2. Preserve all numbers, dates, and codes exactly as they appear
3. Translate field names and descriptions accurately
4. Keep currency symbols as-is (don't convert values)
5. Be precise with business terminology

Text to translate:
---
{text}
---

Provide ONLY the translated English text without any explanations or notes."""

        input_check = self.guardrails.check_input(prompt, agent_id=self.agent_id)
        if not input_check.passed:
            raise ValueError("Input blocked by RAI guardrails")

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )

        translated_text = response.choices[0].message.content.strip()
        return {
            "translated_text": translated_text,
            "confidence": 0.95,
            "translation_method": "LLM_Bedrock_Cohere",
        }

    def normalize_currency(self, text: str) -> str:
        """Normalize currency symbols to ISO codes."""
        currency_map = self.config.get("currency_symbol_map", {})
        for symbol, code in currency_map.items():
            text = text.replace(symbol, code)
        return text

    def translate_tables(
        self, tables: List[List[List[str]]], lang_code: str
    ) -> List[List[List[str]]]:
        """Translate table content."""
        if not tables or lang_code == "en":
            return tables

        translated_tables = []
        for table in tables:
            translated_table = []
            for row in table:
                translated_row = []
                for cell in row:
                    if cell and isinstance(cell, str) and cell.strip():
                        if re.match(r"^[\d\s,.€$£¥₹\-]+$", cell.strip()):
                            translated_row.append(cell)
                            continue
                        try:
                            if self.use_llm and self.model:
                                result = self.translate_with_llm(cell, lang_code)
                            else:
                                result = self.translate_with_marian(cell, lang_code)
                            translated_row.append(result["translated_text"])
                        except Exception:
                            translated_row.append(cell)
                    else:
                        translated_row.append(cell)
                translated_table.append(translated_row)
            translated_tables.append(translated_table)

        return translated_tables
