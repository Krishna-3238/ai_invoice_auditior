"""
ADK Extractor Agent – Data Harvester Tool
==========================================
Google ADK implementation of the Extractor Agent.
Persona: Document Parser
Tool Name: Data Harvester Tool

Extracts text, tables, and structured data from invoice documents
(PDF, DOCX, images) using appropriate parsing tools.

Registered with:
- A2A Protocol as 'adk_extractor_agent'
- MCP as 'data_harvester_tool'
"""

import os
import sys
import logging
from typing import Dict, Any, List

import pdfplumber
import pytesseract
from PIL import Image
from docx import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from agents.adk_agents.base_agent import ADKBaseAgent
from protocols.mcp_protocol import MCPToolDescriptor, ToolCategory
from config.persona_loader import AgentPersona, get_prompt
from guardrails import get_guardrails

logger = logging.getLogger("ADK.ExtractorAgent")


class ADKExtractorAgent(ADKBaseAgent):
    """
    Google ADK Extractor Agent with Data Harvester Tool.

    Extracts text, tables from invoices using pdfplumber,
    python-docx, and pytesseract. Exposes functionality
    through MCP as the "Data Harvester Tool".
    """

    def __init__(self, llm=None):
        super().__init__(
            agent_id="adk_extractor_agent",
            name="Extractor Agent (ADK)",
            description="Extracts text, tables, and structured data from invoice documents",
            capabilities=[
                "pdf_extraction",
                "docx_extraction",
                "ocr_extraction",
                "data_structuring",
            ],
            tools=["data_harvester_tool"],
        )
        self.llm = llm
        self.persona = AgentPersona("extractor_agent")
        self.guardrails = get_guardrails()

        self._register_data_harvester_tool()

    def _register_data_harvester_tool(self):
        """Register the Data Harvester Tool with MCP."""
        descriptor = MCPToolDescriptor(
            tool_id="data_harvester_tool",
            name="Data Harvester Tool",
            description=(
                "Extracts raw text, tables, and metadata from invoice documents. "
                "Supports PDF, DOCX, and image (PNG/JPG) formats. "
                "Uses pdfplumber for PDFs, python-docx for DOCX, "
                "and pytesseract for OCR on scanned documents."
            ),
            category=ToolCategory.EXTRACTION,
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the invoice file",
                    },
                },
                "required": ["file_path"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "raw_text": {"type": "string"},
                    "tables": {"type": "array"},
                    "metadata": {"type": "object"},
                },
            },
        )
        self.register_mcp_tool(descriptor, self._data_harvester_handler)

    def _data_harvester_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """MCP handler for the Data Harvester Tool."""
        return self.run(input_data)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        A2A handler: Extract content from an invoice file.

        Args:
            input_data: Dict with 'file_path' key

        Returns:
            Extraction results with raw_text, tables, metadata
        """
        file_path = input_data.get("file_path", input_data.get("invoice_path", ""))
        if not file_path:
            raise ValueError("No file_path provided for extraction")

        result = self.extract(file_path)

        rai_check = self.guardrails.check_output(
            result.get("raw_text", ""),
            agent_id=self.agent_id,
        )
        if rai_check.violations:
            result["rai_violations"] = [v.message for v in rai_check.violations]

        return result

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract content from invoice file based on extension.

        Args:
            file_path: Path to the invoice file

        Returns:
            Dict containing raw_text, tables, and metadata
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        logger.info(
            f"Data Harvester: Extracting from {os.path.basename(file_path)} ({file_ext})"
        )

        if file_ext == ".pdf":
            return self._extract_pdf(file_path)
        elif file_ext == ".docx":
            return self._extract_docx(file_path)
        elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return self._extract_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _extract_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract from PDF using pdfplumber."""
        raw_text = ""
        tables: List[Any] = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                raw_text += page.extract_text() or ""
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)

        return {
            "raw_text": raw_text.strip(),
            "tables": tables,
            "metadata": {
                "file_name": os.path.basename(file_path),
                "file_type": "pdf",
                "extraction_method": "pdfplumber",
                "tool": "Data Harvester Tool",
                "framework": "google_adk",
            },
        }

    def _extract_docx(self, file_path: str) -> Dict[str, Any]:
        """Extract from DOCX using python-docx."""
        doc = Document(file_path)
        raw_text = "\n".join(p.text for p in doc.paragraphs)
        tables = []

        for table in doc.tables:
            rows = []
            for row in table.rows:
                rows.append([cell.text for cell in row.cells])
            tables.append(rows)

        return {
            "raw_text": raw_text.strip(),
            "tables": tables,
            "metadata": {
                "file_name": os.path.basename(file_path),
                "file_type": "docx",
                "extraction_method": "python-docx",
                "tool": "Data Harvester Tool",
                "framework": "google_adk",
            },
        }

    def _extract_image(self, file_path: str) -> Dict[str, Any]:
        """Extract from image using pytesseract OCR."""
        image = Image.open(file_path)
        raw_text = pytesseract.image_to_string(image)

        return {
            "raw_text": raw_text.strip(),
            "tables": [],
            "metadata": {
                "file_name": os.path.basename(file_path),
                "file_type": "image",
                "extraction_method": "pytesseract",
                "tool": "Data Harvester Tool",
                "framework": "google_adk",
            },
        }

    def structure_invoice_content(self, raw_text: str, tables: list) -> Dict[str, Any]:
        """
        Convert extracted invoice content into structured JSON.
        Requires LLM to be provided in __init__.
        """
        if not self.llm:
            raise ValueError("LLM is required for structure_invoice_content")

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        invoice_schema = (
            get_prompt("extractor_agent", "invoice_schema")
            or self._default_invoice_schema()
        )

        prompt_template = self.persona.get_prompt(
            "structure_invoice",
            raw_text="{raw_text}",
            tables="{tables}",
            invoice_schema="{invoice_schema}",
        )

        if not prompt_template:
            prompt_template = """You are an invoice understanding engine.

Raw invoice text:
{raw_text}

Extracted tables:
{tables}

Expected JSON schema (guideline only, do not validate):
{invoice_schema}

Return ONLY valid JSON."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["raw_text", "tables", "invoice_schema"],
        )

        chain = prompt | self.llm | JsonOutputParser()
        return chain.invoke(
            {
                "raw_text": raw_text,
                "tables": tables,
                "invoice_schema": invoice_schema,
            }
        )

    def _default_invoice_schema(self) -> str:
        return """{
  "invoice_number": string | null,
  "invoice_date": string | null,
  "vendor_name": string | null,
  "currency": string | null,
  "total": number | null,
  "line_items": [{"item_code": string | null, "description": string, "quantity": number | null, "unit_price": number | null, "total": number | null}]
}"""
