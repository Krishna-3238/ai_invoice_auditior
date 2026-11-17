"""
ADK Reporting Agent – Insight Reporter Tool
=============================================
Google ADK implementation of the Reporting Agent.
Persona: Compliance Reporter
Tool Name: Insight Reporter Tool

Generates structured validation reports with audit trails,
including JSON and HTML formats, executive summaries,
and recommendations.

Registered with:
- A2A Protocol as 'adk_reporting_agent'
- MCP as 'insight_reporter_tool'
"""

import os
import sys
import json
import logging
import yaml
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
from litellm import completion

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from agents.adk_agents.base_agent import ADKBaseAgent
from protocols.mcp_protocol import MCPToolDescriptor, ToolCategory
from config.persona_loader import AgentPersona
from guardrails import get_guardrails

logger = logging.getLogger("ADK.ReportingAgent")


class ADKReportingAgent(ADKBaseAgent):
    """
    Google ADK Reporting Agent with Insight Reporter Tool.

    Generates comprehensive validation reports in JSON and HTML formats.
    """

    def __init__(
        self,
        model: str = "bedrock/amazon.nova-lite-v1:0",
        config_path: str = "./config/rules.yaml",
    ):
        super().__init__(
            agent_id="adk_reporting_agent",
            name="Reporting Agent (ADK)",
            description="Generates structured validation reports with audit trails and recommendations",
            capabilities=[
                "json_report",
                "html_report",
                "executive_summary",
                "audit_trail",
            ],
            tools=["insight_reporter_tool"],
        )
        self.model = model
        self.config = self._load_config(config_path)
        self.persona = AgentPersona("reporting_agent")

        self.output_dir = Path(
            self.config.get("reporting", {}).get("output_dir", "./outputs/reports")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._register_insight_reporter_tool()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def _register_insight_reporter_tool(self):
        """Register the Insight Reporter Tool with MCP."""
        descriptor = MCPToolDescriptor(
            tool_id="insight_reporter_tool",
            name="Insight Reporter Tool",
            description=(
                "Generates comprehensive validation reports including JSON data, "
                "HTML visual reports, executive summaries, and audit trails. "
                "Provides APPROVE/MANUAL_REVIEW/REJECT recommendations."
            ),
            category=ToolCategory.REPORTING,
            input_schema={
                "type": "object",
                "properties": {
                    "invoice_data": {"type": "object"},
                    "validation_issues": {"type": "array"},
                    "recommendation": {"type": "string"},
                },
                "required": ["invoice_data"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "report_json": {"type": "object"},
                    "files": {"type": "object"},
                },
            },
        )
        self.register_mcp_tool(descriptor, self._insight_reporter_handler)

    def _insight_reporter_handler(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """MCP handler for the Insight Reporter Tool."""
        return self.run(input_data, save_format="both")

    def run(
        self,
        input_data: Dict[str, Any],
        save_format: str = "both",
    ) -> Dict[str, Any]:
        """
        Generate reports (A2A handler).

        Args:
            input_data: Validation results
            save_format: 'json', 'html', or 'both'

        Returns:
            Report data and file paths
        """
        logger.info("Insight Reporter: Generating reports...")

        json_report = self.generate_json_report(input_data)

        html_content = self.generate_html_report(json_report)

        invoice_no = input_data.get("invoice_data", {}).get("invoice_no", "UNKNOWN")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"report_{invoice_no}_{timestamp}"

        files = {}

        if save_format in ["json", "both"]:
            json_path = self.output_dir / f"{base_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_report, f, indent=2, default=str)
            files["json"] = str(json_path)

        if save_format in ["html", "both"]:
            html_path = self.output_dir / f"{base_name}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            files["html"] = str(html_path)

        logger.info(f"Insight Reporter: Reports saved to {files}")

        return {
            "report_json": json_report,
            "html_content": html_content,
            "files": files,
        }

    def generate_summary(self, validation_result: Dict[str, Any]) -> str:
        """Generate natural language summary using LLM."""
        if not self.model:
            return self._generate_basic_summary(validation_result)

        try:
            invoice_data = validation_result.get("invoice_data", {})
            issues = validation_result.get("validation_issues", [])
            recommendation = validation_result.get("recommendation", "UNKNOWN")

            error_issues = [i for i in issues if i.get("severity") == "error"]
            warning_issues = [i for i in issues if i.get("severity") == "warning"]

            prompt = self.persona.get_prompt(
                "executive_summary",
                invoice_no=invoice_data.get("invoice_no", "N/A"),
                vendor_id=invoice_data.get("vendor_id", "N/A"),
                vendor_name=invoice_data.get("vendor_name", "N/A"),
                po_number=invoice_data.get("po_number", "N/A"),
                invoice_date=invoice_data.get("invoice_date", "N/A"),
                total_amount=invoice_data.get("total_amount", "N/A"),
                currency=invoice_data.get("currency", "N/A"),
                total_issues=len(issues),
                error_count=len(error_issues),
                warning_count=len(warning_issues),
                recommendation=recommendation,
                issues_summary=self._format_issues(issues[:5]),
            )

            if not prompt:
                prompt = f"""Write a concise executive summary for an invoice validation report.

Invoice: {invoice_data.get('invoice_no', 'N/A')} | Vendor: {invoice_data.get('vendor_name', 'N/A')}
Total: {invoice_data.get('total_amount', 'N/A')} {invoice_data.get('currency', 'N/A')}
Issues: {len(error_issues)} errors, {len(warning_issues)} warnings
Recommendation: {recommendation}

Key Issues:
{self._format_issues(issues[:5])}

Write 3-4 sentences. Be professional and concise."""

            input_check = self.guardrails.check_input(prompt, agent_id=self.agent_id)
            if not input_check.passed:
                return self._generate_basic_summary(validation_result)

            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM summary failed: {e}")
            return self._generate_basic_summary(validation_result)

    def _format_issues(self, issues: List[Dict]) -> str:
        if not issues:
            return "No issues found."
        return "\n".join(
            f"{i}. [{issue.get('severity', 'unknown').upper()}] {issue.get('message', '')}"
            for i, issue in enumerate(issues, 1)
        )

    def _generate_basic_summary(self, validation_result: Dict[str, Any]) -> str:
        invoice_data = validation_result.get("invoice_data", {})
        recommendation = validation_result.get("recommendation", "UNKNOWN")
        invoice_no = invoice_data.get("invoice_no", "N/A")
        vendor = invoice_data.get("vendor_id", "N/A")

        if recommendation == "APPROVE":
            return f"Invoice {invoice_no} from vendor {vendor} validated successfully. Recommended for approval."
        elif recommendation == "MANUAL_REVIEW":
            return f"Invoice {invoice_no} from vendor {vendor} has warnings requiring manual review."
        else:
            return f"Invoice {invoice_no} from vendor {vendor} has critical errors. Recommended for rejection."

    def generate_json_report(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured JSON report."""
        invoice_data = validation_result.get("invoice_data", {})
        issues = validation_result.get("validation_issues", [])
        metadata = validation_result.get(
            "metadata", validation_result.get("extraction_metadata", {})
        )

        executive_summary = self.generate_summary(validation_result)

        errors = [i for i in issues if i.get("severity") == "error"]
        warnings = [i for i in issues if i.get("severity") == "warning"]
        missing_fields = [
            i.get("field") for i in issues if i.get("type") == "missing_field"
        ]

        return {
            "report_metadata": {
                "report_id": f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "generated_at": datetime.now().isoformat(),
                "report_version": "2.0",
                "agent": "ADK Reporting Agent (Insight Reporter)",
                "framework": "google_adk",
                "tool": "Insight Reporter Tool",
            },
            "invoice_summary": {
                "invoice_number": invoice_data.get("invoice_no"),
                "invoice_date": invoice_data.get("invoice_date"),
                "vendor_id": invoice_data.get("vendor_id"),
                "vendor_name": invoice_data.get("vendor_name"),
                "po_number": invoice_data.get("po_number"),
                "currency": invoice_data.get("currency"),
                "total_amount": invoice_data.get("total_amount"),
                "tax_amount": invoice_data.get("tax_amount"),
            },
            "validation_summary": {
                "is_valid": validation_result.get("is_valid", False),
                "recommendation": validation_result.get("recommendation", "UNKNOWN"),
                "total_issues": len(issues),
                "errors_count": len(errors),
                "warnings_count": len(warnings),
                "missing_fields": missing_fields,
            },
            "translation_info": {
                "original_language": validation_result.get(
                    "translation_metadata", {}
                ).get("original_language", "unknown"),
                "translation_confidence": validation_result.get(
                    "translation_metadata", {}
                ).get("translation_confidence", 1.0),
            },
            "executive_summary": executive_summary,
            "discrepancy_details": {"errors": errors, "warnings": warnings},
            "audit_trail": {
                "source_file": metadata.get("file_name", "unknown"),
                "extraction_method": metadata.get("extraction_method", "unknown"),
                "validation_timestamp": datetime.now().isoformat(),
                "validators_used": ["DataCompletenessChecker", "BusinessValidation"],
                "framework": "google_adk",
            },
        }

    def generate_html_report(self, json_report: Dict[str, Any]) -> str:
        """Generate HTML report from JSON report."""
        invoice = json_report.get("invoice_summary", {})
        validation = json_report.get("validation_summary", {})

        rec = validation.get("recommendation", "UNKNOWN")
        if rec == "APPROVE":
            color, icon = "#28a745", "✓"
        elif rec == "MANUAL_REVIEW":
            color, icon = "#ffc107", "⚠"
        else:
            color, icon = "#dc3545", "✗"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Invoice Validation Report - {invoice.get('invoice_number', 'N/A')}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }}
        .header {{ border-bottom: 3px solid {color}; padding-bottom: 15px; margin-bottom: 20px; }}
        .header h1 {{ color: #2c3e50; }}
        .badge {{ display: inline-block; background: {color}; color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid {color}; }}
        .section h2 {{ color: #2c3e50; margin-bottom: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
        .item {{ padding: 8px; background: white; border-radius: 4px; }}
        .label {{ font-size: 11px; text-transform: uppercase; color: #666; font-weight: bold; }}
        .value {{ font-size: 15px; color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px 12px; border: 1px solid #dee2e6; text-align: left; }}
        th {{ background: #f8f9fa; }}
        .footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #ddd; color: #888; font-size: 12px; }}
        .framework-badge {{ display: inline-block; background: #e3f2fd; color: #1565c0; padding: 3px 8px; border-radius: 3px; font-size: 11px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Invoice Validation Report</h1>
        <p>Generated: {json_report.get('report_metadata', {}).get('generated_at', 'N/A')}</p>
        <span class="framework-badge">Google ADK – Insight Reporter Tool</span>
    </div>
    
    <div class="badge">{icon} {rec}</div>
    
    <div class="section">
        <h2>Invoice Details</h2>
        <div class="grid">
            <div class="item"><div class="label">Invoice #</div><div class="value">{invoice.get('invoice_number', 'N/A')}</div></div>
            <div class="item"><div class="label">Date</div><div class="value">{invoice.get('invoice_date', 'N/A')}</div></div>
            <div class="item"><div class="label">Vendor</div><div class="value">{invoice.get('vendor_name', 'N/A')}</div></div>
            <div class="item"><div class="label">PO Number</div><div class="value">{invoice.get('po_number', 'N/A')}</div></div>
            <div class="item"><div class="label">Total</div><div class="value">{invoice.get('total_amount', 'N/A')} {invoice.get('currency', '')}</div></div>
        </div>
    </div>
    
    <div class="section">
        <h2>Validation Summary</h2>
        <div class="grid">
            <div class="item"><div class="label">Total Issues</div><div class="value">{validation.get('total_issues', 0)}</div></div>
            <div class="item"><div class="label">Errors</div><div class="value">{validation.get('errors_count', 0)}</div></div>
            <div class="item"><div class="label">Warnings</div><div class="value">{validation.get('warnings_count', 0)}</div></div>
        </div>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>{json_report.get('executive_summary', 'N/A')}</p>
    </div>"""

        errors = json_report.get("discrepancy_details", {}).get("errors", [])
        warnings = json_report.get("discrepancy_details", {}).get("warnings", [])
        all_issues = errors + warnings

        if all_issues:
            html += """
    <div class="section">
        <h2>Issues Found</h2>
        <table>
            <tr><th>Severity</th><th>Type</th><th>Message</th></tr>"""
            for issue in all_issues:
                sev = issue.get("severity", "unknown").upper()
                html += f"""
            <tr><td>{sev}</td><td>{issue.get('type', 'N/A')}</td><td>{issue.get('message', 'N/A')}</td></tr>"""
            html += """
        </table>
    </div>"""

        html += f"""
    <div class="footer">
        <p>Report ID: {json_report.get('report_metadata', {}).get('report_id', 'N/A')} | 
        Agent: ADK Reporting Agent (Insight Reporter Tool) | 
        Framework: Google ADK + LangGraph</p>
    </div>
</div>
</body>
</html>"""

        return html
