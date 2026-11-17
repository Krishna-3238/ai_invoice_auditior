"""
Responsible AI (RAI) Guardrails
================================
Safety layers that ensure AI components behave ethically, securely,
and remain compliant throughout the invoice auditing workflow.

Guardrails are applied at three levels:
1. Input Guardrails - Validate and sanitize inputs to LLMs
2. Output Guardrails - Validate LLM outputs for safety and accuracy
3. Process Guardrails - Ensure workflow compliance and audit trail integrity

Key checks:
- PII Detection & Masking in logs/outputs
- Prompt Injection Detection
- Hallucination flags on structured data
- Data integrity validation
- Compliance boundary enforcement
- Bias detection in validation decisions
"""

import re
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("RAIGuardrails")




class GuardrailLevel(str, Enum):
    """Severity levels for guardrail violations."""

    BLOCK = "block"                            
    WARN = "warn"                                      
    LOG = "log"                               
    SANITIZE = "sanitize"                         


class GuardrailCategory(str, Enum):
    """Categories of guardrail checks."""

    PII_PROTECTION = "pii_protection"
    PROMPT_INJECTION = "prompt_injection"
    HALLUCINATION = "hallucination"
    DATA_INTEGRITY = "data_integrity"
    COMPLIANCE = "compliance"
    BIAS_DETECTION = "bias_detection"
    CONTENT_SAFETY = "content_safety"


@dataclass
class GuardrailViolation:
    """Record of a guardrail violation."""

    violation_id: str
    category: GuardrailCategory
    level: GuardrailLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_id: Optional[str] = None
    remediation: Optional[str] = None


@dataclass
class GuardrailResult:
    """Result of guardrail evaluation."""

    passed: bool
    violations: List[GuardrailViolation] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)



PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "passport": r"\b[A-Z]{1,2}\d{6,9}\b",
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b",
}

BUSINESS_SAFE_FIELDS = {
    "invoice_no",
    "invoice_number",
    "po_number",
    "vendor_id",
    "item_code",
    "sku",
    "currency",
    "invoice_date",
}


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+(?:a|an)\s+",
    r"system\s*:\s*you\s+are",
    r"forget\s+everything",
    r"override\s+(?:your\s+)?instructions",
    r"new\s+instructions?\s*:",
    r"<\s*/?script\s*>",
    r"execute\s+(?:this\s+)?(?:code|command)",
    r"rm\s+-rf",
    r"drop\s+table",
    r";\s*(?:select|insert|update|delete|drop)\s+",
]




class RAIGuardrails:
    """
    Responsible AI Guardrails Engine.

    Wraps around LLM calls and data processing to enforce
    ethical, secure, and compliant behavior.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize guardrails.

        Args:
            strict_mode: If True, WARN-level violations become BLOCK-level
        """
        self.strict_mode = strict_mode
        self.violation_log: List[GuardrailViolation] = []
        self._violation_counter = 0
        logger.info(f"RAI Guardrails initialized (strict_mode={strict_mode})")

    def _create_violation(
        self,
        category: GuardrailCategory,
        level: GuardrailLevel,
        message: str,
        agent_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        remediation: Optional[str] = None,
    ) -> GuardrailViolation:
        """Create and log a violation."""
        self._violation_counter += 1

        if self.strict_mode and level == GuardrailLevel.WARN:
            level = GuardrailLevel.BLOCK

        violation = GuardrailViolation(
            violation_id=f"RAI-{self._violation_counter:04d}",
            category=category,
            level=level,
            message=message,
            details=details or {},
            agent_id=agent_id,
            remediation=remediation,
        )

        self.violation_log.append(violation)

        if level == GuardrailLevel.BLOCK:
            logger.error(f"RAI BLOCK: [{category.value}] {message}")
        elif level == GuardrailLevel.WARN:
            logger.warning(f"RAI WARN: [{category.value}] {message}")
        else:
            logger.info(f"RAI LOG: [{category.value}] {message}")

        return violation


    def check_input(
        self,
        text: str,
        agent_id: str = "unknown",
        context: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """
        Validate input text before sending to LLM.

        Checks for:
        - Prompt injection attempts
        - Excessive PII that shouldn't be sent to LLM
        - Content safety issues
        """
        violations = []

        injection_result = self._check_prompt_injection(text, agent_id)
        violations.extend(injection_result)

        pii_result = self._check_pii(text, agent_id, is_input=True)
        violations.extend(pii_result)

        blocking = [v for v in violations if v.level == GuardrailLevel.BLOCK]
        passed = len(blocking) == 0

        return GuardrailResult(
            passed=passed,
            violations=violations,
            metadata={"agent_id": agent_id, "check_type": "input"},
        )


    def check_output(
        self,
        output: str,
        agent_id: str = "unknown",
        expected_format: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """
        Validate LLM output before using it.

        Checks for:
        - PII leakage in outputs
        - Hallucination indicators
        - Format compliance
        """
        violations = []

        pii_result = self._check_pii(output, agent_id, is_input=False)
        violations.extend(pii_result)

        hallucination_result = self._check_hallucination_indicators(
            output, agent_id, context
        )
        violations.extend(hallucination_result)

        blocking = [v for v in violations if v.level == GuardrailLevel.BLOCK]
        passed = len(blocking) == 0

        sanitized = output
        for violation in violations:
            if violation.level == GuardrailLevel.SANITIZE:
                sanitized = self._sanitize_pii(sanitized)

        return GuardrailResult(
            passed=passed,
            violations=violations,
            sanitized_data={"text": sanitized},
            metadata={"agent_id": agent_id, "check_type": "output"},
        )


    def check_invoice_data_integrity(
        self,
        invoice_data: Dict[str, Any],
        agent_id: str = "unknown",
    ) -> GuardrailResult:
        """
        Validate extracted invoice data for integrity.

        Checks for:
        - Negative or unreasonable monetary amounts
        - Future dates
        - Inconsistent data
        """
        violations = []

        total = invoice_data.get("total_amount")
        if total is not None:
            try:
                total_float = float(total)
                if total_float < 0:
                    violations.append(
                        self._create_violation(
                            GuardrailCategory.DATA_INTEGRITY,
                            GuardrailLevel.WARN,
                            f"Negative total amount: {total_float}",
                            agent_id=agent_id,
                            details={"field": "total_amount", "value": total_float},
                            remediation="Verify with source document",
                        )
                    )
                if total_float > 10_000_000:
                    violations.append(
                        self._create_violation(
                            GuardrailCategory.DATA_INTEGRITY,
                            GuardrailLevel.WARN,
                            f"Unusually large total amount: {total_float}",
                            agent_id=agent_id,
                            details={"field": "total_amount", "value": total_float},
                            remediation="Manual review recommended",
                        )
                    )
            except (ValueError, TypeError):
                pass

        invoice_date = invoice_data.get("invoice_date")
        if invoice_date:
            try:
                from dateutil import parser as date_parser

                parsed_date = date_parser.parse(str(invoice_date))
                if parsed_date.date() > datetime.now().date():
                    violations.append(
                        self._create_violation(
                            GuardrailCategory.DATA_INTEGRITY,
                            GuardrailLevel.WARN,
                            f"Future invoice date: {invoice_date}",
                            agent_id=agent_id,
                            details={
                                "field": "invoice_date",
                                "value": str(invoice_date),
                            },
                        )
                    )
            except Exception:
                pass

        blocking = [v for v in violations if v.level == GuardrailLevel.BLOCK]
        passed = len(blocking) == 0

        return GuardrailResult(
            passed=passed,
            violations=violations,
            metadata={"agent_id": agent_id, "check_type": "data_integrity"},
        )


    def check_compliance(
        self,
        workflow_state: Dict[str, Any],
        agent_id: str = "unknown",
    ) -> GuardrailResult:
        """
        Validate workflow compliance requirements.

        Ensures:
        - Audit trail completeness
        - Required steps have been executed
        - Proper authorization for decisions
        """
        violations = []

        if workflow_state.get("validation_summary") and not workflow_state.get(
            "extracted_text"
        ):
            violations.append(
                self._create_violation(
                    GuardrailCategory.COMPLIANCE,
                    GuardrailLevel.WARN,
                    "Validation performed without extraction step",
                    agent_id=agent_id,
                )
            )

        if workflow_state.get("report_path") and not workflow_state.get(
            "recommendation"
        ):
            violations.append(
                self._create_violation(
                    GuardrailCategory.COMPLIANCE,
                    GuardrailLevel.WARN,
                    "Report generated without recommendation",
                    agent_id=agent_id,
                )
            )

        blocking = [v for v in violations if v.level == GuardrailLevel.BLOCK]
        passed = len(blocking) == 0

        return GuardrailResult(
            passed=passed,
            violations=violations,
            metadata={"agent_id": agent_id, "check_type": "compliance"},
        )


    def check_validation_bias(
        self,
        validation_results: List[Dict[str, Any]],
        agent_id: str = "unknown",
    ) -> GuardrailResult:
        """
        Check for potential bias in validation decisions.

        Flags if a disproportionate number of invoices from
        a specific vendor are being rejected.
        """
        violations = []

        if len(validation_results) < 5:
            return GuardrailResult(
                passed=True,
                violations=[],
                metadata={"agent_id": agent_id, "check_type": "bias_detection"},
            )

        vendor_outcomes = {}
        for result in validation_results:
            vendor = result.get("invoice_data", {}).get("vendor_id", "unknown")
            recommendation = result.get("recommendation", "UNKNOWN")
            vendor_outcomes.setdefault(vendor, []).append(recommendation)

        for vendor, outcomes in vendor_outcomes.items():
            if len(outcomes) >= 3:
                reject_rate = outcomes.count("REJECT") / len(outcomes)
                if reject_rate > 0.8:
                    violations.append(
                        self._create_violation(
                            GuardrailCategory.BIAS_DETECTION,
                            GuardrailLevel.WARN,
                            f"High rejection rate ({reject_rate:.0%}) for vendor {vendor}",
                            agent_id=agent_id,
                            details={
                                "vendor_id": vendor,
                                "total_invoices": len(outcomes),
                                "rejections": outcomes.count("REJECT"),
                                "reject_rate": reject_rate,
                            },
                            remediation="Review validation rules for potential bias",
                        )
                    )

        blocking = [v for v in violations if v.level == GuardrailLevel.BLOCK]
        passed = len(blocking) == 0

        return GuardrailResult(
            passed=passed,
            violations=violations,
            metadata={"agent_id": agent_id, "check_type": "bias_detection"},
        )


    def _check_prompt_injection(
        self, text: str, agent_id: str
    ) -> List[GuardrailViolation]:
        """Check for prompt injection patterns."""
        violations = []
        text_lower = text.lower()

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append(
                    self._create_violation(
                        GuardrailCategory.PROMPT_INJECTION,
                        GuardrailLevel.BLOCK,
                        f"Potential prompt injection detected: pattern '{pattern}'",
                        agent_id=agent_id,
                        details={"pattern": pattern},
                        remediation="Input sanitized; blocked from LLM",
                    )
                )
                break                                    

        return violations

    def _check_pii(
        self, text: str, agent_id: str, is_input: bool
    ) -> List[GuardrailViolation]:
        """Check for PII in text."""
        violations = []

        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                level = GuardrailLevel.LOG if is_input else GuardrailLevel.SANITIZE
                violations.append(
                    self._create_violation(
                        GuardrailCategory.PII_PROTECTION,
                        level,
                        f"Detected {len(matches)} potential {pii_type} in "
                        f"{'input' if is_input else 'output'}",
                        agent_id=agent_id,
                        details={"pii_type": pii_type, "count": len(matches)},
                        remediation="PII will be masked in logs",
                    )
                )

        return violations

    def _check_hallucination_indicators(
        self,
        output: str,
        agent_id: str,
        context: Optional[Dict[str, Any]],
    ) -> List[GuardrailViolation]:
        """Check for hallucination indicators in LLM output."""
        violations = []

        hallucination_phrases = [
            "I believe",
            "I think",
            "probably",
            "likely",
            "it seems",
            "appears to be",
            "might be",
        ]

        if context and context.get("expects_factual", False):
            output_lower = output.lower()
            for phrase in hallucination_phrases:
                if phrase in output_lower:
                    violations.append(
                        self._create_violation(
                            GuardrailCategory.HALLUCINATION,
                            GuardrailLevel.WARN,
                            f"Hallucination indicator in factual response: '{phrase}'",
                            agent_id=agent_id,
                            details={"phrase": phrase},
                            remediation="Cross-validate with source document",
                        )
                    )
                    break

        return violations

    def _sanitize_pii(self, text: str) -> str:
        """Mask PII in text."""
        sanitized = text
        for pii_type, pattern in PII_PATTERNS.items():
            sanitized = re.sub(
                pattern,
                f"[{pii_type.upper()}_REDACTED]",
                sanitized,
            )
        return sanitized


    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of all violations."""
        summary = {
            "total_violations": len(self.violation_log),
            "by_category": {},
            "by_level": {},
            "blocked_count": 0,
        }

        for v in self.violation_log:
            cat = v.category.value
            lvl = v.level.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            summary["by_level"][lvl] = summary["by_level"].get(lvl, 0) + 1
            if v.level == GuardrailLevel.BLOCK:
                summary["blocked_count"] += 1

        return summary

    def get_recent_violations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent violations as dicts."""
        return [
            {
                "violation_id": v.violation_id,
                "category": v.category.value,
                "level": v.level.value,
                "message": v.message,
                "agent_id": v.agent_id,
                "timestamp": v.timestamp,
                "remediation": v.remediation,
            }
            for v in self.violation_log[-limit:]
        ]




class GuardedLLMWrapper:
    """
    Wrapper that applies RAI guardrails around LLM calls.

    Usage:
        guardrails = RAIGuardrails()
        guarded = GuardedLLMWrapper(guardrails)
        result = guarded.call(model, prompt, agent_id="extractor")
    """

    def __init__(self, guardrails: RAIGuardrails):
        self.guardrails = guardrails

    def call(
        self,
        model: str,
        prompt: str,
        agent_id: str = "unknown",
        expects_factual: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a guardrailed LLM call.

        Args:
            model: LiteLLM model string
            prompt: Prompt to send
            agent_id: Calling agent ID
            expects_factual: Whether response should be factual only
            **kwargs: Additional args for litellm.completion

        Returns:
            Dict with 'response', 'guardrail_check', and 'text'
        """
        from litellm import completion

        input_check = self.guardrails.check_input(prompt, agent_id)
        if not input_check.passed:
            return {
                "response": None,
                "text": "",
                "guardrail_check": {
                    "input_passed": False,
                    "violations": [v.message for v in input_check.violations],
                },
                "blocked": True,
            }

        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        response_text = response.choices[0].message.content.strip()

        context = {"expects_factual": expects_factual}
        output_check = self.guardrails.check_output(
            response_text, agent_id, context=context
        )

        final_text = response_text
        if output_check.sanitized_data:
            final_text = output_check.sanitized_data.get("text", response_text)

        return {
            "response": response,
            "text": final_text,
            "guardrail_check": {
                "input_passed": input_check.passed,
                "output_passed": output_check.passed,
                "violations": [
                    v.message for v in input_check.violations + output_check.violations
                ],
            },
            "blocked": False,
        }



_guardrails = RAIGuardrails()


def get_guardrails() -> RAIGuardrails:
    """Get the global RAI guardrails instance."""
    return _guardrails


def get_guarded_llm() -> GuardedLLMWrapper:
    """Get a guardrailed LLM wrapper."""
    return GuardedLLMWrapper(_guardrails)
