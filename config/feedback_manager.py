"""
Human Feedback Persistence Manager
===================================
This module persists human feedback (approvals, corrections, rejections)
to PostgreSQL for durable audit trails and compliance reporting.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid

from config.logging_config import get_logger

DEFAULT_FEEDBACK_DIR = "./outputs/feedback"


class FeedbackManager:
    """
    Manages human feedback persistence for audit trails.

    Stores feedback in PostgreSQL JSONB payloads including:
    - Approvals
    - Corrections
    - Rejections
    - Notes/Comments
    """

    def __init__(self, feedback_dir: str = DEFAULT_FEEDBACK_DIR):
        """
        Initialize the FeedbackManager.

        Args:
            feedback_dir: Accepted for backwards compatibility; PostgreSQL is the store.
        """
        self.feedback_dir = Path(feedback_dir)  # retained for API compatibility; feedback is database-backed
        self.logger = get_logger("FeedbackManager")

    def _connect(self):
        dsn = os.getenv("DATABASE_URL", "")
        if not dsn:
            raise RuntimeError("DATABASE_URL must be configured for durable human feedback")
        import psycopg
        return psycopg.connect(dsn)

    def _ensure_schema(self):
        from integrations.storage import PostgresInvoiceStore
        PostgresInvoiceStore().ensure_schema()

    @staticmethod
    def _row_to_entry(row):
        return {
            "feedback_id": row[0], "workflow_id": row[1], "invoice_id": row[2],
            "feedback_type": row[3], "decision": row[4], "reviewer_id": row[5],
            "user_notes": row[6], "original_data": row[7] or {},
            "corrected_data": row[8] or {}, "metadata": row[9] or {},
            "changes_summary": row[10] or {}, "timestamp": row[11].isoformat(),
            "has_corrections": bool((row[10] or {}).get("changes_detected", False)),
        }

    def _generate_feedback_id(self) -> str:
        """Generate a unique feedback ID."""
        return f"FB-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

    def record_feedback(
        self,
        invoice_id: str,
        feedback_type: str,
        decision: str,
        original_data: Optional[Dict[str, Any]] = None,
        corrected_data: Optional[Dict[str, Any]] = None,
        user_notes: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record human feedback for an invoice.

        Args:
            invoice_id: ID of the invoice being reviewed
            feedback_type: Type of feedback ('approval', 'correction', 'rejection', 'escalation')
            decision: Decision made ('approved', 'corrected', 'rejected', 'escalated')
            original_data: Original extracted/validated data
            corrected_data: Corrected data (if applicable)
            user_notes: Free-text notes from reviewer
            reviewer_id: ID of the reviewer
            metadata: Additional metadata

        Returns:
            Feedback ID
        """
        feedback_id = self._generate_feedback_id()

        feedback_entry = {
            "feedback_id": feedback_id,
            "invoice_id": invoice_id,
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback_type,
            "decision": decision,
            "original_data": original_data or {},
            "corrected_data": corrected_data or {},
            "user_notes": user_notes,
            "reviewer_id": reviewer_id or "anonymous",
            "metadata": metadata or {},
            "changes_summary": self._summarize_changes(original_data, corrected_data),
        }

        from psycopg.types.json import Jsonb
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO audit.human_feedback
                   (feedback_id, workflow_id, invoice_id, feedback_type, decision,
                    reviewer_id, user_notes, original_payload, corrected_payload,
                    metadata, changes_summary)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (feedback_id, (metadata or {}).get("workflow_id"), invoice_id,
                 feedback_type, decision, reviewer_id or "anonymous", user_notes,
                 Jsonb(original_data or {}, dumps=lambda value: json.dumps(value, default=str)),
                 Jsonb(corrected_data or {}, dumps=lambda value: json.dumps(value, default=str)),
                 Jsonb(metadata or {}, dumps=lambda value: json.dumps(value, default=str)),
                 Jsonb(feedback_entry["changes_summary"], dumps=lambda value: json.dumps(value, default=str))),
            )

        self.logger.info(
            f"Recorded {feedback_type} feedback for invoice {invoice_id} | "
            f"Decision: {decision} | Feedback ID: {feedback_id}"
        )

        return feedback_id

    def _summarize_changes(
        self, original: Optional[Dict], corrected: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Summarize the differences between original and corrected data.

        Args:
            original: Original data
            corrected: Corrected data

        Returns:
            Summary of changes
        """
        if not original or not corrected:
            return {"changes_detected": False}

        changes = {"changes_detected": False, "changed_fields": [], "field_changes": {}}

        all_keys = set(original.keys()) | set(corrected.keys())

        for key in all_keys:
            orig_val = original.get(key)
            corr_val = corrected.get(key)

            if orig_val != corr_val:
                changes["changes_detected"] = True
                changes["changed_fields"].append(key)
                changes["field_changes"][key] = {
                    "original": orig_val,
                    "corrected": corr_val,
                }

        return changes

    def record_approval(
        self,
        invoice_id: str,
        validated_data: Dict[str, Any],
        user_notes: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> str:
        """
        Record an approval decision.

        Args:
            invoice_id: Invoice ID
            validated_data: The validated invoice data
            user_notes: Optional notes
            reviewer_id: Reviewer identifier

        Returns:
            Feedback ID
        """
        return self.record_feedback(
            invoice_id=invoice_id,
            feedback_type="approval",
            decision="approved",
            original_data=validated_data,
            corrected_data=validated_data,
            user_notes=user_notes,
            reviewer_id=reviewer_id,
        )

    def record_correction(
        self,
        invoice_id: str,
        original_data: Dict[str, Any],
        corrected_data: Dict[str, Any],
        user_notes: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> str:
        """
        Record a correction decision.

        Args:
            invoice_id: Invoice ID
            original_data: Original extracted data
            corrected_data: Human-corrected data
            user_notes: Optional notes explaining correction
            reviewer_id: Reviewer identifier

        Returns:
            Feedback ID
        """
        return self.record_feedback(
            invoice_id=invoice_id,
            feedback_type="correction",
            decision="corrected",
            original_data=original_data,
            corrected_data=corrected_data,
            user_notes=user_notes,
            reviewer_id=reviewer_id,
        )

    def record_rejection(
        self,
        invoice_id: str,
        rejected_data: Dict[str, Any],
        rejection_reason: str,
        reviewer_id: Optional[str] = None,
    ) -> str:
        """
        Record a rejection decision.

        Args:
            invoice_id: Invoice ID
            rejected_data: The rejected invoice data
            rejection_reason: Reason for rejection
            reviewer_id: Reviewer identifier

        Returns:
            Feedback ID
        """
        return self.record_feedback(
            invoice_id=invoice_id,
            feedback_type="rejection",
            decision="rejected",
            original_data=rejected_data,
            user_notes=rejection_reason,
            reviewer_id=reviewer_id,
        )

    def record_escalation(
        self,
        invoice_id: str,
        invoice_data: Dict[str, Any],
        escalation_reason: str,
        reviewer_id: Optional[str] = None,
    ) -> str:
        """
        Record an escalation request.

        Args:
            invoice_id: Invoice ID
            invoice_data: The invoice data being escalated
            escalation_reason: Reason for escalation
            reviewer_id: Reviewer identifier

        Returns:
            Feedback ID
        """
        return self.record_feedback(
            invoice_id=invoice_id,
            feedback_type="escalation",
            decision="escalated",
            original_data=invoice_data,
            user_notes=escalation_reason,
            reviewer_id=reviewer_id,
            metadata={"requires_attention": True},
        )

    def get_invoice_feedback_history(self, invoice_id: str) -> List[Dict[str, Any]]:
        """
        Get all feedback history for an invoice.

        Args:
            invoice_id: Invoice ID

        Returns:
            List of feedback entries
        """
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT feedback_id, workflow_id, invoice_id, feedback_type, decision,
                          reviewer_id, user_notes, original_payload, corrected_payload,
                          metadata, changes_summary, created_at
                   FROM audit.human_feedback WHERE invoice_id=%s ORDER BY created_at""",
                (invoice_id,),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_audit_trail(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feedback_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get audit trail entries with optional filters.

        Args:
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            feedback_type: Filter by feedback type

        Returns:
            List of audit trail entries
        """
        self._ensure_schema()
        query = """SELECT feedback_id, workflow_id, invoice_id, feedback_type, decision,
                          reviewer_id, user_notes, original_payload, corrected_payload,
                          metadata, changes_summary, created_at
                   FROM audit.human_feedback WHERE TRUE"""
        params = []
        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)
        if end_date:
            query += " AND created_at <= %s"
            params.append(end_date)
        if feedback_type:
            query += " AND feedback_type = %s"
            params.append(feedback_type)
        query += " ORDER BY created_at"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        entries = [self._row_to_entry(row) for row in rows]

        if start_date:
            entries = [e for e in entries if e["timestamp"] >= start_date]
        if end_date:
            entries = [e for e in entries if e["timestamp"] <= end_date]
        if feedback_type:
            entries = [e for e in entries if e["feedback_type"] == feedback_type]

        return entries

    def get_audit_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from the audit trail.

        Returns:
            Dictionary with audit statistics
        """
        entries = self.get_audit_trail()

        if not entries:
            return {
                "total_reviews": 0,
                "by_type": {},
                "by_decision": {},
                "correction_rate": 0.0,
            }

        by_type = {}
        by_decision = {}
        corrections = 0

        for entry in entries:
            fb_type = entry.get("feedback_type", "unknown")
            decision = entry.get("decision", "unknown")

            by_type[fb_type] = by_type.get(fb_type, 0) + 1
            by_decision[decision] = by_decision.get(decision, 0) + 1

            if entry.get("has_corrections", False):
                corrections += 1

        return {
            "total_reviews": len(entries),
            "by_type": by_type,
            "by_decision": by_decision,
            "correction_rate": corrections / len(entries) if entries else 0.0,
            "last_updated": entries[-1].get("timestamp") if entries else None,
        }


_feedback_manager = None


def get_feedback_manager() -> FeedbackManager:
    """Get the singleton FeedbackManager instance."""
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = FeedbackManager()
    return _feedback_manager
