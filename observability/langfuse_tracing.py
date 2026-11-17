"""
LangFuse Observability Integration
====================================
Real-time tracing and monitoring for agentic workflows.

Provides:
1. Trace management for workflow runs
2. Span tracking for individual agent operations
3. Generation tracking for LLM calls
4. Score recording for RAG quality metrics
5. Performance dashboards and debugging visibility

If LangFuse is not available (no server running), the module gracefully
degrades to local logging so the system keeps working.
"""

import os
import time
import logging
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("LangFuseObservability")


_LANGFUSE_AVAILABLE = False
_langfuse_client = None

try:
    from langfuse import Langfuse
    from langfuse.types import TraceContext

    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        _LANGFUSE_AVAILABLE = True
        logger.info(f"LangFuse connected at {LANGFUSE_HOST}")
    else:
        logger.info(
            "LangFuse keys not configured. Using local trace logging. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable."
        )
except ImportError:
    TraceContext = None                
    logger.info(
        "LangFuse package not installed. Using local trace logging. "
        "Install with: pip install langfuse"
    )




@dataclass
class LocalSpan:
    """A local span record for tracing."""

    span_id: str
    name: str
    trace_id: str
    parent_id: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    level: str = "DEFAULT"                                  


@dataclass
class LocalTrace:
    """A local trace record."""

    trace_id: str
    name: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    spans: List[LocalSpan] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class LocalTraceStore:
    """In-memory query cache with an append-only local summary log."""

    def __init__(self, max_traces: int = 1000):
        self.traces: Dict[str, LocalTrace] = {}
        self.max_traces = max_traces

    def create_trace(self, name: str, **kwargs) -> LocalTrace:
        trace = LocalTrace(
            trace_id=f"trace_{uuid.uuid4().hex[:12]}",
            name=name,
            **kwargs,
        )
        self.traces[trace.trace_id] = trace
        self._trim()
        return trace

    def get_trace(self, trace_id: str) -> Optional[LocalTrace]:
        return self.traces.get(trace_id)

    def _trim(self):
        if len(self.traces) > self.max_traces:
            oldest = sorted(self.traces.keys())[: len(self.traces) - self.max_traces]
            for tid in oldest:
                del self.traces[tid]

    def get_recent(self, limit: int = 50) -> List[LocalTrace]:
        all_traces = sorted(
            self.traces.values(),
            key=lambda t: t.start_time,
            reverse=True,
        )
        return all_traces[:limit]

    def persist_summary(self, trace: LocalTrace) -> None:
        """Persist metadata-only fallback traces so process restarts retain history."""
        path = os.getenv("LOCAL_TRACE_FILE", "./logs/local_traces.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        record = {
            "trace_id": trace.trace_id,
            "name": trace.name,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "duration_ms": trace.duration_ms,
            "status": trace.status,
            "metadata": trace.metadata,
            "scores": trace.scores,
            "spans": [
                {"span_id": span.span_id, "name": span.name,
                 "start_time": span.start_time, "end_time": span.end_time,
                 "duration_ms": span.duration_ms, "status": span.status,
                 "level": span.level}
                for span in trace.spans
            ],
        }
        with open(path, "a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, default=str) + "\n")


_local_store = LocalTraceStore()


def _redact_trace_data(value):
    """Strip invoice content and local file paths before exporting telemetry."""
    sensitive_keys = {"text", "raw_text", "translated_text", "text_preview",
                      "translated_text_preview", "file", "file_path", "invoice_path",
                      "path", "prompt", "completion"}
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if str(key).lower() in sensitive_keys else _redact_trace_data(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_trace_data(item) for item in value]
    return value




class InvoiceAuditorObservability:
    """
    Unified observability interface for the Invoice Auditor system.

    Uses LangFuse v3 API when available, falls back to
    local tracing otherwise. Provides the same API regardless.

    LangFuse v3 API patterns:
    - Traces are created implicitly via TraceContext on the first span
    - client.start_span() / client.start_observation(as_type='generation')
    - span.update(output=...) then span.end()
    - client.create_score() for scoring
    """

    def __init__(self):
        self.use_langfuse = _LANGFUSE_AVAILABLE
        self.client = _langfuse_client
        self.local_store = _local_store
        self._active_traces: Dict[str, Any] = {}
        self._active_spans: Dict[str, Any] = {}


    def start_trace(
        self,
        name: str,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new trace for a workflow run.

        In LangFuse v3, traces are created implicitly when the first
        span is created with a TraceContext. We store the context here.
        """
        if self.use_langfuse and self.client:
            trace_id = self.client.create_trace_id(
                seed=workflow_id or uuid.uuid4().hex[:12]
            )
        else:
            trace_id = workflow_id or f"trace_{uuid.uuid4().hex[:12]}"

        if self.use_langfuse and self.client and TraceContext is not None:
            try:
                trace_context = TraceContext(
                    trace_id=trace_id,
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata or {},
                    input=_redact_trace_data(input_data),
                    tags=["invoice_auditor"],
                )
                self._active_traces[trace_id] = trace_context
                logger.info(f"LangFuse trace started: {trace_id}")
            except Exception as e:
                logger.warning(f"LangFuse trace failed: {e}. Using local.")
                self._start_local_trace(trace_id, name, metadata, user_id, session_id)
        else:
            self._start_local_trace(trace_id, name, metadata, user_id, session_id)

        return trace_id

    def attach_trace(self, trace_id: str, name: str = "invoice_audit") -> None:
        """Attach this process to an existing distributed trace ID."""
        if trace_id in self._active_traces:
            return
        if self.use_langfuse and self.client and TraceContext is not None:
            try:
                self._active_traces[trace_id] = TraceContext(trace_id=trace_id, name=name)
                return
            except Exception as exc:
                logger.warning("Could not attach Langfuse trace context: %s", exc)
        trace = LocalTrace(trace_id=trace_id, name=name)
        self.local_store.traces[trace_id] = trace
        self._active_traces[trace_id] = trace

    def _start_local_trace(self, trace_id, name, metadata, user_id, session_id):
        trace = self.local_store.create_trace(
            name=name,
            metadata=metadata or {},
            user_id=user_id,
            session_id=session_id,
        )
        trace.trace_id = trace_id
        self.local_store.traces[trace_id] = trace
        self._active_traces[trace_id] = trace
        logger.info(f"Local trace started: {trace_id}")

    def end_trace(
        self,
        trace_id: str,
        output: Optional[Dict[str, Any]] = None,
        status: str = "completed",
    ):
        """End a trace. In v3, we update the trace via update_current_trace or a span."""
        trace_ctx = self._active_traces.get(trace_id)
        if not trace_ctx:
            return

        if self.use_langfuse and self.client and not isinstance(trace_ctx, LocalTrace):
            try:
                span = self.client.start_span(
                    name=f"{status}_finalize",
                    input={"status": status},
                    trace_context=trace_ctx,
                )
                span.update(output=_redact_trace_data(output), metadata={"status": status})
                span.end()
                self.client.flush()
            except Exception as e:
                logger.warning(f"LangFuse trace end failed: {e}")
        else:
            if isinstance(trace_ctx, LocalTrace):
                trace_ctx.end_time = datetime.now().isoformat()
                trace_ctx.status = status
                trace_ctx.duration_ms = (datetime.fromisoformat(trace_ctx.end_time) - datetime.fromisoformat(trace_ctx.start_time)).total_seconds() * 1000
                self.local_store.persist_summary(trace_ctx)

        self._active_traces.pop(trace_id, None)
        logger.info(f"Trace ended: {trace_id} ({status})")


    def start_span(
        self,
        trace_id: str,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a span within a trace using LangFuse v3 API."""
        span_id = f"span_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        trace_ctx = self._active_traces.get(trace_id)

        if (
            self.use_langfuse
            and self.client
            and trace_ctx
            and not isinstance(trace_ctx, LocalTrace)
        ):
            try:
                span = self.client.start_span(
                    name=name,
                    input=_redact_trace_data(input_data),
                    metadata=metadata or {},
                    trace_context=trace_ctx,
                )
                self._active_spans[span_id] = (span, start_time, trace_id)
            except Exception as e:
                logger.warning(f"LangFuse span failed: {e}. Using local.")
                self._start_local_span(
                    span_id, trace_id, name, input_data, metadata, start_time
                )
        else:
            self._start_local_span(
                span_id, trace_id, name, input_data, metadata, start_time
            )

        logger.debug(f"Span started: {name} [{span_id}]")
        return span_id

    def _start_local_span(
        self, span_id, trace_id, name, input_data, metadata, start_time
    ):
        local_span = LocalSpan(
            span_id=span_id,
            name=name,
            trace_id=trace_id,
            input_data=input_data,
            metadata=metadata or {},
        )
        self._active_spans[span_id] = (local_span, start_time, trace_id)

        local_trace = self.local_store.get_trace(trace_id)
        if local_trace:
            local_trace.spans.append(local_span)

    def end_span(
        self,
        span_id: str,
        output: Optional[Dict[str, Any]] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
    ):
        """End a span. In v3: span.update(output=...) then span.end()."""
        span_data = self._active_spans.get(span_id)
        if not span_data:
            return

        span, start_time, trace_id = span_data
        duration_ms = (time.time() - start_time) * 1000

        if self.use_langfuse and not isinstance(span, LocalSpan):
            try:
                span.update(
                    output=_redact_trace_data(output),
                    level=level,
                    status_message=status_message,
                )
                span.end()
            except Exception as e:
                logger.warning(f"LangFuse span end failed: {e}")
        else:
            if isinstance(span, LocalSpan):
                span.end_time = datetime.now().isoformat()
                span.duration_ms = duration_ms
                span.output_data = output
                span.status = "error" if level == "ERROR" else "completed"
                span.level = level
                local_trace = self.local_store.get_trace(trace_id)
                if local_trace:
                    self.local_store.persist_summary(local_trace)

        self._active_spans.pop(span_id, None)
        logger.debug(f"Span ended: {span_id} ({duration_ms:.1f}ms)")


    def log_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log an LLM generation call using v3 start_observation."""
        trace_ctx = self._active_traces.get(trace_id)

        capture_content = os.getenv("LANGFUSE_CAPTURE_CONTENT", "false").lower() in {"1", "true", "yes"}
        if (
            self.use_langfuse
            and self.client
            and trace_ctx
            and not isinstance(trace_ctx, LocalTrace)
        ):
            try:
                gen = self.client.start_observation(
                    as_type="generation",
                    name=name,
                    model=model,
                    input=prompt[:5000] if capture_content else None,
                    metadata=metadata or {},
                    trace_context=trace_ctx,
                )
                gen.update(
                    output=completion[:5000] if capture_content else None,
                    usage_details=usage or {},
                )
                gen.end()
            except Exception as e:
                logger.warning(f"LangFuse generation log failed: {e}")

        logger.debug(
            f"Generation logged: {name} (model={model}, "
            f"tokens={usage.get('total_tokens', 'N/A') if usage else 'N/A'})"
        )


    def record_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """Record a quality score using v3 create_score."""
        if self.use_langfuse and self.client:
            try:
                self.client.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment=comment,
                )
            except Exception as e:
                logger.warning(f"LangFuse score failed: {e}")

        local_trace = self.local_store.get_trace(trace_id)
        if local_trace:
            local_trace.scores[name] = value
            self.local_store.persist_summary(local_trace)

        logger.debug(f"Score recorded: {name}={value:.3f} for trace {trace_id}")


    @contextmanager
    def span(
        self,
        trace_id: str,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for automatic span management.

        Usage:
            with observability.span(trace_id, "extraction") as span_id:
                result = extract(data)
        """
        span_id = self.start_span(trace_id, name, input_data)
        try:
            yield span_id
            self.end_span(span_id, level="DEFAULT")
        except Exception as e:
            self.end_span(
                span_id,
                output={"error": str(e)},
                level="ERROR",
                status_message=str(e),
            )
            raise


    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a trace."""
        local_trace = self.local_store.get_trace(trace_id)
        if not local_trace:
            return None

        return {
            "trace_id": local_trace.trace_id,
            "name": local_trace.name,
            "status": local_trace.status,
            "start_time": local_trace.start_time,
            "end_time": local_trace.end_time,
            "duration_ms": local_trace.duration_ms,
            "num_spans": len(local_trace.spans),
            "scores": local_trace.scores,
            "spans": [
                {
                    "name": s.name,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "level": s.level,
                }
                for s in local_trace.spans
            ],
        }

    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent traces with summaries."""
        traces = self.local_store.get_recent(limit)
        return [
            {
                "trace_id": t.trace_id,
                "name": t.name,
                "status": t.status,
                "start_time": t.start_time,
                "num_spans": len(t.spans),
                "scores": t.scores,
            }
            for t in traces
        ]

    def flush(self):
        """Flush any pending data to LangFuse."""
        if self.use_langfuse and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.warning(f"LangFuse flush failed: {e}")



_observability = InvoiceAuditorObservability()


def get_observability() -> InvoiceAuditorObservability:
    """Get the global observability instance."""
    return _observability
