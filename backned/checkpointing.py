"""Shared PostgreSQL checkpoint context for the invoice LangGraph stages."""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_saver: ContextVar[Any | None] = ContextVar("invoice_checkpoint_saver", default=None)
_thread_id: ContextVar[str | None] = ContextVar("invoice_checkpoint_thread", default=None)


def set_checkpoint_context(saver: Any, thread_id: str):
    return _saver.set(saver), _thread_id.set(thread_id)


def reset_checkpoint_context(tokens) -> None:
    saver_token, thread_token = tokens
    _saver.reset(saver_token)
    _thread_id.reset(thread_token)


def compile_graph(workflow):
    return workflow.compile(checkpointer=_saver.get())


def invoke_config() -> dict:
    thread_id = _thread_id.get()
    return {"configurable": {"thread_id": thread_id}} if thread_id else {}
