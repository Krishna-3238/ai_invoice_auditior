"""No external tracing: logs go to container stdout/stderr as specified."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps


def observe(*args, **kwargs):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*a, **kw):
            return fn(*a, **kw)
        return wrapped
    return decorate


class _Context:
    def __getattr__(self, _):
        return lambda **__: None


langfuse_context = _Context()


class CallbackHandler:
    def __init__(self, **kwargs):
        pass
    def flush(self):
        pass


def get_callback_handler(**kwargs):
    return CallbackHandler(**kwargs)


def flush_langfuse():
    return None


def get_langfuse_client():
    return None
