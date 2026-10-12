"""Centralised logging configuration — reads settings from rules.yaml."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # → Invoice_Auditor/
_RULES_PATH = Path(__file__).resolve().parent / "rules.yaml"

_LOG_FORMAT = "[%(asctime)s][%(levelname)s][%(module)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure the root logger from rules.yaml.

    Format : [timestamp][level][module] message
    Output: stdout/stderr so the container runtime can collect service logs.
    """
    try:
        with open(_RULES_PATH, encoding="utf-8") as fh:
            rules = yaml.safe_load(fh) or {}
    except Exception:
        rules = {}

    log_cfg = rules.get("logging", {})
    level = getattr(logging, log_cfg.get("log_level", "INFO").upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)

