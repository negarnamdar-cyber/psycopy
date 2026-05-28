"""Structured runtime logging and metadata."""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_run_metadata(app_version: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "app_version": app_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def configure_logging(output_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("psycopy")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

