"""Atomic persistence primitives with error handling."""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger("psycopy.storage")


class StorageError(RuntimeError):
    """Raised when storage operations fail."""

    pass


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically with proper error handling and cleanup.

    Uses temp file + os.replace pattern for atomic writes. Cleans up temp files
    on failure to prevent orphaned files.
    """
    tmp_path: Path | None = None

    try:
        # Ensure parent directory exists
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise StorageError(f"Cannot create directory {path.parent}: {e}") from e

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            encoding="utf-8",
            dir=path.parent,
            newline="",
        ) as handle:
            handle.write(content)
            tmp_path = Path(handle.name)

        # Atomic replace
        os.replace(tmp_path, path)
        tmp_path = None  # Clear after successful replace

    except Exception as e:
        # Clean up temp file on any error
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass  # Best effort cleanup

        if isinstance(e, StorageError):
            raise
        raise StorageError(f"Failed to write {path}: {e}") from e


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON data atomically with error handling."""
    try:
        content = json.dumps(data, indent=2)
    except TypeError as e:
        raise StorageError(f"Cannot serialize data to JSON: {e}") from e
    _atomic_write(path, content)


def atomic_write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        _atomic_write(path, "")
        return

    fieldnames: set[str] = set()
    for row in rows_list:
        fieldnames.update(row.keys())
    ordered = sorted(fieldnames)

    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL)
    writer.writerow(ordered)
    for row in rows_list:
        writer.writerow([row.get(k, "") for k in ordered])

    _atomic_write(path, buffer.getvalue())
