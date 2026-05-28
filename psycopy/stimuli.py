"""Stimuli loading helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from psycopy.types import Stimulus
from psycopy.validation import validate_stimuli


def get_stimuli_path() -> Path:
    candidates = [
        Path("stimuli.csv"),
        Path(__file__).parent / "stimuli.csv",
        Path(__file__).parent.parent / "stimuli.csv",
        Path(__file__).parent.parent / "psycopy" / "stimuli.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_stimuli(stimuli_file: Path) -> list[Stimulus]:
    rows: list[dict[str, str]] = []
    with open(stimuli_file, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"trial_id", "text"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("stimuli.csv must contain trial_id and text columns.")
        for row in reader:
            rows.append({"trial_id": row.get("trial_id", ""), "text": row.get("text", "")})
    return validate_stimuli(rows)

