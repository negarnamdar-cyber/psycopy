"""Validation logic for config and stimuli."""

from __future__ import annotations

from typing import Iterable, Mapping

from psycopy.types import Stimulus


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: object) -> None:
    participant_id = str(getattr(config, "participant_id")).strip()
    session_id = str(getattr(config, "session_id")).strip()
    sample_rate = int(getattr(config, "sample_rate"))
    random_seed = str(getattr(config, "random_seed")).strip()

    _ensure(participant_id != "", "participant_id must not be empty.")
    _ensure(session_id != "", "session_id must not be empty.")
    _ensure(sample_rate >= 8000, "sample_rate must be >= 8000.")
    if random_seed:
        int(random_seed)


def validate_stimuli(rows: Iterable[Mapping[str, str]]) -> list[Stimulus]:
    parsed: list[Stimulus] = []
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        trial_id = str(row.get("trial_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not trial_id:
            raise ValueError(f"Stimulus row {idx} has empty trial_id.")
        if trial_id in seen_ids:
            raise ValueError(f"Duplicate trial_id in stimuli: {trial_id}")
        if not text:
            raise ValueError(f"Stimulus row {idx} has empty text.")
        seen_ids.add(trial_id)
        parsed.append(Stimulus(trial_id=trial_id, text=text))
    if not parsed:
        raise ValueError("Stimuli file is empty.")
    return parsed
