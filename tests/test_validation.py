from dataclasses import replace

import pytest

from psycopy.config import ExperimentConfig
from psycopy.validation import validate_config, validate_stimuli


def test_validate_config_accepts_defaults() -> None:
    validate_config(ExperimentConfig())


def test_validate_config_rejects_bad_duration() -> None:
    config = replace(ExperimentConfig(), trial_duration_sec=1.0, num_switches=10)
    with pytest.raises(ValueError):
        validate_config(config)


def test_validate_config_allows_unbounded_duration_when_segmentation_disabled() -> None:
    config = replace(
        ExperimentConfig(),
        trial_duration_sec=1.0,
        num_switches=10,
        go_segmentation_enabled=False,
    )
    validate_config(config)


def test_validate_stimuli_rejects_duplicates() -> None:
    rows = [
        {"trial_id": "1", "text": "hello"},
        {"trial_id": "1", "text": "world"},
    ]
    with pytest.raises(ValueError):
        validate_stimuli(rows)


def test_validate_stimuli_parses_records() -> None:
    rows = [{"trial_id": "1", "text": "hello"}]
    result = validate_stimuli(rows)
    assert result[0].trial_id == "1"
    assert result[0].text == "hello"


def test_validate_config_practice_trial_cap() -> None:
    config = replace(ExperimentConfig(), practice_mode=True, practice_trials_per_block=3)
    with pytest.raises(ValueError):
        validate_config(config)
