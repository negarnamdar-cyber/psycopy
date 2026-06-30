"""Tests for the short practice demo mode (no Medoc, no audio)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock heavy dependencies before imports (mirrors test_e2e_medoc.py).
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("psychopy", MagicMock())
sys.modules.setdefault("psychopy.core", MagicMock())
sys.modules.setdefault("psychopy.event", MagicMock())
sys.modules.setdefault("psychopy.visual", MagicMock())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psycopy.config import ExperimentConfig, ExperimentMode
from psycopy.practice_demo import (
    DEMO_QUESTIONS,
    VOWEL_GO_SEGMENTS,
    PracticeDemo,
)
from psycopy.run_experiment import build_config
from psycopy.types import TaskState


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        practice_demo=False,
        practice_no_medoc=False,
        practice_with_medoc=False,
        speech=False,
        normal=False,
        participant_id="001",
        session_id="01",
        age="",
        sex="",
        ethnicity="",
        first_language="",
        random_seed="",
        fullscreen="false",
        vad_enabled="true",
        medoc_ip="10.196.94.38",
        medoc_port=20121,
        medoc_timeout=5.0,
        questions_file=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_demo() -> tuple[PracticeDemo, MagicMock]:
    """Construct a PracticeDemo with a mocked PsychoPyUI (no real window)."""
    config = ExperimentConfig(
        participant_id="DEMO",
        session_id="01",
        fullscreen=False,
        mode=ExperimentMode.PRACTICE,
    )
    mock_ui = MagicMock()

    def spy_apply(state: TaskState) -> None:
        mock_ui._states.append(state)

    mock_ui._states: list[TaskState] = []
    mock_ui.apply_state = spy_apply
    mock_ui.show_rate_pain_prompt = MagicMock()
    mock_ui.close = MagicMock()

    with patch("psycopy.practice_demo.PsychoPyUI", return_value=mock_ui):
        demo = PracticeDemo(config)
    return demo, mock_ui


def test_build_config_practice_demo_mode():
    args = _make_args(practice_demo=True)
    config = build_config(args)
    assert config.mode == ExperimentMode.PRACTICE
    assert config.medoc_config is None


def test_build_config_practice_demo_disables_medoc():
    """--practice-demo must never construct a MedocConfig even with medoc args."""
    args = _make_args(practice_demo=True, medoc_ip="10.196.94.38", medoc_port=20121)
    config = build_config(args)
    assert config.medoc_config is None


def test_practice_demo_runs_without_audio_or_medoc():
    demo, mock_ui = _build_demo()

    with (
        patch("psycopy.audio.AudioService") as mock_audio,
        patch("psycopy.medoc.MedocClient") as mock_medoc,
    ):
        demo.run()
        mock_audio.assert_not_called()
        mock_medoc.assert_not_called()

    # Vowel: STOP,GO,STOP,GO,STOP (VOWEL_GO_SEGMENTS GO periods)
    # Speech: per question STOP,GO then a final STOP -> STOP,GO,STOP,GO,STOP
    states = mock_ui._states
    expected = 2 * (VOWEL_GO_SEGMENTS * 2 + 1)
    assert len(states) == expected, f"Expected {expected} state displays, got {len(states)}"
    assert states.count(TaskState.STOP) == states.count(TaskState.GO) + 2
    assert mock_ui.show_rate_pain_prompt.call_count == len(DEMO_QUESTIONS)
    mock_ui.close.assert_called_once()


def test_practice_demo_no_output_directory_created(tmp_path, monkeypatch):
    """PracticeDemo must not create a data output directory or audio dir."""
    monkeypatch.chdir(tmp_path)
    demo, _ = _build_demo()
    demo.run()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "audio").exists()


def test_practice_demo_vowel_state_sequence():
    """The vowel demo alternates STOP/GO starting and ending with STOP."""
    demo, mock_ui = _build_demo()
    # Run only the vowel demo.
    demo._run_vowel_demo()
    expected = [TaskState.STOP, TaskState.GO] * VOWEL_GO_SEGMENTS + [TaskState.STOP]
    assert mock_ui._states == expected
