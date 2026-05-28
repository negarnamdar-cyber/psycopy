"""End-to-end validation tests for experiment runtime behavior.

Tests verify:
- Spacebar termination: end_reason="completed"
- ESC abort: end_reason="interrupted"
- STOP segment tracking: stop_cue_index increments
- VAD speech_end linked to stop_cue_index
- trial_instance_id in all output files
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from psycopy.models import (
    EndReason,
    EventType,
    TrialRecord,
    TrialStatus,
    generate_trial_instance_id,
)
from psycopy.config import ExperimentConfig
from psycopy.session import (
    EventLogger,
    SessionPaths,
    TrialSegmentsLogger,
    VADLogger,
)
from psycopy.stimuli import Stimulus


# ==============================================================================
# Mock Factory Classes
# ==============================================================================


class MockPsychoPyClock:
    """Mock clock with incrementing time."""

    def __init__(self, start_time: float = 0.0):
        self._time = start_time
        self._increment = 0.016

    def getTime(self):
        current = self._time
        self._time += self._increment
        return current

    def reset(self):
        self._time = 0.0


class MockAudioService:
    """Mock AudioService that avoids importing sounddevice."""

    def __init__(self, vad_events: list[dict] = None):
        self._recording = False
        self._recording_path = None
        self._vad_events = vad_events or []
        self._vad_enabled = False
        self._stop_cue_time = None
        self._preflight_called = False

    def preflight(self):
        self._preflight_called = True

    def start(self, path):
        self._recording = True
        self._recording_path = path

    def stop(self):
        self._recording = False

    def abort(self):
        self._recording = False

    def enable_vad(self, config):
        self._vad_enabled = config.vad_enabled

    def start_vad_monitoring(self):
        pass

    def stop_vad_monitoring(self):
        return self._vad_events.copy()

    def set_stop_cue_time(self):
        self._stop_cue_time = 0.5
        return 0.5

    def get_speech_cessation_latency(self):
        return None

    @property
    def is_recording(self):
        return self._recording


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_config():
    """Minimal config for testing."""
    return ExperimentConfig(
        participant_id="TEST001",
        session_id="01",
        block_order="baseline_then_pain",
        trial_duration_sec=12.0,
        num_switches=6,
        start_state="GO",
        min_segment_sec=0.8,
        max_segment_sec=2.0,
        go_segmentation_enabled=True,
        random_seed="42",
        fullscreen=False,
        practice_mode=True,
        practice_trials_per_block=1,
        vowel_trials_per_block=1,
        rt_trials_per_block=1,
        vad_enabled=False,
    )


@pytest.fixture
def vad_config(minimal_config):
    """Config with VAD enabled."""
    return replace(minimal_config, vad_enabled=True)


@pytest.fixture
def session_paths(temp_output_dir):
    """Create session paths for temp dir."""
    (temp_output_dir / "audio").mkdir(exist_ok=True)
    (temp_output_dir / "audio_16k").mkdir(exist_ok=True)
    return SessionPaths(
        output_dir=temp_output_dir,
        audio_dir=temp_output_dir / "audio",
        audio_16k_dir=temp_output_dir / "audio_16k",
        events_file=temp_output_dir / "events.csv",
        trials_file=temp_output_dir / "trials.csv",
        rt_file=temp_output_dir / "rt_trials.csv",
        config_file=temp_output_dir / "config.json",
        features_file=temp_output_dir / "features.csv",
        features_manifest_file=temp_output_dir / "features_manifest.json",
        vad_file=temp_output_dir / "vad_events.csv",
        blocks_file=temp_output_dir / "blocks.csv",
        segments_file=temp_output_dir / "segments.csv",
    )


# ==============================================================================
# Fixtures
# ==============================================================================


def create_mock_ui(abort_on_segment: bool = False, complete_all: bool = True):
    """Create mock UI for testing.

    Args:
        abort_on_segment: If True, raises UserAbort on first segment
        complete_all: If True, all segments complete. If False, first terminates.
    """
    from psycopy.runtime import UserAbort

    mock_ui = MagicMock()
    mock_ui.exp_clock = MockPsychoPyClock()
    mock_ui.show_fixation = MagicMock()
    mock_ui.show_instructions = MagicMock()
    mock_ui.show_pain_warning = MagicMock()
    mock_ui.show_vowel_instructions = MagicMock()
    mock_ui.show_rt_instructions = MagicMock()
    mock_ui.show_completion = MagicMock()
    mock_ui.run_rt_trial = MagicMock(return_value=(100.0, False))
    mock_ui.close = MagicMock()
    mock_ui.win = MagicMock()

    segment_call_count = [0]

    if abort_on_segment:

        def mock_run_segment_abort(sentence, state, duration):
            segment_call_count[0] += 1
            raise UserAbort()

        mock_ui.run_segment_with_termination = mock_run_segment_abort
    elif not complete_all:

        def mock_run_segment_terminate(sentence, state, duration):
            segment_call_count[0] += 1
            if segment_call_count[0] == 1:
                return (False, 100.0)
            return (True, 200.0)

        mock_ui.run_segment_with_termination = mock_run_segment_terminate
    else:

        def mock_run_segment_complete(sentence, state, duration):
            segment_call_count[0] += 1
            return (True, 200.0)

        mock_ui.run_segment_with_termination = mock_run_segment_complete

    return mock_ui


# ==============================================================================
# Module setup fixture for sounddevice mock
# ==============================================================================


@pytest.fixture(autouse=True)
def mock_sounddevice():
    """Mock sounddevice and scipy.io imports for all tests in this module."""
    old_sounddevice = sys.modules.get("sounddevice")
    old_scipy = sys.modules.get("scipy.io")
    sys.modules["sounddevice"] = MagicMock()
    sys.modules["scipy.io"] = MagicMock()
    yield
    if old_sounddevice is not None:
        sys.modules["sounddevice"] = old_sounddevice
    else:
        sys.modules.pop("sounddevice", None)
    if old_scipy is not None:
        sys.modules["scipy.io"] = old_scipy
    else:
        sys.modules.pop("scipy.io", None)


# ==============================================================================
# Test: End Reason Model
# ==============================================================================


class TestEndReasonModel:
    """Unit tests for EndReason and TrialRecord."""

    def test_end_reason_values(self):
        assert EndReason.COMPLETED.value == "completed"
        assert EndReason.INTERRUPTED.value == "interrupted"

    def test_trial_record_defaults(self):
        record = TrialRecord(
            participant_id="P001",
            session_id="S01",
            timestamp="2024-01-01T00:00:00",
            block="baseline",
            block_index=1,
            trial_number=1,
            trial_id="test_001",
            sentence_text="Hello",
            trial_duration_sec=10.0,
            actual_duration_sec=5.0,
            audio_filename="test.wav",
            num_switches=3,
            start_state="GO",
        )
        assert record.end_reason == "completed"
        assert record.num_stop_cues == 0

    def test_trial_record_with_stop_cues(self):
        record = TrialRecord(
            participant_id="P001",
            session_id="S01",
            timestamp="2024-01-01T00:00:00",
            block="baseline",
            block_index=1,
            trial_number=1,
            trial_id="test_001",
            sentence_text="Hello",
            trial_duration_sec=10.0,
            actual_duration_sec=5.0,
            audio_filename="test.wav",
            num_switches=3,
            start_state="GO",
            num_stop_cues=5,
        )
        assert record.num_stop_cues == 5

    def test_trial_instance_id_generation(self):
        trial_id = generate_trial_instance_id("P001", "S01", "baseline", 1)
        assert trial_id == "P001_S01_baseline_001"

        trial_id_2 = generate_trial_instance_id("P001", "S01", "pain", 10)
        assert trial_id_2 == "P001_S01_pain_010"


# ==============================================================================
# Test: VAD Events and stop_cue_index Linkage
# ==============================================================================


class TestVADSpeechEndLinkage:
    """Tests for VAD speech_end events linked to stop_cue_index."""

    def test_vad_speech_end_includes_stop_cue_index(self, temp_output_dir):
        vad_file = temp_output_dir / "vad_events.csv"
        vad_logger = VADLogger(vad_file)
        vad_logger.set_start_time(0.0)
        vad_logger.set_context("P001_S01_baseline_001", "baseline", 1)

        vad_logger.log_event(event_type="stop_cue", timestamp=0.5, stop_cue_index=0)
        vad_logger.log_event(event_type="stop_cue", timestamp=1.5, stop_cue_index=1)
        vad_logger.log_event(
            event_type="speech_end",
            timestamp=0.8,
            is_speech=False,
            latency_ms=300.0,
            stop_cue_index=0,
        )
        vad_logger.save(force=True)

        content = vad_file.read_text()
        lines = content.strip().split("\n")
        header = lines[0]
        assert "stop_cue_index" in header

        for line in lines[1:]:
            parts = line.split(",")
            if "speech_end" in parts:
                assert any(p in ["0", "1"] for p in parts)

    def test_vad_logger_tracks_multiple_stop_cues(self, temp_output_dir):
        vad_file = temp_output_dir / "vad_events.csv"
        vad_logger = VADLogger(vad_file)
        vad_logger.set_start_time(0.0)
        vad_logger.set_context("P001_S01_baseline_001", "baseline", 1)

        vad_logger.log_event(event_type="speech_start", timestamp=0.0)
        vad_logger.log_event(event_type="stop_cue", timestamp=0.5, stop_cue_index=0)
        vad_logger.log_event(
            event_type="speech_end",
            timestamp=0.8,
            latency_ms=300.0,
            stop_cue_index=0,
        )
        vad_logger.log_event(event_type="stop_cue", timestamp=1.0, stop_cue_index=1)
        vad_logger.log_event(
            event_type="speech_end",
            timestamp=1.3,
            latency_ms=300.0,
            stop_cue_index=1,
        )
        vad_logger.save(force=True)

        events = vad_logger.events
        assert len(events) == 5

        stop_cue_events = [e for e in events if e["event_type"] == "stop_cue"]
        assert len(stop_cue_events) == 2

        speech_end_events = [e for e in events if e["event_type"] == "speech_end"]
        assert len(speech_end_events) == 2
        for evt in speech_end_events:
            assert evt["stop_cue_index"] in [0, 1]


# ==============================================================================
# Test: trial_instance_id Propagation
# ==============================================================================


class TestTrialInstanceIdPropagation:
    """Tests for trial_instance_id in all outputs."""

    def test_trial_instance_id_in_vad_events(self, temp_output_dir):
        vad_file = temp_output_dir / "vad_events.csv"
        vad_logger = VADLogger(vad_file)
        vad_logger.set_start_time(0.0)
        vad_logger.set_context("P001_S01_baseline_001", "baseline", 1)
        vad_logger.log_event(event_type="speech_start", timestamp=0.0)
        vad_logger.log_event(event_type="speech_end", timestamp=0.5)
        vad_logger.save(force=True)

        content = vad_file.read_text()
        assert "trial_instance_id" in content.lower()
        assert "P001_S01_baseline_001" in content

    def test_trial_instance_id_in_segments(self, temp_output_dir):
        segments_file = temp_output_dir / "trial_segments.csv"
        segments_logger = TrialSegmentsLogger(segments_file)

        segments_logger.log_segment(
            trial_instance_id="P001_S01_baseline_001",
            segment_index=0,
            state="GO",
            start_time=0.0,
            end_time=0.5,
            duration=0.5,
        )
        segments_logger.log_segment(
            trial_instance_id="P001_S01_baseline_001",
            segment_index=1,
            state="STOP",
            start_time=0.5,
            end_time=1.0,
            duration=0.5,
        )
        segments_logger.save(force=True)

        content = segments_file.read_text()
        assert "trial_instance_id" in content.lower()
        assert "P001_S01_baseline_001" in content


# ==============================================================================
# Test: Event Logger Integration
# ==============================================================================


class TestEventLogger:
    """Tests for EventLogger trial_instance_id handling."""

    def test_event_logger_includes_trial_instance_id(self, temp_output_dir):
        events_file = temp_output_dir / "events.csv"
        logger = EventLogger(events_file)
        logger.set_start_time()
        trial_id = "P001_S01_baseline_001"

        logger.log(
            event_type=EventType.TRIAL_START,
            trial_instance_id=trial_id,
            block="baseline",
            event_data={"trial_id": "test_001"},
        )
        logger.log(
            event_type=EventType.STATE_CHANGE,
            trial_instance_id=trial_id,
            block="baseline",
            event_data={"segment_index": 0, "state": "GO"},
        )
        logger.log(
            event_type=EventType.STOP_CUE_APPEAR,
            trial_instance_id=trial_id,
            block="baseline",
            event_data={"stop_cue_index": 0},
        )
        logger.log(
            event_type=EventType.TRIAL_END,
            trial_instance_id=trial_id,
            block="baseline",
            event_data={"actual_duration": 5.0, "end_reason": "completed"},
        )
        logger.save(force=True)

        content = events_file.read_text()
        lines = content.strip().split("\n")
        for line in lines[1:]:
            assert trial_id in line

    def test_event_logger_header(self, temp_output_dir):
        events_file = temp_output_dir / "events.csv"
        logger = EventLogger(events_file)
        logger.set_start_time()
        logger.log(
            event_type=EventType.TRIAL_START,
            trial_instance_id="test_001",
            block="baseline",
        )
        logger.save(force=True)

        content = events_file.read_text()
        header = content.split("\n")[0]
        assert "event_type" in header
        assert "timestamp" in header
        assert "trial_instance_id" in header
        assert "block" in header


# ==============================================================================
# Test: Experiment Trial Logic (Integration with Mocks)
# ==============================================================================


class TestExperimentTrialLogic:
    """Integration tests using full Experiment class with mocked dependencies."""

    def test_spacebar_termination_sets_completed(self, minimal_config, session_paths):
        sys.modules["sounddevice"] = MagicMock()
        sys.modules["scipy.io"] = MagicMock()

        import psycopy.runtime
        import psycopy.audio

        def terminate_fn(sentence, state, duration):
            return (False, 100.0)

        mock_ui = MagicMock()
        mock_ui.exp_clock = MockPsychoPyClock()
        mock_ui.show_fixation = MagicMock()
        mock_ui.show_instructions = MagicMock()
        mock_ui.show_pain_warning = MagicMock()
        mock_ui.show_vowel_instructions = MagicMock()
        mock_ui.show_rt_instructions = MagicMock()
        mock_ui.show_completion = MagicMock()
        mock_ui.run_rt_trial = MagicMock(return_value=(100.0, False))
        mock_ui.close = MagicMock()
        mock_ui.win = MagicMock()
        mock_ui.run_segment_with_termination = terminate_fn

        with (
            patch.object(psycopy.runtime, "PsychoPyUI") as MockUI,
            patch.object(psycopy.audio, "AudioService") as MockAudio,
            patch("psycopy.experiment.create_output_directory") as mock_create,
            patch("psycopy.experiment.FeatureExtractor") as MockFeatureExtractor,
            patch("psycopy.experiment.configure_logging") as mock_log,
            patch("psycopy.experiment.get_run_metadata") as mock_meta,
            patch("psycopy.experiment.save_config_snapshot"),
            patch("psycopy.experiment.load_stimuli") as mock_stimuli,
        ):
            mock_create.return_value = session_paths
            mock_log.return_value = MagicMock()
            mock_meta.return_value = {"version": "0.3.0"}
            mock_stimuli.return_value = [Stimulus(trial_id="test", text="Test")]
            MockUI.return_value = mock_ui

            mock_audio = MockAudioService()
            MockAudio.return_value = mock_audio

            from psycopy.experiment import Experiment

            experiment = Experiment(minimal_config)

            with patch("psycopy.experiment.generate_schedule") as mock_schedule:
                mock_schedule.return_value = [
                    {"state": "GO", "duration": 1.0},
                    {"state": "STOP", "duration": 1.0},
                ]

                trial_record = experiment.run_trial(
                    trial_idx=1,
                    stimulus=Stimulus(trial_id="test_001", text="Test"),
                    block_name="baseline",
                    block_index=1,
                )

                assert trial_record.end_reason == EndReason.COMPLETED.value

    def test_esc_abort_sets_interrupted(self, minimal_config, session_paths):
        sys.modules["sounddevice"] = MagicMock()
        sys.modules["scipy.io"] = MagicMock()

        # Force fresh imports
        for mod in list(sys.modules.keys()):
            if "psycopy" in mod:
                del sys.modules[mod]

        sys.modules["sounddevice"] = MagicMock()
        sys.modules["scipy.io"] = MagicMock()

        import psycopy.runtime
        import psycopy.audio
        from psycopy.runtime import UserAbort

        def abort_fn(sentence, state, duration):
            raise UserAbort()

        mock_ui = MagicMock()
        mock_ui.exp_clock = MockPsychoPyClock()
        mock_ui.show_fixation = MagicMock()
        mock_ui.show_instructions = MagicMock()
        mock_ui.show_pain_warning = MagicMock()
        mock_ui.show_vowel_instructions = MagicMock()
        mock_ui.show_rt_instructions = MagicMock()
        mock_ui.show_completion = MagicMock()
        mock_ui.run_rt_trial = MagicMock(return_value=(100.0, False))
        mock_ui.close = MagicMock()
        mock_ui.win = MagicMock()
        mock_ui.run_segment_with_termination = abort_fn

        with (
            patch.object(psycopy.runtime, "PsychoPyUI") as MockUI,
            patch.object(psycopy.audio, "AudioService") as MockAudio,
            patch("psycopy.experiment.create_output_directory") as mock_create,
            patch("psycopy.experiment.FeatureExtractor") as MockFeatureExtractor,
            patch("psycopy.experiment.configure_logging") as mock_log,
            patch("psycopy.experiment.get_run_metadata") as mock_meta,
            patch("psycopy.experiment.save_config_snapshot"),
            patch("psycopy.experiment.load_stimuli") as mock_stimuli,
        ):
            mock_create.return_value = session_paths
            mock_log.return_value = MagicMock()
            mock_meta.return_value = {"version": "0.3.0"}
            mock_stimuli.return_value = [Stimulus(trial_id="test", text="Test")]
            MockUI.return_value = mock_ui

            mock_audio = MockAudioService()
            MockAudio.return_value = mock_audio

            from psycopy.experiment import Experiment

            experiment = Experiment(minimal_config)

            with patch("psycopy.experiment.generate_schedule") as mock_schedule:
                mock_schedule.return_value = [
                    {"state": "GO", "duration": 1.0},
                    {"state": "STOP", "duration": 1.0},
                ]

                trial_record = experiment.run_trial(
                    trial_idx=1,
                    stimulus=Stimulus(trial_id="test_001", text="Test"),
                    block_name="baseline",
                    block_index=1,
                )

                assert trial_record.end_reason == EndReason.INTERRUPTED.value
                assert trial_record.status == TrialStatus.INTERRUPTED

    def test_stop_cue_index_increments(self, minimal_config, session_paths):
        sys.modules["sounddevice"] = MagicMock()
        sys.modules["scipy.io"] = MagicMock()

        # Force fresh imports
        for mod in list(sys.modules.keys()):
            if "psycopy" in mod:
                del sys.modules[mod]

        sys.modules["sounddevice"] = MagicMock()
        sys.modules["scipy.io"] = MagicMock()

        import psycopy.runtime
        import psycopy.audio

        call_count = [0]

        def complete_fn(sentence, state, duration):
            call_count[0] += 1
            return (True, 200.0)

        mock_ui = MagicMock()
        mock_ui.exp_clock = MockPsychoPyClock()
        mock_ui.show_fixation = MagicMock()
        mock_ui.show_instructions = MagicMock()
        mock_ui.show_pain_warning = MagicMock()
        mock_ui.show_vowel_instructions = MagicMock()
        mock_ui.show_rt_instructions = MagicMock()
        mock_ui.show_completion = MagicMock()
        mock_ui.run_rt_trial = MagicMock(return_value=(100.0, False))
        mock_ui.close = MagicMock()
        mock_ui.win = MagicMock()
        mock_ui.run_segment_with_termination = complete_fn

        with (
            patch.object(psycopy.runtime, "PsychoPyUI") as MockUI,
            patch.object(psycopy.audio, "AudioService") as MockAudio,
            patch("psycopy.experiment.create_output_directory") as mock_create,
            patch("psycopy.experiment.FeatureExtractor") as MockFeatureExtractor,
            patch("psycopy.experiment.configure_logging") as mock_log,
            patch("psycopy.experiment.get_run_metadata") as mock_meta,
            patch("psycopy.experiment.save_config_snapshot"),
            patch("psycopy.experiment.load_stimuli") as mock_stimuli,
        ):
            mock_create.return_value = session_paths
            mock_log.return_value = MagicMock()
            mock_meta.return_value = {"version": "0.3.0"}
            mock_stimuli.return_value = [Stimulus(trial_id="test", text="Test")]
            MockUI.return_value = mock_ui

            mock_audio = MockAudioService()
            MockAudio.return_value = mock_audio

            from psycopy.experiment import Experiment

            experiment = Experiment(minimal_config)

            with patch("psycopy.experiment.generate_schedule") as mock_schedule:
                mock_schedule.return_value = [
                    {"state": "GO", "duration": 1.0},
                    {"state": "STOP", "duration": 1.0},
                    {"state": "GO", "duration": 1.0},
                    {"state": "STOP", "duration": 1.0},
                ]

                trial_record = experiment.run_trial(
                    trial_idx=1,
                    stimulus=Stimulus(trial_id="test_001", text="Test"),
                    block_name="baseline",
                    block_index=1,
                )

                assert trial_record.num_stop_cues == 2
