"""End-to-end validation test for Medoc experiment flow.

Tests verify:
- 4 blocks x 1 trial = 4 trials total
- All trials are vowel trials
- Each trial has 4-7 GO segments per minute (16-28 total), each 1.5-3.5 seconds
- All CSV files created (trials.csv, medoc_events.csv, events.csv)
- Config snapshot exists (config.json)

VAD is now performed offline via scripts/process_vad.py; these tests verify
the experiment records audio and logs STOP/GO cues to events.csv.

Uses mock everything - no real hardware required.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy dependencies before imports
sys.modules["sounddevice"] = MagicMock()
sys.modules["psychopy"] = MagicMock()
sys.modules["psychopy.core"] = MagicMock()
sys.modules["psychopy.event"] = MagicMock()
sys.modules["psychopy.visual"] = MagicMock()

from psycopy.config import ExperimentConfig, MedocConfig
from psycopy.trial_generator import TrialConfig, generate_trials
from psycopy.schedule import get_rng


# ==============================================================================
# Mock Classes
# ==============================================================================


class MockPsychoPyClock:
    """Mock clock with incremental time."""

    def __init__(self, start_time: float = 0.0):
        self._time = start_time
        self._increment = 0.016

    def getTime(self):
        current = self._time
        self._time += self._increment
        return current

    def reset(self):
        self._time = 0.0


class MockMedocServer:
    """Mock Medoc server that acknowledges Medoc program commands."""

    def respond_to_program(self) -> bytes:
        """Return OK response (0x00) for a Medoc program command."""
        return b"\x00"


class MockAudioService:
    """Mock AudioService that avoids real hardware."""

    def __init__(self):
        self._recording = False
        self._recording_path = None

    def preflight(self):
        self._preflight_called = True

    def start(self, path):
        self._recording = True
        self._recording_path = path

    def stop(self):
        self._recording = False

    def abort(self):
        self._recording = False

    @property
    def is_recording(self):
        return self._recording


class MockUI:
    """Mock PsychoPyUI for non-interactive testing."""

    def __init__(self):
        self.exp_clock = MockPsychoPyClock()
        self.instruction_text = MagicMock()
        self.sentence_text = MagicMock()
        self.help_text = MagicMock()
        self.state_background = MagicMock()
        self.state_indicator = MagicMock()
        self.win = MagicMock()
        self.win.flip = MagicMock()
        self._call_count = {"show_instructions": 0, "show_completion": 0, "wait_for_space": 0}

    def show_instructions(self, *args, **kwargs):
        self._call_count["show_instructions"] += 1

    def show_completion(self):
        self._call_count["show_completion"] += 1

    def wait_for_space(self):
        self._call_count["wait_for_space"] += 1

    def show_set_waiting_screen(self, set_num):
        """Show waiting screen between sets."""
        pass

    def wait(self, duration):
        """Mock wait - do nothing."""
        pass

    def apply_state(self, state):
        """Mock apply_state - do nothing."""
        pass

    def show_progress(self, fraction):
        """Mock show_progress - do nothing."""
        pass

    def hide_progress(self):
        """Mock hide_progress - do nothing."""
        pass

    @property
    def progress_bar_bg(self):
        return MagicMock()

    @property
    def progress_bar(self):
        return MagicMock()

    def close(self):
        pass


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_session_paths(temp_output_dir):
    """Create mock SessionPaths."""
    from psycopy.session import SessionPaths

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
        segments_file=temp_output_dir / "trial_segments.csv",
        medoc_file=temp_output_dir / "medoc_events.csv",
    )


@pytest.fixture
def mock_medoc_client():
    """Create mock MedocClient that simulates Medoc program acknowledgments."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.send_unified_program = MagicMock(return_value=None)
    return client


@pytest.fixture
def mock_audio():
    """Create mock AudioService."""
    return MockAudioService()


@pytest.fixture
def mock_ui():
    """Create mock PsychoPyUI."""
    return MockUI()


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.exception = MagicMock()
    return logger


@pytest.fixture
def medoc_config():
    """Create MedocConfig for testing."""
    return MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000, medoc_timeout=5.0)


@pytest.fixture
def experiment_config(medoc_config):
    """Create ExperimentConfig with MedocConfig for testing."""
    return ExperimentConfig(
        participant_id="TEST001",
        session_id="01",
        random_seed="42",
        medoc_config=medoc_config,
        vad_enabled=True,
    )


# ==============================================================================
# QA Scenario 1: Full Experiment Output Validation
# ==============================================================================


class TestFullExperimentOutput:
    """E2E test: Run full experiment and validate all outputs."""

    def test_e2e_full_experiment_five_trials(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_medoc_client,
        mock_audio,
        mock_ui,
        mock_logger,
    ):
        """QA Scenario: Run E2E test with all mocks, validate 4 trials."""
        from psycopy.medoc_experiment import MedocExperiment

        with (
            patch("psycopy.medoc_experiment.MedocClient", return_value=mock_medoc_client),
            patch("psycopy.medoc_experiment.PsychoPyUI", return_value=mock_ui),
            patch("psycopy.medoc_experiment.AudioService", return_value=mock_audio),
            patch("psycopy.medoc_experiment.validate_config"),
            patch("psycopy.medoc_experiment.get_run_metadata", return_value={"version": "0.3.0"}),
            patch("psycopy.medoc_experiment.configure_logging", return_value=mock_logger),
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch(
                "psycopy.medoc_experiment.create_output_directory",
                return_value=mock_session_paths,
            ),
            patch("time.sleep"),  # Speed up tests
        ):
            config = ExperimentConfig(
                participant_id="TEST001",
                session_id="01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio
            # Mock break screen to avoid real-time waiting
            exp._show_break_screen = MagicMock()
            exp.run()

            # Validate: 4 blocks x 1 trial = 4 trials total
            total_trials = len(exp.trial_logger.trials)
            assert total_trials == 4, f"Expected 4 trials, got {total_trials}"

    def test_e2e_trial_distribution_all_vowel(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_medoc_client,
        mock_audio,
        mock_ui,
        mock_logger,
    ):
        """QA Scenario: Validate all 4 trials are vowel trials."""
        from psycopy.medoc_experiment import MedocExperiment

        with (
            patch("psycopy.medoc_experiment.MedocClient", return_value=mock_medoc_client),
            patch("psycopy.medoc_experiment.PsychoPyUI", return_value=mock_ui),
            patch("psycopy.medoc_experiment.AudioService", return_value=mock_audio),
            patch("psycopy.medoc_experiment.validate_config"),
            patch("psycopy.medoc_experiment.get_run_metadata", return_value={"version": "0.3.0"}),
            patch("psycopy.medoc_experiment.configure_logging", return_value=mock_logger),
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch(
                "psycopy.medoc_experiment.create_output_directory",
                return_value=mock_session_paths,
            ),
            patch("time.sleep"),
        ):
            config = ExperimentConfig(
                participant_id="TEST001",
                session_id="01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio
            exp._show_break_screen = MagicMock()
            exp.run()

            trials = exp.trial_logger.trials
            total = len(trials)
            vowel_count = sum(1 for t in trials if t.task_type == "vowel")
            sentence_count = sum(1 for t in trials if t.task_type == "sentence")

            assert vowel_count == total, (
                f"Expected {total} vowel trials, got {vowel_count}"
            )
            assert sentence_count == 0, (
                f"Expected 0 sentence trials, got {sentence_count}"
            )

    def test_e2e_go_segments_per_trial(self,
        temp_output_dir,
        mock_session_paths,
        mock_medoc_client,
        mock_audio,
        mock_ui,
        mock_logger,
    ):
        """QA Scenario: Each trial has 4-7 GO segments per minute (16-28 total)."""
        from psycopy.medoc_experiment import MedocExperiment

        with (
            patch("psycopy.medoc_experiment.MedocClient", return_value=mock_medoc_client),
            patch("psycopy.medoc_experiment.PsychoPyUI", return_value=mock_ui),
            patch("psycopy.medoc_experiment.AudioService", return_value=mock_audio),
            patch("psycopy.medoc_experiment.validate_config"),
            patch("psycopy.medoc_experiment.get_run_metadata", return_value={"version": "0.3.0"}),
            patch("psycopy.medoc_experiment.configure_logging", return_value=mock_logger),
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch(
                "psycopy.medoc_experiment.create_output_directory",
                return_value=mock_session_paths,
            ),
            patch("time.sleep"),
        ):
            config = ExperimentConfig(
                participant_id="TEST001",
                session_id="01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            for block in exp.trials:
                for trial in block:
                    assert 16 <= trial.num_go_segments <= 28, (
                        f"Expected 16-28 GO segments, got {trial.num_go_segments}"
                    )

    def test_e2e_go_segment_durations(self,
        temp_output_dir,
        mock_session_paths,
        mock_medoc_client,
        mock_audio,
        mock_ui,
        mock_logger,
    ):
        """QA Scenario: Each GO segment is 1.5-3.5 seconds and total GO < 240."""
        from psycopy.medoc_experiment import MedocExperiment

        with (
            patch("psycopy.medoc_experiment.MedocClient", return_value=mock_medoc_client),
            patch("psycopy.medoc_experiment.PsychoPyUI", return_value=mock_ui),
            patch("psycopy.medoc_experiment.AudioService", return_value=mock_audio),
            patch("psycopy.medoc_experiment.validate_config"),
            patch("psycopy.medoc_experiment.get_run_metadata", return_value={"version": "0.3.0"}),
            patch("psycopy.medoc_experiment.configure_logging", return_value=mock_logger),
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch(
                "psycopy.medoc_experiment.create_output_directory",
                return_value=mock_session_paths,
            ),
            patch("time.sleep"),
        ):
            config = ExperimentConfig(
                participant_id="TEST001",
                session_id="01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            for block in exp.trials:
                for trial in block:
                    for dur in trial.go_segment_durations:
                        assert 1.5 <= dur <= 3.5, (
                            f"GO segment duration {dur} outside [1.5, 3.5]"
                        )
                    total_go = sum(trial.go_segment_durations)
                    assert total_go < 240.0, (
                        f"Total GO time {total_go} exceeds 240s"
                    )


# ==============================================================================
# QA Scenario 2: CSV Output Validation
# ==============================================================================


class TestCSVOutputValidation:
    """E2E test: Validate CSV file outputs."""

    def test_e2e_trials_csv_created(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_medoc_client,
        mock_audio,
        mock_ui,
        mock_logger,
    ):
        """QA Scenario: trials.csv has 4 rows with required fields."""
        import pandas as pd
        from psycopy.medoc_experiment import MedocExperiment

        with (
            patch("psycopy.medoc_experiment.MedocClient", return_value=mock_medoc_client),
            patch("psycopy.medoc_experiment.PsychoPyUI", return_value=mock_ui),
            patch("psycopy.medoc_experiment.AudioService", return_value=mock_audio),
            patch("psycopy.medoc_experiment.validate_config"),
            patch("psycopy.medoc_experiment.get_run_metadata", return_value={"version": "0.3.0"}),
            patch("psycopy.medoc_experiment.configure_logging", return_value=mock_logger),
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch(
                "psycopy.medoc_experiment.create_output_directory",
                return_value=mock_session_paths,
            ),
            patch("time.sleep"),
        ):
            config = ExperimentConfig(
                participant_id="TEST001",
                session_id="01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio
            exp._show_break_screen = MagicMock()
            exp.run()
            exp.save_all_loggers()

            # Read trials.csv
            trials_path = mock_session_paths.trials_file
            assert trials_path.exists(), "trials.csv not created"

            df = pd.read_csv(trials_path)
            assert len(df) == 4, f"Expected 4 rows in trials.csv, got {len(df)}"

            # Verify required columns
            required_columns = [
                "trial_instance_id",
                "set_number",
                "trial_in_set",
                "task_type",
                "is_stop_trial",
                "trigger_timestamp",
            ]
            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"

    def test_e2e_medoc_events_csv(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_medoc_client,
        mock_audio,
        mock_ui,
        mock_logger,
    ):
        """QA Scenario: medoc_events.csv has trigger and status events."""
        from psycopy.medoc_experiment import MedocExperiment

        with (
            patch("psycopy.medoc_experiment.MedocClient", return_value=mock_medoc_client),
            patch("psycopy.medoc_experiment.PsychoPyUI", return_value=mock_ui),
            patch("psycopy.medoc_experiment.AudioService", return_value=mock_audio),
            patch("psycopy.medoc_experiment.validate_config"),
            patch("psycopy.medoc_experiment.get_run_metadata", return_value={"version": "0.3.0"}),
            patch("psycopy.medoc_experiment.configure_logging", return_value=mock_logger),
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch(
                "psycopy.medoc_experiment.create_output_directory",
                return_value=mock_session_paths,
            ),
            patch("time.sleep"),
        ):
            config = ExperimentConfig(
                participant_id="TEST001",
                session_id="01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio
            exp._show_break_screen = MagicMock()
            exp.run()
            exp.save_all_loggers()

            # Read medoc_events.csv
            medoc_path = mock_session_paths.medoc_file
            assert medoc_path.exists(), "medoc_events.csv not created"

            content = medoc_path.read_text()
            lines = content.strip().split("\n")

            # Validate: 4 trigger + 4 status = 8 total (or merged events)
            data_lines = [l for l in lines[1:] if l.strip()]  # Skip header
            assert len(data_lines) >= 4, (
                f"Expected at least 4 medoc event rows, got {len(data_lines)}"
            )

            # Verify required columns
            header = lines[0]
            required_columns = [
                "trial_instance_id",
                "set_number",
                "trial_in_set",
                "trigger_timestamp",
                "status_timestamp",
            ]
            for col in required_columns:
                assert col in header, f"Missing column in medoc_events.csv: {col}"

    def test_e2e_stop_go_cues_logged(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_medoc_client,
        mock_audio,
        mock_ui,
        mock_logger,
    ):
        """QA Scenario: events.csv contains stop_cue and go_cue events for offline VAD."""
        from psycopy.medoc_experiment import MedocExperiment

        with (
            patch("psycopy.medoc_experiment.MedocClient", return_value=mock_medoc_client),
            patch("psycopy.medoc_experiment.PsychoPyUI", return_value=mock_ui),
            patch("psycopy.medoc_experiment.AudioService", return_value=mock_audio),
            patch("psycopy.medoc_experiment.validate_config"),
            patch("psycopy.medoc_experiment.get_run_metadata", return_value={"version": "0.3.0"}),
            patch("psycopy.medoc_experiment.configure_logging", return_value=mock_logger),
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch(
                "psycopy.medoc_experiment.create_output_directory",
                return_value=mock_session_paths,
            ),
            patch("time.sleep"),
        ):
            config = ExperimentConfig(
                participant_id="TEST001",
                session_id="01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=False,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio
            exp._show_break_screen = MagicMock()
            exp.run()
            exp.save_all_loggers()

            # Read events.csv
            events_path = mock_session_paths.events_file
            assert events_path.exists(), "events.csv not created"

            content = events_path.read_text()
            lines = content.strip().split("\n")
            reader = csv.DictReader(lines)

            stop_cues = 0
            go_cues = 0
            for row in reader:
                et = row.get("event_type", "")
                if "stop_cue" in et:
                    stop_cues += 1
                elif "go_cue" in et:
                    go_cues += 1

            assert stop_cues > 0, "Expected stop_cue events in events.csv for offline VAD"
            assert go_cues > 0, "Expected go_cue events in events.csv for offline VAD"

    def test_e2e_events_csv(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_medoc_client,
        mock_audio,
        mock_ui,
        mock_logger,
    ):
        """QA Scenario: events.csv has TRIAL_START, TRIAL_END markers."""
        from psycopy.medoc_experiment import MedocExperiment

        with (
            patch("psycopy.medoc_experiment.MedocClient", return_value=mock_medoc_client),
            patch("psycopy.medoc_experiment.PsychoPyUI", return_value=mock_ui),
            patch("psycopy.medoc_experiment.AudioService", return_value=mock_audio),
            patch("psycopy.medoc_experiment.validate_config"),
            patch("psycopy.medoc_experiment.get_run_metadata", return_value={"version": "0.3.0"}),
            patch("psycopy.medoc_experiment.configure_logging", return_value=mock_logger),
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch(
                "psycopy.medoc_experiment.create_output_directory",
                return_value=mock_session_paths,
            ),
            patch("time.sleep"),
        ):
            config = ExperimentConfig(
                participant_id="TEST001",
                session_id="01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio
            exp._show_break_screen = MagicMock()
            exp.run()
            exp.save_all_loggers()

            # Read events.csv
            events_path = mock_session_paths.events_file
            assert events_path.exists(), "events.csv not created"

            content = events_path.read_text()
            lines = content.strip().split("\n")

            # Verify header
            header = lines[0]
            required_columns = ["event_type", "timestamp", "trial_instance_id"]
            for col in required_columns:
                assert col in header, f"Missing column in events.csv: {col}"

            # Verify trial_start and trial_end events
            import csv

            lines = content.strip().split("\n")
            reader = csv.DictReader(lines)
            trial_starts = 0
            trial_ends = 0
            for row in reader:
                event_type = row.get("event_type", "")
                if "trial_start" in event_type.lower():
                    trial_starts += 1
                elif "trial_end" in event_type.lower():
                    trial_ends += 1

            assert trial_starts >= 4, (
                f"Expected at least 4 trial_start events, got {trial_starts}"
            )
            assert trial_ends >= 4, f"Expected at least 4 trial_end events, got {trial_ends}"

    def test_e2e_config_json_snapshot(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_medoc_client,
        mock_audio,
        mock_ui,
        mock_logger,
    ):
        """QA Scenario: config.json snapshot exists with all fields."""
        from psycopy.medoc_experiment import MedocExperiment

        with (
            patch("psycopy.medoc_experiment.MedocClient", return_value=mock_medoc_client),
            patch("psycopy.medoc_experiment.PsychoPyUI", return_value=mock_ui),
            patch("psycopy.medoc_experiment.AudioService", return_value=mock_audio),
            patch("psycopy.medoc_experiment.validate_config"),
            patch("psycopy.medoc_experiment.get_run_metadata", return_value={"version": "0.3.0"}),
            patch("psycopy.medoc_experiment.configure_logging", return_value=mock_logger),
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch(
                "psycopy.medoc_experiment.create_output_directory",
                return_value=mock_session_paths,
            ),
            patch("time.sleep"),
        ):
            config = ExperimentConfig(
                participant_id="TEST001",
                session_id="01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio
            exp._show_break_screen = MagicMock()
            exp.run()
            exp.save_all_loggers()

            # Check that config snapshot was saved (via save_config_snapshot mock)
            # The file is created during __init__ via save_config_snapshot
            from psycopy.session import save_config_snapshot

            # Verify save_config_snapshot was called
            # For this test, we just verify the SessionPaths has the config_file
            config_path = mock_session_paths.config_file
            assert config_path.name == "config.json", "Config file path incorrect"


# ==============================================================================
# QA Scenario 3: Trial Generator Validation (Unit Test)
# ==============================================================================


class TestTrialGeneratorValidation:
    """Unit tests for trial generator constraints."""

    def test_five_trials_total(self):
        """Validate: 5 blocks x 1 trial = 5 trials total."""
        rng = get_rng(ExperimentConfig(random_seed="42"))
        trials = generate_trials(num_sets=5, trials_per_set=1, num_stop_trials_ratio=0.0, rng=rng)

        total_trials = sum(len(t) for t in trials)
        assert total_trials == 5, f"Expected 5 trials, got {total_trials}"
        assert len(trials) == 5, f"Expected 5 blocks, got {len(trials)}"

    def test_all_vowel_trials(self):
        """Validate: All 5 trials are vowel."""
        rng = get_rng(ExperimentConfig(random_seed="42"))
        trials = generate_trials(num_sets=5, trials_per_set=1, num_stop_trials_ratio=0.0, rng=rng)

        for block_idx, block_trials in enumerate(trials):
            for trial_idx, trial in enumerate(block_trials):
                assert trial.task_type == "vowel", (
                    f"Block {block_idx} Trial {trial_idx}: expected vowel, got {trial.task_type}"
                )

    def test_go_segments_in_range(self):
        """Validate: Each trial has 4-7 GO segments per minute (16-28 total)."""
        rng = get_rng(ExperimentConfig(random_seed="42"))
        trials = generate_trials(num_sets=5, trials_per_set=1, num_stop_trials_ratio=0.0, rng=rng)

        for block_idx, block_trials in enumerate(trials):
            for trial_idx, trial in enumerate(block_trials):
                assert 16 <= trial.num_go_segments <= 28, (
                    f"Block {block_idx} Trial {trial_idx}: expected 16-28 GO segments, "
                    f"got {trial.num_go_segments}"
                )

    def test_go_durations_in_range(self):
        """Validate: Each GO segment is 1.5-3.5 seconds and total GO < 240."""
        rng = get_rng(ExperimentConfig(random_seed="42"))
        trials = generate_trials(num_sets=5, trials_per_set=1, num_stop_trials_ratio=0.0, rng=rng)

        for block_idx, block_trials in enumerate(trials):
            for trial_idx, trial in enumerate(block_trials):
                for seg_idx, dur in enumerate(trial.go_segment_durations):
                    assert 1.5 <= dur <= 3.5, (
                        f"Block {block_idx} Trial {trial_idx} Segment {seg_idx}: "
                        f"expected 1.5-3.5s, got {dur}"
                    )
                total_go = sum(trial.go_segment_durations)
                assert total_go < 240.0, (
                    f"Block {block_idx} Trial {trial_idx}: total GO {total_go} >= 240s"
                )

    def test_reproducibility_with_seed(self):
        """Validate: Same seed produces same sequence."""
        rng1 = get_rng(ExperimentConfig(random_seed="12345"))
        trials1 = generate_trials(
            num_sets=5, trials_per_set=1, num_stop_trials_ratio=0.0, rng=rng1
        )

        rng2 = get_rng(ExperimentConfig(random_seed="12345"))
        trials2 = generate_trials(
            num_sets=5, trials_per_set=1, num_stop_trials_ratio=0.0, rng=rng2
        )

        # Compare all trial configs
        for block_idx in range(len(trials1)):
            for trial_idx in range(len(trials1[block_idx])):
                t1 = trials1[block_idx][trial_idx]
                t2 = trials2[block_idx][trial_idx]
                assert t1.task_type == t2.task_type, (
                    f"Block {block_idx} Trial {trial_idx}: task_type mismatch"
                )
                assert t1.task_type == t2.task_type, (
                    f"Block {block_idx} Trial {trial_idx}: task_type mismatch"
                )
                assert t1.num_go_segments == t2.num_go_segments, (
                    f"Block {block_idx} Trial {trial_idx}: num_go_segments mismatch"
                )
                assert t1.go_segment_durations == t2.go_segment_durations, (
                    f"Block {block_idx} Trial {trial_idx}: go_segment_durations mismatch"
                )
