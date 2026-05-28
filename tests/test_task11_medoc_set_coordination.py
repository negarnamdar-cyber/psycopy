"""Test suite for Task 11: MedocExperiment set coordination.

Tests:
- run_set() executes 12 trials
- run() executes 8 sets
- User abort saves data
"""

import sys
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock heavy dependencies before imports
sys.modules["sounddevice"] = MagicMock()
sys.modules["psychopy"] = MagicMock()
sys.modules["psychopy.core"] = MagicMock()
sys.modules["psychopy.event"] = MagicMock()
sys.modules["psychopy.visual"] = MagicMock()

import pytest

from psycopy.config import ExperimentConfig, MedocConfig
from psycopy.trial_generator import TrialConfig
from psycopy.models import MedocTrialRecord


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test output."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


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
        rt_file=temp_output_dir / "rt.csv",
        config_file=temp_output_dir / "config.json",
        features_file=temp_output_dir / "features.csv",
        features_manifest_file=temp_output_dir / "features_manifest.json",
        vad_file=temp_output_dir / "vad.csv",
        blocks_file=temp_output_dir / "blocks.csv",
        segments_file=temp_output_dir / "segments.csv",
        medoc_file=temp_output_dir / "medoc.csv",
    )


@pytest.fixture
def mock_ui():
    """Create mock PsychoPyUI."""
    ui = MagicMock()
    ui.show_instructions = MagicMock()
    ui.show_completion = MagicMock()
    ui.wait_for_space = MagicMock()
    ui.close = MagicMock()
    ui.instruction_text = MagicMock()
    ui.win = MagicMock()
    ui.win.flip = MagicMock()
    ui.exp_clock = MagicMock()
    ui.exp_clock.getTime = MagicMock(return_value=0.0)
    ui.help_text = MagicMock()
    return ui


@pytest.fixture
def mock_audio():
    """Create mock AudioService."""
    audio = MagicMock()
    audio.vad_enabled = True
    audio.start = MagicMock()
    audio.stop = MagicMock()
    audio.start_vad_monitoring = MagicMock()
    audio.stop_vad_monitoring = MagicMock(return_value=[])
    audio.set_stop_cue_time = MagicMock(return_value=0.5)
    audio.get_speech_cessation_latency = MagicMock(return_value=None)
    audio.preflight = MagicMock()
    audio.enable_vad = MagicMock()
    return audio


@pytest.fixture
def mock_medoc_client():
    """Create mock MedocClient."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.send_program = MagicMock(return_value=b"\x00")
    return client


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.exception = MagicMock()
    return logger


@pytest.fixture
def mock_time():
    """Mock time.sleep to speed up tests."""
    with patch("time.sleep"):
        yield


class TestRunSetTwelveTrials:
    """QA Scenario 1: Set executes 12 trials."""

    def test_run_set_executes_twelve_trials(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_ui,
        mock_audio,
        mock_medoc_client,
        mock_logger,
        mock_time,
    ):
        """Test that run_set() executes exactly 12 trials."""
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
                "psycopy.medoc_experiment.create_output_directory", return_value=mock_session_paths
            ),
        ):
            config = ExperimentConfig(
                participant_id="P001",
                session_id="S01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio

            # Run first set
            trials = exp.trials[0]
            exp.run_set(0, trials)

            # Verify 12 trials were logged
            trial_count = len(exp.trial_logger.trials)
            assert trial_count == 12, f"Expected 12 trials, got {trial_count}"

    def test_run_set_correct_trial_distribution(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_ui,
        mock_audio,
        mock_medoc_client,
        mock_logger,
        mock_time,
    ):
        """Test that run_set() has correct 6V+6S distribution."""
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
                "psycopy.medoc_experiment.create_output_directory", return_value=mock_session_paths
            ),
        ):
            config = ExperimentConfig(
                participant_id="P001",
                session_id="S01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio

            # Run first set
            trials = exp.trials[0]
            exp.run_set(0, trials)

            # Verify trial distribution
            task_types = [t.task_type for t in exp.trial_logger.trials]
            vowel_count = task_types.count("vowel")
            sentence_count = task_types.count("sentence")

            assert vowel_count == 6, f"Expected 6 vowel trials, got {vowel_count}"
            assert sentence_count == 6, f"Expected 6 sentence trials, got {sentence_count}"


class TestRunEightSets:
    """QA Scenario 2: Full experiment 8 sets."""

    def test_run_sets_waiting_screen_between_sets(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_ui,
        mock_audio,
        mock_medoc_client,
        mock_logger,
        mock_time,
    ):
        """Test that waiting screen is shown between sets (7 times for 8 sets)."""
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
                "psycopy.medoc_experiment.create_output_directory", return_value=mock_session_paths
            ),
        ):
            config = ExperimentConfig(
                participant_id="P001",
                session_id="S01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio

            # Run full experiment
            exp.run()

            # Verify waiting screen was called 7 times (between 8 sets)
            assert mock_ui.wait_for_space.call_count == 7, (
                f"Expected 7 waiting screen calls (between 8 sets), got {mock_ui.wait_for_space.call_count}"
            )

    def test_run_full_experiment_ninety_six_trials(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_ui,
        mock_audio,
        mock_medoc_client,
        mock_logger,
        mock_time,
    ):
        """Test that run() executes all 96 trials (8 sets x 12 trials)."""
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
                "psycopy.medoc_experiment.create_output_directory", return_value=mock_session_paths
            ),
        ):
            config = ExperimentConfig(
                participant_id="P001",
                session_id="S01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio

            # Run full experiment
            exp.run()

            # Verify total trials = 8 sets x 12 trials = 96
            total_trials = len(exp.trial_logger.trials)
            assert total_trials == 96, f"Expected 96 trials, got {total_trials}"


class TestUserAbortSavesData:
    """QA Scenario 3: User abort saves data."""

    def test_user_abort_saves_loggers(
        self,
        temp_output_dir,
        mock_session_paths,
        mock_ui,
        mock_audio,
        mock_medoc_client,
        mock_logger,
        mock_time,
    ):
        """Test that UserAbort triggers save_all_loggers with partial data."""
        from psycopy.medoc_experiment import MedocExperiment
        from psycopy.runtime import UserAbort

        with (
            patch("psycopy.medoc_experiment.MedocClient", return_value=mock_medoc_client),
            patch("psycopy.medoc_experiment.PsychoPyUI", return_value=mock_ui),
            patch("psycopy.medoc_experiment.AudioService", return_value=mock_audio),
            patch("psycopy.medoc_experiment.validate_config"),
            patch("psycopy.medoc_experiment.get_run_metadata", return_value={"version": "0.3.0"}),
            patch("psycopy.medoc_experiment.configure_logging", return_value=mock_logger),
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch(
                "psycopy.medoc_experiment.create_output_directory", return_value=mock_session_paths
            ),
        ):
            config = ExperimentConfig(
                participant_id="P001",
                session_id="S01",
                random_seed="42",
                medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
                vad_enabled=True,
            )

            exp = MedocExperiment(config)
            exp.audio = mock_audio

            # Mock save_all_loggers to track its call
            save_spy = MagicMock()
            exp.save_all_loggers = save_spy

            # Patch run_trial to raise UserAbort on trial 5 (second set, trial 5)
            original_run_trial = exp.run_trial
            call_count = [0]

            def abort_on_trial_5(set_num, trial_num, trial_config):
                call_count[0] += 1
                if call_count[0] == 5:  # Abort on 5th overall trial
                    raise UserAbort()
                return original_run_trial(set_num, trial_num, trial_config)

            exp.run_trial = abort_on_trial_5

            # Run should abort
            with pytest.raises(UserAbort):
                exp.run()

            # Verify save_all_loggers was called on abort
            save_spy.assert_called_once()
