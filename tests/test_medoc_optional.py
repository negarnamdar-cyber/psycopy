"""Tests for Medoc device optional mode (require_connection=False)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from psycopy.config import ExperimentConfig, MedocConfig
from psycopy.medoc import MedocConnectionError
from psycopy.medoc_experiment import MedocExperiment


class TestMedocOptionalMode:
    """Tests for running without physical Medoc device."""

    def test_config_with_require_connection_false(self):
        """MedocConfig should accept require_connection=False."""
        config = MedocConfig(require_connection=False)
        assert config.require_connection is False

    def test_config_require_connection_true_by_default(self):
        """MedocConfig should have require_connection=True by default."""
        config = MedocConfig()
        assert config.require_connection is True

    def test_medoc_experiment_initializes_without_device(self, tmp_path):
        """MedocExperiment should initialize when device not connected with require_connection=False."""
        # Patch all external dependencies that require real hardware
        with (
            patch("psycopy.audio.AudioService") as mock_audio,
            patch("psycopy.medoc_experiment.create_output_directory") as mock_output,
            patch("psycopy.medoc_experiment.configure_logging") as mock_logging,
            patch("psycopy.medoc_experiment.get_run_metadata") as mock_metadata,
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch("psycopy.medoc_experiment.EventLogger"),
            patch("psycopy.medoc_experiment.MedocTrialLogger"),
            patch("psycopy.medoc_experiment.MedocLogger"),
            patch("psycopy.medoc_experiment.PsychoPyUI"),
            patch("psycopy.schedule.get_rng") as mock_rng,
        ):
            # Configure mock RNG
            mock_rng.return_value = MagicMock()

            # Configure mock output directory
            mock_output.return_value = MagicMock(
                output_dir=tmp_path,
                events_file=tmp_path / "events.csv",
                trials_file=tmp_path / "trials.csv",
                vad_file=tmp_path / "vad.csv",
                medoc_file=tmp_path / "medoc.csv",
                audio_dir=tmp_path / "audio",
            )

            # Configure mock logging
            mock_logger = MagicMock()
            mock_logging.return_value = mock_logger

            # Configure mock metadata
            mock_metadata.return_value = {"version": "0.3.0"}

            # Configure mock audio service
            mock_audio_instance = MagicMock()
            mock_audio_instance.vad_enabled = False
            mock_audio.return_value = mock_audio_instance

            # Configure MedocClient to raise MedocConnectionError
            with patch("psycopy.medoc_experiment.MedocClient") as mock_medoc_client:
                mock_medoc_client.side_effect = MedocConnectionError(
                    ip="192.168.1.100", port=5000, message="Connection refused"
                )

                # Create config with require_connection=False
                medoc_config = MedocConfig(
                    medoc_ip="192.168.1.100",
                    medoc_port=5000,
                    require_connection=False,
                )
                config = ExperimentConfig(
                    participant_id="001",
                    session_id="01",
                    medoc_config=medoc_config,
                    vad_enabled=False,
                )

                # Should not raise - should initialize with medoc_client=None
                experiment = MedocExperiment(config)
                assert experiment.medoc_client is None

    def test_medoc_experiment_raises_when_require_connection_true(self, tmp_path):
        """MedocExperiment should raise MedocConnectionError when require_connection=True."""
        with (
            patch("psycopy.audio.AudioService") as mock_audio,
            patch("psycopy.medoc_experiment.create_output_directory") as mock_output,
            patch("psycopy.medoc_experiment.configure_logging") as mock_logging,
            patch("psycopy.medoc_experiment.get_run_metadata") as mock_metadata,
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch("psycopy.medoc_experiment.EventLogger"),
            patch("psycopy.medoc_experiment.MedocTrialLogger"),
            patch("psycopy.medoc_experiment.MedocLogger"),
            patch("psycopy.medoc_experiment.PsychoPyUI"),
            patch("psycopy.schedule.get_rng") as mock_rng,
        ):
            mock_rng.return_value = MagicMock()
            mock_output.return_value = MagicMock(
                output_dir=tmp_path,
                events_file=tmp_path / "events.csv",
                trials_file=tmp_path / "trials.csv",
                vad_file=tmp_path / "vad.csv",
                medoc_file=tmp_path / "medoc.csv",
                audio_dir=tmp_path / "audio",
            )
            mock_logging.return_value = MagicMock()
            mock_metadata.return_value = {"version": "0.3.0"}
            mock_audio_instance = MagicMock()
            mock_audio_instance.vad_enabled = False
            mock_audio.return_value = mock_audio_instance

            # Configure MedocClient to raise MedocConnectionError
            with patch("psycopy.medoc_experiment.MedocClient") as mock_medoc_client:
                mock_medoc_client.side_effect = MedocConnectionError(
                    ip="192.168.1.100", port=5000, message="Connection refused"
                )

                medoc_config = MedocConfig(
                    medoc_ip="192.168.1.100",
                    medoc_port=5000,
                    require_connection=True,
                )
                config = ExperimentConfig(
                    participant_id="001",
                    session_id="01",
                    medoc_config=medoc_config,
                    vad_enabled=False,
                )

                with pytest.raises(MedocConnectionError):
                    MedocExperiment(config)

    def test_trial_skips_medoc_commands_when_no_device(self, tmp_path):
        """Trial should skip Medoc commands when medoc_client is None."""
        with (
            patch("psycopy.audio.AudioService") as mock_audio,
            patch("psycopy.medoc_experiment.create_output_directory") as mock_output,
            patch("psycopy.medoc_experiment.configure_logging") as mock_logging,
            patch("psycopy.medoc_experiment.get_run_metadata") as mock_metadata,
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch("psycopy.medoc_experiment.EventLogger"),
            patch("psycopy.medoc_experiment.MedocTrialLogger"),
            patch("psycopy.medoc_experiment.MedocLogger"),
            patch("psycopy.medoc_experiment.PsychoPyUI"),
            patch("psycopy.schedule.get_rng") as mock_rng,
        ):
            mock_rng.return_value = MagicMock()
            mock_output.return_value = MagicMock(
                output_dir=tmp_path,
                events_file=tmp_path / "events.csv",
                trials_file=tmp_path / "trials.csv",
                vad_file=tmp_path / "vad.csv",
                medoc_file=tmp_path / "medoc.csv",
                audio_dir=tmp_path / "audio",
            )
            mock_logger = MagicMock()
            mock_logging.return_value = mock_logger
            mock_metadata.return_value = {"version": "0.3.0"}

            mock_audio_instance = MagicMock()
            mock_audio_instance.vad_enabled = False
            mock_audio.return_value = mock_audio_instance

            with patch("psycopy.medoc_experiment.MedocClient") as mock_medoc_client:
                mock_medoc_client.side_effect = MedocConnectionError(
                    ip="192.168.1.100", port=5000, message="Connection refused"
                )

                medoc_config = MedocConfig(
                    medoc_ip="192.168.1.100",
                    medoc_port=5000,
                    require_connection=False,
                )
                config = ExperimentConfig(
                    participant_id="001",
                    session_id="01",
                    medoc_config=medoc_config,
                    vad_enabled=False,
                )

                experiment = MedocExperiment(config)
                assert experiment.medoc_client is None

                # Create a trial config
                from psycopy.trial_generator import TrialConfig

                trial_config = TrialConfig(
                    task_type="vowel",
                    num_go_segments=1,
                    go_segment_durations=(1.0,),
                )

                # Run trial - should not call medoc_client methods
                with patch.object(experiment, "audio", mock_audio_instance):
                    with patch.object(experiment, "medoc_logger", MagicMock()):
                        with patch.object(experiment, "event_logger", MagicMock()):
                            record = experiment.run_trial(0, 0, trial_config)

                # Verify no medoc client methods were called
                mock_medoc_client.return_value.__enter__.return_value.send_unified_program.assert_not_called()

                # Verify record was created with no temperature data
                assert record.temperature_celsius is None
                assert record.temperature_raw is None

    def test_warning_logged_when_device_not_connected(self, tmp_path, caplog):
        """Warning should be logged when device not connected with require_connection=False."""
        import logging

        with (
            patch("psycopy.audio.AudioService") as mock_audio,
            patch("psycopy.medoc_experiment.create_output_directory") as mock_output,
            patch("psycopy.medoc_experiment.configure_logging") as mock_logging,
            patch("psycopy.medoc_experiment.get_run_metadata") as mock_metadata,
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch("psycopy.medoc_experiment.EventLogger"),
            patch("psycopy.medoc_experiment.MedocTrialLogger"),
            patch("psycopy.medoc_experiment.MedocLogger"),
            patch("psycopy.medoc_experiment.PsychoPyUI"),
            patch("psycopy.schedule.get_rng") as mock_rng,
        ):
            mock_rng.return_value = MagicMock()
            mock_output.return_value = MagicMock(
                output_dir=tmp_path,
                events_file=tmp_path / "events.csv",
                trials_file=tmp_path / "trials.csv",
                vad_file=tmp_path / "vad.csv",
                medoc_file=tmp_path / "medoc.csv",
                audio_dir=tmp_path / "audio",
            )
            mock_logger = MagicMock()
            mock_logging.return_value = mock_logger
            mock_metadata.return_value = {"version": "0.3.0"}

            mock_audio_instance = MagicMock()
            mock_audio_instance.vad_enabled = False
            mock_audio.return_value = mock_audio_instance

            with patch("psycopy.medoc_experiment.MedocClient") as mock_medoc_client:
                mock_medoc_client.side_effect = MedocConnectionError(
                    ip="192.168.1.100", port=5000, message="Connection refused"
                )

                medoc_config = MedocConfig(
                    medoc_ip="192.168.1.100",
                    medoc_port=5000,
                    require_connection=False,
                )
                config = ExperimentConfig(
                    participant_id="001",
                    session_id="01",
                    medoc_config=medoc_config,
                    vad_enabled=False,
                )

                with caplog.at_level(logging.WARNING, logger="psycopy.medoc_experiment"):
                    experiment = MedocExperiment(config)

                # Check that warning was logged about running in testing mode
                assert any("testing mode" in record.message.lower() for record in caplog.records)

    def test_trial_logs_skip_message_when_no_device(self, tmp_path):
        """Trial should log skip message when medoc_client is None."""
        with (
            patch("psycopy.audio.AudioService") as mock_audio,
            patch("psycopy.medoc_experiment.create_output_directory") as mock_output,
            patch("psycopy.medoc_experiment.configure_logging") as mock_logging,
            patch("psycopy.medoc_experiment.get_run_metadata") as mock_metadata,
            patch("psycopy.medoc_experiment.save_config_snapshot"),
            patch("psycopy.medoc_experiment.EventLogger"),
            patch("psycopy.medoc_experiment.MedocTrialLogger"),
            patch("psycopy.medoc_experiment.MedocLogger"),
            patch("psycopy.medoc_experiment.PsychoPyUI"),
            patch("psycopy.schedule.get_rng") as mock_rng,
        ):
            mock_rng.return_value = MagicMock()
            mock_output.return_value = MagicMock(
                output_dir=tmp_path,
                events_file=tmp_path / "events.csv",
                trials_file=tmp_path / "trials.csv",
                vad_file=tmp_path / "vad.csv",
                medoc_file=tmp_path / "medoc.csv",
                audio_dir=tmp_path / "audio",
            )
            mock_logger = MagicMock()
            mock_logging.return_value = mock_logger
            mock_metadata.return_value = {"version": "0.3.0"}

            mock_audio_instance = MagicMock()
            mock_audio_instance.vad_enabled = False
            mock_audio.return_value = mock_audio_instance

            with patch("psycopy.medoc_experiment.MedocClient") as mock_medoc_client:
                mock_medoc_client.side_effect = MedocConnectionError(
                    ip="192.168.1.100", port=5000, message="Connection refused"
                )

                medoc_config = MedocConfig(
                    medoc_ip="192.168.1.100",
                    medoc_port=5000,
                    require_connection=False,
                )
                config = ExperimentConfig(
                    participant_id="001",
                    session_id="01",
                    medoc_config=medoc_config,
                    vad_enabled=False,
                )

                experiment = MedocExperiment(config)

                from psycopy.trial_generator import TrialConfig

                trial_config = TrialConfig(
                    task_type="vowel",
                    num_go_segments=1,
                    go_segment_durations=(1.0,),
                )

                with patch.object(experiment, "audio", mock_audio_instance):
                    with patch.object(experiment, "medoc_logger", MagicMock()):
                        with patch.object(experiment, "event_logger", MagicMock()):
                            experiment.run_trial(0, 0, trial_config)

                # Verify warning was logged during trial about skipping Medoc
                warning_calls = [
                    call
                    for call in mock_logger.warning.call_args_list
                    if "skipping medoc" in str(call).lower()
                ]
                assert len(warning_calls) > 0, "Expected warning about skipping Medoc"
