"""Medoc experiment controller for thermal stimulation with speech production.

Integrates Medoc thermode device communication with the existing speech
experiment infrastructure. Manages trial randomization, device communication,
VAD coordination, and data logging.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from random import Random

from psycopy.audio import AudioService
from psycopy.config import ExperimentConfig, ExperimentMode
from psycopy.medoc import MedocClient, MedocConnectionError
from psycopy.models import MedocTrialRecord, TaskState, generate_trial_instance_id
from psycopy.schedule import get_rng
from psycopy.session import (
    EventLogger,
    MedocLogger,
    MedocTrialLogger,
    VADLogger,
    create_output_directory,
    save_config_snapshot,
)
from psycopy.runtime import PsychoPyUI, UserAbort
from psycopy.stimuli import get_stimuli_path, load_stimuli
from psycopy.trial_generator import TrialConfig, generate_trials
from psycopy.validation import validate_config

# Trial timing constants
TRIAL_DURATION_SEC = 30.0  # Total trial duration in seconds
COOLDOWN_DURATION_SEC = 15.0  # Fixed cooldown duration at end of trial
SPEAKING_DURATION_SEC = TRIAL_DURATION_SEC - COOLDOWN_DURATION_SEC  # Active GO/STOP window before cooldown
STOP_CUE_MIN_SEC = 10.0  # Earliest STOP cue time from trial start
STOP_CUE_MAX_SEC = 15.0  # Latest STOP cue time from trial start
MIN_STOP_DISPLAY_SEC = 2.0  # Minimum time STOP cue must remain visible
INITIAL_STOP_SEC = 2.0  # Initial STOP screen before GO onset
VOWEL_TEXT = "Ahh"  # Text displayed for vowel trials

logger = logging.getLogger("psycopy.medoc_experiment")


class TrialState(Enum):
    """Trial execution state machine for Medoc experiments.

    Tracks the state of a single trial's lifecycle:
    - PENDING: Trial has not started
    - TRIGGER_SENT: TRIGGER command sent to Medoc, awaiting status
    - STATUS_SENT: GET_STATUS sent, trial in progress
    - COMPLETED: Trial finished successfully
    - FAILED: Trial failed due to error or abort
    """

    PENDING = "pending"
    TRIGGER_SENT = "trigger_sent"
    STATUS_SENT = "status_sent"
    COMPLETED = "completed"
    FAILED = "failed"


class MedocExperiment:
    """Experiment controller for Medoc thermal stimulation.

    Orchestrates 8 sets of 8 trials each, with per-trial Medoc device
    communication, VAD measurements, and data logging.

    Trial structure:
    - 4 vowel + 4 sentence trials per set (one of each pain level per task)
    - 1 xlow + 1 low + 1 medium + 1 high pain per task per set
    - ~25% stop trials per set by default

    Example:
        >>> config = ExperimentConfig(
        ...     participant_id="001",
        ...     session_id="01",
        ...     medoc_config=MedocConfig(medoc_ip="192.168.1.100"),
        ... )
        >>> experiment = MedocExperiment(config)
        >>> len(experiment.trials)  # 8 sets
        8
        >>> len(experiment.trials[0])  # 8 trials per set
        8
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize Medoc experiment with all dependencies.

        Args:
            config: Experiment configuration including Medoc settings.

        Raises:
            ValueError: If config validation fails.
            AudioServiceError: If audio device unavailable.
        """
        validate_config(config)
        self.config = config
        self.rng: Random = get_rng(config)

        self.session_paths = create_output_directory(config.to_dict())
        self.logger = configure_logging(self.session_paths.output_dir)
        self.logger.info("Initializing Medoc experiment")

        self.metadata = get_run_metadata(app_version="0.3.0")
        save_config_snapshot(config.to_dict(), self.metadata, self.session_paths.config_file)

        self.event_logger = EventLogger(self.session_paths.events_file)
        self.trial_logger = MedocTrialLogger(self.session_paths.trials_file)
        self.vad_logger = VADLogger(self.session_paths.vad_file)
        self.medoc_logger = MedocLogger(self.session_paths.medoc_file)

        self.audio = AudioService(sample_rate=config.sample_rate, retries=2)
        self.audio.preflight()
        if config.vad_enabled:
            self.audio.enable_vad(config)

        if config.medoc_config is None:
            logger.info(
                "Medoc disabled - running in testing mode. Temperature data will not be recorded."
            )
            self.medoc_client = None
        else:
            try:
                self.medoc_client = MedocClient(config.medoc_config)
            except MedocConnectionError:
                if config.medoc_config.require_connection:
                    raise
                logger.warning(
                    "Medoc device not connected - running in testing mode. Temperature data will not be recorded."
                )
                self.medoc_client = None

        self.ui = PsychoPyUI(fullscreen=config.fullscreen)

        # Determine number of sets based on mode (practice = 1 set, normal = 8 sets)
        num_sets = (
            1
            if config.mode in (ExperimentMode.PRACTICE_NO_MEDOC, ExperimentMode.PRACTICE_WITH_MEDOC)
            else 8
        )
        self.trials = generate_trials(
            num_sets=num_sets,
            trials_per_set=8,
            num_stop_trials_ratio=0.25,
            rng=self.rng,
        )

        self.currently_recording = False
        self._current_set = 0
        self._current_trial_in_set = 0

        self.event_logger.set_start_time()
        self.logger.info(
            "Generated %d sets with %d trials each",
            len(self.trials),
            len(self.trials[0]) if self.trials else 0,
        )

    def _setup_trial(self, trial_config: TrialConfig) -> None:
        """Prepare for trial execution.

        Validates trial configuration and resets trial-specific state.
        Does NOT connect to Medoc device (per-trial connection).

        Args:
            trial_config: Configuration for this trial (task_type, pain_condition,
                is_stop_trial).

        Note:
            This method prepares internal state only. The actual trial execution
            (connect, send command, disconnect) happens in run_trial.
        """
        logger.debug(
            "Setting up trial: type=%s pain=%s stop=%s",
            trial_config.task_type,
            trial_config.pain_condition,
            trial_config.is_stop_trial,
        )
        if self.audio.vad_enabled:
            self.vad_logger.reset()

    def _load_stimulus_text(self, trial_config: TrialConfig, trial_idx: int) -> str:
        """Load stimulus text for a trial.

        For vowel trials, returns 'Ahh'.
        For sentence trials, loads text from stimuli.csv (cycling through available sentences).

        Args:
            trial_config: Trial configuration containing task_type.
            trial_idx: Trial index for selecting sentence stimulus.

        Returns:
            Stimulus text string.
        """
        if trial_config.task_type == "vowel":
            return VOWEL_TEXT

        stimuli_file = get_stimuli_path()
        stimuli = load_stimuli(stimuli_file)
        if not stimuli:
            return "Sentence stimulus not available."
        stimulus_idx = trial_idx % len(stimuli)
        return stimuli[stimulus_idx].text

    def run_trial(
        self, set_num: int, trial_num: int, trial_config: TrialConfig
    ) -> MedocTrialRecord:
        """Execute a single trial with Medoc thermal stimulation.

        Trial timing sequence:
        1. Connect to Medoc (per-trial connection using context manager)
        2. Record trigger_timestamp = time.monotonic()
        3. Send the Medoc command mapped to this trial's pain condition
        4. Start audio recording via audio.start(audio_path)
        5. Start VAD monitoring via audio.start_vad_monitoring()
        6. Show stimulus text (vowel="Ahh" or sentence from stimuli.csv)
        7. For STOP trials (75%): show STOP cue via UI, VAD measures latency
        8. Wait until 38s total elapsed from trigger_timestamp
        9. Stop audio via audio.stop(), stop VAD via audio.stop_vad_monitoring()
        10. Disconnect from Medoc (context manager handles this)
        11. Return MedocTrialRecord with trigger fields populated

        Args:
            set_num: Set number (0-indexed).
            trial_num: Trial number within set (0-indexed).
            trial_config: Trial configuration (task_type, pain_condition, is_stop_trial).

        Returns:
            MedocTrialRecord with all trial data populated.

        Raises:
            MedocConnectionError: If connection to Medoc fails.
            MedocTimeoutError: If Medoc communication times out.
            MedocResponseError: If Medoc returns invalid response.
        """

        trial_instance_id = generate_trial_instance_id(
            self.config.participant_id,
            self.config.session_id,
            f"set{set_num}",
            trial_num,
        )

        # Compute global trial index based on actual per-set length
        per_set_len = len(self.trials[0]) if self.trials and len(self.trials[0]) > 0 else 1
        global_trial_idx = set_num * per_set_len + trial_num
        stimulus_text = self._load_stimulus_text(trial_config, global_trial_idx)

        audio_filename = f"sub-{self.config.participant_id}_set-{set_num}_trial-{trial_num:03d}.wav"
        audio_path = self.session_paths.audio_dir / audio_filename

        self.event_logger.log(
            event_type="trial_start",
            trial_instance_id=trial_instance_id,
            block=f"set{set_num}",
            event_data={
                "task_type": trial_config.task_type,
                "pain_condition": trial_config.pain_condition,
                "is_stop_trial": trial_config.is_stop_trial,
            },
        )

        trigger_timestamp: float = 0.0
        status_timestamp: float | None = None
        temperature_raw: bytes | None = None
        temperature_celsius: float | None = None
        device_state: int | None = None
        test_state: int | None = None
        response_code: int | None = None
        latency_ms: float | None = None
        vad_started = False
        audio_started = False

        try:
            if trial_config.task_type == "vowel":
                self.ui.apply_state(TaskState.STOP)
                self.ui.sentence_text.text = stimulus_text
                self.ui.sentence_text.draw()
                self.ui.state_background.draw()
                self.ui.state_indicator.draw()
                self.ui.win.flip()
                self.ui.wait(INITIAL_STOP_SEC)

            self.ui.apply_state(TaskState.GO)
            self.ui.sentence_text.text = stimulus_text
            self.ui.sentence_text.draw()
            self.ui.state_background.draw()
            self.ui.state_indicator.draw()
            self.ui.win.flip()

            if self.medoc_client is not None:
                with self.medoc_client as client:
                    trigger_timestamp = time.monotonic()
                    client.send_program(trial_config.pain_condition)
                    self.medoc_logger.log_trigger(
                        trial_instance_id=trial_instance_id,
                        set_number=set_num,
                        trial_in_set=trial_num,
                        timestamp=trigger_timestamp,
                    )
                    self.logger.info(
                        "Medoc command sent for trial %d.%d (%s) at %.3f",
                        set_num,
                        trial_num,
                        trial_config.pain_condition,
                        trigger_timestamp,
                    )

                    self.audio.start(audio_path)
                    audio_started = True
                    self.currently_recording = True

                    if self.config.vad_enabled:
                        self.audio.start_vad_monitoring()
                        self.vad_logger.set_context(trial_instance_id, f"set{set_num}", set_num)
                        vad_started = True

                    self.event_logger.log(
                        event_type="recording_start",
                        trial_instance_id=trial_instance_id,
                        block=f"set{set_num}",
                    )

                    elapsed = time.monotonic() - trigger_timestamp
                    remaining = TRIAL_DURATION_SEC - elapsed
            else:
                trigger_timestamp = time.monotonic()
                self.logger.warning(
                    "Skipping Medoc communication for trial %d.%d - running in testing mode",
                    set_num,
                    trial_num,
                )

                self.audio.start(audio_path)
                audio_started = True
                self.currently_recording = True

                if self.config.vad_enabled:
                    self.audio.start_vad_monitoring()
                    self.vad_logger.set_context(trial_instance_id, f"set{set_num}", set_num)
                    vad_started = True

                self.event_logger.log(
                    event_type="recording_start",
                    trial_instance_id=trial_instance_id,
                    block=f"set{set_num}",
                )

                elapsed = time.monotonic() - trigger_timestamp
                remaining = TRIAL_DURATION_SEC - elapsed

            stop_cue_time: float | None = None
            adjusted_speaking_duration = SPEAKING_DURATION_SEC

            if trial_config.is_stop_trial:
                stop_cue_elapsed = time.monotonic() - trigger_timestamp
                stop_cue_delay = self.rng.uniform(STOP_CUE_MIN_SEC, STOP_CUE_MAX_SEC) - stop_cue_elapsed
                if stop_cue_delay > 0:
                    self.ui.wait(stop_cue_delay)

                    self.ui.apply_state(TaskState.STOP)
                    self.ui.sentence_text.text = stimulus_text
                    self.ui.sentence_text.draw()
                    self.ui.state_background.draw()
                    self.ui.state_indicator.draw()
                    self.ui.win.flip()
                    stop_cue_time = time.monotonic() - trigger_timestamp
                    adjusted_speaking_duration = max(
                        SPEAKING_DURATION_SEC,
                        stop_cue_time + MIN_STOP_DISPLAY_SEC,
                    )

                    if self.config.vad_enabled and self.audio.vad_enabled:
                        relative_stop_cue_time = self.audio.set_stop_cue_time()
                        if relative_stop_cue_time is not None:
                            self.vad_logger.log_event(
                                event_type="stop_cue",
                                timestamp=relative_stop_cue_time,
                            )
                            self.logger.info(
                                "STOP cue displayed at %.3f for trial %d.%d",
                                relative_stop_cue_time,
                                set_num,
                                trial_num,
                            )

            speaking_elapsed = time.monotonic() - trigger_timestamp
            speaking_remaining = adjusted_speaking_duration - speaking_elapsed
            if speaking_remaining > 0:
                self.ui.wait(speaking_remaining)

            if audio_started or self.currently_recording:
                try:
                    self.audio.stop()
                    self.currently_recording = False
                    audio_started = False
                    self.event_logger.log(
                        event_type="recording_end",
                        trial_instance_id=trial_instance_id,
                        block=f"set{set_num}",
                    )
                except Exception as exc:
                    self.logger.warning("Error stopping audio: %s", exc)

            if vad_started and self.config.vad_enabled:
                try:
                    vad_events = self.audio.stop_vad_monitoring()
                    for event in vad_events:
                        self.vad_logger.log_event(
                            event_type=event["type"],
                            timestamp=event["timestamp"],
                            is_speech=event.get("is_speech"),
                        )

                    if trial_config.is_stop_trial:
                        latency = self.audio.get_speech_cessation_latency()
                        if latency is not None:
                            latency_ms = latency * 1000.0
                            self.logger.info(
                                "VAD speech cessation latency: %.1f ms for trial %d.%d",
                                latency_ms,
                                set_num,
                                trial_num,
                            )

                    self.vad_logger.save()
                    self.vad_logger.reset()
                    vad_started = False
                except Exception as exc:
                    self.logger.warning("Error stopping VAD: %s", exc)

            cooldown_start = time.monotonic()
            cooldown_duration = TRIAL_DURATION_SEC - (cooldown_start - trigger_timestamp)
            if cooldown_duration > 0:
                self._show_cooldown_screen(cooldown_duration)

        except Exception as exc:
            self.logger.exception("Error during trial %d.%d: %s", set_num, trial_num, exc)
            raise

        finally:
            if audio_started or self.currently_recording:
                try:
                    self.audio.stop()
                    self.currently_recording = False
                    self.event_logger.log(
                        event_type="recording_end",
                        trial_instance_id=trial_instance_id,
                        block=f"set{set_num}",
                    )
                except Exception as exc:
                    self.logger.warning("Error stopping audio: %s", exc)

            if vad_started and self.config.vad_enabled:
                try:
                    vad_events = self.audio.stop_vad_monitoring()
                    for event in vad_events:
                        self.vad_logger.log_event(
                            event_type=event["type"],
                            timestamp=event["timestamp"],
                            is_speech=event.get("is_speech"),
                        )

                    if trial_config.is_stop_trial:
                        latency = self.audio.get_speech_cessation_latency()
                        if latency is not None:
                            latency_ms = latency * 1000.0
                            self.logger.info(
                                "VAD speech cessation latency: %.1f ms for trial %d.%d",
                                latency_ms,
                                set_num,
                                trial_num,
                            )

                    self.vad_logger.save()
                    self.vad_logger.reset()
                except Exception as exc:
                    self.logger.warning("Error stopping VAD: %s", exc)

        trial_end_timestamp = time.monotonic()
        actual_duration = trial_end_timestamp - trigger_timestamp

        self.event_logger.log(
            event_type="trial_end",
            trial_instance_id=trial_instance_id,
            block=f"set{set_num}",
            event_data={
                "actual_duration_sec": round(actual_duration, 3),
                "is_stop_trial": trial_config.is_stop_trial,
            },
        )

        self.logger.info(
            "Trial %d.%d complete: type=%s pain=%s stop=%s duration=%.3fs",
            set_num,
            trial_num,
            trial_config.task_type,
            trial_config.pain_condition,
            trial_config.is_stop_trial,
            actual_duration,
        )

        return MedocTrialRecord(
            trial_instance_id=trial_instance_id,
            set_number=set_num,
            trial_in_set=trial_num,
            task_type=trial_config.task_type,
            pain_condition=trial_config.pain_condition,
            is_stop_trial=trial_config.is_stop_trial,
            trigger_timestamp=trigger_timestamp,
            status_timestamp=status_timestamp,
            temperature_raw=temperature_raw,
            temperature_celsius=temperature_celsius,
            device_state=device_state,
            test_state=test_state,
            response_code=response_code,
        )

    def run_set(self, set_num: int, trials: list[TrialConfig]) -> None:
        """Execute all trials in a set.

        Iterates through trials in a set, calling `run_trial` for each.

        Logs trial records to `MedocTrialLogger` and handles errors gracefully
        (failed trials are logged and the set continues).

        Args:
            set_num: Set number (0-indexed).
            trials: List of `TrialConfig` for this set (length depends on
                experimental configuration; typically 8 or 12).

        Note:
            Does NOT auto-advance to next set. Caller handles waiting screen.
        """
        self._current_set = set_num
        self.event_logger.log(
            event_type="set_start",
            trial_instance_id="",
            block=f"set{set_num}",
            event_data={"set_number": set_num, "num_trials": len(trials)},
        )

        for trial_num, trial_config in enumerate(trials):
            self._current_trial_in_set = trial_num
            try:
                record = self.run_trial(set_num, trial_num, trial_config)
                self.trial_logger.log_trial(record)
                self.logger.info(
                    "Logged trial %d.%d: type=%s pain=%s",
                    set_num,
                    trial_num,
                    trial_config.task_type,
                    trial_config.pain_condition,
                )
            except UserAbort:
                raise
            except Exception as exc:
                self.logger.exception(
                    "Failed trial %d.%d: %s - continuing to next trial",
                    set_num,
                    trial_num,
                    exc,
                )
                self.event_logger.log(
                    event_type="trial_error",
                    trial_instance_id="",
                    block=f"set{set_num}",
                    event_data={
                        "trial_num": trial_num,
                        "error": str(exc),
                    },
                )

        self.event_logger.log(
            event_type="set_end",
            trial_instance_id="",
            block=f"set{set_num}",
            event_data={"set_number": set_num},
        )
        self.logger.info("Set %d complete: %d trials executed", set_num, len(trials))

    def save_all_loggers(self) -> None:
        """Flush all loggers to disk.

        Writes pending data from all loggers:
        - TrialLogger
        - VADLogger
        - MedocLogger
        - EventLogger
        """
        self.trial_logger.save()
        self.vad_logger.save()
        self.medoc_logger.save()
        self.event_logger.save()
        self.logger.info("All loggers saved to disk")

    def run(self) -> None:
        """Run the complete 8-set experiment.

        Sequence:
        1. Show instruction screen at start
        2. Iterate through all 8 sets
        3. Call run_set() for each set
        4. After each set (except last), show waiting screen for manual advance
        5. Handle UserAbort exception (ESC key) - save data and exit gracefully
        6. Show completion screen at end

        Note:
            Manual advance required between sets. Does NOT auto-advance.
        """
        try:
            self.event_logger.log(
                event_type="experiment_start",
                trial_instance_id="",
                block="",
                event_data={"participant_id": self.config.participant_id},
            )

            self.ui.show_instructions(
                go_segmentation_enabled=False, medoc_enabled=self.medoc_client is not None
            )

            for set_num, set_trials in enumerate(self.trials):
                self.logger.info("Starting set %d of %d", set_num + 1, len(self.trials))
                self.run_set(set_num, set_trials)

                if set_num < len(self.trials) - 1:
                    self.ui.show_set_waiting_screen(set_num)

            total_per_set = len(self.trials[0]) if self.trials and len(self.trials[0]) > 0 else 0
            self.event_logger.log(
                event_type="experiment_complete",
                trial_instance_id="",
                block="",
                event_data={"total_trials": len(self.trials) * total_per_set},
            )

            self.ui.show_completion()
            self.save_all_loggers()

        except UserAbort:
            self.logger.warning("User aborted experiment with ESC key")
            self.event_logger.log(
                event_type="experiment_abort",
                trial_instance_id="",
                block="",
                event_data={"reason": "user_abort"},
            )
            self.save_all_loggers()
            self.ui.close()
            raise

        except Exception as exc:
            self.logger.exception("Unhandled error during experiment: %s", exc)
            self.event_logger.log(
                event_type="experiment_error",
                trial_instance_id="",
                block="",
                event_data={"error": str(exc)},
            )
            self.save_all_loggers()
            self.ui.close()
            raise

        finally:
            self.ui.close()

    def _show_cooldown_screen(self, duration: float) -> None:
        self.ui.instruction_text.text = "Cooling down"
        self.ui.instruction_text.draw()
        self.ui.win.flip()
        self.ui.wait(duration)

    def _show_set_waiting_screen(self, set_num: int) -> None:
        """Show waiting screen between sets for manual advance.

        Displays: "Set {set_num+1}/8 complete. Press SPACE when ready."

        Args:
            set_num: Just-completed set number (0-indexed).
        """
        message = (
            f"Set {set_num + 1}/{len(self.trials)} complete.\n\nPress SPACE when ready to continue."
        )
        self.ui.instruction_text.text = message
        self.ui.instruction_text.draw()
        self.ui.help_text.draw()
        self.ui.win.flip()
        self.ui.wait_for_space()


def configure_logging(output_dir) -> logging.Logger:
    """Configure logging for the Medoc experiment.

    Args:
        output_dir: Directory for log files.

    Returns:
        Configured logger instance.
    """
    from psycopy.runtime_logging import configure_logging as _configure_logging

    return _configure_logging(output_dir)


def get_run_metadata(app_version: str) -> dict:
    """Get run metadata including app version.

    Args:
        app_version: Application version string.

    Returns:
        Dict with metadata including version and timestamp.
    """
    from psycopy.runtime_logging import get_run_metadata as _get_run_metadata

    return _get_run_metadata(app_version)
