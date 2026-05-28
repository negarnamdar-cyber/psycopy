"""Medoc experiment controller for thermal stimulation with vowel production.

Integrates Medoc thermode device communication with vowel speech
experiment infrastructure. Manages trial randomization, device communication,
VAD coordination, and data logging.

Two experiment modes are supported:

1. Vowel mode (NORMAL / PRACTICE):
   - 5 blocks of 6 trials each = 30 total trials
   - Each trial: 60 seconds of alternating STOP/GO segments
     - 3-7 GO segments per trial, each 3-7 seconds
     - STOP segments are brief (~0.5s) between GOs
     - Pattern: STOP -> GO -> STOP -> GO -> ... -> STOP
   - 1-minute break between blocks
   - Pain conditions: xlow (8), low (8), medium (7), high (7)
   - Total experiment time: ~35 minutes (30 trials x 60s + 4 breaks x 60s)

2. Speech mode (SPEECH):
   - Free speech interview with thermal stimulation
   - Researchers ask questions while the screen shows "SPEAK"
   - Behind the scenes: randomized pain schedule with 6 minutes on,
     1 minute off (repeating until researcher stops)
   - Only graceful shutdown (Q + 12345) stops the experiment
   - ESC is disabled — only coded shutdown works
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
from psycopy.trial_generator import TrialConfig, generate_trials
from psycopy.validation import validate_config

# Trial timing constants
TRIAL_DURATION_SEC = 60.0  # Total trial duration in seconds
BLOCK_BREAK_SEC = 60.0  # Break duration between blocks in seconds
STOP_TRANSITION_SEC = 0.5  # Brief STOP display between GO segments
VOWEL_TEXT = "Ahh"  # Text displayed for all trials

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
    """Experiment controller for Medoc thermal stimulation with vowel production.

    Orchestrates 5 blocks of 6 trials each, with per-trial Medoc device
    communication, VAD measurements, and data logging.

    Trial structure:
    - 30 total trials across 5 blocks
    - Each trial: 60 seconds of alternating STOP/GO
    - 3-7 GO segments per trial, each 3-7 seconds
    - 1-minute break between blocks
    - Pain conditions: xlow (8), low (8), medium (7), high (7)

    Example:
        >>> config = ExperimentConfig(
        ...     participant_id="001",
        ...     session_id="01",
        ...     medoc_config=MedocConfig(medoc_ip="192.168.1.100"),
        ... )
        >>> experiment = MedocExperiment(config)
        >>> len(experiment.trials)  # 5 blocks
        5
        >>> len(experiment.trials[0])  # 6 trials per block
        6
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

        # Determine number of blocks based on mode (practice = 1 block, normal = 5 blocks)
        num_blocks = (
            1
            if config.mode in (ExperimentMode.PRACTICE_NO_MEDOC, ExperimentMode.PRACTICE_WITH_MEDOC)
            else 5
        )

        # Speech mode: no structured trials; pain schedule generated later
        if config.mode == ExperimentMode.SPEECH:
            self.trials = []
            self.speech_pain_schedule = self._generate_speech_pain_schedule()
        else:
            self.trials = generate_trials(
                num_sets=num_blocks,
                trials_per_set=6,
                num_stop_trials_ratio=0.0,
                rng=self.rng,
            )
            self.speech_pain_schedule = []

        self.currently_recording = False
        self._current_set = 0
        self._current_trial_in_set = 0

        self.event_logger.set_start_time()
        if self.trials:
            self.logger.info(
                "Generated %d blocks with %d trials each",
                len(self.trials),
                len(self.trials[0]) if self.trials else 0,
            )
        else:
            self.logger.info("Speech mode: no structured trials. Pain schedule ready.")

    def _setup_trial(self, trial_config: TrialConfig) -> None:
        """Prepare for trial execution.

        Validates trial configuration and resets trial-specific state.
        Does NOT connect to Medoc device (per-trial connection).

        Args:
            trial_config: Configuration for this trial (task_type, pain_condition,
                num_go_segments, go_segment_durations).

        Note:
            This method prepares internal state only. The actual trial execution
            (connect, send command, disconnect) happens in run_trial.
        """
        logger.debug(
            "Setting up trial: type=%s pain=%s segments=%d",
            trial_config.task_type,
            trial_config.pain_condition,
            trial_config.num_go_segments,
        )
        if self.audio.vad_enabled:
            self.vad_logger.reset()

    def _load_stimulus_text(self, trial_config: TrialConfig, trial_idx: int) -> str:
        """Load stimulus text for a trial.

        All trials are vowel trials and display 'Ahh'.

        Args:
            trial_config: Trial configuration containing task_type.
            trial_idx: Trial index (unused, kept for API compatibility).

        Returns:
            Stimulus text string.
        """
        return VOWEL_TEXT

    def _display_state(self, state: TaskState, stimulus_text: str) -> None:
        """Display the current state (GO or STOP) with stimulus text.

        Args:
            state: TaskState.GO or TaskState.STOP
            stimulus_text: Text to display (always "Ahh")
        """
        self.ui.apply_state(state)
        self.ui.sentence_text.text = stimulus_text
        self.ui.sentence_text.draw()
        self.ui.state_background.draw()
        self.ui.state_indicator.draw()
        self.ui.win.flip()

    def run_trial(
        self, set_num: int, trial_num: int, trial_config: TrialConfig
    ) -> MedocTrialRecord:
        """Execute a single 60-second trial with alternating STOP/GO segments.

        Trial timing sequence:
        1. Connect to Medoc (per-trial connection using context manager)
        2. Record trigger_timestamp = time.monotonic()
        3. Send the Medoc command mapped to this trial's pain condition
        4. Start audio recording via audio.start(audio_path)
        5. Start VAD monitoring via audio.start_vad_monitoring()
        6. Display alternating STOP/GO segments:
           - STOP -> GO (3-7s) -> STOP -> GO (3-7s) -> ... -> STOP
           - 3-7 GO segments per trial
           - Each GO segment: 3-7 seconds
           - STOP between segments: ~0.5 seconds
        7. Stop audio and VAD monitoring
        8. Disconnect from Medoc (context manager handles this)
        9. Return MedocTrialRecord with trigger fields populated

        Args:
            set_num: Block number (0-indexed).
            trial_num: Trial number within block (0-indexed).
            trial_config: Trial configuration (task_type, pain_condition,
                num_go_segments, go_segment_durations).

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
            f"block{set_num}",
            trial_num,
        )

        stimulus_text = self._load_stimulus_text(trial_config, trial_num)

        audio_filename = f"sub-{self.config.participant_id}_block-{set_num}_trial-{trial_num:03d}.wav"
        audio_path = self.session_paths.audio_dir / audio_filename

        self.event_logger.log(
            event_type="trial_start",
            trial_instance_id=trial_instance_id,
            block=f"block{set_num}",
            event_data={
                "task_type": trial_config.task_type,
                "pain_condition": trial_config.pain_condition,
                "num_go_segments": trial_config.num_go_segments,
                "go_durations": list(trial_config.go_segment_durations),
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
            # Send Medoc command and start recording
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
                        self.vad_logger.set_context(trial_instance_id, f"block{set_num}", set_num)
                        vad_started = True

                    self.event_logger.log(
                        event_type="recording_start",
                        trial_instance_id=trial_instance_id,
                        block=f"block{set_num}",
                    )
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
                    self.vad_logger.set_context(trial_instance_id, f"block{set_num}", set_num)
                    vad_started = True

                self.event_logger.log(
                    event_type="recording_start",
                    trial_instance_id=trial_instance_id,
                    block=f"block{set_num}",
                )

            # Run the alternating STOP/GO segments
            # Pattern: STOP -> GO -> STOP -> GO -> ... -> STOP
            for seg_idx, go_duration in enumerate(trial_config.go_segment_durations):
                # Brief STOP before GO (or between GOs)
                self._display_state(TaskState.STOP, stimulus_text)
                self.ui.wait(STOP_TRANSITION_SEC)

                # GO segment - participant says "Ahh"
                self._display_state(TaskState.GO, stimulus_text)

                # Wait for the GO duration
                self.ui.wait(go_duration)

                # Log segment info
                self.logger.debug(
                    "Trial %d.%d segment %d: GO for %.2fs",
                    set_num,
                    trial_num,
                    seg_idx,
                    go_duration,
                )

            # Final STOP after last GO segment
            self._display_state(TaskState.STOP, stimulus_text)

            # Wait until full 60 seconds have elapsed
            elapsed = time.monotonic() - trigger_timestamp
            remaining = TRIAL_DURATION_SEC - elapsed
            if remaining > 0:
                self.ui.wait(remaining)

            # Stop audio
            if audio_started or self.currently_recording:
                try:
                    self.audio.stop()
                    self.currently_recording = False
                    audio_started = False
                    self.event_logger.log(
                        event_type="recording_end",
                        trial_instance_id=trial_instance_id,
                        block=f"block{set_num}",
                    )
                except Exception as exc:
                    self.logger.warning("Error stopping audio: %s", exc)

            # Stop VAD and save events
            if vad_started and self.config.vad_enabled:
                try:
                    vad_events = self.audio.stop_vad_monitoring()
                    for event in vad_events:
                        self.vad_logger.log_event(
                            event_type=event["type"],
                            timestamp=event["timestamp"],
                            is_speech=event.get("is_speech"),
                        )

                    self.vad_logger.save()
                    self.vad_logger.reset()
                    vad_started = False
                except Exception as exc:
                    self.logger.warning("Error stopping VAD: %s", exc)

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
                        block=f"block{set_num}",
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

                    self.vad_logger.save()
                    self.vad_logger.reset()
                except Exception as exc:
                    self.logger.warning("Error stopping VAD: %s", exc)

        trial_end_timestamp = time.monotonic()
        actual_duration = trial_end_timestamp - trigger_timestamp

        self.event_logger.log(
            event_type="trial_end",
            trial_instance_id=trial_instance_id,
            block=f"block{set_num}",
            event_data={
                "actual_duration_sec": round(actual_duration, 3),
                "num_go_segments": trial_config.num_go_segments,
            },
        )

        self.logger.info(
            "Trial %d.%d complete: pain=%s segments=%d duration=%.3fs",
            set_num,
            trial_num,
            trial_config.pain_condition,
            trial_config.num_go_segments,
            actual_duration,
        )

        return MedocTrialRecord(
            trial_instance_id=trial_instance_id,
            set_number=set_num,
            trial_in_set=trial_num,
            task_type=trial_config.task_type,
            pain_condition=trial_config.pain_condition,
            is_stop_trial=False,  # All trials have internal stop/go; this field retained for compat
            trigger_timestamp=trigger_timestamp,
            status_timestamp=status_timestamp,
            temperature_raw=temperature_raw,
            temperature_celsius=temperature_celsius,
            device_state=device_state,
            test_state=test_state,
            response_code=response_code,
        )

    def run_set(self, set_num: int, trials: list[TrialConfig]) -> None:
        """Execute all trials in a block.

        Iterates through trials in a block, calling `run_trial` for each.

        Logs trial records to `MedocTrialLogger` and handles errors gracefully
        (failed trials are logged and the block continues).

        Args:
            set_num: Block number (0-indexed).
            trials: List of `TrialConfig` for this block (6 trials).

        Note:
            Does NOT show break screen. Caller handles breaks between blocks.
        """
        self._current_set = set_num
        self.event_logger.log(
            event_type="block_start",
            trial_instance_id="",
            block=f"block{set_num}",
            event_data={"block_number": set_num, "num_trials": len(trials)},
        )

        for trial_num, trial_config in enumerate(trials):
            self._current_trial_in_set = trial_num
            try:
                record = self.run_trial(set_num, trial_num, trial_config)
                self.trial_logger.log_trial(record)
                self.logger.info(
                    "Logged trial %d.%d: pain=%s segments=%d",
                    set_num,
                    trial_num,
                    trial_config.pain_condition,
                    trial_config.num_go_segments,
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
                    block=f"block{set_num}",
                    event_data={
                        "trial_num": trial_num,
                        "error": str(exc),
                    },
                )

        self.event_logger.log(
            event_type="block_end",
            trial_instance_id="",
            block=f"block{set_num}",
            event_data={"block_number": set_num},
        )
        self.logger.info("Block %d complete: %d trials executed", set_num, len(trials))

    def _show_break_screen(self, duration: float = BLOCK_BREAK_SEC) -> None:
        """Show a break screen with countdown timer.

        Displays: "BREAK - Rest your voice\n\nNext block in: {countdown}s"

        Args:
            duration: Break duration in seconds (default: 60).
        """
        start_time = time.monotonic()
        while True:
            elapsed = time.monotonic() - start_time
            remaining = duration - elapsed
            if remaining <= 0:
                break

            message = (
                "BREAK\n\nRest your voice.\n\n"
                f"Next block in: {int(remaining)}s"
            )
            self.ui.instruction_text.text = message
            self.ui.instruction_text.draw()
            self.ui.help_text.draw()
            self.ui.win.flip()
            self.ui.wait(0.5)

    def _generate_speech_pain_schedule(self) -> list[str]:
        """Generate a randomized pain condition schedule for speech mode.

        Creates a repeating pool of pain conditions: 8 xlow + 8 low + 7 medium + 7 high,
        shuffles them, and returns as a flat list. The schedule loops as needed.

        Returns:
            List of pain condition strings in randomized order.
        """
        pain_pool = (
            ["xlow"] * 8
            + ["low"] * 8
            + ["medium"] * 7
            + ["high"] * 7
        )
        self.rng.shuffle(pain_pool)
        self.logger.info("Speech mode: generated pain schedule with %d conditions", len(pain_pool))
        return pain_pool

    def _run_speech_pain_cycle(self, pain_condition: str, cycle_idx: int) -> None:
        """Run one 6-minute pain + 1-minute pause cycle for speech mode.

        Args:
            pain_condition: Pain level to send to Medoc (xlow/low/medium/high).
            cycle_idx: Zero-based cycle index for logging.
        """
        self.logger.info(
            "Speech mode: starting pain cycle %d with condition %s",
            cycle_idx,
            pain_condition,
        )
        self.event_logger.log(
            event_type="pain_cycle_start",
            trial_instance_id="",
            block="speech",
            event_data={
                "cycle_idx": cycle_idx,
                "pain_condition": pain_condition,
            },
        )

        trigger_timestamp = time.monotonic()

        if self.medoc_client is not None:
            with self.medoc_client as client:
                client.send_program(pain_condition)
                self.medoc_logger.log_trigger(
                    trial_instance_id=f"speech_cycle_{cycle_idx:03d}",
                    set_number=cycle_idx,
                    trial_in_set=0,
                    timestamp=trigger_timestamp,
                )

        # Pain on for 6 minutes (360 seconds)
        self._show_speech_screen_with_timer(360.0, pain_condition, cycle_idx)

        # 1-minute pause (break) between cycles
        self.logger.info("Speech mode: 1-minute pause after cycle %d", cycle_idx)
        self._show_break_screen(60.0)

        self.event_logger.log(
            event_type="pain_cycle_end",
            trial_instance_id="",
            block="speech",
            event_data={"cycle_idx": cycle_idx},
        )

    def _show_speech_screen_with_timer(
        self, duration: float, pain_condition: str, cycle_idx: int
    ) -> None:
        """Show the speech screen with a countdown timer.

        Displays "SPEAK" and the remaining time. Only Q + shutdown code can exit.
        Polls for the shutdown code every 0.1s without blocking the timer.

        Args:
            duration: Duration in seconds (6 minutes = 360).
            pain_condition: Current pain condition being applied.
            cycle_idx: Current cycle index.
        """
        start_time = time.monotonic()
        while True:
            elapsed = time.monotonic() - start_time
            remaining = duration - elapsed
            if remaining <= 0:
                break

            # Show SPEAK screen with countdown
            minutes = int(remaining) // 60
            seconds = int(remaining) % 60
            self.ui.sentence_text.text = (
                f"SPEAK\n\n{minutes:01d}:{seconds:02d}\n\n{pain_condition.upper()}"
            )
            self.ui.sentence_text.pos = (0, 0.05)
            self.ui.sentence_text.height = 0.08
            self.ui.sentence_text.draw()
            self.ui.help_text.text = "Press Q + 12345 to stop"
            self.ui.help_text.draw()
            self.ui.win.flip()

            # Short wait; shutdown code check happens via _check_escape in UI's normal flow,
            # but we bypass it here to avoid ESC handling. We just flip and wait.
            self.ui.core.wait(0.1)

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
        """Run the complete experiment.

        For NORMAL/PRACTICE modes: runs the 5-block vowel experiment.
        For SPEECH mode: runs free speech interview with thermal stimulation
            until researcher enters shutdown code (Q + 12345).

        Note:
            ESC is disabled in speech mode. Only Q + 12345 stops gracefully.
        """
        try:
            self.event_logger.log(
                event_type="experiment_start",
                trial_instance_id="",
                block="",
                event_data={
                    "participant_id": self.config.participant_id,
                    "mode": self.config.mode.value,
                },
            )

            # ------------------------------------------------------------------
            # SPEECH MODE
            # ------------------------------------------------------------------
            if self.config.mode == ExperimentMode.SPEECH:
                self.logger.info("Running in SPEECH mode")
                self.ui.show_instructions(
                    go_segmentation_enabled=False, medoc_enabled=self.medoc_client is not None
                )

                cycle_idx = 0
                while True:
                    pain = self.speech_pain_schedule[
                        cycle_idx % len(self.speech_pain_schedule)
                    ]
                    try:
                        self._run_speech_pain_cycle(pain, cycle_idx)
                    except UserAbort:
                        self.logger.info(
                            "Speech mode: graceful shutdown requested after cycle %d", cycle_idx
                        )
                        break
                    cycle_idx += 1

                self.event_logger.log(
                    event_type="experiment_complete",
                    trial_instance_id="",
                    block="",
                    event_data={"total_cycles": cycle_idx + 1, "mode": "speech"},
                )
                self.ui.show_completion()
                self.save_all_loggers()
                return

            # ------------------------------------------------------------------
            # VOWEL MODE (NORMAL / PRACTICE)
            # ------------------------------------------------------------------
            self.ui.show_instructions(
                go_segmentation_enabled=False, medoc_enabled=self.medoc_client is not None
            )

            for block_num, block_trials in enumerate(self.trials):
                self.logger.info("Starting block %d of %d", block_num + 1, len(self.trials))
                self.run_set(block_num, block_trials)

                # Show 1-minute break between blocks (except after last block)
                if block_num < len(self.trials) - 1:
                    self.logger.info("Starting 1-minute break after block %d", block_num + 1)
                    self._show_break_screen(BLOCK_BREAK_SEC)

            total_trials = sum(len(block) for block in self.trials)
            self.event_logger.log(
                event_type="experiment_complete",
                trial_instance_id="",
                block="",
                event_data={"total_trials": total_trials},
            )

            self.ui.show_completion()
            self.save_all_loggers()

        except UserAbort:
            self.logger.warning("User requested graceful shutdown")
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
