"""Medoc experiment controller for thermal stimulation with vowel production.

Integrates Medoc thermode device communication with vowel speech
experiment infrastructure. Manages trial randomization, device communication,
VAD coordination, and data logging.

Two experiment modes are supported:

1. Vowel mode (NORMAL / PRACTICE):
   - 5 blocks of 1 trial each = 5 total trials
   - Each trial: 4 minutes (240 seconds) of alternating STOP/GO segments
   - 1-minute break between blocks
   - Total experiment time: ~25 minutes

2. Speech mode (SPEECH):
   - Free speech interview with thermal stimulation
   - Repeating cycles of 4 minutes speaking + 1 minute break
   - Continues until the researcher triggers graceful shutdown (Q + 12345)
   - Continuous unified Medoc program (experiment 192)
   - ESC is disabled — only coded shutdown works
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from random import Random
from typing import Any

from psycopy.audio import AudioService
from psycopy.config import ExperimentConfig, ExperimentMode
from psycopy.medoc import MedocClient, MedocConnectionError
from psycopy.models import MedocTrialRecord, TaskState, generate_trial_instance_id
from psycopy.schedule import get_rng
from psycopy.session import (
    EventLogger,
    MedocLogger,
    MedocTrialLogger,
    create_output_directory,
    save_config_snapshot,
)
from psycopy.runtime import PsychoPyUI, UserAbort
from psycopy.trial_generator import TrialConfig, generate_trials
from psycopy.validation import validate_config

TRIAL_DURATION_SEC = 240.0
BLOCK_BREAK_SEC = 60.0
VOWEL_TEXT = "Ahh"

logger = logging.getLogger("psycopy.medoc_experiment")


class TrialState(Enum):
    PENDING = "pending"
    TRIGGER_SENT = "trigger_sent"
    STATUS_SENT = "status_sent"
    COMPLETED = "completed"
    FAILED = "failed"


class MedocExperiment:
    """Experiment controller for Medoc thermal stimulation with vowel production."""

    def __init__(self, config: ExperimentConfig) -> None:
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
        self.medoc_logger = MedocLogger(self.session_paths.medoc_file)

        self.audio = AudioService(sample_rate=config.sample_rate, retries=2)
        self.audio.preflight()

        if config.medoc_config is None:
            logger.info("Medoc disabled - running in testing mode.")
            self.medoc_client = None
        else:
            try:
                self.medoc_client = MedocClient(config.medoc_config)
            except MedocConnectionError:
                if config.medoc_config.require_connection:
                    raise
                logger.warning("Medoc device not connected - running in testing mode.")
                self.medoc_client = None

        self.ui = PsychoPyUI(fullscreen=config.fullscreen)

        num_blocks = (
            1
            if config.mode in (ExperimentMode.PRACTICE_NO_MEDOC, ExperimentMode.PRACTICE_WITH_MEDOC)
            else 5
        )

        if config.mode == ExperimentMode.SPEECH:
            self.trials = []
        else:
            self.trials = generate_trials(
                num_sets=num_blocks,
                trials_per_set=1,
                num_stop_trials_ratio=0.0,
                rng=self.rng,
            )

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
            self.logger.info("Speech mode: no structured trials.")

    def _display_state(self, state: TaskState, stimulus_text: str) -> None:
        self.ui.apply_state(state)
        self.ui.sentence_text.text = stimulus_text
        self.ui.sentence_text.draw()
        self.ui.state_background.draw()
        self.ui.state_indicator.draw()
        self.ui.win.flip()

    def run_trial(
        self,
        set_num: int,
        trial_num: int,
        trial_config: TrialConfig,
        client: MedocClient | None = None,
    ) -> MedocTrialRecord:
        trial_instance_id = generate_trial_instance_id(
            self.config.participant_id,
            self.config.session_id,
            f"block{set_num}",
            trial_num,
        )

        audio_filename = f"sub-{self.config.participant_id}_block-{set_num}_trial-{trial_num:03d}.wav"
        audio_path = self.session_paths.audio_dir / audio_filename

        self.event_logger.log(
            event_type="trial_start",
            trial_instance_id=trial_instance_id,
            block=f"block{set_num}",
            event_data={
                "task_type": trial_config.task_type,
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
        audio_started = False

        try:
            trigger_timestamp = time.monotonic()

            if client is not None:
                self.medoc_logger.log_trigger(
                    trial_instance_id=trial_instance_id,
                    set_number=set_num,
                    trial_in_set=trial_num,
                    timestamp=trigger_timestamp,
                )
                self.logger.info(
                    "Trial %d.%d started at %.3f",
                    set_num,
                    trial_num,
                    trigger_timestamp,
                )
            else:
                self.logger.warning(
                    "Skipping Medoc communication for trial %d.%d - testing mode",
                    set_num,
                    trial_num,
                )

            self.audio.start(audio_path)
            audio_started = True
            self.currently_recording = True

            self.event_logger.log(
                event_type="recording_start",
                trial_instance_id=trial_instance_id,
                block=f"block{set_num}",
                event_data={"audio_type": "vowel"},
            )

            total_go_time = sum(trial_config.go_segment_durations)
            num_stop_periods = trial_config.num_go_segments + 1
            stop_duration = (TRIAL_DURATION_SEC - total_go_time) / num_stop_periods

            # Poll Medoc every 5 seconds during trials for higher-resolution
            # temperature data that can be matched to individual GO/STOP segments.
            next_poll_at = 5.0
            current_temperature: float | None = None

            for seg_idx, go_duration in enumerate(trial_config.go_segment_durations):
                # --- Poll before STOP cue if interval hit -------------------
                elapsed = time.monotonic() - trigger_timestamp
                if client is not None and elapsed >= next_poll_at:
                    status = self._poll_and_log(
                        client,
                        trial_instance_id,
                        set_num,
                        trial_num,
                        time.monotonic(),
                    )
                    if status is not None:
                        current_temperature = status.get("temperature_celsius")
                        temperature_celsius = current_temperature
                        temperature_raw = status.get("raw_bytes")
                        device_state = status.get("device_state")
                        test_state = status.get("test_state")
                        response_code = status.get("response_code")
                    next_poll_at += 5.0

                # Log STOP cue with most recent temperature
                stop_cue_ts = time.monotonic()
                self.event_logger.log(
                    event_type="stop_cue",
                    trial_instance_id=trial_instance_id,
                    block=f"block{set_num}",
                    event_data={
                        "segment_index": seg_idx,
                        "cue_type": "stop",
                        "cue_duration_sec": round(stop_duration, 3),
                        "trial_elapsed_sec": round(stop_cue_ts - trigger_timestamp, 3),
                        "temperature_celsius": current_temperature,
                    },
                )
                self._display_state(TaskState.STOP, VOWEL_TEXT)
                self.ui.wait(stop_duration)

                # --- Poll during STOP if interval hit ----------------------
                elapsed = time.monotonic() - trigger_timestamp
                if client is not None and elapsed >= next_poll_at:
                    status = self._poll_and_log(
                        client,
                        trial_instance_id,
                        set_num,
                        trial_num,
                        time.monotonic(),
                    )
                    if status is not None:
                        current_temperature = status.get("temperature_celsius")
                        temperature_celsius = current_temperature
                        temperature_raw = status.get("raw_bytes")
                        device_state = status.get("device_state")
                        test_state = status.get("test_state")
                        response_code = status.get("response_code")
                    next_poll_at += 5.0

                # Log GO cue with most recent temperature
                go_cue_ts = time.monotonic()
                self.event_logger.log(
                    event_type="go_cue",
                    trial_instance_id=trial_instance_id,
                    block=f"block{set_num}",
                    event_data={
                        "segment_index": seg_idx,
                        "cue_type": "go",
                        "cue_duration_sec": round(go_duration, 3),
                        "trial_elapsed_sec": round(go_cue_ts - trigger_timestamp, 3),
                        "temperature_celsius": current_temperature,
                    },
                )
                self._display_state(TaskState.GO, VOWEL_TEXT)
                self.ui.wait(go_duration)

                self.logger.debug(
                    "Trial %d.%d segment %d: GO for %.2fs temp=%s°C",
                    set_num,
                    trial_num,
                    seg_idx,
                    go_duration,
                    current_temperature,
                )

            # Final STOP cue
            final_stop_ts = time.monotonic()
            self.event_logger.log(
                event_type="stop_cue",
                trial_instance_id=trial_instance_id,
                block=f"block{set_num}",
                event_data={
                    "segment_index": trial_config.num_go_segments,
                    "cue_type": "stop",
                    "cue_duration_sec": round(stop_duration, 3),
                    "trial_elapsed_sec": round(final_stop_ts - trigger_timestamp, 3),
                    "temperature_celsius": current_temperature,
                },
            )
            self._display_state(TaskState.STOP, VOWEL_TEXT)

            elapsed = time.monotonic() - trigger_timestamp
            remaining = TRIAL_DURATION_SEC - elapsed
            if remaining > 0:
                self.ui.wait(remaining)

            # Final poll at end of trial
            if client is not None:
                status = self._poll_and_log(
                    client,
                    trial_instance_id,
                    set_num,
                    trial_num,
                    time.monotonic(),
                )
                if status is not None:
                    temperature_celsius = status.get("temperature_celsius")
                    temperature_raw = status.get("raw_bytes")
                    device_state = status.get("device_state")
                    test_state = status.get("test_state")
                    response_code = status.get("response_code")

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

        except UserAbort:
            self.logger.info("Trial %d.%d interrupted by graceful shutdown", set_num, trial_num)
            raise
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
            "Trial %d.%d complete: segments=%d duration=%.3fs",
            set_num,
            trial_num,
            trial_config.num_go_segments,
            actual_duration,
        )

        return MedocTrialRecord(
            trial_instance_id=trial_instance_id,
            set_number=set_num,
            trial_in_set=trial_num,
            task_type=trial_config.task_type,
            is_stop_trial=False,
            trigger_timestamp=trigger_timestamp,
            status_timestamp=status_timestamp,
            temperature_raw=temperature_raw,
            temperature_celsius=temperature_celsius,
            device_state=device_state,
            test_state=test_state,
            response_code=response_code,
        )

    def run_set(self, set_num: int, trials: list[TrialConfig], client: MedocClient | None = None) -> None:
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
                record = self.run_trial(set_num, trial_num, trial_config, client=client)
                self.trial_logger.log_trial(record)
                self.logger.info(
                    "Logged trial %d.%d: segments=%d",
                    set_num,
                    trial_num,
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
        start_time = time.monotonic()
        last_displayed_second: int | None = None
        while True:
            elapsed = time.monotonic() - start_time
            remaining = duration - elapsed
            if remaining <= 0:
                break

            displayed_second = int(remaining)
            if displayed_second != last_displayed_second:
                message = (
                    "BREAK\n\nRest your voice.\n\n"
                    f"Next block in: {displayed_second}s"
                )
                self.ui.instruction_text.text = message
                self.ui.instruction_text.draw()
                self.ui.help_text.draw()
                self.ui.win.flip()
                last_displayed_second = displayed_second

            self.ui._check_escape()
            self.ui.core.wait(0.1)

    def _poll_and_log(
        self,
        client: MedocClient,
        trial_instance_id: str,
        set_number: int,
        trial_in_set: int,
        timestamp: float,
    ) -> dict[str, Any] | None:
        """Poll Medoc for temperature/state, log the result, and return status."""
        try:
            status = client.poll_status()
            self.medoc_logger.log_poll(
                trial_instance_id=trial_instance_id,
                set_number=set_number,
                trial_in_set=trial_in_set,
                timestamp=timestamp,
                raw_bytes=status.get("raw_bytes"),
                state_dict=status,
            )
            self.logger.debug(
                "Polled Medoc: temp=%s°C state=%s",
                status.get("temperature_celsius"),
                status.get("device_state"),
            )
            return status
        except Exception as exc:
            self.logger.warning("Medoc poll failed: %s", exc)
            return None

    def _run_speech_pain_cycle(
        self,
        cycle_idx: int,
        client: MedocClient | None,
    ) -> None:
        self.logger.info("Speech mode: starting pain cycle %d", cycle_idx)
        self.event_logger.log(
            event_type="pain_cycle_start",
            trial_instance_id="",
            block="speech",
            event_data={"cycle_idx": cycle_idx},
        )

        trigger_timestamp = time.monotonic()
        trial_instance_id = f"speech_cycle_{cycle_idx:03d}"

        if client is not None:
            self.medoc_logger.log_trigger(
                trial_instance_id=trial_instance_id,
                set_number=cycle_idx,
                trial_in_set=0,
                timestamp=trigger_timestamp,
            )

        self._show_speech_screen_with_timer(240.0, cycle_idx, client, trial_instance_id)

        self.logger.info("Speech mode: 1-minute pause after cycle %d", cycle_idx)
        self._show_break_screen(60.0)

        self.event_logger.log(
            event_type="pain_cycle_end",
            trial_instance_id="",
            block="speech",
            event_data={"cycle_idx": cycle_idx},
        )

    def _show_speech_screen_with_timer(
        self,
        duration: float,
        cycle_idx: int,
        client: MedocClient | None,
        trial_instance_id: str,
    ) -> None:
        start_time = time.monotonic()
        next_poll_at = 30.0
        last_displayed_second: int | None = None
        while True:
            elapsed = time.monotonic() - start_time
            remaining = duration - elapsed
            if remaining <= 0:
                break

            if client is not None and elapsed >= next_poll_at:
                self._poll_and_log(
                    client,
                    trial_instance_id,
                    cycle_idx,
                    0,
                    time.monotonic(),
                )
                next_poll_at += 30.0

            displayed_second = int(remaining)
            if displayed_second != last_displayed_second:
                minutes = displayed_second // 60
                seconds = displayed_second % 60
                self.ui.sentence_text.text = f"SPEAK\n\n{minutes:01d}:{seconds:02d}"
                self.ui.sentence_text.pos = (0, 0.05)
                self.ui.sentence_text.height = 0.08
                self.ui.sentence_text.draw()
                self.ui.help_text.text = "Press Q + 12345 to stop"
                self.ui.help_text.draw()
                self.ui.win.flip()
                last_displayed_second = displayed_second

            self.ui._check_escape()
            self.ui.core.wait(0.1)

    def save_all_loggers(self) -> None:
        self.trial_logger.save()
        self.medoc_logger.save()
        self.event_logger.save()
        self.logger.info("All loggers saved to disk")

    def run(self) -> None:
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

            # SPEECH MODE
            if self.config.mode == ExperimentMode.SPEECH:
                self.logger.info("Running in SPEECH mode")
                self.ui.show_speech_instructions(medoc_enabled=self.medoc_client is not None)

                speech_audio_path = (
                    self.session_paths.audio_dir
                    / f"sub-{self.config.participant_id}_speech.wav"
                )
                self.audio.start(speech_audio_path)
                self.currently_recording = True
                self.event_logger.log(
                    event_type="recording_start",
                    trial_instance_id="",
                    block="speech",
                    event_data={"audio_type": "speech"},
                )

                cycle_idx = 0
                while True:
                    speech_client = self.medoc_client

                    if speech_client is not None:
                        try:
                            speech_client.connect()
                            speech_client.send_unified_program()
                            self.logger.info(
                                "Medoc unified program 11000000 started for speech cycle %d",
                                cycle_idx,
                            )
                        except Exception as exc:
                            self.logger.warning(
                                "Failed to start unified Medoc program for cycle %d: %s",
                                cycle_idx,
                                exc,
                            )

                    try:
                        self._run_speech_pain_cycle(cycle_idx, speech_client)
                    except UserAbort:
                        self.logger.info(
                            "Speech mode: graceful shutdown requested after cycle %d", cycle_idx
                        )
                        break
                    finally:
                        if speech_client is not None:
                            try:
                                speech_client.stop_unified_program()
                                speech_client.disconnect()
                            except Exception as exc:
                                self.logger.warning("Error disconnecting Medoc: %s", exc)

                    cycle_idx += 1

                if self.currently_recording:
                    try:
                        self.audio.stop()
                        self.currently_recording = False
                        self.event_logger.log(
                            event_type="recording_end",
                            trial_instance_id="",
                            block="speech",
                        )
                    except Exception as exc:
                        self.logger.warning("Error stopping audio: %s", exc)

                self.event_logger.log(
                    event_type="experiment_complete",
                    trial_instance_id="",
                    block="",
                    event_data={"total_cycles": cycle_idx + 1, "mode": "speech"},
                )
                self.ui.show_completion()
                self.save_all_loggers()
                return

            # VOWEL MODE (NORMAL / PRACTICE)
            self.ui.show_instructions(
                go_segmentation_enabled=True, medoc_enabled=self.medoc_client is not None
            )

            for block_num, block_trials in enumerate(self.trials):
                vowel_client = self.medoc_client
                if vowel_client is not None:
                    try:
                        vowel_client.connect()
                        vowel_client.send_unified_program()
                        self.logger.info(
                            "Medoc unified program 11000000 started for block %d",
                            block_num,
                        )
                    except Exception as exc:
                        self.logger.warning(
                            "Failed to start unified Medoc program for block %d: %s",
                            block_num,
                            exc,
                        )

                self.logger.info("Starting block %d of %d", block_num + 1, len(self.trials))
                self.run_set(block_num, block_trials, client=vowel_client)

                if vowel_client is not None:
                    try:
                        vowel_client.stop_unified_program()
                        vowel_client.disconnect()
                        self.logger.info("Medoc disconnected before break")
                    except Exception as exc:
                        self.logger.warning("Error disconnecting Medoc: %s", exc)

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


def configure_logging(output_dir) -> logging.Logger:
    from psycopy.runtime_logging import configure_logging as _configure_logging
    return _configure_logging(output_dir)


def get_run_metadata(app_version: str) -> dict:
    from psycopy.runtime_logging import get_run_metadata as _get_run_metadata
    return _get_run_metadata(app_version)
