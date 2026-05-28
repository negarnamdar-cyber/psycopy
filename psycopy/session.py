"""Session paths and batched loggers for reduced I/O."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from psycopy.models import (
    EventRecord,
    EventType,
    MarkerRecord,
    MarkerType,
    MedocTrialRecord,
)
from psycopy.storage import StorageError, atomic_write_csv, atomic_write_json

logger = logging.getLogger("psycopy.session")


@dataclass(frozen=True, slots=True)
class SessionPaths:
    output_dir: Path
    audio_dir: Path
    audio_16k_dir: Path

    events_file: Path
    trials_file: Path
    rt_file: Path
    config_file: Path
    features_file: Path
    features_manifest_file: Path
    vad_file: Path
    blocks_file: Path
    segments_file: Path
    medoc_file: Path


def create_output_directory(config: dict[str, Any]) -> SessionPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    participant = config["participant_id"]
    session = config["session_id"]

    output_dir = Path("data") / f"{timestamp}_sub-{participant}_session-{session}"
    audio_dir = output_dir / "audio"
    audio_16k_dir = output_dir / "audio_16k"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    audio_16k_dir.mkdir(exist_ok=True)

    return SessionPaths(
        output_dir=output_dir,
        audio_dir=audio_dir,
        audio_16k_dir=audio_16k_dir,
        events_file=output_dir / "events.csv",
        trials_file=output_dir / "trials.csv",
        rt_file=output_dir / "rt_trials.csv",
        config_file=output_dir / "config.json",
        features_file=output_dir / f"features_matrix_{participant}.csv",
        features_manifest_file=output_dir / "features_manifest.json",
        vad_file=output_dir / "vad_events.csv",
        blocks_file=output_dir / "blocks.csv",
        segments_file=output_dir / "trial_segments.csv",
        medoc_file=output_dir / "medoc_events.csv",
    )


def save_config_snapshot(
    config: dict[str, Any], metadata: dict[str, Any], config_file: Path
) -> None:
    atomic_write_json(
        config_file,
        {
            **config,
            "metadata": metadata,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


class EventLogger:
    """Logger for experiment events using trial_instance_id.

    Uses a simplified schema focused on trial_instance_id for tracking
    experiment lifecycle events.
    """

    def __init__(self, events_file: Path):
        self.events_file = events_file
        self.events: list[EventRecord] = []
        self.start_time: float | None = None
        self._dirty: bool = False

    def set_start_time(self, timestamp: float | None = None) -> None:
        self.start_time = monotonic() if timestamp is None else timestamp

    def log(
        self,
        event_type: EventType | str,
        trial_instance_id: str = "",
        event_data: dict[str, Any] | None = None,
        block: str = "",
    ) -> EventRecord:
        import json

        timestamp = monotonic()
        if self.start_time is not None:
            timestamp -= self.start_time
        event_type_str = event_type.value if isinstance(event_type, EventType) else event_type
        event_data_str = json.dumps(event_data) if event_data else ""
        record = EventRecord(
            event_type=event_type_str,
            timestamp=timestamp,
            trial_instance_id=trial_instance_id,
            block=block,
            event_data=event_data_str,
        )
        self.events.append(record)
        self._dirty = True
        return record

    def save(self, force: bool = False) -> None:
        if not force and not self._dirty:
            return
        try:
            atomic_write_csv(self.events_file, [event.to_dict() for event in self.events])
            self._dirty = False
        except StorageError as e:
            logger.error("Failed to save events: %s", e)
            raise


class VADLogger:
    """Logger for Voice Activity Detection events per trial."""

    def __init__(self, vad_file: Path):
        self.vad_file = vad_file
        self.events: list[dict[str, Any]] = []
        self._start_time: float | None = None
        self._trial_number: int = 0
        self._trial_instance_id: str = ""
        self._block: str = ""
        self._block_index: int = 0
        self._stop_cue_index: int = 0
        self._dirty: bool = False

    def set_start_time(self, timestamp: float) -> None:
        """Set the reference time for VAD timestamps."""
        self._start_time = timestamp

    def set_trial_number(self, trial_number: int) -> None:
        """Set the current trial number."""
        self._trial_number = trial_number

    def set_block(self, block: str, block_index: int) -> None:
        """Set the current block context for events."""
        self._block = block
        self._block_index = block_index

    def set_context(self, trial_instance_id: str, block: str = "", block_index: int = 0) -> None:
        """Set trial context for VAD events.

        Args:
            trial_instance_id: Unique trial identifier
            block: Block name
            block_index: Block order index
        """
        self._trial_instance_id = trial_instance_id
        self._block = block
        self._block_index = block_index

    def increment_stop_cue(self) -> int:
        """Increment stop cue counter and return the new index.

        Returns:
            The stop_cue_index for this STOP cue (0-indexed)
        """
        self._stop_cue_index += 1
        return self._stop_cue_index - 1

    def log_event(
        self,
        event_type: str,
        timestamp: float,
        is_speech: bool | None = None,
        latency_ms: float | None = None,
        stop_cue_index: int | None = None,
    ) -> None:
        """Log a VAD event.

        Args:
            event_type: 'speech_start', 'speech_end', 'stop_cue'
            timestamp: Absolute timestamp (will be made relative to start)
            is_speech: Whether frame contained speech
            latency_ms: For speech_end after stop cue
            stop_cue_index: Index of the associated STOP cue (for speech_end events)
        """
        relative_time = timestamp - self._start_time if self._start_time is not None else 0.0
        event = {
            "trial_instance_id": self._trial_instance_id,
            "trial_number": self._trial_number,
            "block": self._block,
            "block_index": self._block_index,
            "event_type": event_type,
            "timestamp": timestamp,
            "relative_time": round(relative_time, 3),
            "is_speech": is_speech if is_speech is not None else "",
            "latency_ms": round(latency_ms, 1) if latency_ms is not None else "",
            "stop_cue_index": stop_cue_index if stop_cue_index is not None else "",
        }
        self.events.append(event)
        self._dirty = True

    def log_speech_cessation(self, stop_cue_time: float, speech_end_time: float) -> dict[str, Any]:
        """Log speech cessation latency.

        Returns:
            Dict with stop_cue_time, speech_end_time, latency_ms
        """
        latency_ms = (speech_end_time - stop_cue_time) * 1000
        self.log_event(
            event_type="speech_end",
            timestamp=speech_end_time,
            is_speech=False,
            latency_ms=latency_ms,
            stop_cue_index=self._stop_cue_index - 1 if self._stop_cue_index > 0 else None,
        )
        return {
            "stop_cue_time": stop_cue_time,
            "speech_end_time": speech_end_time,
            "latency_ms": latency_ms,
        }

    def save(self, force: bool = False) -> None:
        """Write events to CSV file. Use force=True for explicit flush."""
        if not force and not self._dirty:
            return
        try:
            atomic_write_csv(self.vad_file, self.events)
            self._dirty = False
        except StorageError as e:
            logger.error("Failed to save VAD events: %s", e)
            raise

    def reset(self) -> None:
        """Reset trial-specific state for new trial.

        Note: Does NOT clear events - they accumulate across trials for the
        final save, matching the behavior of TrialLogger.
        """
        self._trial_number = 0
        self._trial_instance_id = ""
        self._block = ""
        self._block_index = 0
        self._stop_cue_index = 0


class MedocLogger:
    """Logger for Medoc thermode device events.

    Logs trigger and status events from the Medoc device during trials.
    Uses batched writes with _dirty flag pattern.
    """

    HEADER = [
        "trial_instance_id",
        "set_number",
        "trial_in_set",
        "trigger_timestamp",
        "status_timestamp",
        "temperature_raw",
        "temperature_celsius",
        "device_state",
        "test_state",
        "response_code",
    ]

    def __init__(self, medoc_file: Path):
        self.medoc_file = medoc_file
        self.events: list[dict[str, Any]] = []
        self._dirty: bool = False

    def log_trigger(
        self,
        trial_instance_id: str,
        set_number: int,
        trial_in_set: int,
        timestamp: float,
    ) -> None:
        """Log a TRIGGER event sent to the Medoc device.

        Args:
            trial_instance_id: Unique trial identifier
            set_number: Set number (1-N)
            trial_in_set: Trial index within set (1-N)
            timestamp: Trigger timestamp (monotonic time)
        """
        event = {
            "trial_instance_id": trial_instance_id,
            "set_number": set_number,
            "trial_in_set": trial_in_set,
            "trigger_timestamp": str(timestamp),
            "status_timestamp": "",
            "temperature_raw": "",
            "temperature_celsius": "",
            "device_state": "",
            "test_state": "",
            "response_code": "",
        }
        self.events.append(event)
        self._dirty = True

    def log_status(
        self,
        trial_instance_id: str,
        timestamp: float,
        raw_bytes: bytes | None,
        state_dict: dict | None,
    ) -> None:
        trigger_event = None
        for event in reversed(self.events):
            if event["trial_instance_id"] == trial_instance_id and event["status_timestamp"] == "":
                trigger_event = event
                break

        if trigger_event is None:
            trigger_event = {
                "trial_instance_id": trial_instance_id,
                "set_number": "",
                "trial_in_set": "",
                "trigger_timestamp": "",
                "status_timestamp": "",
                "temperature_raw": "",
                "temperature_celsius": "",
                "device_state": "",
                "test_state": "",
                "response_code": "",
            }
            self.events.append(trigger_event)

        temp_raw = raw_bytes.hex() if raw_bytes else ""
        trigger_event["status_timestamp"] = str(timestamp)
        trigger_event["temperature_raw"] = temp_raw

        if state_dict:
            if state_dict.get("temperature_celsius") is not None:
                trigger_event["temperature_celsius"] = state_dict["temperature_celsius"]
            if state_dict.get("device_state") is not None:
                trigger_event["device_state"] = state_dict["device_state"]
            if state_dict.get("test_state") is not None:
                trigger_event["test_state"] = state_dict["test_state"]
            if state_dict.get("response_code") is not None:
                trigger_event["response_code"] = state_dict["response_code"]
        self._dirty = True

    def save(self, force: bool = False) -> None:
        """Write events to CSV file. Use force=True for explicit flush."""
        if not force and not self._dirty:
            return
        if not self.events:
            return
        try:
            atomic_write_csv(self.medoc_file, self.events)
            self._dirty = False
        except StorageError as e:
            logger.error("Failed to save Medoc events: %s", e)
            raise


class MedocTrialLogger:
    """Logger for Medoc trial records with batched writes."""

    def __init__(self, trials_file: Path):
        self.trials_file = trials_file
        self.trials: list[MedocTrialRecord] = []
        self._dirty: bool = False

    def log_trial(self, trial_data: MedocTrialRecord) -> None:
        self.trials.append(trial_data)
        self._dirty = True

    def save(self, force: bool = False) -> None:
        if not force and not self._dirty:
            return
        try:
            atomic_write_csv(self.trials_file, [trial.to_dict() for trial in self.trials])
            self._dirty = False
        except StorageError as e:
            logger.error("Failed to save Medoc trials: %s", e)
            raise
