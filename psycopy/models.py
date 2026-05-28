"""Typed runtime models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any


class TaskState(str, Enum):
    GO = "GO"
    STOP = "STOP"


class ExperimentState(str, Enum):
    INIT = "init"
    INSTRUCTIONS = "instructions"
    BLOCK = "block"
    TRIAL = "trial"
    COMPLETE = "complete"
    ABORTED = "aborted"


def generate_trial_instance_id(
    participant_id: str, session_id: str, block_name: str, trial_number: int
) -> str:
    """Generate a unique trial instance ID.

    Format: {participant_id}_{session_id}_{block_name}_{trial_number:03d}
    Example: P001_S01_baseline_003
    """
    return f"{participant_id}_{session_id}_{block_name}_{trial_number:03d}"


def generate_block_id(participant_id: str, session_id: str, block_name: str) -> str:
    """Generate a unique block ID.

    Format: {participant_id}_{session_id}_{block_name}
    Example: P001_S01_baseline
    """
    return f"{participant_id}_{session_id}_{block_name}"


class MarkerType(str, Enum):
    TRIAL_START = "trial_start"
    TRIAL_END = "trial_end"
    RECORDING_START = "recording_start"
    STATE_CHANGE = "state_change"


class EventType(str, Enum):
    """Event types for EventLogger.

    These are the core experiment lifecycle events that use trial_instance_id
    for tracking instead of separate trial number fields.
    """

    BLOCK_START = "block_start"
    BLOCK_END = "block_end"
    TRIAL_START = "trial_start"
    TRIAL_END = "trial_end"
    RECORDING_START = "recording_start"
    RECORDING_END = "recording_end"
    STATE_CHANGE = "state_change"
    STOP_CUE_APPEAR = "stop_cue_appear"


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_type: str
    timestamp: float
    trial_instance_id: str = ""
    block: str = ""
    event_data: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Stimulus:
    trial_id: str
    text: str


@dataclass(frozen=True, slots=True)
class MarkerRecord:
    marker_type: MarkerType
    timestamp: float
    block: str = ""
    trial: int = 0
    state: str = ""
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra_info")
        payload["marker_type"] = self.marker_type.value
        payload.update(extra)
        return payload


class MedocState(IntEnum):
    IDLE = 0
    READY = 1
    TEST_IN_PROGRESS = 2


class MedocTestState(IntEnum):
    IDLE = 0
    RUNNING = 1
    PAUSED = 2
    READY = 3


class MedocResponseCode(IntEnum):
    OK = 0
    ILLEGAL_PARAMETER = 1
    ILLEGAL_STATE = 2
    NOT_PROPER_STATE = 3
    COMM_ERROR = 4096
    SAFETY_WARNING = 8192
    SAFETY_ERROR = 16384


@dataclass(frozen=True, slots=True)
class MedocTrialRecord:
    trial_instance_id: str
    set_number: int
    trial_in_set: int
    task_type: str  # "vowel" or "sentence"
    pain_condition: str  # "xlow", "low", "medium", "high"
    is_stop_trial: bool
    trigger_timestamp: float
    status_timestamp: float | None = None
    temperature_raw: bytes | None = None
    temperature_celsius: float | None = None
    device_state: int | None = None
    test_state: int | None = None
    response_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["temperature_raw"] is not None:
            payload["temperature_raw"] = payload["temperature_raw"].hex()
        return payload
