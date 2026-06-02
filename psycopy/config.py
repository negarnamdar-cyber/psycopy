"""Configuration model and startup dialog."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from psycopy.validation import validate_config

# IPv4 validation regex
_IPV4_REGEX = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")


class ExperimentMode(Enum):
    """Experiment execution mode."""

    NORMAL = "normal"  # Full experiment with Medoc
    PRACTICE_NO_MEDOC = "practice_no_medoc"  # Practice mode without Medoc device
    PRACTICE_WITH_MEDOC = "practice_with_medoc"  # Practice mode with Medoc device
    SPEECH = "speech"  # Free speech interview with thermal stimulation


def _validate_medoc_config(medoc_ip: str, medoc_port: int, medoc_timeout: float) -> None:
    if not _IPV4_REGEX.match(medoc_ip):
        raise ValueError(f"Invalid IP: {medoc_ip}")
    if not (1 <= medoc_port <= 65535):
        raise ValueError(f"Invalid port: {medoc_port}")
    if medoc_timeout <= 0:
        raise ValueError(f"Invalid timeout: {medoc_timeout}")


@dataclass(frozen=True, slots=True)
class MedocConfig:
    medoc_ip: str = "10.196.94.38"
    medoc_port: int = 20121
    medoc_timeout: float = 5.0
    baseline_temp: float = 30.0
    require_connection: bool = True

    def __post_init__(self) -> None:
        _validate_medoc_config(self.medoc_ip, self.medoc_port, self.medoc_timeout)


DEFAULT_SPEECH_QUESTIONS = [
    "How's your pain today?",
    "Tell me about your favorite hobby.",
    "Describe what you had for breakfast.",
    "What are your plans for the weekend?",
    "Tell me about a place you'd like to visit.",
]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    participant_id: str = "001"
    session_id: str = "01"
    random_seed: str = ""
    sample_rate: int = 44100
    fullscreen: bool = True
    # WebRTC VAD settings
    vad_enabled: bool = True  # Enable WebRTC VAD recording
    vad_aggressiveness: int = 2  # VAD sensitivity (0-3, where 3 is most aggressive)
    vad_frame_duration_ms: int = 30  # Frame duration in ms (10, 20, or 30)
    vad_silence_frames: int = 10  # Consecutive silence frames to trigger speech cessation
    vad_target_rate: int = 16000  # Target sample rate for VAD processing
    mode: ExperimentMode = ExperimentMode.NORMAL  # Experiment execution mode
    medoc_config: Optional[MedocConfig] = None
    speech_questions: tuple[str, ...] = tuple(DEFAULT_SPEECH_QUESTIONS)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Convert enum to string for JSON serialization
        if "mode" in data and isinstance(data["mode"], ExperimentMode):
            data["mode"] = data["mode"].value
        # Convert nested MedocConfig to dict if present
        if data.get("medoc_config") is not None and hasattr(data["medoc_config"], "__dataclass_fields__"):
            data["medoc_config"] = asdict(data["medoc_config"])
        return data

    def save(self, filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "ExperimentConfig":
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        # Convert mode string back to enum
        if "mode" in data and isinstance(data["mode"], str):
            data["mode"] = ExperimentMode(data["mode"])
        # Convert medoc_config dict back to MedocConfig if present
        if data.get("medoc_config") is not None and isinstance(data["medoc_config"], dict):
            data["medoc_config"] = MedocConfig(**data["medoc_config"])
        # Convert speech_questions list back to tuple if present
        if "speech_questions" in data and isinstance(data["speech_questions"], list):
            data["speech_questions"] = tuple(data["speech_questions"])
        cfg = cls(**data)
        validate_config(cfg)
        return cfg


def _parse_dialog_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _str_to_bool(value: str) -> bool:
    """Convert string 'True'/'False' to boolean."""
    return value.strip().lower() == "true"


def show_startup_dialog() -> ExperimentConfig:
    """Show startup dialog for participant/session setup with mode selection."""
    from psychopy import core, gui

    defaults = ExperimentConfig()

    # Main configuration dialog
    dialog = gui.Dlg(title="Speech Gating Experiment - Setup")
    dialog.addText("=" * 50)
    dialog.addText("Vowel Speech Gating Experiment")
    dialog.addText("=" * 50)
    dialog.addText("")

    # Participant Information
    dialog.addText("Participant Information")
    dialog.addField("Participant ID:", defaults.participant_id)
    dialog.addField("Session ID:", defaults.session_id)
    dialog.addText("")

    # Experiment Mode Selection
    dialog.addText("Experiment Mode")
    dialog.addField(
        "Mode:",
        choices=[
            "Normal (Full experiment)",
            "Practice (no Medoc device)",
            "Practice (with Medoc device)",
            "Speech interview (free speech + thermal)",
        ],
        initial="Normal (Full experiment)",
    )
    dialog.addText("")

    # Reproducibility
    dialog.addText("Reproducibility")
    dialog.addField("Random Seed (blank = random):", defaults.random_seed)
    dialog.addText("")

    # Display Settings
    dialog.addText("Display Settings")
    dialog.addField(
        "Fullscreen Mode:",
        choices=["True", "False"],
        initial="True",
    )
    dialog.addText("")

    # VAD Settings
    dialog.addText("Voice Activity Detection (VAD)")
    dialog.addField(
        "Enable VAD (Voice Activity Detection):",
        choices=["True", "False"],
        initial="True",
    )
    dialog.addText("")

    # Medoc Pain Device (always configured, mode determines usage)
    dialog.addText("Medoc Device Configuration")
    dialog.addField(
        "Medoc IP Address:",
        "10.196.94.38",
    )
    dialog.addField("Medoc Port:", "20121")
    dialog.addField(
        "Medoc Timeout (sec):",
        "5.0",
    )

    values = dialog.show()
    if not dialog.OK or values is None:
        core.quit()
        raise SystemExit()  # Ensures type checker knows we exit

    # Type narrowing: values is guaranteed to be a list here
    assert values is not None  # for type checker

    participant_id = str(values[0]).strip() if values[0] else "001"
    session_id = str(values[1]).strip() if values[1] else "01"
    mode_str = str(values[2]).strip() if values[2] else "Normal (Full experiment)"
    random_seed = str(values[3]).strip() if values[3] else ""
    fullscreen = _str_to_bool(str(values[4])) if values[4] else True
    vad_enabled = _str_to_bool(str(values[5])) if values[5] else True
    medoc_ip = str(values[6]).strip() if values[6] else "192.168.1.100"
    medoc_port = int(values[7]) if values[7] else 5000
    medoc_timeout = float(values[8]) if values[8] else 5.0

    # Determine mode
    mode_str_lower = mode_str.lower()
    if "speech" in mode_str_lower:
        mode = ExperimentMode.SPEECH
    elif "no medoc" in mode_str_lower:
        mode = ExperimentMode.PRACTICE_NO_MEDOC
    elif "with medoc" in mode_str_lower:
        mode = ExperimentMode.PRACTICE_WITH_MEDOC
    else:
        mode = ExperimentMode.NORMAL

    # MedocConfig is always created, but require_connection depends on mode
    # - NORMAL: require_connection=True (must connect)
    # - PRACTICE_NO_MEDOC: medoc_config=None (no Medoc at all)
    # - PRACTICE_WITH_MEDOC / SPEECH: require_connection=False (try to connect, but don't fail)
    if mode == ExperimentMode.PRACTICE_NO_MEDOC:
        medoc_config = None
    else:
        medoc_config = MedocConfig(
            medoc_ip=medoc_ip,
            medoc_port=medoc_port,
            medoc_timeout=medoc_timeout,
            require_connection=(mode == ExperimentMode.NORMAL),
        )

    config = ExperimentConfig(
        participant_id=participant_id,
        session_id=session_id,
        random_seed=random_seed,
        fullscreen=fullscreen,
        vad_enabled=vad_enabled,
        mode=mode,
        medoc_config=medoc_config,
    )
    validate_config(config)
    return config


# Backward-compatible exports expected by existing callers/tests.
from psycopy.schedule import generate_schedule, get_rng  # noqa: E402
