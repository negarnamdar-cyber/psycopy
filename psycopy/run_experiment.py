from __future__ import annotations

import argparse
import sys
from pathlib import Path

from psycopy.config import ExperimentConfig, ExperimentMode, MedocConfig, show_startup_dialog
from psycopy.medoc_experiment import MedocExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the psycopy speech experiment with optional Medoc device support."
    )

    parser.add_argument(
        "--no-dialog",
        action="store_true",
        help="Use command-line arguments instead of the PsychoPy startup dialog.",
    )
    parser.add_argument(
        "--normal",
        action="store_true",
        help="Run full experiment mode with Medoc required.",
    )
    parser.add_argument(
        "--practice-no-medoc",
        action="store_true",
        help="Run practice mode without Medoc.",
    )
    parser.add_argument(
        "--practice-with-medoc",
        action="store_true",
        help="Run practice mode and attempt to connect to Medoc if available.",
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        help="Run speech Q&A mode with thermal stimulation (20 s per question, 4-min blocks, 5 blocks).",
    )
    parser.add_argument("--participant-id", default="001", help="Participant identifier.")
    parser.add_argument("--session-id", default="01", help="Session identifier.")
    parser.add_argument("--random-seed", default="", help="Optional random seed for reproducibility.")
    parser.add_argument(
        "--fullscreen",
        choices=["true", "false"],
        default="false",
        help="Run PsychoPy in fullscreen mode.",
    )
    parser.add_argument(
        "--vad-enabled",
        choices=["true", "false"],
        default="true",
        help="Enable WebRTC VAD if available.",
    )
    parser.add_argument("--medoc-ip", default="10.196.94.38", help="Medoc device IP address.")
    parser.add_argument("--medoc-port", type=int, default=20121, help="Medoc device TCP port.")
    parser.add_argument(
        "--medoc-timeout",
        type=float,
        default=5.0,
        help="Medoc socket timeout in seconds.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="Path to a text file with one speech question per line (speech mode only).",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.practice_no_medoc:
        mode = ExperimentMode.PRACTICE_NO_MEDOC
    elif args.practice_with_medoc:
        mode = ExperimentMode.PRACTICE_WITH_MEDOC
    elif args.speech:
        mode = ExperimentMode.SPEECH
    elif args.normal:
        mode = ExperimentMode.NORMAL
    else:
        # Default to practice mode without Medoc so the experiment is safe to run
        # if the Medoc device is not available.
        mode = ExperimentMode.PRACTICE_NO_MEDOC

    medoc_config = None
    if mode != ExperimentMode.PRACTICE_NO_MEDOC:
        medoc_config = MedocConfig(
            medoc_ip=args.medoc_ip,
            medoc_port=args.medoc_port,
            medoc_timeout=args.medoc_timeout,
            require_connection=(mode == ExperimentMode.NORMAL),
        )

    speech_questions = None
    if args.questions_file is not None:
        lines = [
            line.strip()
            for line in args.questions_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if lines:
            speech_questions = tuple(lines)

    return ExperimentConfig(
        participant_id=args.participant_id,
        session_id=args.session_id,
        random_seed=args.random_seed,
        fullscreen=parse_bool(args.fullscreen),
        vad_enabled=parse_bool(args.vad_enabled),
        mode=mode,
        medoc_config=medoc_config,
        speech_questions=speech_questions,
    )


def main() -> int:
    args = parse_args()

    if args.no_dialog:
        config = build_config(args)
    else:
        config = show_startup_dialog()

    print("Initializing experiment with config:")
    print(config.to_dict())

    experiment = MedocExperiment(config)
    if config.mode == ExperimentMode.SPEECH:
        print("Experiment initialized. Speech mode: press Q + 12345 to stop gracefully.")
    else:
        print("Experiment initialized. Press Q + 12345 for graceful shutdown.")

    try:
        experiment.run()
        print("Experiment complete. Data saved under the generated data directory.")
        return 0
    except KeyboardInterrupt:
        print("Experiment interrupted by keyboard.")
        return 1
    except Exception as exc:
        print(f"Experiment failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
