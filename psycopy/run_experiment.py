from __future__ import annotations

import argparse
import sys

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
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.practice_no_medoc:
        mode = ExperimentMode.PRACTICE_NO_MEDOC
    elif args.practice_with_medoc:
        mode = ExperimentMode.PRACTICE_WITH_MEDOC
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

    return ExperimentConfig(
        participant_id=args.participant_id,
        session_id=args.session_id,
        random_seed=args.random_seed,
        fullscreen=parse_bool(args.fullscreen),
        vad_enabled=parse_bool(args.vad_enabled),
        mode=mode,
        medoc_config=medoc_config,
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
    print("Experiment initialized. Press ESC anytime to abort.")

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
