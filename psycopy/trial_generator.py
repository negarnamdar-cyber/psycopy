"""Trial randomization for Medoc experiments.

Generates trial schedules with constrained randomization.

Supported configurations:
- 12-trial sets (legacy): 6 vowel + 6 sentence, 3 each of xlow/low/medium/high per set.
- 8-trial sets: 4 vowel + 4 sentence (one vowel+one sentence per pain level).

STOP/no-go allocation is handled per-configuration; for 8-trial sets we default
to ~25% stop trials per set.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TrialConfig:
    """Configuration for a single trial.

    Attributes:
        task_type: "vowel" or "sentence"
        pain_condition: "xlow", "low", "medium", or "high"
        is_stop_trial: Whether this is a STOP trial
    """

    task_type: str
    pain_condition: str
    is_stop_trial: bool

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrialConfig):
            return NotImplemented
        return (
            self.task_type == other.task_type
            and self.pain_condition == other.pain_condition
            and self.is_stop_trial == other.is_stop_trial
        )

    def __hash__(self) -> int:
        return hash((self.task_type, self.pain_condition, self.is_stop_trial))


def generate_trials(
    num_sets: int,
    trials_per_set: int,
    num_stop_trials_ratio: float,
    rng: random.Random,
) -> list[list[TrialConfig]]:
    """Generate randomized trial schedule with constraints.

        Supported per-set layouts:
        - 12-trial sets (legacy): 6 vowel + 6 sentence and 3 each of
            xlow/low/medium/high pain conditions.
        - 8-trial sets: 4 vowel + 4 sentence with one trial per pain level
            for each task type.

        STOP/no-go allocation is handled per-configuration (see implementation).

    Args:
        num_sets: Number of trial sets (typically 8)
        trials_per_set: Number of trials per set (supported: 8 or 12)
        num_stop_trials_ratio: Retained for backward compatibility; ignored.
        rng: Seeded Random instance for reproducibility

    Returns:
        List of sets, each set is a list of TrialConfig instances

    Raises:
        ValueError: If unsupported `trials_per_set` is provided
    """
    all_sets: list[list[TrialConfig]] = []

    # Support two common configurations: 12-trial sets (legacy) and 8-trial sets
    # (one vowel+one sentence per pain level). Other values are rejected.
    if trials_per_set == 12:
        sentence_stop_per_set = 3
        vowel_trials_per_set = 6
        total_vowel_trials = num_sets * vowel_trials_per_set
        total_vowel_stop_trials = round(total_vowel_trials * 0.75)
        vowel_stop_per_set = total_vowel_stop_trials // num_sets
        vowel_stop_remainder = total_vowel_stop_trials % num_sets

        for set_idx in range(num_sets):
            # Build task list: exactly 6 vowel + 6 sentence
            task_list = ["vowel"] * 6 + ["sentence"] * 6
            rng.shuffle(task_list)

            # Build pain list: exactly 3 each of xlow/low/medium/high
            pain_list = ["xlow"] * 3 + ["low"] * 3 + ["medium"] * 3 + ["high"] * 3
            rng.shuffle(pain_list)

            set_trials = [
                TrialConfig(
                    task_type=task_list[i],
                    pain_condition=pain_list[i],
                    is_stop_trial=False,
                )
                for i in range(trials_per_set)
            ]

            sentence_indices = [idx for idx, trial in enumerate(set_trials) if trial.task_type == "sentence"]
            vowel_indices = [idx for idx, trial in enumerate(set_trials) if trial.task_type == "vowel"]

            rng.shuffle(sentence_indices)
            rng.shuffle(vowel_indices)

            vowel_stop_this_set = vowel_stop_per_set + (1 if set_idx < vowel_stop_remainder else 0)
            stop_indices = set(sentence_indices[:sentence_stop_per_set] + vowel_indices[:vowel_stop_this_set])

            set_trials = [
                TrialConfig(
                    task_type=trial.task_type,
                    pain_condition=trial.pain_condition,
                    is_stop_trial=(idx in stop_indices),
                )
                for idx, trial in enumerate(set_trials)
            ]

            all_sets.append(set_trials)

        return all_sets

    if trials_per_set == 8:
        # Create a balanced set: for each pain level, include one vowel and one sentence trial
        pain_levels = ["xlow", "low", "medium", "high"]
        for _ in range(num_sets):
            set_trials = [
                TrialConfig(task_type=task, pain_condition=p, is_stop_trial=False)
                for p in pain_levels
                for task in ("vowel", "sentence")
            ]
            rng.shuffle(set_trials)

            # Allocate stop trials: default to ~25% of trials per set (rounded)
            num_stop = max(1, round(trials_per_set * 0.25))
            stop_indices = set(rng.sample(range(trials_per_set), num_stop))

            set_trials = [
                TrialConfig(
                    task_type=trial.task_type,
                    pain_condition=trial.pain_condition,
                    is_stop_trial=(idx in stop_indices),
                )
                for idx, trial in enumerate(set_trials)
            ]

            all_sets.append(set_trials)

        return all_sets

    raise ValueError(
        f"Unsupported trials_per_set {trials_per_set}. Supported values: 8 or 12."
    )
