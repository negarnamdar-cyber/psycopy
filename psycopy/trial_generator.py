"""Trial randomization for Medoc experiments.

Generates trial schedules with constrained randomization.

Structure: 5 blocks of 6 trials = 30 total trials.
Pain conditions: xlow (8), low (8), medium (7), high (7) across all blocks.
Each trial is a 60-second vowel task with alternating STOP/GO segments.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TrialConfig:
    """Configuration for a single trial.

    Attributes:
        task_type: Always "vowel"
        pain_condition: "xlow", "low", "medium", or "high"
        num_go_segments: Number of GO segments within the 60s trial (3-7)
        go_segment_durations: Tuple of GO segment durations (seconds), each
            between 3 and 7. Sum must be < 60 so STOP periods can fill the rest.
    """

    task_type: str
    pain_condition: str
    num_go_segments: int
    go_segment_durations: tuple[float, ...]

    def to_dict(self) -> dict[str, str | int | tuple[float, ...]]:
        return asdict(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrialConfig):
            return NotImplemented
        return (
            self.task_type == other.task_type
            and self.pain_condition == other.pain_condition
            and self.num_go_segments == other.num_go_segments
            and self.go_segment_durations == other.go_segment_durations
        )

    def __hash__(self) -> int:
        return hash((self.task_type, self.pain_condition, self.num_go_segments, self.go_segment_durations))


def _generate_go_durations(
    num_segments: int,
    min_seg_sec: float,
    max_seg_sec: float,
    total_max: float,
    rng: random.Random,
) -> tuple[float, ...]:
    """Generate GO segment durations where each is in [min, max] and sum <= total_max.

    With min=3, max=7 and total_max reserved from 60s, the max possible sum
    (7 * 7 = 49) is always < total_max (52 for the worst case of 7 segments).
    Therefore simple uniform random draws in [min, max] always satisfy the
    constraints and no rescaling is needed.

    Args:
        num_segments: Number of GO segments.
        min_seg_sec: Minimum duration per GO segment.
        max_seg_sec: Maximum duration per GO segment.
        total_max: Maximum total sum of all GO durations (unused, kept for API).
        rng: Random instance.

    Returns:
        Tuple of GO segment durations.
    """
    # Simple uniform random draws - with the chosen parameters, the total
    # will always be well under total_max, so no scaling is needed.
    return tuple(round(rng.uniform(min_seg_sec, max_seg_sec), 2) for _ in range(num_segments))


def generate_trials(
    num_sets: int,
    trials_per_set: int,
    num_stop_trials_ratio: float,
    rng: random.Random,
) -> list[list[TrialConfig]]:
    """Generate randomized trial schedule.

    Generates 5 blocks of 6 trials each (30 total). Pain conditions are:
    - xlow: 8 trials
    - low: 8 trials
    - medium: 7 trials
    - high: 7 trials

    Each trial has 3-7 GO segments (each 3-7 seconds) within a 60-second window.
    `num_stop_trials_ratio` and the `num_sets` / `trials_per_set` arguments are
    ignored (retained for API compatibility).

    Args:
        num_sets: Ignored (always 5 blocks).
        trials_per_set: Ignored (always 6 trials per block).
        num_stop_trials_ratio: Retained for API compatibility; ignored.
        rng: Seeded Random instance for reproducibility

    Returns:
        List of blocks, each block is a list of TrialConfig instances.
    """
    num_blocks = 5
    trials_per_block = 6

    # Pain pool: 8 xlow + 8 low + 7 medium + 7 high = 30
    pain_pool = (
        ["xlow"] * 8
        + ["low"] * 8
        + ["medium"] * 7
        + ["high"] * 7
    )
    rng.shuffle(pain_pool)

    all_blocks: list[list[TrialConfig]] = []
    idx = 0

    for _ in range(num_blocks):
        block: list[TrialConfig] = []
        for _ in range(trials_per_block):
            pain = pain_pool[idx]
            idx += 1

            num_go = rng.randint(3, 7)
            # Reserve ~1 s per STOP period so there is always some STOP time.
            # With (num_go + 1) STOP periods, reserve that many seconds.
            reserved_for_stop = float(num_go + 1)
            max_go_total = 60.0 - reserved_for_stop

            go_durs = _generate_go_durations(
                num_segments=num_go,
                min_seg_sec=3.0,
                max_seg_sec=7.0,
                total_max=max_go_total,
                rng=rng,
            )

            block.append(
                TrialConfig(
                    task_type="vowel",
                    pain_condition=pain,
                    num_go_segments=num_go,
                    go_segment_durations=go_durs,
                )
            )

        rng.shuffle(block)
        all_blocks.append(block)

    return all_blocks
