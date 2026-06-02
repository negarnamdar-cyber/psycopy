"""Trial randomization for Medoc experiments.

Generates trial schedules with constrained randomization.

Default structure: 5 blocks of 1 trial = 5 total trials.
Each trial is a 4-minute (240-second) vowel task with alternating STOP/GO segments.
All trials use the unified Medoc program (experiment 192).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TrialConfig:
    """Configuration for a single trial.

    Attributes:
        task_type: Always "vowel"
        num_go_segments: Number of GO segments within the 240s trial (12-28)
        go_segment_durations: Tuple of GO segment durations (seconds), each
            between 3 and 7. Sum must be < 240 so STOP periods can fill the rest.
    """

    task_type: str
    num_go_segments: int
    go_segment_durations: tuple[float, ...]

    def to_dict(self) -> dict[str, str | int | tuple[float, ...]]:
        return asdict(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrialConfig):
            return NotImplemented
        return (
            self.task_type == other.task_type
            and self.num_go_segments == other.num_go_segments
            and self.go_segment_durations == other.go_segment_durations
        )

    def __hash__(self) -> int:
        return hash((self.task_type, self.num_go_segments, self.go_segment_durations))


def _generate_go_durations(
    num_segments: int,
    min_seg_sec: float,
    max_seg_sec: float,
    total_max: float,
    rng: random.Random,
) -> tuple[float, ...]:
    """Generate GO segment durations where each is in [min, max] and sum <= total_max.

    With min=3, max=7 and total_max reserved from 240s, the max possible sum
    (28 * 7 = 196) is always < total_max (211 for the worst case of 28 segments).
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

    Generates 5 blocks of 1 trial each (5 total).
    Each trial has 12-28 GO segments (3-7 per minute, drawn independently for each
    of the 4 minutes) within a 240-second window.
    All trials use the unified Medoc program (experiment 192).
    `num_stop_trials_ratio` and the `num_sets` / `trials_per_set` arguments are
    ignored (retained for API compatibility).

    Args:
        num_sets: Number of blocks.
        trials_per_set: Number of trials per block.
        num_stop_trials_ratio: Retained for API compatibility; ignored.
        rng: Seeded Random instance for reproducibility

    Returns:
        List of blocks, each block is a list of TrialConfig instances.
    """
    all_blocks: list[list[TrialConfig]] = []

    for _ in range(num_sets):
        block: list[TrialConfig] = []
        for _ in range(trials_per_set):
            # 3-7 GO segments per minute, independently for each of 4 minutes.
            # This guarantees density and avoids edge cases where all segments
            # cluster in one part of the 4-minute trial.
            num_go_per_minute = [rng.randint(3, 7) for _ in range(4)]
            num_go = sum(num_go_per_minute)
            # Reserve ~1 s per STOP period so there is always some STOP time.
            # With (num_go + 1) STOP periods, reserve that many seconds.
            reserved_for_stop = float(num_go + 1)
            max_go_total = 240.0 - reserved_for_stop

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
                    num_go_segments=num_go,
                    go_segment_durations=go_durs,
                )
            )

        rng.shuffle(block)
        all_blocks.append(block)

    return all_blocks
