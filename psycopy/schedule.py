"""Pure schedule generation functions.

Probabilistic GO/STOP scheduling:
- Trial starts with GO segment
- Each ~1 second during GO, check prob_stop_per_sec to trigger STOP
- STOP duration sampled from [min_segment_sec, max_segment_sec]
- After STOP, return to GO and continue until trial_duration_sec
"""

from __future__ import annotations

import random
from typing import Any

from psycopy.config import ExperimentConfig
from psycopy.types import TaskState


def get_rng(config: ExperimentConfig) -> random.Random:
    if config.random_seed and config.random_seed.strip():
        return random.Random(int(config.random_seed))
    return random.Random()


def generate_schedule(
    config: ExperimentConfig, rng: random.Random | None = None, *, duration_sec: float | None = None
) -> list[dict[str, Any]]:
    """Generate a probabilistic GO/STOP schedule.

    Starts with GO segment. Each second during GO, checks prob_stop_per_sec
    to determine if STOP should trigger. When STOP triggers, samples duration
    from [min_segment_sec, max_segment_sec]. Continues alternating until
    trial duration is reached.

    Args:
        config: Experiment configuration with timing parameters
        rng: Random number generator (optional, created if not provided)
        duration_sec: Override for trial duration in seconds (optional, defaults to config.trial_duration_sec)

    Returns:
        List of segment dicts with 'state', 'start', 'end', 'duration' keys
    """
    if rng is None:
        rng = random.Random()

    target = duration_sec if duration_sec is not None else config.trial_duration_sec

    # No segmentation: single continuous GO window
    if not config.go_segmentation_enabled:
        return [
            {
                "state": TaskState.GO.value,
                "start": 0.0,
                "end": round(target, 4),
                "duration": round(target, 4),
            }
        ]

    prob_stop = config.prob_stop_per_sec
    min_seg = config.min_segment_sec
    max_seg = config.max_segment_sec

    schedule: list[dict[str, Any]] = []
    t = 0.0

    while t < target:
        remaining = target - t

        # --- GO PHASE ---
        go_start = t

        # Need enough time for min GO + min STOP to consider a switch
        if remaining < 2 * min_seg:
            # Not enough room for proper GO+STOP alternation, run GO to end
            schedule.append(
                {
                    "state": TaskState.GO.value,
                    "start": round(go_start, 4),
                    "end": round(target, 4),
                }
            )
            t = target
            continue

        # Find when STOP should trigger (if at all)
        # Check each second after min_seg of GO, leaving room for min STOP
        earliest_stop_check = t + min_seg
        latest_stop_check = target - min_seg  # Need room for STOP

        stop_time = None
        check_time = earliest_stop_check

        while check_time < latest_stop_check:
            if rng.random() < prob_stop:
                stop_time = check_time
                break
            check_time += 1.0

        if stop_time is not None:
            # Add GO segment up to STOP trigger
            schedule.append(
                {
                    "state": TaskState.GO.value,
                    "start": round(go_start, 4),
                    "end": round(stop_time, 4),
                }
            )

            # --- STOP PHASE ---
            # Sample STOP duration, clamp to remaining time
            raw_stop_duration = rng.uniform(min_seg, max_seg)
            stop_end = min(stop_time + raw_stop_duration, target)

            schedule.append(
                {
                    "state": TaskState.STOP.value,
                    "start": round(stop_time, 4),
                    "end": round(stop_end, 4),
                }
            )
            t = stop_end
            # Loop continues - next iteration starts in GO state
        else:
            # No STOP triggered, GO runs to end of trial
            schedule.append(
                {
                    "state": TaskState.GO.value,
                    "start": round(go_start, 4),
                    "end": round(target, 4),
                }
            )
            t = target

    # Calculate durations for all segments
    for seg in schedule:
        seg["duration"] = round(seg["end"] - seg["start"], 4)

    return schedule
