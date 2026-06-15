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
        task_type: "vowel" or "speech"
        num_go_segments: Number of GO segments within the 240s trial
        go_segment_durations: Tuple of GO segment durations (seconds).
            Sum must be < 240 so STOP periods can fill the rest.
        segment_texts: Optional tuple of stimulus text for each GO segment.
            Used by speech mode to show questions during GO periods.
        stop_segment_durations: Optional tuple of explicit STOP segment
            durations (seconds).  Empty means "calculate automatically"
            (used by vowel mode).  When provided, the final STOP fills
            the remainder of the 240-second trial.
    """

    task_type: str
    num_go_segments: int
    go_segment_durations: tuple[float, ...]
    segment_texts: tuple[str, ...] = ()
    stop_segment_durations: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrialConfig):
            return NotImplemented
        return (
            self.task_type == other.task_type
            and self.num_go_segments == other.num_go_segments
            and self.go_segment_durations == other.go_segment_durations
            and self.segment_texts == other.segment_texts
            and self.stop_segment_durations == other.stop_segment_durations
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.task_type,
                self.num_go_segments,
                self.go_segment_durations,
                self.segment_texts,
                self.stop_segment_durations,
            )
        )


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

    Generates 4 blocks of 1 trial each (4 total).
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


def generate_speech_trials(
    questions: list[str],
    rng: random.Random,
    num_blocks: int = 4,
    min_read: float = 7.0,
    max_read: float = 13.0,
    min_answer: float = 17.0,
    max_answer: float = 23.0,
) -> list[list[TrialConfig]]:
    """Generate speech Q&A trial schedule.

    Each block is a 240-second trial made of:

        READ (STOP, 7--13 s)  ->  ANSWER (GO, 17--23 s)  ->  repeat ...

    Each question is pinned to 30 s total so that 8 questions fill the block
    and 1-minute Medoc temperature steps fall between questions, never during
    one.  No final rest period is needed.

    The session is always exactly ``num_blocks`` blocks (default 4).  If you
    supply more questions than fit, extras are truncated (no recycling).  If
    you supply fewer, they are spread evenly across the 4 blocks.

    With the default bounds, **~32 questions is the sweet spot** (8 per block).

    Args:
        questions: List of question strings.  Easily swappable by the caller.
        rng: Seeded Random instance for reproducibility.
        num_blocks: Number of blocks (default 4 for a ~20 min session).
        min_read: Minimum question-read time in seconds (default 7).
        max_read: Maximum question-read time in seconds (default 13).
        min_answer: Minimum answer time in seconds (unused; computed as 30 - read).
        max_answer: Maximum answer time in seconds (unused; computed as 30 - read).

    Returns:
        List of blocks, each block is a list of one TrialConfig.
    """
    if not questions:
        questions = ["Please speak freely."]

    # Target: 32 questions / 4 blocks = 8 per block.
    # Each question is pinned to 30 s total (read + answer = 30) so that
    # temperature steps at 1-minute boundaries never fall inside a question.
    #   8 * 30 = 240 s  =>  no final rest needed.
    max_per_block = 8

    # Always produce exactly ``num_blocks`` blocks (default 4)
    total_q = len(questions)
    qpb = total_q // num_blocks  # floor
    remainder = total_q % num_blocks

    # First ``remainder`` blocks get one extra question
    block_counts = [
        min(max_per_block, qpb + (1 if i < remainder else 0))
        for i in range(num_blocks)
    ]
    # Truncate if total would exceed available questions
    used = 0
    final_counts: list[int] = []
    for c in block_counts:
        actual = min(c, total_q - used)
        if actual <= 0:
            break
        final_counts.append(actual)
        used += actual

    # If somehow we ended up with fewer blocks (e.g. < 4 questions),
    # pad with single-question blocks so we always hit num_blocks
    while len(final_counts) < num_blocks and used < total_q:
        final_counts.append(1)
        used += 1
    while len(final_counts) < num_blocks:
        final_counts.append(1)  # will recycle the last question as fallback

    # Shuffle question order once, then slice into blocks
    shuffled = list(questions)
    rng.shuffle(shuffled)

    def _fit_durations(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Generate n read durations and n answer durations.

        Returns (read_durs, answer_durs).  Durations are paired by index so
        each question keeps its matched read / answer times (no independent
        shuffling).  Each pair sums to exactly 30 s so that 8 questions fill
        a 240-second block and 1-minute Medoc temperature steps land between
        questions, never during one.
        """
        reads = [round(rng.uniform(min_read, max_read), 2) for _ in range(n)]
        answers = [round(30.0 - r, 2) for r in reads]
        return tuple(reads), tuple(answers)

    all_blocks: list[list[TrialConfig]] = []
    idx = 0
    for n in final_counts:
        block_qs = shuffled[idx : idx + n]
        idx += n
        if not block_qs:
            block_qs = [shuffled[-1] if shuffled else "Please speak freely."]
            n = 1

        read_durs, answer_durs = _fit_durations(n)
        all_blocks.append(
            [
                TrialConfig(
                    task_type="speech",
                    num_go_segments=n,
                    go_segment_durations=answer_durs,
                    segment_texts=tuple(block_qs),
                    stop_segment_durations=read_durs,
                )
            ]
        )

    return all_blocks
