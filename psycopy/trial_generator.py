"""Trial randomization for Medoc experiments.

Generates trial schedules with constrained randomization.

Default structure: 5 blocks of 1 trial = 5 total trials.
Each trial is a 4-minute (240-second) vowel task built from four 60-second
minute-blocks of alternating STOP/GO segments.  Because every minute starts
and ends on a STOP, the 60/120/180 s Medoc temperature steps always land on a
STOP period and never inside a GO (speaking) segment.
All trials use the unified Medoc program (experiment 192).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

# Vowel task timing.  Each 240 s trial is split into four 60 s minute-blocks so
# that Medoc temperature steps (every 60 s) land on a STOP, never inside a GO.
# Within every minute the trial draws 3-7 GO segments of 1.5-3.5 s each; the
# remaining time is split evenly across the STOP periods that bracket and
# separate the GO segments.
VOWEL_GO_MIN_SEC = 1.5
VOWEL_GO_MAX_SEC = 3.5
VOWEL_GO_PER_MINUTE_MIN = 3
VOWEL_GO_PER_MINUTE_MAX = 7
MINUTE_SEC = 60.0
NUM_MINUTES_PER_TRIAL = 4


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
        speech_read_duration: For speech mode, the constant question-read
            (STOP) sub-duration in seconds.  The STOP segment is split into
            this read window plus the rate-pain popup.  Zero for vowel mode.
        speech_rate_pain_duration: For speech mode, the constant "Rate your
            pain" popup (STOP) sub-duration in seconds.  Zero for vowel mode.
    """

    task_type: str
    num_go_segments: int
    go_segment_durations: tuple[float, ...]
    segment_texts: tuple[str, ...] = ()
    stop_segment_durations: tuple[float, ...] = ()
    speech_read_duration: float = 0.0
    speech_rate_pain_duration: float = 0.0

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
            and self.speech_read_duration == other.speech_read_duration
            and self.speech_rate_pain_duration == other.speech_rate_pain_duration
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.task_type,
                self.num_go_segments,
                self.go_segment_durations,
                self.segment_texts,
                self.stop_segment_durations,
                self.speech_read_duration,
                self.speech_rate_pain_duration,
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

    With the vowel defaults (min=1.5, max=3.5) and total_max reserved from a
    60 s minute, the max possible sum (7 * 3.5 = 24.5) is always well under
    total_max, so simple uniform random draws in [min, max] always satisfy the
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


def _generate_vowel_minute_schedule(
    rng: random.Random,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Build one 240 s vowel trial from four independent 60 s minute-blocks.

    Each minute draws 3-7 GO segments of 1.5-3.5 s.  The leftover time in the
    minute is split evenly across the (k+1) STOP periods that bracket and
    separate the GO segments, so every minute starts and ends on a STOP.  The
    four minute-blocks are concatenated, merging the trailing STOP of one
    minute with the leading STOP of the next into a single STOP that spans the
    60 s boundary.  This guarantees the 60/120/180 s Medoc temperature steps
    land on a STOP, never inside a GO (speaking) segment.

    Returns:
        ``(go_durations, stop_durations)`` where ``stop_durations`` holds one
        STOP duration per GO segment (the STOP that precedes it).  The final
        trailing STOP is left for the runtime to fill from the elapsed time.
    """
    flat: list[tuple[str, float]] = []
    for _ in range(NUM_MINUTES_PER_TRIAL):
        k = rng.randint(VOWEL_GO_PER_MINUTE_MIN, VOWEL_GO_PER_MINUTE_MAX)
        go_durs = _generate_go_durations(
            num_segments=k,
            min_seg_sec=VOWEL_GO_MIN_SEC,
            max_seg_sec=VOWEL_GO_MAX_SEC,
            total_max=MINUTE_SEC,
            rng=rng,
        )
        stop_per = (MINUTE_SEC - sum(go_durs)) / (k + 1)
        # Minute sub-schedule: STOP, GO, STOP, GO, ..., STOP (k GOs, k+1 STOPs).
        for go_dur in go_durs:
            flat.append(("stop", stop_per))
            flat.append(("go", go_dur))
        flat.append(("stop", stop_per))  # trailing STOP closes the minute

    # Merge consecutive STOPs at minute boundaries into single STOP periods.
    merged: list[tuple[str, float]] = []
    for state, dur in flat:
        if merged and merged[-1][0] == "stop" and state == "stop":
            merged[-1] = ("stop", merged[-1][1] + dur)
        else:
            merged.append((state, dur))

    go_durations = tuple(d for s, d in merged if s == "go")
    # Every STOP except the final trailing one precedes a GO segment.
    stops = [d for s, d in merged if s == "stop"]
    stop_durations = tuple(stops[:-1])
    return go_durations, stop_durations


def generate_trials(
    num_sets: int,
    trials_per_set: int,
    num_stop_trials_ratio: float,
    rng: random.Random,
) -> list[list[TrialConfig]]:
    """Generate randomized trial schedule.

    Generates ``num_sets`` blocks of ``trials_per_set`` trial each.
    Each trial is a 240-second vowel task built from four independent
    60-second minute-blocks.  Within every minute the trial draws 3-7 GO
    segments of 1.5-3.5 s each; the remaining time is split evenly across the
    STOP periods that bracket and separate them.  Because each minute starts
    and ends on a STOP, the 60/120/180 s Medoc temperature steps always land
    on a STOP period and never inside a GO (speaking) segment.

    Explicit STOP durations are emitted (``stop_segment_durations``) so the
    runtime places each GO exactly inside its minute-block.
    `num_stop_trials_ratio` is retained for API compatibility and ignored.

    Args:
        num_sets: Number of blocks.
        trials_per_set: Number of trials per block.
        num_stop_trials_ratio: Retained for API compatibility; ignored.
        rng: Seeded Random instance for reproducibility.

    Returns:
        List of blocks, each block is a list of TrialConfig instances.
    """
    all_blocks: list[list[TrialConfig]] = []

    for _ in range(num_sets):
        block: list[TrialConfig] = []
        for _ in range(trials_per_set):
            go_durations, stop_durations = _generate_vowel_minute_schedule(rng)
            block.append(
                TrialConfig(
                    task_type="vowel",
                    num_go_segments=len(go_durations),
                    go_segment_durations=go_durations,
                    stop_segment_durations=stop_durations,
                )
            )

        rng.shuffle(block)
        all_blocks.append(block)

    return all_blocks


def generate_speech_trials(
    questions: list[str],
    rng: random.Random,
    num_blocks: int = 4,
    read_duration: float = 13.0,
    rate_pain_duration: float = 5.0,
    answer_duration: float = 12.0,
    trial_duration_sec: float = 240.0,
) -> list[list[TrialConfig]]:
    """Generate speech Q&A trial schedule with constant per-question timing.

    Each block is a ``trial_duration_sec``-second trial (default 240 s) made of
    identical 30-second question cycles:

        READ (STOP, question shown)          ->  ``read_duration`` s  (default 13)
        ANSWER (GO, screen turns green)      ->  ``answer_duration`` s  (default 12)
        "Rate your pain" prompt (STOP)      ->  ``rate_pain_duration`` s (default 5)

    Pause (STOP) periods therefore total 18 s (longer) and the GO speaking
    period is 12 s (shorter).  Durations are constant -- there is no
    randomization.

    The Medoc thermode changes temperature every 60 s.  Each cycle must divide
    60 s evenly so that temperature steps land exactly on cycle boundaries
    (between questions) and never inside a GO speaking period.  With the
    default 13 + 5 + 12 = 30 s cycle, 8 questions fill a 240 s block and the
    60 s / 120 s / 180 s temperature steps fall between questions, so the
    temperature never changes while the participant is speaking.

    The session is always exactly ``num_blocks`` blocks (default 4).  Extra
    questions beyond what fits are truncated (no recycling); fewer questions
    are spread evenly across the blocks.

    Args:
        questions: List of question strings.  Easily swappable by the caller.
        rng: Seeded Random instance for question shuffling/reproducibility.
        num_blocks: Number of blocks (default 4 for a ~20 min session).
        read_duration: Constant question-read (STOP) time in seconds.
        rate_pain_duration: Constant "Rate your pain" prompt (STOP) time in seconds.
        answer_duration: Constant answer (GO) time in seconds.
        trial_duration_sec: Total trial/block length in seconds (default 240).

    Returns:
        List of blocks, each block is a list of one TrialConfig.
    """
    if not questions:
        questions = ["Please speak freely."]

    cycle_sec = round(read_duration + rate_pain_duration + answer_duration, 6)
    if cycle_sec <= 0:
        raise ValueError("Speech cycle length must be positive.")
    # The cycle must divide 60 s so Medoc temperature steps (every 60 s) land
    # on cycle boundaries instead of inside a GO speaking period.
    if abs(60.0 % cycle_sec) > 1e-6:
        raise ValueError(
            f"Speech cycle length ({cycle_sec}s) must divide 60s evenly so "
            "Medoc temperature steps never fall inside a GO speaking period."
        )

    max_per_block = max(1, int(round(trial_duration_sec / cycle_sec)))
    stop_per_q = round(read_duration + rate_pain_duration, 6)
    answer_per_q = round(answer_duration, 6)

    total_q = len(questions)
    qpb = total_q // num_blocks  # floor
    remainder = total_q % num_blocks

    block_counts = [
        min(max_per_block, qpb + (1 if i < remainder else 0))
        for i in range(num_blocks)
    ]
    used = 0
    final_counts: list[int] = []
    for c in block_counts:
        actual = min(c, total_q - used)
        if actual <= 0:
            break
        final_counts.append(actual)
        used += actual

    while len(final_counts) < num_blocks and used < total_q:
        final_counts.append(1)
        used += 1
    while len(final_counts) < num_blocks:
        final_counts.append(1)  # recycle the last question as fallback

    shuffled = list(questions)
    rng.shuffle(shuffled)

    all_blocks: list[list[TrialConfig]] = []
    idx = 0
    for n in final_counts:
        block_qs = shuffled[idx : idx + n]
        idx += n
        if not block_qs:
            block_qs = [shuffled[-1] if shuffled else "Please speak freely."]
            n = 1

        stop_durs = tuple(stop_per_q for _ in range(n))
        go_durs = tuple(answer_per_q for _ in range(n))
        all_blocks.append(
            [
                TrialConfig(
                    task_type="speech",
                    num_go_segments=n,
                    go_segment_durations=go_durs,
                    segment_texts=tuple(block_qs),
                    stop_segment_durations=stop_durs,
                    speech_read_duration=read_duration,
                    speech_rate_pain_duration=rate_pain_duration,
                )
            ]
        )

    return all_blocks
