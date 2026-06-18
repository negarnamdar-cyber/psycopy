"""ML Segmenter — cut session audio into individual GO segments using event timestamps.

Usage:
    python scripts/ml_segmenter.py <participant_id> [--data-dir data]

Looks in data/ for session folder(s) matching the participant,
then slices every WAV in the session's audio/ folder into GO segments
using the go_cue timestamps from events.csv.

    Produces:
        data/..._segments/
            segment_0001.wav
            segment_0002.wav
            ...
        segments.csv   (columns: source_file, trial_instance_id, audio_type,
                        segment_index, segment_filename, start_sec, end_sec,
                        duration_sec, temperature_celsius, pain)

    `temperature_celsius` is auto-filled from the GO-cue event data.
    `audio_type` is auto-filled ('vowel' or 'speech') from recording_start events.
    `pain` is left BLANK — fill it in manually with a 1-10 pain rating per
    segment before running scripts/cnn_analyze.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

logger = logging.getLogger("psycopy.ml_segmenter")

TARGET_SAMPLE_RATE = 16000


def _find_session_dirs(data_dir: Path, participant_id: str) -> list[Path]:
    """Return sorted list of session dirs whose name contains the participant id."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []

    pid = participant_id.strip().lower()
    candidates: list[Path] = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name.endswith("_segments"):
            continue
        if pid in name:
            candidates.append(child)

    candidates.sort(key=lambda p: p.name)
    return candidates


def _load_events_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _parse_event_data(data_str: str) -> dict[str, Any]:
    if not data_str:
        return {}
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return {}


def _extract_go_segments(events_csv: Path) -> dict[str, list[dict[str, Any]]]:
    """Parse events.csv and return GO segments per trial_instance_id.

    Each segment has:
        segment_index, start_sec, end_sec, duration_sec, temperature_celsius, pain
    """
    events = _load_events_csv(events_csv)
    go_segments: dict[str, list[dict[str, Any]]] = {}

    for row in events:
        if row.get("event_type") != "go_cue":
            continue
        trial_id = row.get("trial_instance_id", "").strip()
        if not trial_id:
            continue
        data = _parse_event_data(row.get("event_data", ""))

        elapsed = data.get("trial_elapsed_sec")
        duration = data.get("cue_duration_sec")
        temp = data.get("temperature_celsius")
        seg_idx = data.get("segment_index", 0)

        if elapsed is None or duration is None:
            continue

        go_segments.setdefault(trial_id, []).append(
            {
                "segment_index": int(seg_idx),
                "start_sec": round(float(elapsed), 3),
                "end_sec": round(float(elapsed) + float(duration), 3),
                "duration_sec": round(float(duration), 3),
                "temperature_celsius": temp if temp is not None else "",
                "pain": "",
            }
        )

    # Sort by segment_index within each trial
    for trial_id in go_segments:
        go_segments[trial_id].sort(key=lambda x: x["segment_index"])

    return go_segments


def _load_trial_audio_types(events_csv: Path) -> dict[str, str]:
    """Map trial_instance_id -> audio_type ('vowel' or 'speech').

    Reads `recording_start` events, whose event_data carries the audio_type
    emitted by the experiment runtime (psycopy/medoc_experiment.py).
    """
    events = _load_events_csv(events_csv)
    types: dict[str, str] = {}
    for row in events:
        if row.get("event_type") != "recording_start":
            continue
        trial_id = row.get("trial_instance_id", "").strip()
        if not trial_id:
            continue
        data = _parse_event_data(row.get("event_data", ""))
        types[trial_id] = data.get("audio_type", "vowel")
    return types


def _resolve_wav(audio_dir: Path, trial_instance_id: str) -> Path | None:
    """Match a trial_instance_id to its WAV file."""
    # trial_instance_id format: {participant_id}_{session_id}_block{set_num}_{trial_num:03d}
    parts = trial_instance_id.split("_")
    if len(parts) >= 4:
        participant_id = parts[0]
        block_num = parts[2].replace("block", "")
        trial_num = parts[3]
        expected = f"sub-{participant_id}_block-{block_num}_trial-{trial_num}.wav"
        wav_path = audio_dir / expected
        if wav_path.exists():
            return wav_path

    # Fallback: any WAV containing the trial id
    candidates = list(audio_dir.glob(f"*{trial_instance_id}*.wav"))
    if candidates:
        return candidates[0]

    # Last fallback: single WAV in directory
    all_wavs = list(audio_dir.glob("*.wav"))
    if len(all_wavs) == 1:
        return all_wavs[0]

    return None


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file and return float32 samples + sample rate."""
    rate, audio = wavfile.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / float(max_val)
    else:
        audio = audio.astype(np.float32)
    return audio, rate


def _write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    """Write float32 mono audio to 16-bit WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(path), rate, pcm)


def _resample_to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == TARGET_SAMPLE_RATE:
        return audio
    g = np.gcd(rate, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // g
    down = rate // g
    return resample_poly(audio, up, down).astype(np.float32)


def segment_session(
    session_dir: Path,
    output_dir: Path | None = None,
) -> Path:
    """Segment all WAVs in session_dir/audio/ using events.csv GO cues.

    Returns the path to the CSV that was written.
    """
    session_dir = Path(session_dir)
    audio_dir = session_dir / "audio"
    events_csv = session_dir / "events.csv"

    if not audio_dir.is_dir():
        raise FileNotFoundError(f"No audio/ folder in {session_dir}")
    if not events_csv.exists():
        raise FileNotFoundError(f"No events.csv in {session_dir}")

    go_segments = _extract_go_segments(events_csv)
    if not go_segments:
        logger.warning("No go_cue events found in %s", events_csv)
        return Path()

    audio_types = _load_trial_audio_types(events_csv)

    if output_dir is None:
        output_dir = Path(str(session_dir) + "_segments")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    global_idx = 0

    for trial_id, segments in go_segments.items():
        wav_path = _resolve_wav(audio_dir, trial_id)
        if wav_path is None:
            logger.warning("No WAV found for trial %s in %s", trial_id, audio_dir)
            continue

        logger.info("Processing %s (%d segments)", wav_path.name, len(segments))
        samples, rate = _read_wav(wav_path)
        samples = _resample_to_16k(samples, rate)
        rate = TARGET_SAMPLE_RATE
        duration_sec = len(samples) / float(rate)

        for seg in segments:
            global_idx += 1
            start_sec = max(0.0, seg["start_sec"])
            end_sec = min(duration_sec, seg["end_sec"])
            if end_sec <= start_sec:
                logger.warning(
                    "Skipping empty/invalid segment %d for %s (%.3f - %.3f)",
                    seg["segment_index"],
                    wav_path.name,
                    start_sec,
                    end_sec,
                )
                continue

            s0 = int(round(start_sec * rate))
            s1 = int(round(end_sec * rate))
            clip = samples[s0:s1]

            out_name = f"segment_{global_idx:04d}.wav"
            out_path = output_dir / out_name
            _write_wav(out_path, clip, rate)

            rows.append(
                {
                    "source_file": wav_path.name,
                    "trial_instance_id": trial_id,
                    "audio_type": audio_types.get(trial_id, "vowel"),
                    "segment_index": seg["segment_index"],
                    "segment_filename": out_name,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round(end_sec - start_sec, 3),
                    "temperature_celsius": seg["temperature_celsius"],
                    "pain": seg["pain"],
                }
            )
            logger.info(
                "  -> %s (%.3f s)", out_name, end_sec - start_sec
            )

    csv_path = output_dir / "segments.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %d segments to %s", len(rows), csv_path)
    else:
        logger.info("No segments written.")

    return csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Slice session audio into individual GO segments for ML."
    )
    parser.add_argument(
        "participant_id",
        help="Participant ID (e.g. 001, P001). The script scans data/ for matching session folder(s).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing session folders (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write segments and CSV (default: <session_dir>_segments)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="If multiple sessions exist, pick this session number (e.g. 01).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sessions = _find_session_dirs(args.data_dir, args.participant_id)
    if not sessions:
        logger.error(
            "No session folders found for participant '%s' in %s",
            args.participant_id,
            args.data_dir,
        )
        return 1

    if len(sessions) > 1 and args.session:
        filtered = [s for s in sessions if f"session-{args.session}" in s.name.lower()]
        if filtered:
            sessions = filtered

    if len(sessions) > 1:
        logger.info("Found %d sessions for participant %s:", len(sessions), args.participant_id)
        for s in sessions:
            logger.info("  %s", s.name)
        logger.info("Using the most recent one: %s", sessions[-1].name)

    session_dir = sessions[-1]
    logger.info("Processing session: %s", session_dir)

    try:
        csv_path = segment_session(session_dir, output_dir=args.output_dir)
        if csv_path.exists():
            print(f"Segments written to: {csv_path.parent}")
            print(f"CSV catalog: {csv_path}")
        else:
            print("No segments were produced.")
    except Exception as exc:
        logger.error("Failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
