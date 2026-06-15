"""ML Segmenter — chop session audio into individual speech segments via VAD.

Usage:
    python scripts/ml_segmenter.py <participant_id> [--data-dir data]

Looks in data/ for session folder(s) matching the participant,
then slices every WAV in the session's audio/ folder into speech clips.

Produces:
    data/..._segments/
        segment_0001.wav
        segment_0002.wav
        ...
    segments.csv   (columns: source_file, segment_index, start_sec, end_sec,
                    duration_sec, temperature_celsius)
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

logger = logging.getLogger("psycopy.ml_segmenter")

TARGET_SAMPLE_RATE = 16000

# VAD defaults
_VAD_CONFIG: dict[str, int] = {
    "aggressiveness": 2,
    "frame_duration_ms": 30,
    "silence_frames": 10,
}

try:
    import webrtcvad
except ImportError:
    webrtcvad = None  # type: ignore[assignment]


def _find_session_dirs(data_dir: Path, participant_id: str) -> list[Path]:
    """Return sorted list of session dirs whose name contains the participant id."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []

    # Normalize participant id for matching
    pid = participant_id.strip().lower()
    # Session folder pattern: data/YYYYMMDD_HHMMSS_sub-{participant}_session-{session}/
    # The participant part could be like "001", "P001", etc.
    candidates: list[Path] = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        # Skip segment-output dirs so they don't look like sessions
        if name.endswith("_segments"):
            continue
        # Check if participant id appears in folder name
        if pid in name:
            candidates.append(child)

    # Sort by name (timestamp prefix means chronological)
    candidates.sort(key=lambda p: p.name)
    return candidates


def read_wav(path: Path) -> tuple[np.ndarray, int]:
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


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    """Write float32 mono audio to 16-bit WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(path), rate, pcm)


def resample_to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    """Resample audio to 16 kHz."""
    if rate == TARGET_SAMPLE_RATE:
        return audio
    g = np.gcd(rate, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // g
    down = rate // g
    return resample_poly(audio, up, down).astype(np.float32)


def find_speech_segments(
    wav_path: Path,
    aggressiveness: int = _VAD_CONFIG["aggressiveness"],
    frame_duration_ms: int = _VAD_CONFIG["frame_duration_ms"],
    silence_frames: int = _VAD_CONFIG["silence_frames"],
    min_segment_sec: float = 0.3,
) -> list[dict[str, Any]]:
    """Run WebRTC VAD and return list of speech segments.

    Each segment has: start_sec, end_sec, duration_sec.
    Segments shorter than min_segment_sec are dropped.
    """
    if webrtcvad is None:
        logger.warning("webrtcvad not installed — run: pip install webrtcvad-wheels")
        return []

    samples, rate = read_wav(wav_path)
    samples = resample_to_16k(samples, rate)
    rate = TARGET_SAMPLE_RATE

    frame_size = int(rate * frame_duration_ms / 1000)
    vad = webrtcvad.Vad(aggressiveness)

    # Collect contiguous speech regions
    is_speaking = False
    consecutive_speech = 0
    consecutive_silence = 0
    speech_start_time: float | None = None

    regions: list[dict[str, float]] = []

    num_frames = len(samples) // frame_size
    for i in range(num_frames):
        frame = samples[i * frame_size : (i + 1) * frame_size]
        pcm = (frame * 32767.0).astype(np.int16).tobytes()
        ts = i * frame_duration_ms / 1000.0

        try:
            speech = vad.is_speech(pcm, rate)
        except Exception as exc:
            logger.warning("VAD error at %.3f s in %s: %s", ts, wav_path.name, exc)
            continue

        if speech:
            consecutive_speech += 1
            consecutive_silence = 0
            if not is_speaking and consecutive_speech >= 2:
                is_speaking = True
                speech_start_time = ts
        else:
            consecutive_silence += 1
            consecutive_speech = 0
            if is_speaking and consecutive_silence >= silence_frames:
                is_speaking = False
                if speech_start_time is not None:
                    duration = ts - speech_start_time
                    if duration >= min_segment_sec:
                        regions.append(
                            {
                                "start_sec": round(speech_start_time, 3),
                                "end_sec": round(ts, 3),
                                "duration_sec": round(duration, 3),
                            }
                        )
                speech_start_time = None

    # Close trailing speech
    if is_speaking and speech_start_time is not None:
        end_time = num_frames * frame_duration_ms / 1000.0
        duration = end_time - speech_start_time
        if duration >= min_segment_sec:
            regions.append(
                {
                    "start_sec": round(speech_start_time, 3),
                    "end_sec": round(end_time, 3),
                    "duration_sec": round(duration, 3),
                }
            )

    # Merge very close segments (gaps < 0.15 sec)
    merged: list[dict[str, float]] = []
    for r in regions:
        if merged and r["start_sec"] - merged[-1]["end_sec"] < 0.15:
            merged[-1]["end_sec"] = r["end_sec"]
            merged[-1]["duration_sec"] = merged[-1]["end_sec"] - merged[-1]["start_sec"]
        else:
            merged.append(dict(r))

    return merged


def segment_session(
    session_dir: Path,
    output_dir: Path | None = None,
    aggressiveness: int = _VAD_CONFIG["aggressiveness"],
    frame_duration_ms: int = _VAD_CONFIG["frame_duration_ms"],
    silence_frames: int = _VAD_CONFIG["silence_frames"],
    min_segment_sec: float = 0.3,
) -> Path:
    """Segment all WAVs in session_dir/audio/ and write CSV + clips.

    Returns the path to the CSV that was written.
    """
    session_dir = Path(session_dir)
    audio_dir = session_dir / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"No audio/ folder in {session_dir}")

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("No .wav files found in %s", audio_dir)
        return Path()

    if output_dir is None:
        output_dir = Path(str(session_dir) + "_segments")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    global_idx = 0

    for wav_path in wav_files:
        logger.info("Processing %s", wav_path.name)
        segments = find_speech_segments(
            wav_path,
            aggressiveness=aggressiveness,
            frame_duration_ms=frame_duration_ms,
            silence_frames=silence_frames,
            min_segment_sec=min_segment_sec,
        )
        if not segments:
            logger.info("  -> no speech segments found")
            continue

        samples, rate = read_wav(wav_path)
        samples = resample_to_16k(samples, rate)
        rate = TARGET_SAMPLE_RATE

        for seg_idx, seg in enumerate(segments, start=1):
            global_idx += 1
            s0 = max(0, int(round(seg["start_sec"] * rate)))
            s1 = min(len(samples), int(round(seg["end_sec"] * rate)))
            clip = samples[s0:s1]

            out_name = f"segment_{global_idx:04d}.wav"
            out_path = output_dir / out_name
            write_wav(out_path, clip, rate)

            rows.append(
                {
                    "source_file": wav_path.name,
                    "segment_index": seg_idx,
                    "segment_filename": out_name,
                    "start_sec": seg["start_sec"],
                    "end_sec": seg["end_sec"],
                    "duration_sec": seg["duration_sec"],
                    "temperature_celsius": "",
                }
            )
            logger.info(
                "  -> %s (%.3f s)", out_name, seg["duration_sec"]
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
        description="Slice session audio into individual speech segments for ML."
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
        "--aggressiveness",
        type=int,
        default=_VAD_CONFIG["aggressiveness"],
        choices=[0, 1, 2, 3],
        help=f"WebRTC VAD aggressiveness (default: {_VAD_CONFIG['aggressiveness']})",
    )
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=_VAD_CONFIG["frame_duration_ms"],
        choices=[10, 20, 30],
        help=f"VAD frame duration in ms (default: {_VAD_CONFIG['frame_duration_ms']})",
    )
    parser.add_argument(
        "--silence-frames",
        type=int,
        default=_VAD_CONFIG["silence_frames"],
        help=f"Consecutive silent frames to end speech (default: {_VAD_CONFIG['silence_frames']})",
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=0.3,
        help="Minimum segment duration in seconds (default: 0.3)",
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

    # 1. Find matching session dirs
    sessions = _find_session_dirs(args.data_dir, args.participant_id)
    if not sessions:
        logger.error(
            "No session folders found for participant '%s' in %s",
            args.participant_id,
            args.data_dir,
        )
        return 1

    # 2. If multiple sessions, optionally filter by --session
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
        csv_path = segment_session(
            session_dir,
            output_dir=args.output_dir,
            aggressiveness=args.aggressiveness,
            frame_duration_ms=args.frame_duration_ms,
            silence_frames=args.silence_frames,
            min_segment_sec=args.min_segment_sec,
        )
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
