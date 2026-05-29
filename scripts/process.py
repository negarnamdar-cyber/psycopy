"""Unified offline post-processing pipeline for vowel experiment recordings.

Scans the data/ directory for sessions that have not yet been processed and
runs all post-hoc analyses:

    1. VAD (WebRTC) on vowel trials  — speech_start/end + stop-cue latencies
    2. openSMILE ComParE_2016 on vowel trials  — acoustic features
    3. openSMILE ComParE_2016 on speech recordings  — acoustic features
    4. Speaker diarization placeholder on speech recordings  — turn-taking stats

Output files per session:
    - vad_events.csv
    - vowel_features_ComParE.csv
    - speech_features_ComParE.csv
    - speech_diarization.csv
    - summary.csv
    - processed.json   (status tracker)

Usage:
    python scripts/process.py              # process all unprocessed sessions
    python scripts/process.py --force      # re-process everything
    python scripts/process.py <path>       # process a single session folder
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

logger = logging.getLogger("psycopy.process")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE = 16000
STATUS_FILE = "processed.json"

# openSMILE config sets
_VOWEL_FEATURE_SET = "ComParE_2016"
_SPEECH_FEATURE_SET = "ComParE_2016"

# VAD defaults
_VAD_CONFIG: dict[str, int] = {
    "aggressiveness": 2,
    "frame_duration_ms": 30,
    "silence_frames": 10,
}

# openSMILE segment / window settings for vowel trials
GO_CONTEXT_PRE_SEC = 0.35
GO_CONTEXT_POST_SEC = 0.35
SLIDING_WINDOW_SEC = 10.0
SLIDING_HOP_SEC = 1.0

# ---------------------------------------------------------------------------
# Optional dependency imports (fail gracefully)
# ---------------------------------------------------------------------------

try:
    import opensmile
except ImportError:
    opensmile = None  # type: ignore[assignment]

try:
    import webrtcvad
except ImportError:
    webrtcvad = None  # type: ignore[assignment]


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True, slots=True)
class AudioRecording:
    """Represents one recorded audio file with its metadata."""

    wav_path: Path
    trial_instance_id: str
    block: str
    audio_type: str  # "vowel" or "speech"
    event_data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Parsed session metadata."""

    session_dir: Path
    audio_dir: Path
    events_csv: Path
    output_dir: Path
    recordings: list[AudioRecording]


# =============================================================================
# Status tracking
# =============================================================================


def is_processed(session_dir: Path) -> bool:
    return (session_dir / STATUS_FILE).exists()


def mark_processed(session_dir: Path, meta: dict[str, Any]) -> None:
    (session_dir / STATUS_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_status(session_dir: Path) -> dict[str, Any]:
    path = session_dir / STATUS_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


# =============================================================================
# CSV / event helpers
# =============================================================================


def load_events_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
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


def discover_recordings(session_dir: Path) -> list[AudioRecording]:
    """Parse events.csv and match recording_start rows to WAV files."""
    events_csv = session_dir / "events.csv"
    audio_dir = session_dir / "audio"
    if not events_csv.exists() or not audio_dir.exists():
        return []

    events = load_events_csv(events_csv)
    recordings: list[AudioRecording] = []

    # Map recording_start rows to trial_instance_id + audio_type
    rec_starts = [r for r in events if r.get("event_type") == "recording_start"]

    for rec in rec_starts:
        trial_id = rec.get("trial_instance_id", "")
        block = rec.get("block", "")
        data = _parse_event_data(rec.get("event_data", ""))
        audio_type = data.get("audio_type", "vowel")  # default legacy sessions to vowel

        # Find matching WAV file
        wav_path = _resolve_wav(audio_dir, trial_id, block)
        if wav_path is None:
            logger.warning("No WAV found for trial %s in %s", trial_id, audio_dir)
            continue

        recordings.append(
            AudioRecording(
                wav_path=wav_path,
                trial_instance_id=trial_id,
                block=block,
                audio_type=audio_type,
                event_data=data,
            )
        )

    return recordings


def _resolve_wav(audio_dir: Path, trial_instance_id: str, block: str) -> Path | None:
    """Match a trial_instance_id to its WAV file."""
    # Direct match by trial_id in filename (skip if trial_id is empty)
    if trial_instance_id:
        candidates = list(audio_dir.glob(f"*{trial_instance_id}*.wav"))
        if candidates:
            return candidates[0]

    # Parse trial_instance_id: {participant}_{session}_block{set_num}_{trial_num:03d}
    parts = trial_instance_id.split("_")
    if len(parts) >= 2:
        trial_num_str = parts[-1]
        block_str = parts[-2]
        if block_str.startswith("block"):
            block_num = block_str[5:]
            candidates = list(
                audio_dir.glob(f"sub-*_block-{block_num}_trial-{trial_num_str}.wav")
            )
            if candidates:
                return candidates[0]

    # Fallback: any WAV in directory (only one WAV = speech mode)
    all_wavs = list(audio_dir.glob("*.wav"))
    if len(all_wavs) == 1:
        return all_wavs[0]

    return None


def find_cues(events_csv: Path, trial_instance_id: str) -> list[dict[str, Any]]:
    """Return all GO and STOP cues for a trial from events.csv.

    New-format sessions have explicit ``go_cue`` / ``stop_cue`` events.
    Old-format sessions have ``trial_start`` with ``go_durations`` in
    ``event_data`` — we reconstruct the cue times from that.

    Returns list of dicts with keys: ``cue_type`` ("go"|"stop"),
    ``timestamp_sec``, ``segment_index``.
    """
    cues: list[dict[str, Any]] = []
    events = load_events_csv(events_csv)

    # --- New format: explicit go_cue / stop_cue events --------------------
    for row in events:
        if row.get("trial_instance_id") != trial_instance_id:
            continue
        et = row.get("event_type", "")
        if et not in ("go_cue", "stop_cue"):
            continue
        data = _parse_event_data(row.get("event_data", ""))
        elapsed = data.get("trial_elapsed_sec")
        if elapsed is not None:
            cues.append(
                {
                    "cue_type": "go" if et == "go_cue" else "stop",
                    "timestamp_sec": float(elapsed),
                    "segment_index": data.get("segment_index", 0),
                }
            )

    if cues:
        cues.sort(key=lambda x: x["timestamp_sec"])
        return cues

    # --- Old format: reconstruct from trial_start event_data --------------
    for row in events:
        if row.get("trial_instance_id") != trial_instance_id:
            continue
        if row.get("event_type") != "trial_start":
            continue
        data = _parse_event_data(row.get("event_data", ""))
        go_durations = data.get("go_durations", [])
        if not go_durations:
            continue

        total_go = sum(float(d) for d in go_durations)
        num_stop_periods = len(go_durations) + 1
        stop_duration = (60.0 - total_go) / num_stop_periods

        elapsed = 0.0
        for idx, go_dur in enumerate(go_durations):
            # STOP cue
            cues.append(
                {
                    "cue_type": "stop",
                    "timestamp_sec": round(elapsed, 3),
                    "segment_index": idx,
                }
            )
            elapsed += stop_duration
            # GO cue
            cues.append(
                {
                    "cue_type": "go",
                    "timestamp_sec": round(elapsed, 3),
                    "segment_index": idx,
                }
            )
            elapsed += float(go_dur)

        # Final STOP cue
        cues.append(
            {
                "cue_type": "stop",
                "timestamp_sec": round(elapsed, 3),
                "segment_index": len(go_durations),
            }
        )

    cues.sort(key=lambda x: x["timestamp_sec"])
    return cues


def find_go_cue_durations(events_csv: Path, trial_instance_id: str) -> list[dict[str, float]]:
    """Return GO segment start/end/duration info for a trial."""
    segments: list[dict[str, float]] = []
    for row in load_events_csv(events_csv):
        if row.get("trial_instance_id") != trial_instance_id:
            continue
        if row.get("event_type") != "go_cue":
            continue
        data = _parse_event_data(row.get("event_data", ""))
        elapsed = data.get("trial_elapsed_sec")
        duration = data.get("cue_duration_sec")
        if elapsed is not None and duration is not None:
            segments.append(
                {
                    "start_sec": float(elapsed),
                    "end_sec": float(elapsed) + float(duration),
                    "duration_sec": float(duration),
                }
            )
    return segments


# =============================================================================
# Audio I/O
# =============================================================================


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


def standardize_16k_mono(input_path: Path, output_path: Path) -> Path:
    """Resample to 16 kHz mono int16 and write a WAV file."""
    audio, rate = read_wav(input_path)
    if rate != TARGET_SAMPLE_RATE:
        g = np.gcd(rate, TARGET_SAMPLE_RATE)
        up = TARGET_SAMPLE_RATE // g
        down = rate // g
        audio = resample_poly(audio, up, down).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_path), TARGET_SAMPLE_RATE, pcm)
    return output_path


# =============================================================================
# 1. VAD (vowel trials only)
# =============================================================================


def run_vad(
    wav_path: Path,
    cues: list[dict[str, Any]],
    aggressiveness: int = _VAD_CONFIG["aggressiveness"],
    frame_duration_ms: int = _VAD_CONFIG["frame_duration_ms"],
    silence_frames: int = _VAD_CONFIG["silence_frames"],
) -> list[dict[str, Any]]:
    """Run WebRTC VAD on a WAV file and compute GO-onset and STOP-cue latencies.

    For each GO cue, finds the first speech_start after the cue → ``go_latency_ms``.
    For each STOP cue, finds the first speech_end after the cue → ``stop_latency_ms``.

    Returns rows with keys:
        event_type, timestamp, speech_duration, cue_type, cue_index,
        latency_ms, go_latency_ms, stop_latency_ms
    """
    if webrtcvad is None:
        logger.warning("webrtcvad not installed — skipping VAD")
        return []

    samples, rate = read_wav(wav_path)
    if rate != TARGET_SAMPLE_RATE:
        g = np.gcd(rate, TARGET_SAMPLE_RATE)
        up = TARGET_SAMPLE_RATE // g
        down = rate // g
        samples = resample_poly(samples, up, down).astype(np.float32)
        rate = TARGET_SAMPLE_RATE

    frame_size = int(rate * frame_duration_ms / 1000)
    vad = webrtcvad.Vad(aggressiveness)

    speech_events: list[dict[str, Any]] = []
    is_speaking = False
    consecutive_speech = 0
    consecutive_silence = 0
    speech_start_time: float | None = None

    num_frames = len(samples) // frame_size
    for i in range(num_frames):
        frame = samples[i * frame_size : (i + 1) * frame_size]
        pcm = (frame * 32767.0).astype(np.int16).tobytes()
        ts = i * frame_duration_ms / 1000.0
        try:
            speech = vad.is_speech(pcm, rate)
        except Exception as exc:
            logger.warning("VAD error at %.3f s: %s", ts, exc)
            continue

        if speech:
            consecutive_speech += 1
            consecutive_silence = 0
            if not is_speaking and consecutive_speech >= 2:
                is_speaking = True
                speech_start_time = ts
                speech_events.append({"type": "speech_start", "timestamp": round(ts, 3)})
        else:
            consecutive_silence += 1
            consecutive_speech = 0
            if is_speaking and consecutive_silence >= silence_frames:
                is_speaking = False
                duration = ts - speech_start_time if speech_start_time is not None else None
                speech_events.append(
                    {
                        "type": "speech_end",
                        "timestamp": round(ts, 3),
                        "speech_duration": round(duration, 3) if duration is not None else None,
                    }
                )
                speech_start_time = None

    if is_speaking:
        end_time = num_frames * frame_duration_ms / 1000.0
        duration = end_time - speech_start_time if speech_start_time is not None else None
        speech_events.append(
            {
                "type": "speech_end",
                "timestamp": round(end_time, 3),
                "speech_duration": round(duration, 3) if duration is not None else None,
            }
        )

    # Build latency rows from cues
    go_cues = [c for c in cues if c["cue_type"] == "go"]
    stop_cues = [c for c in cues if c["cue_type"] == "stop"]

    rows: list[dict[str, Any]] = []

    # GO latencies: first speech_start after each GO cue
    for idx, cue in enumerate(go_cues):
        best_start = next(
            (
                ev
                for ev in speech_events
                if ev["type"] == "speech_start" and ev["timestamp"] >= cue["timestamp_sec"]
            ),
            None,
        )
        latency_ms = (
            (best_start["timestamp"] - cue["timestamp_sec"]) * 1000.0
            if best_start
            else None
        )
        rows.append(
            {
                "event_type": "speech_start",
                "timestamp": best_start["timestamp"] if best_start else "",
                "speech_duration": "",
                "cue_type": "go",
                "cue_index": idx,
                "latency_ms": round(latency_ms, 1) if latency_ms is not None else "",
                "go_latency_ms": round(latency_ms, 1) if latency_ms is not None else "",
                "stop_latency_ms": "",
            }
        )

    # STOP latencies: first speech_end after each STOP cue
    for idx, cue in enumerate(stop_cues):
        best_end = next(
            (
                ev
                for ev in speech_events
                if ev["type"] == "speech_end" and ev["timestamp"] >= cue["timestamp_sec"]
            ),
            None,
        )
        latency_ms = (
            (best_end["timestamp"] - cue["timestamp_sec"]) * 1000.0 if best_end else None
        )
        rows.append(
            {
                "event_type": "speech_end",
                "timestamp": best_end["timestamp"] if best_end else "",
                "speech_duration": best_end.get("speech_duration", "") if best_end else "",
                "cue_type": "stop",
                "cue_index": idx,
                "latency_ms": round(latency_ms, 1) if latency_ms is not None else "",
                "go_latency_ms": "",
                "stop_latency_ms": round(latency_ms, 1) if latency_ms is not None else "",
            }
        )

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for ev in rows:
        key = f"{ev['cue_type']}:{ev['cue_index']}:{ev.get('timestamp')}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(ev)

    return unique


def process_vad_for_session(session: SessionInfo) -> dict[str, Any]:
    """Run VAD on all vowel trial recordings in a session."""
    vad_rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0

    for rec in session.recordings:
        if rec.audio_type != "vowel":
            continue
        if not rec.wav_path.exists():
            skipped += 1
            continue

        cues = find_cues(session.events_csv, rec.trial_instance_id)
        events = run_vad(rec.wav_path, cues)
        for ev in events:
            vad_rows.append({"trial_instance_id": rec.trial_instance_id, **ev})
        processed += 1

    vad_csv = session.output_dir / "vad_events.csv"
    if vad_rows:
        _write_csv(vad_csv, vad_rows)
        logger.info("Wrote %d VAD events to %s", len(vad_rows), vad_csv)
    else:
        logger.info("No VAD events generated for %s", session.session_dir.name)

    return {"processed_trials": processed, "skipped_trials": skipped, "events": len(vad_rows)}


# =============================================================================
# 2. openSMILE ComParE_2016 (vowel + speech)
# =============================================================================


def _has_opensmile() -> bool:
    return opensmile is not None


def _init_smile(feature_set_name: str) -> Any:
    """Initialize an openSMILE SmILE extractor."""
    if opensmile is None:
        return None
    fs = getattr(opensmile.FeatureSet, feature_set_name, None)
    if fs is None:
        fs = getattr(opensmile.FeatureSet, "ComParE_2016", None)
    return opensmile.Smile(
        feature_set=fs or opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def _extract_opensmile_features(smile: Any, signal: np.ndarray) -> dict[str, float]:
    """Run openSMILE on a 16k mono float32 signal."""
    if smile is None:
        return {}
    frame = smile.process_signal(signal, sampling_rate=TARGET_SAMPLE_RATE)
    if frame.empty:
        return {}
    return {str(k): float(v) for k, v in frame.iloc[0].to_dict().items()}


def _build_sliding_windows(
    intervals: list[dict[str, float]],
    window_sec: float = SLIDING_WINDOW_SEC,
    hop_sec: float = SLIDING_HOP_SEC,
) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    for interval_idx, interval in enumerate(intervals, start=1):
        i_start = float(interval["start_sec"])
        i_end = float(interval["end_sec"])
        duration = i_end - i_start
        if duration <= 0:
            continue
        if duration <= window_sec:
            windows.append(
                {
                    "window_index": len(windows) + 1,
                    "interval_index": interval_idx,
                    "interval_start_sec": i_start,
                    "interval_end_sec": i_end,
                    "start_sec": i_start,
                    "end_sec": i_end,
                    "duration_sec": duration,
                }
            )
            continue
        cursor = i_start
        last_start = i_end - window_sec
        while cursor <= (last_start + 1e-9):
            windows.append(
                {
                    "window_index": len(windows) + 1,
                    "interval_index": interval_idx,
                    "interval_start_sec": i_start,
                    "interval_end_sec": i_end,
                    "start_sec": cursor,
                    "end_sec": cursor + window_sec,
                    "duration_sec": window_sec,
                }
            )
            cursor += hop_sec
        # tail window
        last = windows[-1]
        if last["interval_index"] == interval_idx and i_end - last["end_sec"] > 1e-9:
            windows.append(
                {
                    "window_index": len(windows) + 1,
                    "interval_index": interval_idx,
                    "interval_start_sec": i_start,
                    "interval_end_sec": i_end,
                    "start_sec": i_end - window_sec,
                    "end_sec": i_end,
                    "duration_sec": window_sec,
                }
            )
    return windows


def extract_vowel_features(session: SessionInfo) -> dict[str, Any]:
    """Run ComParE_2016 on vowel trial GO segments."""
    smile = _init_smile("ComParE_2016")
    if smile is None:
        logger.warning("openSMILE not installed — skipping vowel features")
        return {"status": "skipped", "reason": "openSMILE not installed"}

    rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    standardized_dir = session.output_dir / "audio_16k"
    standardized_dir.mkdir(exist_ok=True)

    for rec in session.recordings:
        if rec.audio_type != "vowel":
            continue
        if not rec.wav_path.exists():
            skipped += 1
            continue

        # Standardize to 16k
        std_name = rec.wav_path.stem + "_16k.wav"
        std_path = standardized_dir / std_name
        standardize_16k_mono(rec.wav_path, std_path)

        _, audio_16k = wavfile.read(str(std_path))
        if audio_16k.dtype != np.int16:
            audio_16k = (audio_16k.astype(np.float32) * 32767).astype(np.int16)

        duration_sec = len(audio_16k) / float(TARGET_SAMPLE_RATE)

        # Build GO intervals from events
        go_segments = find_go_cue_durations(session.events_csv, rec.trial_instance_id)
        if not go_segments:
            # Fallback: whole file
            go_segments = [{"start_sec": 0.0, "end_sec": duration_sec, "duration_sec": duration_sec}]

        # Expand with context
        expanded = []
        for seg in go_segments:
            start = max(0.0, seg["start_sec"] - GO_CONTEXT_PRE_SEC)
            end = min(duration_sec, seg["end_sec"] + GO_CONTEXT_POST_SEC)
            if end > start:
                expanded.append({"start_sec": start, "end_sec": end, "duration_sec": end - start})

        windows = _build_sliding_windows(expanded)
        if not windows:
            skipped += 1
            continue

        for w in windows:
            s0 = max(0, int(round(w["start_sec"] * TARGET_SAMPLE_RATE)))
            s1 = min(len(audio_16k), int(round(w["end_sec"] * TARGET_SAMPLE_RATE)))
            if s1 <= s0:
                continue
            seg = audio_16k[s0:s1].astype(np.float32) / 32767.0
            feats = _extract_opensmile_features(smile, seg)
            if not feats:
                continue
            row: dict[str, Any] = {
                "trial_instance_id": rec.trial_instance_id,
                "block": rec.block,
                "audio_type": "vowel",
                "audio_filename": rec.wav_path.name,
                "audio_16k_filename": std_name,
                "window_index": int(w["window_index"]),
                "window_start_sec": round(w["start_sec"], 4),
                "window_end_sec": round(w["end_sec"], 4),
                "window_duration_sec": round(w["duration_sec"], 4),
                "interval_index": int(w["interval_index"]),
                "interval_start_sec": round(w["interval_start_sec"], 4),
                "interval_end_sec": round(w["interval_end_sec"], 4),
            }
            row.update(feats)
            rows.append(row)

        processed += 1

    out_csv = session.output_dir / "vowel_features_ComParE.csv"
    if rows:
        _write_csv(out_csv, rows)
        logger.info("Wrote %d vowel feature rows to %s", len(rows), out_csv)
    else:
        logger.info("No vowel features extracted for %s", session.session_dir.name)

    return {
        "status": "completed",
        "processed_trials": processed,
        "skipped_trials": skipped,
        "rows": len(rows),
    }


def extract_speech_features(session: SessionInfo) -> dict[str, Any]:
    """Run ComParE_2016 on free-speech recordings."""
    smile = _init_smile("ComParE_2016")
    if smile is None:
        logger.warning("openSMILE not installed — skipping speech features")
        return {"status": "skipped", "reason": "openSMILE not installed"}

    rows: list[dict[str, Any]] = []
    processed = 0
    standardized_dir = session.output_dir / "audio_16k"
    standardized_dir.mkdir(exist_ok=True)

    for rec in session.recordings:
        if rec.audio_type != "speech":
            continue
        if not rec.wav_path.exists():
            continue

        std_name = rec.wav_path.stem + "_16k.wav"
        std_path = standardized_dir / std_name
        standardize_16k_mono(rec.wav_path, std_path)

        _, audio_16k = wavfile.read(str(std_path))
        if audio_16k.dtype != np.int16:
            audio_16k = (audio_16k.astype(np.float32) * 32767).astype(np.int16)

        duration_sec = len(audio_16k) / float(TARGET_SAMPLE_RATE)

        # Use whole recording as one window (no GO segments for speech)
        seg = audio_16k.astype(np.float32) / 32767.0
        feats = _extract_opensmile_features(smile, seg)
        if not feats:
            continue

        row: dict[str, Any] = {
            "trial_instance_id": rec.trial_instance_id or rec.wav_path.stem,
            "block": rec.block or "speech",
            "audio_type": "speech",
            "audio_filename": rec.wav_path.name,
            "audio_16k_filename": std_name,
            "window_index": 1,
            "window_start_sec": 0.0,
            "window_end_sec": round(duration_sec, 4),
            "window_duration_sec": round(duration_sec, 4),
            "interval_index": 1,
            "interval_start_sec": 0.0,
            "interval_end_sec": round(duration_sec, 4),
        }
        row.update(feats)
        rows.append(row)
        processed += 1

    out_csv = session.output_dir / "speech_features_ComParE.csv"
    if rows:
        _write_csv(out_csv, rows)
        logger.info("Wrote %d speech feature rows to %s", len(rows), out_csv)
    else:
        logger.info("No speech features extracted for %s", session.session_dir.name)

    return {"status": "completed", "processed_recordings": processed, "rows": len(rows)}


# =============================================================================
# 3. Speaker diarization (speech only) — placeholder
# =============================================================================


def process_speech_diarization(session: SessionInfo) -> dict[str, Any]:
    """Placeholder for speaker diarization on free-speech recordings.

    Currently produces a placeholder CSV with one row per speech recording
    so the pipeline structure is ready. Future integration can fill in actual
    turn-taking statistics (speech / non-speech ratio, speaker switch rate, etc.).
    """
    rows: list[dict[str, Any]] = []
    for rec in session.recordings:
        if rec.audio_type != "speech":
            continue
        if not rec.wav_path.exists():
            continue

        # Placeholder metrics
        audio, rate = read_wav(rec.wav_path)
        duration = len(audio) / float(rate)
        # Simple energy-based voice activity as a placeholder
        energy = audio ** 2
        threshold = energy.mean() * 0.1
        voiced_frames = (energy > threshold).sum()
        voiced_ratio = voiced_frames / len(audio) if len(audio) > 0 else 0.0

        rows.append(
            {
                "trial_instance_id": rec.trial_instance_id or rec.wav_path.stem,
                "block": rec.block or "speech",
                "audio_filename": rec.wav_path.name,
                "duration_sec": round(duration, 3),
                "voiced_ratio": round(float(voiced_ratio), 4),
                "speaker_turns": "",  # placeholder for future diarization
                "avg_turn_duration_sec": "",
                "primary_speaker_ratio": "",
                "diarization_model": "placeholder",
            }
        )

    out_csv = session.output_dir / "speech_diarization.csv"
    if rows:
        _write_csv(out_csv, rows)
        logger.info("Wrote %d speech diarization placeholder rows to %s", len(rows), out_csv)

    return {"status": "completed", "processed_recordings": len(rows)}


# =============================================================================
# 4. Summary CSV
# =============================================================================


def write_summary(session: SessionInfo, stats: dict[str, Any]) -> None:
    """Write a single-row summary.csv for easy data-frame assembly across sessions."""
    summary = {
        "session_id": session.session_dir.name,
        "participant_id": session.recordings[0].trial_instance_id.split("_")[0]
        if session.recordings
        else "",
        "num_recordings": len(session.recordings),
        "num_vowel_trials": sum(1 for r in session.recordings if r.audio_type == "vowel"),
        "num_speech_recordings": sum(1 for r in session.recordings if r.audio_type == "speech"),
        "vad_events": stats.get("vad", {}).get("events", 0),
        "vowel_feature_rows": stats.get("vowel_features", {}).get("rows", 0),
        "speech_feature_rows": stats.get("speech_features", {}).get("rows", 0),
        "diarization_recordings": stats.get("diarization", {}).get("processed_recordings", 0),
    }
    _write_csv(session.output_dir / "summary.csv", [summary])


# =============================================================================
# Session runner
# =============================================================================


def process_session(session_dir: Path, force: bool = False) -> dict[str, Any]:
    """Run the full offline pipeline on one session directory."""
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise ValueError(f"Not a directory: {session_dir}")

    if not force and is_processed(session_dir):
        logger.info("Skipping already-processed session: %s", session_dir.name)
        return {"status": "skipped", "session": session_dir.name}

    events_csv = session_dir / "events.csv"
    if not events_csv.exists():
        raise FileNotFoundError(f"Missing events.csv in {session_dir}")

    recordings = discover_recordings(session_dir)
    if not recordings:
        logger.warning("No recordings discovered in %s", session_dir.name)
        return {"status": "no_recordings", "session": session_dir.name}

    session = SessionInfo(
        session_dir=session_dir,
        audio_dir=session_dir / "audio",
        events_csv=events_csv,
        output_dir=session_dir,
        recordings=recordings,
    )

    logger.info("Processing session: %s (%d recordings)", session_dir.name, len(recordings))

    # Run each pipeline
    vad_stats = process_vad_for_session(session)
    vowel_stats = extract_vowel_features(session)
    speech_stats = extract_speech_features(session)
    diarization_stats = process_speech_diarization(session)

    stats = {
        "vad": vad_stats,
        "vowel_features": vowel_stats,
        "speech_features": speech_stats,
        "diarization": diarization_stats,
    }
    write_summary(session, stats)

    meta = {
        "status": "completed",
        "session": session_dir.name,
        "num_recordings": len(recordings),
        **{k: v for k, v in stats.items() if isinstance(v, dict)},
    }
    mark_processed(session_dir, meta)
    logger.info("Session complete: %s", session_dir.name)
    return meta


# =============================================================================
# Utilities
# =============================================================================


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts to a CSV file (atomically via temp file)."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    import tempfile

    tmp = Path(tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8", newline="", dir=path.parent
    ).name)
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def discover_sessions(data_dir: Path) -> list[Path]:
    """Find all session directories under data/."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    sessions = [p for p in data_dir.iterdir() if p.is_dir()]
    sessions.sort(key=lambda p: p.name)
    return sessions


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified offline post-processing for vowel experiment recordings."
    )
    parser.add_argument(
        "session_dir",
        nargs="?",
        type=Path,
        help="Process a single session directory. If omitted, scan data/ for all unprocessed sessions.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory to scan for session folders (default: data/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process sessions even if already marked processed.",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=_VAD_CONFIG["aggressiveness"],
        choices=[0, 1, 2, 3],
        help=f"WebRTC VAD aggressiveness (default: {_VAD_CONFIG['aggressiveness']})",
    )
    parser.add_argument(
        "--vad-frame-duration-ms",
        type=int,
        default=_VAD_CONFIG["frame_duration_ms"],
        choices=[10, 20, 30],
        help=f"VAD frame duration in ms (default: {_VAD_CONFIG['frame_duration_ms']})",
    )
    parser.add_argument(
        "--vad-silence-frames",
        type=int,
        default=_VAD_CONFIG["silence_frames"],
        help=f"Consecutive silent frames to end speech (default: {_VAD_CONFIG['silence_frames']})",
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

    # Override VAD defaults from CLI
    _VAD_CONFIG["aggressiveness"] = args.vad_aggressiveness
    _VAD_CONFIG["frame_duration_ms"] = args.vad_frame_duration_ms
    _VAD_CONFIG["silence_frames"] = args.vad_silence_frames

    if args.session_dir:
        try:
            meta = process_session(args.session_dir, force=args.force)
            print(json.dumps(meta, indent=2))
        except Exception as exc:
            logger.error("Failed to process %s: %s", args.session_dir, exc)
            return 1
    else:
        sessions = discover_sessions(args.data_dir)
        if not sessions:
            logger.info("No sessions found in %s", args.data_dir)
            return 0

        completed = 0
        for session_dir in sessions:
            try:
                meta = process_session(session_dir, force=args.force)
                if meta.get("status") == "completed":
                    completed += 1
            except Exception as exc:
                logger.error("Failed to process %s: %s", session_dir.name, exc)

        logger.info(
            "Batch complete: %d/%d sessions processed this run",
            completed,
            len(sessions),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
