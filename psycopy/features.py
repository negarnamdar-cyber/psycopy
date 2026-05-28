"""Post-run feature extraction pipeline using openSMILE eGeMAPSv02."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from psycopy.storage import atomic_write_json
from psycopy.types import TrialRecord, TrialStatus


TARGET_SAMPLE_RATE = 16000
GO_CONTEXT_PRE_SEC = 0.35
GO_CONTEXT_POST_SEC = 0.35
SLIDING_WINDOW_SEC = 10.0
SLIDING_HOP_SEC = 1.0


def parse_go_segments(schedule_json: str) -> list[dict[str, float]]:
    raw = json.loads(schedule_json)
    go_segments: list[dict[str, float]] = []
    for seg in raw:
        if str(seg.get("state", "")).upper() != "GO":
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        go_segments.append({"start_sec": start, "end_sec": end, "duration_sec": end - start})
    return go_segments


def _merge_intervals(intervals: list[dict[str, float]]) -> list[dict[str, float]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x["start_sec"])
    merged: list[dict[str, float]] = [dict(sorted_intervals[0])]
    for interval in sorted_intervals[1:]:
        last = merged[-1]
        if interval["start_sec"] <= last["end_sec"]:
            last["end_sec"] = max(last["end_sec"], interval["end_sec"])
            last["duration_sec"] = last["end_sec"] - last["start_sec"]
        else:
            merged.append(dict(interval))
    return merged


def expand_go_segments_with_context(
    go_segments: list[dict[str, float]],
    audio_duration_sec: float,
    pre_sec: float = GO_CONTEXT_PRE_SEC,
    post_sec: float = GO_CONTEXT_POST_SEC,
) -> list[dict[str, float]]:
    expanded: list[dict[str, float]] = []
    audio_end = max(audio_duration_sec, 0.0)
    for seg in go_segments:
        start = max(0.0, float(seg["start_sec"]) - pre_sec)
        end = min(audio_end, float(seg["end_sec"]) + post_sec)
        if end <= start:
            continue
        expanded.append({"start_sec": start, "end_sec": end, "duration_sec": end - start})
    return _merge_intervals(expanded)


def build_feature_intervals(
    trial: TrialRecord,
    audio_duration_sec: float,
) -> list[dict[str, float]]:
    if not trial.go_segmentation_enabled:
        duration = max(audio_duration_sec, 0.0)
        return [{"start_sec": 0.0, "end_sec": duration, "duration_sec": duration}]
    go_segments = parse_go_segments(trial.schedule_json)
    return expand_go_segments_with_context(go_segments, audio_duration_sec=audio_duration_sec)


def build_sliding_windows(
    intervals: list[dict[str, float]],
    window_sec: float = SLIDING_WINDOW_SEC,
    hop_sec: float = SLIDING_HOP_SEC,
) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    if window_sec <= 0 or hop_sec <= 0:
        return windows

    for interval_idx, interval in enumerate(intervals, start=1):
        interval_start = float(interval["start_sec"])
        interval_end = float(interval["end_sec"])
        interval_duration = interval_end - interval_start
        if interval_duration <= 0:
            continue

        if interval_duration <= window_sec:
            windows.append(
                {
                    "window_index": len(windows) + 1,
                    "interval_index": interval_idx,
                    "interval_start_sec": interval_start,
                    "interval_end_sec": interval_end,
                    "start_sec": interval_start,
                    "end_sec": interval_end,
                    "duration_sec": interval_duration,
                }
            )
            continue

        cursor = interval_start
        last_start = interval_end - window_sec
        while cursor <= (last_start + 1e-9):
            windows.append(
                {
                    "window_index": len(windows) + 1,
                    "interval_index": interval_idx,
                    "interval_start_sec": interval_start,
                    "interval_end_sec": interval_end,
                    "start_sec": cursor,
                    "end_sec": cursor + window_sec,
                    "duration_sec": window_sec,
                }
            )
            cursor += hop_sec

        last = windows[-1]
        if last["interval_index"] == interval_idx and interval_end - last["end_sec"] > 1e-9:
            tail_start = interval_end - window_sec
            windows.append(
                {
                    "window_index": len(windows) + 1,
                    "interval_index": interval_idx,
                    "interval_start_sec": interval_start,
                    "interval_end_sec": interval_end,
                    "start_sec": tail_start,
                    "end_sec": interval_end,
                    "duration_sec": window_sec,
                }
            )

    return windows


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _to_float(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        return audio.astype(np.float32) / float(max_val)
    return audio.astype(np.float32)


def standardize_wav_16k_mono(input_path: Path, output_path: Path) -> Path:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sample_rate, audio = wavfile.read(input_path)
    mono = _to_mono(audio)
    mono_f32 = _to_float(mono)

    if sample_rate != TARGET_SAMPLE_RATE:
        gcd = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
        up = TARGET_SAMPLE_RATE // gcd
        down = sample_rate // gcd
        mono_f32 = resample_poly(mono_f32, up, down).astype(np.float32)

    mono_f32 = np.clip(mono_f32, -1.0, 1.0)
    pcm = (mono_f32 * 32767.0).astype(np.int16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, TARGET_SAMPLE_RATE, pcm)
    return output_path


class FeatureExtractor:
    """Build one feature matrix per run from trial WAV recordings."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("psycopy.features")
        self.disabled_reason: str | None = None
        try:
            import opensmile
        except ImportError:
            self.opensmile = None
            self.smile = None
            self.disabled_reason = "openSMILE python package not available"
            self.logger.error("Feature extraction disabled: %s", self.disabled_reason)
            return

        self.opensmile = opensmile
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def _extract_one_segment(self, signal_int16: np.ndarray) -> dict[str, float]:
        if self.smile is None:
            return {}
        signal_f32 = signal_int16.astype(np.float32) / 32767.0
        frame = self.smile.process_signal(signal_f32, sampling_rate=TARGET_SAMPLE_RATE)
        if frame.empty:
            return {}
        return {str(k): float(v) for k, v in frame.iloc[0].to_dict().items()}

    def extract_run_matrix(
        self,
        trials: list[TrialRecord],
        audio_dir: Path,
        standardized_audio_dir: Path,
        features_csv_path: Path,
        manifest_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        if self.smile is None:
            atomic_write_json(
                manifest_path,
                {
                    "metadata": metadata,
                    "status": "disabled",
                    "reason": self.disabled_reason or "openSMILE backend not initialized",
                    "feature_set": "eGeMAPSv02",
                    "feature_level": "Functionals",
                    "rows_written": 0,
                    "processed_trials": 0,
                    "processed_windows": 0,
                    "skipped_windows": 0,
                    "missing_audio_files": 0,
                    "skipped_trials": 0,
                },
            )
            return

        rows: list[dict[str, Any]] = []
        processed_trials = 0
        processed_windows = 0
        skipped_trials = 0
        skipped_windows = 0
        missing_audio_files = 0

        from scipy.io import wavfile

        for trial in trials:
            if trial.status is not TrialStatus.SUCCESS:
                skipped_trials += 1
                continue

            source_path = audio_dir / trial.audio_filename
            if not source_path.exists():
                missing_audio_files += 1
                continue

            standardized_name = f"{Path(trial.audio_filename).stem}_16k.wav"
            standardized_path = standardized_audio_dir / standardized_name
            standardize_wav_16k_mono(source_path, standardized_path)
            sample_rate, audio_16k = wavfile.read(standardized_path)
            if sample_rate != TARGET_SAMPLE_RATE:
                skipped_trials += 1
                continue

            duration_sec = len(audio_16k) / float(TARGET_SAMPLE_RATE)
            intervals = build_feature_intervals(trial, duration_sec)
            windows = build_sliding_windows(intervals)
            if not windows:
                skipped_trials += 1
                continue

            for window in windows:
                start_sample = max(0, int(round(window["start_sec"] * TARGET_SAMPLE_RATE)))
                end_sample = min(len(audio_16k), int(round(window["end_sec"] * TARGET_SAMPLE_RATE)))
                if end_sample <= start_sample:
                    skipped_windows += 1
                    continue

                seg_audio = audio_16k[start_sample:end_sample]
                features = self._extract_one_segment(seg_audio)
                if not features:
                    skipped_windows += 1
                    continue

                row: dict[str, Any] = {
                    "participant_id": trial.participant_id,
                    "session_id": trial.session_id,
                    "block": trial.block,
                    "block_index": trial.block_index,
                    "trial_number": trial.trial_number,
                    "trial_id": trial.trial_id,
                    "audio_filename": trial.audio_filename,
                    "audio_16k_filename": standardized_name,
                    "trial_status": trial.status.value,
                    "go_segmentation_enabled": trial.go_segmentation_enabled,
                    "window_index": int(window["window_index"]),
                    "window_start_sec": round(window["start_sec"], 4),
                    "window_end_sec": round(window["end_sec"], 4),
                    "window_duration_sec": round(window["duration_sec"], 4),
                    "interval_index": int(window["interval_index"]),
                    "interval_start_sec": round(window["interval_start_sec"], 4),
                    "interval_end_sec": round(window["interval_end_sec"], 4),
                }
                row.update(features)
                rows.append(row)
                processed_windows += 1

            processed_trials += 1

        if not rows:
            atomic_write_json(
                manifest_path,
                {
                    "metadata": metadata,
                    "feature_set": "eGeMAPSv02",
                    "feature_level": "Functionals",
                    "target_sample_rate_hz": TARGET_SAMPLE_RATE,
                    "go_context_pre_sec": GO_CONTEXT_PRE_SEC,
                    "go_context_post_sec": GO_CONTEXT_POST_SEC,
                    "sliding_window_sec": SLIDING_WINDOW_SEC,
                    "sliding_hop_sec": SLIDING_HOP_SEC,
                    "rows_written": 0,
                    "processed_trials": processed_trials,
                    "processed_windows": processed_windows,
                    "skipped_windows": skipped_windows,
                    "missing_audio_files": missing_audio_files,
                    "skipped_trials": skipped_trials,
                },
            )
            return

        import csv

        fieldnames: list[str] = list(rows[0].keys())
        for row in rows[1:]:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        features_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(features_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        atomic_write_json(
            manifest_path,
            {
                "metadata": metadata,
                "feature_set": "eGeMAPSv02",
                "feature_level": "Functionals",
                "target_sample_rate_hz": TARGET_SAMPLE_RATE,
                "go_context_pre_sec": GO_CONTEXT_PRE_SEC,
                "go_context_post_sec": GO_CONTEXT_POST_SEC,
                "sliding_window_sec": SLIDING_WINDOW_SEC,
                "sliding_hop_sec": SLIDING_HOP_SEC,
                "rows_written": len(rows),
                "processed_trials": processed_trials,
                "processed_windows": processed_windows,
                "skipped_windows": skipped_windows,
                "missing_audio_files": missing_audio_files,
                "skipped_trials": skipped_trials,
                "output_csv": str(features_csv_path),
                "standardized_audio_dir": str(standardized_audio_dir),
            },
        )

