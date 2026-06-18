"""CNN pain-score analyzer — evaluate the pretrained SimpleCNN against a
participant's GO segments using manually-entered 1-10 pain ratings.

Pipeline position:
    scripts/ml_segmenter.py  ->  (you fill in segments.csv `pain` 1-10)
    scripts/cnn_analyze.py   ->  per-segment predictions + accuracy report

Usage:
    python scripts/cnn_analyze.py <participant_id> [--data-dir data]
    python scripts/cnn_analyze.py --segments-dir data/<session>_segments

Reads the segments + segments.csv produced by ml_segmenter.py from the
participant's ..._segments/ folder. The `pain` column in segments.csv must
be filled in manually with a 1-10 pain rating for each segment; rows with a
blank/unparseable `pain` are skipped (with a count reported).

Outputs (written into the segments folder):
    cnn_predictions.csv   per-segment predicted vs true pain
    cnn_accuracy.json     regression + classification metrics

NOTE on mel-spectrogram params:
    The CNN expects a (1, 1, 128, 300) log-mel spectrogram normalized with
    portable_pain_cnn/spec_norm_stats.npz. These params replicate
    experiments/05-spectrogram_cnn exactly and must NOT be changed:
      sr=16000, n_mels=128, n_fft=2048, hop_length=512, duration=10.0 s,
      log via librosa.power_to_db(..., ref=np.max)   # per-utterance self-ref
      pad/truncate to 300 frames (front-truncate, pad with -80 dB)
      normalize: (spec - mean) / std  with per-mel-bin stats from the npz
    `torch`, `librosa`, and `numpy` must be installed on the running machine.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from portable_pain_cnn.model import SimpleCNN  # noqa: E402

logger = logging.getLogger("psycopy.cnn_analyze")

# ---------------------------------------------------------------------------
# Mel-spectrogram params — must match experiments/05-spectrogram_cnn exactly
# ---------------------------------------------------------------------------
SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
DURATION_SEC = 10.0  # librosa.load duration cap
MAX_FRAMES = 300  # time axis of the model input
PAD_VALUE = -80.0  # dB pad value for short utterances

PAIN_MIN = 1.0
PAIN_MAX = 10.0

CNN_DIR = REPO_ROOT / "portable_pain_cnn"
DEFAULT_MODEL_PATH = CNN_DIR / "cnn_model.pt"
DEFAULT_STATS_PATH = CNN_DIR / "spec_norm_stats.npz"


# =============================================================================
# Discovery
# =============================================================================


def find_segments_dir(data_dir: Path, participant_id: str) -> Path | None:
    """Find the most recent ..._segments folder matching the participant."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return None
    pid = participant_id.strip().lower()
    candidates: list[Path] = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name.endswith("_segments") and pid in name:
            candidates.append(child)
    candidates.sort(key=lambda p: p.name)
    return candidates[-1] if candidates else None


# =============================================================================
# Audio -> mel-spectrogram
# =============================================================================


def _wav_to_logmel(
    wav_path: Path,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    duration: float = DURATION_SEC,
    max_frames: int = MAX_FRAMES,
    pad_value: float = PAD_VALUE,
) -> np.ndarray:
    """Load a WAV and return a (1, 128, 300) dB log-mel spectrogram.

    Replicates experiments/05-spectrogram_cnn exactly:
      - librosa.load(sr=16000, mono=True, duration=10.0)
      - melspectrogram(n_mels=128, n_fft=2048, hop_length=512), power=2.0
      - power_to_db(ref=np.max)   # per-utterance self-normalization
      - front-truncate to [:, :300], else pad with -80 dB
      - add channel dim -> (1, 128, 300)
    """
    import librosa

    y, _ = librosa.load(str(wav_path), sr=sr, mono=True, duration=duration)
    if y.size == 0:
        y = np.zeros(int(sr * duration), dtype=np.float32)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Defensive: replace any non-finite values (e.g. from fully-silent clips)
    # with the pad floor. Normally a no-op since power_to_db clamps via its
    # default top_db=80, but keeps the model from ever seeing NaN/inf.
    if not np.all(np.isfinite(S_db)):
        S_db = np.nan_to_num(S_db, nan=pad_value, posinf=pad_value, neginf=pad_value)

    # Front-truncate or right-pad along the time axis (axis=1) to max_frames.
    frames = S_db.shape[1]
    if frames > max_frames:
        S_db = S_db[:, :max_frames]
    elif frames < max_frames:
        pad = max_frames - frames
        S_db = np.pad(S_db, ((0, 0), (0, pad)), mode="constant", constant_values=pad_value)

    S_db = np.ascontiguousarray(S_db, dtype=np.float32)
    return S_db[np.newaxis, ...]  # (1, 128, 300)


def _normalize(spec: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize with per-mel-bin stats.

    spec has shape (1, 128, 300); mean/std have shape (1, 128, 1).
    Matches training: (spec - mean) / std (std already guarded at train time).
    """
    return (spec - mean) / std


# =============================================================================
# Model
# =============================================================================


def load_model(model_path: Path, stats_path: Path) -> tuple[Any, np.ndarray, np.ndarray]:
    """Load the SimpleCNN weights + normalization stats."""
    import torch

    model = SimpleCNN()
    state = torch.load(str(model_path), map_location="cpu")
    if isinstance(state, dict) and any("state_dict" in k for k in state):
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    stats = np.load(str(stats_path))
    mean = stats["mean"].astype(np.float32)  # (1, 128, 1)
    std = stats["std"].astype(np.float32)
    return model, mean, std


def predict_pain(
    model: Any,
    wav_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    """Run the CNN on one segment WAV and return a pain score in [1, 10]."""
    import torch

    spec = _wav_to_logmel(wav_path)
    spec = _normalize(spec, mean, std)
    x = torch.from_numpy(spec).unsqueeze(0)  # (1, 1, 128, 300)
    with torch.no_grad():
        out = model(x)
    score = float(out.squeeze().item())
    return float(min(PAIN_MAX, max(PAIN_MIN, score)))


# =============================================================================
# Metrics
# =============================================================================


def _build_bins(num_bins: int, lo: int = 1, hi: int = 10) -> list[tuple[int, int]]:
    """Split the integer range [lo, hi] into num_bins contiguous bins."""
    total = hi - lo + 1
    base = total // num_bins
    rem = total % num_bins
    bins: list[tuple[int, int]] = []
    cur = lo
    for i in range(num_bins):
        size = base + (1 if i < rem else 0)
        bins.append((cur, cur + size - 1))
        cur += size
    return bins


def _bin_index(val: float, bins: list[tuple[int, int]]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= val <= hi:
            return i
    return len(bins) - 1 if val > bins[-1][1] else 0


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if n > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = float("nan")
    within_1 = float(np.mean(np.abs(err) <= 1.0)) if n else 0.0
    within_2 = float(np.mean(np.abs(err) <= 2.0)) if n else 0.0
    return {
        "n": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "pearson_r": round(r, 4),
        "within_1": round(within_1, 4),
        "within_2": round(within_2, 4),
        "mean_true": round(float(np.mean(y_true)), 4) if n else 0.0,
        "mean_pred": round(float(np.mean(y_pred)), 4) if n else 0.0,
        "std_true": round(float(np.std(y_true)), 4) if n else 0.0,
        "std_pred": round(float(np.std(y_pred)), 4) if n else 0.0,
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: list[tuple[int, int]],
) -> dict[str, Any]:
    n_classes = len(bins)
    t_labels = np.array([_bin_index(v, bins) for v in y_true])
    p_labels = np.array([_bin_index(v, bins) for v in y_pred])
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(t_labels, p_labels):
        cm[t, p] += 1

    per_class: dict[int, dict[str, Any]] = {}
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "range": list(bins[c]),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[c, :].sum()),
        }

    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    return {
        "bins": [list(b) for b in bins],
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


# =============================================================================
# CSV I/O
# =============================================================================


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Main analysis
# =============================================================================


def analyze(
    segments_dir: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    stats_path: Path = DEFAULT_STATS_PATH,
    num_bins: int = 3,
) -> dict[str, Any]:
    """Run the CNN over all rated segments and write predictions + metrics."""
    segments_dir = Path(segments_dir)
    csv_path = segments_dir / "segments.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No segments.csv in {segments_dir}")

    model, mean, std = load_model(model_path, stats_path)

    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    bins = _build_bins(num_bins)
    pred_rows: list[dict[str, Any]] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    modalities: list[str] = []
    n_skipped_no_pain = 0
    n_skipped_no_wav = 0
    max_pain_seen: float | None = None

    for row in rows:
        seg_name = (row.get("segment_filename") or "").strip()
        if not seg_name:
            continue
        wav_path = segments_dir / seg_name

        pain_str = (row.get("pain") or "").strip()
        try:
            true_pain = float(pain_str)
        except ValueError:
            n_skipped_no_pain += 1
            continue

        if max_pain_seen is None or true_pain > max_pain_seen:
            max_pain_seen = true_pain

        if not wav_path.exists():
            n_skipped_no_wav += 1
            logger.warning("Missing WAV: %s", wav_path.name)
            continue

        audio_type = (row.get("audio_type") or "").strip() or "unknown"
        pred = predict_pain(model, wav_path, mean, std)
        y_true.append(true_pain)
        y_pred.append(pred)
        modalities.append(audio_type)

        pred_rows.append(
            {
                "segment_filename": seg_name,
                "trial_instance_id": row.get("trial_instance_id", ""),
                "audio_type": audio_type,
                "segment_index": row.get("segment_index", ""),
                "true_pain": round(true_pain, 2),
                "predicted_pain": round(pred, 2),
                "abs_error": round(abs(pred - true_pain), 2),
                "temperature_celsius": row.get("temperature_celsius", ""),
                "duration_sec": row.get("duration_sec", ""),
            }
        )

    n_total = len(rows)
    n_eval = len(y_true)

    if max_pain_seen is not None and max_pain_seen > PAIN_MAX:
        logger.warning(
            "Found a `pain` value of %.2f (>10). The `pain` column may still hold "
            "temperature_celsius — re-run ml_segmenter.py to get the new schema, "
            "then fill in 1-10 ratings manually.",
            max_pain_seen,
        )

    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    # Per-modality breakdown (vowel vs speech) so you can see whether the
    # CNN generalizes across both audio types.
    by_modality: dict[str, Any] = {}
    mod_set = sorted(set(modalities))
    for mod in mod_set:
        idx = [i for i, m in enumerate(modalities) if m == mod]
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        by_modality[mod] = {
            "n": len(idx),
            "regression": _regression_metrics(yt, yp),
            "classification": _classification_metrics(yt, yp, bins),
        }

    report: dict[str, Any] = {
        "segments_dir": str(segments_dir),
        "model_path": str(model_path),
        "n_segments": n_total,
        "n_evaluated": n_eval,
        "n_skipped_no_pain": n_skipped_no_pain,
        "n_skipped_no_wav": n_skipped_no_wav,
        "regression": _regression_metrics(y_true_arr, y_pred_arr),
        "classification": _classification_metrics(y_true_arr, y_pred_arr, bins),
        "by_modality": by_modality,
    }

    pred_csv = segments_dir / "cnn_predictions.csv"
    if pred_rows:
        _write_csv(pred_csv, pred_rows)
        logger.info("Wrote %d predictions to %s", len(pred_rows), pred_csv)

    report_path = segments_dir / "cnn_accuracy.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Wrote accuracy report to %s", report_path)

    return report


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the pretrained Pain CNN against a participant's segments."
    )
    parser.add_argument(
        "participant_id",
        nargs="?",
        help="Participant ID (e.g. 001). Scans data/ for the matching ..._segments folder.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing ..._segments folders (default: data/)",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=None,
        help="Point directly at a ..._segments folder instead of scanning by participant.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to cnn_model.pt (default: portable_pain_cnn/cnn_model.pt)",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=DEFAULT_STATS_PATH,
        help="Path to spec_norm_stats.npz (default: portable_pain_cnn/spec_norm_stats.npz)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=3,
        help="Number of contiguous pain bins for classification (default: 3).",
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

    if args.segments_dir:
        segments_dir = Path(args.segments_dir)
    elif args.participant_id:
        segments_dir = find_segments_dir(args.data_dir, args.participant_id)
        if segments_dir is None:
            logger.error(
                "No ..._segments folder for participant '%s' in %s",
                args.participant_id,
                args.data_dir,
            )
            return 1
    else:
        parser.error("Provide a participant_id or --segments-dir.")

    if not segments_dir.is_dir():
        logger.error("Not a directory: %s", segments_dir)
        return 1

    logger.info("Analyzing: %s", segments_dir)
    try:
        report = analyze(
            segments_dir,
            model_path=args.model,
            stats_path=args.stats,
            num_bins=args.num_bins,
        )
    except Exception as exc:
        logger.error("Failed: %s", exc)
        return 1

    reg = report["regression"]
    clf = report["classification"]
    print()
    print(f"Segments dir : {segments_dir}")
    print(f"Evaluated    : {report['n_evaluated']}/{report['n_segments']} "
          f"(skipped {report['n_skipped_no_pain']} unrated, "
          f"{report['n_skipped_no_wav']} missing WAV)")
    print(f"MAE          : {reg['mae']}")
    print(f"RMSE         : {reg['rmse']}")
    print(f"R^2          : {reg['r2']}")
    print(f"Pearson r    : {reg['pearson_r']}")
    print(f"Within +/-1  : {reg['within_1']:.1%}")
    print(f"Within +/-2  : {reg['within_2']:.1%}")
    print(f"Class acc    : {clf['accuracy']:.1%} over {len(clf['bins'])} bins")

    by_mod = report.get("by_modality", {})
    if len(by_mod) > 1 or (len(by_mod) == 1 and "unknown" not in by_mod):
        print("By modality  :")
        for mod, mstats in by_mod.items():
            mr = mstats["regression"]
            mc = mstats["classification"]
            print(f"  {mod:<8} n={mstats['n']:<4} "
                  f"MAE={mr['mae']:<6} r={mr['pearson_r']:<6} "
                  f"acc={mc['accuracy']:.1%}")

    print(f"Predictions  : {segments_dir / 'cnn_predictions.csv'}")
    print(f"Report       : {segments_dir / 'cnn_accuracy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
