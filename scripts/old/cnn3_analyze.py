"""CNN 3-class pain-level analyzer — evaluate the pretrained Keras 3-class CNN
against a participant's GO segments using manually-entered 1-10 pain ratings.

Pipeline position:
    scripts/ml_segmenter.py  ->  (you fill in segments.csv `pain` 1-10)
    scripts/cnn3_analyze.py ->  per-segment predictions + classification report

Usage:
    python scripts/cnn3_analyze.py <participant_id> [--data-dir data]
    python scripts/cnn3_analyze.py --segments-dir data/<session>_segments

Reads the segments + segments.csv produced by ml_segmenter.py from the
participant's ..._segments/ folder. The `pain` column in segments.csv must
be filled in manually with a 1-10 pain rating for each segment; rows with a
blank/unparseable `pain` are skipped (with a count reported).

Pain-to-class mapping:
    Low    : 1-3  -> class 0
    Medium : 4-7  -> class 1
    High   : 8-10 -> class 2

Outputs (written into the segments folder):
    cnn3_predictions.csv   per-segment predicted vs true class + probabilities
    cnn3_accuracy.json     classification metrics + confusion matrix
    cnn3_confusion_matrix.csv
    cnn3_worst_predictions.csv   top-k segments with wrong predictions

Preprocessing (must match TAME training exactly):
    sr=16000, n_mels=64, n_fft=2048, hop_length=1024, duration=4.0 s
    log-mel via librosa.power_to_db(..., ref=np.max)
    pad audio to exactly 4.0 s (silence pad)
    StandardScaler normalization from spec_norm_stats.npz
    Input to Keras model: (1, 64, 63, 1)

`tensorflow`, `librosa`, `numpy`, and `scikit-learn` must be installed.
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

# Optional plotting (install matplotlib if you want PNG figures)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("psycopy.cnn3_analyze")

# ---------------------------------------------------------------------------
# Preprocessing params — must match TAME training exactly
# ---------------------------------------------------------------------------
SR = 16000
N_FFT = 2048
HOP_LENGTH = 1024
N_MELS = 64
DURATION_SEC = 4.0
N_SAMPLES = int(SR * DURATION_SEC)  # 64000

PAIN_MIN = 1.0
PAIN_MAX = 10.0

CNN_DIR = REPO_ROOT / "portable_pain_cnn"
DEFAULT_MODEL_PATH = CNN_DIR / "pain_cnn_model.h5"
DEFAULT_STATS_PATH = CNN_DIR / "spec_norm_stats.npz"

CLASS_LABELS = {0: "low", 1: "medium", 2: "high"}
N_CLASSES = 3


# =============================================================================
# Helpers
# =============================================================================


def _pain_to_class(pain: float) -> int:
    """Map a 1-10 pain rating to a 0-2 class index."""
    pain_int = int(round(pain))
    if 1 <= pain_int <= 3:
        return 0
    if 4 <= pain_int <= 7:
        return 1
    if 8 <= pain_int <= 10:
        return 2
    raise ValueError(f"Pain value {pain} out of range [{PAIN_MIN}, {PAIN_MAX}]")


def _class_to_label(cls: int) -> str:
    return CLASS_LABELS.get(cls, "unknown")


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
    n_samples: int = N_SAMPLES,
) -> np.ndarray:
    """Load a WAV and return a (n_mels, n_frames) dB log-mel spectrogram.

    Replicates TAME training exactly:
      - librosa.load(sr=16000, mono=True, duration=4.0)
      - pad with silence to exactly 4.0 s (64000 samples)
      - melspectrogram(n_mels=64, n_fft=2048, hop_length=1024), power=2.0
      - power_to_db(ref=np.max)   # per-utterance self-normalization
    """
    import librosa

    y, _ = librosa.load(str(wav_path), sr=sr, mono=True, duration=DURATION_SEC)
    # Pad to exact duration
    if y.size < n_samples:
        pad = n_samples - y.size
        y = np.pad(y, (0, pad), mode="constant", constant_values=0.0)
    elif y.size > n_samples:
        y = y[:n_samples]

    if y.size == 0:
        y = np.zeros(n_samples, dtype=np.float32)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Defensive: replace any non-finite values
    if not np.all(np.isfinite(S_db)):
        S_db = np.nan_to_num(S_db, nan=-80.0, posinf=-80.0, neginf=-80.0)

    return np.ascontiguousarray(S_db, dtype=np.float32)  # (64, ~63)


# =============================================================================
# Normalization (StandardScaler from training)
# =============================================================================


def _load_scaler(stats_path: Path) -> Any:
    """Rebuild a StandardScaler from the saved npz."""
    from sklearn.preprocessing import StandardScaler

    stats = np.load(str(stats_path))
    mean = stats["mean"].astype(np.float64)
    scale = stats["scale"].astype(np.float64)

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.n_features_in_ = mean.shape[0]
    return scaler


def _normalize(spec: np.ndarray, scaler: Any) -> np.ndarray:
    """StandardScaler normalize a spectrogram.

    Flatten -> scale -> reshape back to original 2D shape.
    """
    original_shape = spec.shape
    flat = spec.flatten()
    norm = (flat - scaler.mean_) / scaler.scale_
    return norm.reshape(original_shape)


# =============================================================================
# Model
# =============================================================================


def load_model(model_path: Path, stats_path: Path) -> tuple[Any, Any]:
    """Load the Keras model + StandardScaler."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "tensorflow is required for cnn3_analyze. "
            "Install it with: pip install tensorflow"
        ) from exc

    tf.get_logger().setLevel("ERROR")
    model = tf.keras.models.load_model(str(model_path))
    scaler = _load_scaler(stats_path)
    return model, scaler


def predict_class(
    model: Any,
    wav_path: Path,
    scaler: Any,
) -> tuple[int, np.ndarray]:
    """Run the CNN on one segment WAV and return (predicted_class, probabilities)."""
    spec = _wav_to_logmel(wav_path)
    spec = _normalize(spec, scaler)
    x = spec[np.newaxis, ..., np.newaxis]  # (1, 64, n_frames, 1)
    probs = model.predict(x, verbose=0)[0]  # (3,)
    pred_class = int(np.argmax(probs))
    return pred_class, probs


# =============================================================================
# Metrics
# =============================================================================


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = N_CLASSES,
) -> dict[str, Any]:
    """Compute classification metrics from integer class labels."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
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
            "label": CLASS_LABELS[c],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[c, :].sum()),
        }

    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    macro_prec = float(
        np.mean([per_class[c]["precision"] for c in range(n_classes)])
    )
    macro_rec = float(
        np.mean([per_class[c]["recall"] for c in range(n_classes)])
    )
    macro_f1 = float(np.mean([per_class[c]["f1"] for c in range(n_classes)]))

    # Cohen's kappa
    po = accuracy
    pe = sum(cm[i, :].sum() * cm[:, i].sum() for i in range(n_classes)) / (total**2) if total > 0 else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    return {
        "n_classes": n_classes,
        "n_evaluated": total,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def _print_confusion_matrix(cm: list[list[int]], labels: list[str]) -> None:
    """Pretty-print a confusion matrix to stdout."""
    n_classes = len(cm)
    max_val = max(max(row) for row in cm)
    col_width = max(8, len(str(max_val)) + 1)
    header = "Pred ->"
    print(
        f"\n{'Confusion matrix':<{col_width + 4}} "
        f"{header:>{(n_classes * (col_width + 1)) // 2}}"
    )
    print(f"{'True  |':<{col_width + 4}}", end="")
    for lbl in labels:
        print(f"{lbl:>{col_width}}", end=" ")
    print()
    print("-" * ((col_width + 4) + n_classes * (col_width + 1)))
    for i, row in enumerate(cm):
        print(f"{labels[i] + ' |':<{col_width + 4}}", end="")
        for val in row:
            print(f"{val:>{col_width}}", end=" ")
        print()
    print()


# =============================================================================
# Plots (optional — requires matplotlib)
# =============================================================================


def _plot_confusion_heatmap(
    cm: list[list[int]],
    labels: list[str],
    out_path: Path,
) -> None:
    """Confusion matrix heatmap PNG."""
    if plt is None:
        logger.warning("matplotlib not installed; skipping confusion heatmap")
        return
    cm_arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            text = ax.text(
                j, i, cm_arr[i, j], ha="center", va="center", color="black"
            )
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote confusion heatmap to %s", out_path)


def _plot_per_class_bar(
    metrics: dict[str, Any],
    out_path: Path,
) -> None:
    """Bar chart of per-class precision/recall/f1."""
    if plt is None:
        logger.warning("matplotlib not installed; skipping per-class bar chart")
        return
    per_class = metrics["per_class"]
    labels = [per_class[c]["label"] for c in range(N_CLASSES)]
    precisions = [per_class[c]["precision"] for c in range(N_CLASSES)]
    recalls = [per_class[c]["recall"] for c in range(N_CLASSES)]
    f1s = [per_class[c]["f1"] for c in range(N_CLASSES)]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width, precisions, width, label="Precision", color="steelblue")
    ax.bar(x, recalls, width, label="Recall", color="coral")
    ax.bar(x + width, f1s, width, label="F1", color="seagreen")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote per-class bar chart to %s", out_path)


def _plot_probability_distribution(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Stacked bar of predicted class probabilities by true class."""
    if plt is None or not rows:
        return

    # Aggregate probabilities by true class
    probs_by_true: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    for row in rows:
        true_cls = row["true_class"]
        probs = np.array([row["prob_low"], row["prob_medium"], row["prob_high"]])
        probs_by_true[true_cls].append(probs)

    labels = [CLASS_LABELS[c] for c in range(N_CLASSES)]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(N_CLASSES)
    width = 0.6

    bottoms = np.zeros(N_CLASSES)
    colors = ["steelblue", "coral", "seagreen"]
    for pred_c in range(N_CLASSES):
        vals = []
        for true_c in range(N_CLASSES):
            if probs_by_true[true_c]:
                mean_prob = np.mean([p[pred_c] for p in probs_by_true[true_c]])
            else:
                mean_prob = 0.0
            vals.append(mean_prob)
        ax.bar(x, vals, width, bottom=bottoms, label=CLASS_LABELS[pred_c], color=colors[pred_c])
        bottoms += np.array(vals)

    ax.set_ylabel("Mean Predicted Probability")
    ax.set_title("Mean Predicted Probabilities by True Class")
    ax.set_xticks(x)
    ax.set_xticklabels([f"True: {lbl}" for lbl in labels])
    ax.set_ylim(0, 1.05)
    ax.legend(title="Predicted", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote probability distribution plot to %s", out_path)


# =============================================================================
# Worst predictions
# =============================================================================


def _write_worst_predictions(
    pred_rows: list[dict[str, Any]],
    out_path: Path,
    top_k: int = 5,
) -> None:
    """Write a CSV of the top-k most confidently wrong predictions."""
    if not pred_rows:
        return
    # Sort by confidence of the *wrong* predicted class (descending)
    wrong = [r for r in pred_rows if not r["correct"]]
    if not wrong:
        return
    wrong.sort(key=lambda r: r.get("max_prob", 0), reverse=True)
    worst = wrong[:top_k]
    fieldnames = list(worst[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(worst)
    logger.info("Wrote worst %d predictions to %s", len(worst), out_path)


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
    plots: bool = True,
) -> dict[str, Any]:
    """Run the 3-class CNN over all rated segments and write predictions + metrics."""
    segments_dir = Path(segments_dir)
    csv_path = segments_dir / "segments.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No segments.csv in {segments_dir}")

    model, scaler = load_model(model_path, stats_path)

    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    pred_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    modalities: list[str] = []
    seg_labels: list[str] = []
    n_skipped_no_pain = 0
    n_skipped_no_wav = 0
    n_skipped_out_of_range = 0
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

        # Validate pain range before mapping to class
        if not (PAIN_MIN <= true_pain <= PAIN_MAX):
            n_skipped_out_of_range += 1
            logger.warning(
                "Pain value %.2f out of range [%.0f, %.0f] for %s — skipping",
                true_pain, PAIN_MIN, PAIN_MAX, seg_name,
            )
            continue

        try:
            true_class = _pain_to_class(true_pain)
        except ValueError:
            n_skipped_out_of_range += 1
            continue

        if not wav_path.exists():
            n_skipped_no_wav += 1
            logger.warning("Missing WAV: %s", wav_path.name)
            continue

        audio_type = (row.get("audio_type") or "").strip() or "unknown"
        pred_class, probs = predict_class(model, wav_path, scaler)

        y_true.append(true_class)
        y_pred.append(pred_class)
        modalities.append(audio_type)
        seg_labels.append(seg_name)

        correct = pred_class == true_class
        pred_rows.append(
            {
                "segment_filename": seg_name,
                "trial_instance_id": row.get("trial_instance_id", ""),
                "audio_type": audio_type,
                "segment_index": row.get("segment_index", ""),
                "true_pain": round(true_pain, 2),
                "true_class": _class_to_label(true_class),
                "predicted_class": _class_to_label(pred_class),
                "prob_low": round(float(probs[0]), 4),
                "prob_medium": round(float(probs[1]), 4),
                "prob_high": round(float(probs[2]), 4),
                "max_prob": round(float(np.max(probs)), 4),
                "correct": correct,
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

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    # Per-modality breakdown
    by_modality: dict[str, Any] = {}
    mod_set = sorted(set(modalities))
    for mod in mod_set:
        idx = [i for i, m in enumerate(modalities) if m == mod]
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        by_modality[mod] = {
            "n": len(idx),
            "metrics": _classification_metrics(yt, yp),
        }

    report: dict[str, Any] = {
        "segments_dir": str(segments_dir),
        "model_path": str(model_path),
        "stats_path": str(stats_path),
        "n_segments": n_total,
        "n_evaluated": n_eval,
        "n_skipped_no_pain": n_skipped_no_pain,
        "n_skipped_no_wav": n_skipped_no_wav,
        "n_skipped_out_of_range": n_skipped_out_of_range,
        "metrics": _classification_metrics(y_true_arr, y_pred_arr),
        "by_modality": by_modality,
        "plot_paths": {},
        "worst_predictions_path": None,
    }

    pred_csv = segments_dir / "cnn3_predictions.csv"
    if pred_rows:
        _write_csv(pred_csv, pred_rows)
        logger.info("Wrote %d predictions to %s", len(pred_rows), pred_csv)

    # Confusion matrix CSV
    cm = report["metrics"]["confusion_matrix"]
    cm_csv = segments_dir / "cnn3_confusion_matrix.csv"
    labels = [_class_to_label(c) for c in range(N_CLASSES)]
    with open(cm_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["True \\ Pred"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + row)
    logger.info("Wrote confusion matrix to %s", cm_csv)

    # Worst predictions
    worst_csv = segments_dir / "cnn3_worst_predictions.csv"
    _write_worst_predictions(pred_rows, worst_csv, top_k=5)
    report["worst_predictions_path"] = str(worst_csv)

    # Optional figures
    if plots and n_eval > 0:
        plot_dir = segments_dir
        cm_png = plot_dir / "cnn3_confusion_heatmap.png"
        bar_png = plot_dir / "cnn3_per_class_metrics.png"
        prob_png = plot_dir / "cnn3_probability_distribution.png"
        _plot_confusion_heatmap(cm, labels, cm_png)
        _plot_per_class_bar(report["metrics"], bar_png)
        _plot_probability_distribution(pred_rows, prob_png)
        report["plot_paths"] = {
            "confusion_heatmap": str(cm_png),
            "per_class_metrics": str(bar_png),
            "probability_distribution": str(prob_png),
        }

    report_path = segments_dir / "cnn3_accuracy.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Wrote accuracy report to %s", report_path)

    return report


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the pretrained 3-class Pain CNN against a participant's segments."
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
        help="Path to pain_cnn_model.h5 (default: portable_pain_cnn/pain_cnn_model.h5)",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=DEFAULT_STATS_PATH,
        help="Path to spec_norm_stats.npz (default: portable_pain_cnn/spec_norm_stats.npz)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating PNG plots (default: plots are generated).",
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
            plots=not args.no_plots,
        )
    except Exception as exc:
        logger.error("Failed: %s", exc)
        return 1

    m = report["metrics"]
    n_eval = m["n_evaluated"]
    print()
    print(f"Segments dir : {segments_dir}")
    print(f"Evaluated    : {n_eval}/{report['n_segments']} "
          f"(skipped {report['n_skipped_no_pain']} unrated, "
          f"{report['n_skipped_no_wav']} missing WAV, "
          f"{report['n_skipped_out_of_range']} out-of-range)")
    if n_eval < 10:
        print("WARNING      : Only %d segments evaluated. Metrics may be unstable; "
              "do not over-interpret." % n_eval)
    print(f"Accuracy     : {m['accuracy']:.1%}")
    print(f"Macro P/R/F1 : {m['macro_precision']:.3f} / {m['macro_recall']:.3f} / {m['macro_f1']:.3f}")
    print(f"Cohen's kappa: {m['cohens_kappa']:.4f}")

    _print_confusion_matrix(m["confusion_matrix"], [CLASS_LABELS[c] for c in range(N_CLASSES)])

    print("Per-class    :")
    for c in range(N_CLASSES):
        pc = m["per_class"][c]
        print(f"  {pc['label']:<8} n={pc['support']:<4} "
              f"P={pc['precision']:<6} R={pc['recall']:<6} F1={pc['f1']:<6}")

    by_mod = report.get("by_modality", {})
    if len(by_mod) > 1 or (len(by_mod) == 1 and "unknown" not in by_mod):
        print("By modality  :")
        for mod, mstats in by_mod.items():
            mm = mstats["metrics"]
            print(f"  {mod:<8} n={mstats['n']:<4} "
                  f"acc={mm['accuracy']:.1%} "
                  f"F1={mm['macro_f1']:.3f}")

    print(f"Predictions  : {segments_dir / 'cnn3_predictions.csv'}")
    print(f"Worst preds  : {segments_dir / 'cnn3_worst_predictions.csv'}")
    print(f"Confusion    : {segments_dir / 'cnn3_confusion_matrix.csv'}")
    print(f"Report       : {segments_dir / 'cnn3_accuracy.json'}")

    plot_paths = report.get("plot_paths", {})
    if plot_paths:
        print("Plots        :")
        for name, pth in plot_paths.items():
            print(f"  {name:<25} {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
