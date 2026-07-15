"""Organize + merge multi-session participant data into a single pipeline input.

When a Medoc failure or crash forces a re-run, one logical session ends up
scattered across multiple timestamped folders (each with a *different* session
ID).  This script reassembles them *before* downstream post-processing runs,
run, so the rest of the pipeline can treat the merged folder as a single
session.

Layout produced::

    data/p001/                          # original raw — never touched
        {ts}_sub-001_session-01_task-speech/
        {ts}_sub-001_session-02_task-speech/
        {ts}_sub-001_session-01_task-vowel/

    data/p001-processed/                # everything processed lives here
        participant_info.json          # shared demographics (one per participant)
        questionnaires.csv             # shared blank scores (temps + PCS + PANAS)
        pain_ratings_speech.csv        # speech GO segments; blank pain_rating
        pain_ratings_vowel.csv         # vowel GO segments; blank pain_rating
        raw/                           # audit-trail copy of originals
            {ts}_sub-001_session-01_task-speech/
            ...
        merged_task-speech/             # organize output -> downstream pipeline input
            audio/
            events.csv
            medoc_events.csv
            trials.csv
            config.json
            merge_report.json
        merged_task-vowel/

Usage::

    python scripts/organize_sessions.py data/p001
    python scripts/organize_sessions.py data/p001 --task speech
    python scripts/organize_sessions.py data/p001 --dry-run
    python scripts/organize_sessions.py data/p001 --force
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("psycopy.organize")

# Folder-name pattern: {timestamp}_sub-{participant}_session-{session}_task-{task}
_FOLDER_RE = re.compile(
    r"^(?P<ts>\d{8}_\d{6})_"
    r"sub-(?P<participant>[^_]+)_"
    r"session-(?P<session>[^_]+)_"
    r"task-(?P<task>[^_]+)$"
)

# Expected blocks for non-practice modes (4 blocks, 1 trial each).
_EXPECTED_BLOCKS = 4
_TRIALS_PER_BLOCK = 1


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True, slots=True)
class SessionFolder:
    """One timestamped session directory discovered under the participant folder."""

    path: Path
    participant: str
    session: str
    task: str  # "speech" or "vowel"


@dataclass(slots=True)
class TrialInfo:
    """Health + provenance info for a single trial across all source folders."""

    trial_instance_id: str
    source_folder: str  # folder name
    session: str
    block: str
    trial_num: int
    has_recording_start: bool = False
    has_recording_end: bool = False
    has_trial_end: bool = False
    expected_go_segments: int = 0
    actual_go_cues: int = 0
    wav_filename: str = ""
    wav_duration_sec: float | None = None
    medoc_present: bool = False
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Folder discovery
# =============================================================================


def _parse_folder_name(path: Path) -> SessionFolder | None:
    """Parse a session folder name into a SessionFolder, or None if no match."""
    m = _FOLDER_RE.match(path.name)
    if m is None:
        return None
    return SessionFolder(
        path=path,
        participant=m.group("participant"),
        session=m.group("session"),
        task=m.group("task"),
    )


def discover_session_folders(participant_dir: Path) -> list[SessionFolder]:
    """Find all session folders directly inside *participant_dir*."""
    folders: list[SessionFolder] = []
    for child in sorted(participant_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "raw" or child.name.endswith("-processed"):
            continue
        parsed = _parse_folder_name(child)
        if parsed is None:
            logger.debug("Skipping non-session folder: %s", child.name)
            continue
        folders.append(parsed)
    return folders


def group_by_task(folders: list[SessionFolder]) -> dict[str, list[SessionFolder]]:
    """Group session folders by task type ('speech', 'vowel', etc.)."""
    groups: dict[str, list[SessionFolder]] = {}
    for f in folders:
        groups.setdefault(f.task, []).append(f)
    for task in groups:
        groups[task].sort(key=lambda x: x.session)
    return groups


# =============================================================================
# CSV helpers (read-only; merge writes are done with the storage module)
# =============================================================================


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def _wav_duration(path: Path) -> float | None:
    """Return WAV duration in seconds without loading the full signal."""
    if not path.exists():
        return None
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception as exc:
        logger.warning("Could not read WAV %s: %s", path.name, exc)
        return None


# =============================================================================
# Trial indexing + health checks
# =============================================================================


def _extract_trial_num(trial_instance_id: str) -> int:
    """Parse the trial number from a trial_instance_id (last underscore part)."""
    parts = trial_instance_id.split("_")
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def index_trials(
    folders: list[SessionFolder],
    expected_blocks: int = _EXPECTED_BLOCKS,
) -> tuple[list[TrialInfo], list[dict[str, Any]]]:
    """Index every trial across all folders and run health checks.

    Returns ``(trials, gaps)`` where *trials* is a list of TrialInfo for every
    trial that has at least a ``trial_start`` or ``recording_start`` event,
    and *gaps* lists blocks for which no trial was found in any folder.
    """
    trial_map: dict[str, TrialInfo] = {}

    for folder in folders:
        events_csv = folder.path / "events.csv"
        events = _load_csv_rows(events_csv)
        medoc_rows = _load_csv_rows(folder.path / "medoc_events.csv")

        # Collect medoc trial IDs that have at least one poll with temperature
        medoc_trial_ids: set[str] = set()
        for row in medoc_rows:
            tid = row.get("trial_instance_id", "").strip()
            temp = row.get("temperature_celsius", "").strip()
            if tid and temp:
                medoc_trial_ids.add(tid)

        # Gather per-trial event presence
        seen_trials: set[str] = set()
        for row in events:
            tid = row.get("trial_instance_id", "").strip()
            if not tid:
                continue
            seen_trials.add(tid)
            if tid not in trial_map:
                trial_map[tid] = TrialInfo(
                    trial_instance_id=tid,
                    source_folder=folder.path.name,
                    session=folder.session,
                    block=row.get("block", ""),
                    trial_num=_extract_trial_num(tid),
                )

            info = trial_map[tid]
            et = row.get("event_type", "")

            if et == "trial_start":
                data = _parse_event_data(row.get("event_data", ""))
                info.expected_go_segments = int(data.get("num_go_segments", 0))
            elif et == "recording_start":
                info.has_recording_start = True
                data = _parse_event_data(row.get("event_data", ""))
                # Keep the audio_type from event_data for reference
            elif et == "recording_end":
                info.has_recording_end = True
            elif et == "trial_end":
                info.has_trial_end = True
            elif et == "go_cue":
                info.actual_go_cues += 1

        # Resolve WAV for each trial discovered in this folder
        audio_dir = folder.path / "audio"
        for tid in seen_trials:
            info = trial_map[tid]
            wav = _resolve_wav_in_dir(audio_dir, tid)
            if wav is not None and not info.wav_filename:
                info.wav_filename = wav.name
                info.wav_duration_sec = _wav_duration(wav)
            if tid in medoc_trial_ids:
                info.medoc_present = True

    # Run health checks
    trials = sorted(trial_map.values(), key=lambda t: (t.session, t.block, t.trial_num))
    for info in trials:
        _check_trial_health(info, expected_blocks)

    # Detect gaps: expected blocks that have no trial in any folder
    found_blocks: set[str] = {t.block for t in trials}
    gaps: list[dict[str, Any]] = []
    for i in range(expected_blocks):
        block_name = f"block{i}"
        if block_name not in found_blocks:
            gaps.append(
                {
                    "block": block_name,
                    "trial_num": 0,
                    "reason": "no trial_start or recording_start found in any source folder",
                }
            )

    return trials, gaps


def _resolve_wav_in_dir(audio_dir: Path, trial_instance_id: str) -> Path | None:
    """Match a trial_instance_id to a WAV file inside *audio_dir*."""
    if not audio_dir.is_dir():
        return None

    # Direct match by trial_id substring
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

    # Fallback: single WAV in directory
    all_wavs = list(audio_dir.glob("*.wav"))
    if len(all_wavs) == 1:
        return all_wavs[0]

    return None


def _check_trial_health(info: TrialInfo, expected_blocks: int) -> None:
    """Append warnings to TrialInfo based on completeness checks."""
    if not info.has_recording_start:
        info.warnings.append("missing recording_start")
    if not info.has_recording_end:
        info.warnings.append("missing recording_end")
    if not info.has_trial_end:
        info.warnings.append("missing trial_end (trial may have crashed)")

    if info.expected_go_segments > 0 and info.actual_go_cues < info.expected_go_segments:
        info.warnings.append(
            f"go_cue count {info.actual_go_cues} < expected {info.expected_go_segments}"
        )

    if info.wav_duration_sec is not None:
        # Speech/vowel trials are ~240 s; flag anything under 200 s
        if info.wav_duration_sec < 200.0:
            info.warnings.append(
                f"short WAV ({info.wav_duration_sec:.1f}s, expected ~240s)"
            )
    elif info.has_recording_start:
        info.warnings.append("recording_start logged but WAV file not found")

    if not info.medoc_present:
        info.warnings.append("no medoc temperature data for this trial")


# =============================================================================
# Merging
# =============================================================================


def _rename_wav(original: Path, session: str) -> str:
    """Build a session-qualified filename to avoid collisions across sessions.

    Original:  sub-001_block-0_trial-000.wav
    Renamed:   sub-001_session-01_block-0_trial-000.wav
    """
    stem = original.stem
    if f"session-{session}" in stem:
        return original.name  # already qualified
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].startswith("sub-"):
        return f"{parts[0]}_session-{session}_" + "_".join(parts[1:]) + ".wav"
    return f"session-{session}_{original.name}"


def _merge_csv(
    output_path: Path,
    folders: list[SessionFolder],
    csv_filename: str,
) -> int:
    """Concatenate rows from each folder's CSV into a single merged CSV.

    Returns the total number of data rows written.
    """
    all_rows: list[dict[str, str]] = []
    for folder in folders:
        rows = _load_csv_rows(folder.path / csv_filename)
        all_rows.extend(rows)

    if not all_rows:
        return 0

    # Collect union of all fieldnames, preserving first-seen order
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in all_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(all_rows)

    return len(all_rows)


def _copy_trial_audio(
    trials: list[TrialInfo],
    folders: list[SessionFolder],
    dest_audio_dir: Path,
) -> dict[str, str]:
    """Copy each trial's WAV into *dest_audio_dir* with session-qualified names.

    Returns a mapping of ``trial_instance_id -> renamed_filename``.
    """
    dest_audio_dir.mkdir(parents=True, exist_ok=True)
    renamed: dict[str, str] = {}

    # Map source_folder name -> SessionFolder for quick lookup
    folder_map = {f.path.name: f for f in folders}

    for info in trials:
        if not info.wav_filename:
            continue
        folder = folder_map.get(info.source_folder)
        if folder is None:
            continue
        src_wav = folder.path / "audio" / info.wav_filename
        if not src_wav.exists():
            logger.warning("WAV not found on disk: %s", src_wav)
            continue
        new_name = _rename_wav(src_wav, info.session)
        dst_wav = dest_audio_dir / new_name
        shutil.copy2(src_wav, dst_wav)
        renamed[info.trial_instance_id] = new_name
        logger.debug("Copied %s -> %s", src_wav.name, new_name)

    return renamed


_DEMOGRAPHIC_FIELDS = ("age", "sex", "ethnicity", "first_language")

# Hand-recorded initializing temperatures (4 per session, unrelated to medoc).
# Left blank in the output for manual entry.
_INITIALIZING_TEMP_FIELDS = ("initializing_temp_1", "initializing_temp_2",
                             "initializing_temp_3", "initializing_temp_4")

# Pain Catastrophizing Scale (PCS) — 13 items, scored 0-4.
# Subscales: Rumination (8,9,10,11), Magnification (6,7,13), Helplessness (1,2,3,4,5,12).
_PCS_ITEMS = tuple(f"pcs_{i}" for i in range(1, 14))

# Positive and Negative Affect Schedule (PANAS) — 20 items, scored 1-5.
# Positive Affect: 1,3,5,9,10,12,14,16,17,19. Negative Affect: 2,4,6,7,8,11,13,15,18,20.
_PANAS_ITEMS = tuple(f"panas_{i}" for i in range(1, 21))

# Self-documenting reference for the blank questionnaire fields in participant_info.json.
_QUESTIONNAIRE_REFERENCE: dict[str, Any] = {
    "pcs": {
        "name": "Pain Catastrophizing Scale",
        "source_file": "required/Pain-catastrophizing-scale-questionnaire.pdf",
        "num_items": 13,
        "scale": "0 (not at all) to 4 (all the time)",
        "subscales": {
            "rumination": [8, 9, 10, 11],
            "magnification": [6, 7, 13],
            "helplessness": [1, 2, 3, 4, 5, 12],
        },
        "total_range": "0-52",
        "field_prefix": "pcs_",
    },
    "panas": {
        "name": "Positive and Negative Affect Schedule",
        "source_file": "required/Panas_questionnaire_scale_positve_negative_affect.pdf",
        "num_items": 20,
        "scale": "1 (very slightly or not at all) to 5 (extremely)",
        "subscales": {
            "positive_affect": [1, 3, 5, 9, 10, 12, 14, 16, 17, 19],
            "negative_affect": [2, 4, 6, 7, 8, 11, 13, 15, 18, 20],
        },
        "total_range": "PA 10-50, NA 10-50",
        "field_prefix": "panas_",
    },
}


def _subscale_of(item_num: int, subscales: dict[str, list[int]]) -> str:
    """Return the readable subscale name an item belongs to, or ''."""
    for name, items in subscales.items():
        if item_num in items:
            return name.replace("_", " ")
    return ""


def _questionnaire_rows() -> list[tuple[str, str, str]]:
    """Build ``(field, value, note)`` rows for hand-entry into questionnaires.csv.

    The middle ``value`` column is left blank so you can open the CSV in a
    spreadsheet, click the first empty cell, and arrow straight down the column
    typing one score per row.
    """
    rows: list[tuple[str, str, str]] = []
    pcs_subs = _QUESTIONNAIRE_REFERENCE["pcs"]["subscales"]
    panas_subs = _QUESTIONNAIRE_REFERENCE["panas"]["subscales"]

    for temp_field in _INITIALIZING_TEMP_FIELDS:
        rows.append((temp_field, "", "Celsius (hand-recorded initializing temp)"))
    for i in range(1, 14):
        rows.append((f"pcs_{i}", "", f"0-4 ({_subscale_of(i, pcs_subs)})"))
    for i in range(1, 21):
        rows.append((f"panas_{i}", "", f"1-5 ({_subscale_of(i, panas_subs)})"))
    return rows


def write_questionnaire_csv(path: Path) -> int:
    """Write a blank questionnaires.csv for fast manual score entry.

    Returns the number of data rows written.
    """
    rows = _questionnaire_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["field", "value", "note"])
        writer.writerows(rows)
    return len(rows)


# Columns for pain_ratings.csv.  Everything but the trailing ``pain_rating``
# is pre-filled from the experiment logs so the file is a drop-in join table for
# downstream ML: ``trial_instance_id`` + ``segment_index`` (or ``go_id``) join
# to events.csv / medoc_events.csv / trials.csv, and ``wav_filename`` joins to
# the merged audio dir.
_PAIN_RATING_COLUMNS = (
    "participant",
    "task",
    "session",
    "trial_instance_id",
    "block",
    "segment_index",
    "go_id",
    "go_duration_sec",
    "trial_elapsed_sec",
    "temp_at_go_celsius",
    "pain_rating",
    "wav_filename",
)


def _fmt_temp(value: Any) -> str:
    """Format a temperature float as a string, or '' when absent."""
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _build_pain_rating_rows(
    folders: list[SessionFolder],
    participant: str,
    task: str,
    renamed: dict[str, str],
) -> list[dict[str, str]]:
    """One row per GO segment across all source folders, for pain_ratings.csv.

    Cross-reference columns are pre-filled from the ``go_cue`` events;
    ``pain_rating`` is left blank for manual entry (one rating per GO segment).
    """
    go_rows: dict[tuple[str, str], dict[str, str]] = {}

    for folder in folders:
        events = _load_csv_rows(folder.path / "events.csv")
        for row in events:
            et = row.get("event_type", "").strip()
            tid = row.get("trial_instance_id", "").strip()
            if not tid:
                continue
            data = _parse_event_data(row.get("event_data", ""))
            seg = data.get("segment_index")
            seg_key = str(seg) if seg is not None else ""
            key = (tid, seg_key)

            if et == "go_cue":
                # First-seen wins (matches index_trials union semantics).
                if key in go_rows:
                    continue
                go_id = f"{tid}_go{int(seg):03d}" if seg is not None else f"{tid}_go"
                go_rows[key] = {
                    "participant": participant,
                    "task": task,
                    "session": folder.session,
                    "trial_instance_id": tid,
                    "block": row.get("block", "").strip(),
                    "segment_index": seg_key,
                    "go_id": go_id,
                    "go_duration_sec": "" if data.get("cue_duration_sec") is None
                        else str(data["cue_duration_sec"]),
                    "trial_elapsed_sec": "" if data.get("trial_elapsed_sec") is None
                        else str(data["trial_elapsed_sec"]),
                    "temp_at_go_celsius": _fmt_temp(data.get("temperature_celsius")),
                    "wav_filename": renamed.get(tid, ""),
                    "pain_rating": "",
                }

    rows = list(go_rows.values())
    rows.sort(
        key=lambda r: (
            r["session"],
            r["trial_instance_id"],
            int(r["segment_index"]) if r["segment_index"] else 0,
        )
    )
    return rows


def write_pain_ratings_csv(
    path: Path, rows: list[dict[str, str]]
) -> int:
    """Write pain_ratings.csv; the trailing ``pain_rating`` column is blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PAIN_RATING_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _consolidate_config(
    folders: list[SessionFolder],
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str]]:
    """Merge config.json files across sessions, consolidating demographics.

    For each demographic field (age, sex, ethnicity, first_language), takes the
    first non-empty value found across all sessions.  The base config is taken
    from the folder with the most events.csv rows (most complete run), then
    demographic fields are overwritten with the consolidated values.

    Returns ``(base_config, participant_info, sources)`` where:
        - base_config: the full merged config dict (or None if no config.json found)
        - participant_info: just the demographic fields, consolidated
        - sources: list of "field -> value (from <folder>)" strings for the report
    """
    configs: list[tuple[SessionFolder, dict[str, Any]]] = []
    for folder in folders:
        cfg_path = folder.path / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read config.json in %s: %s", folder.path.name, exc)
            continue
        configs.append((folder, cfg))

    if not configs:
        return None, {}, []

    # Pick base config from the folder with the most events rows
    best_cfg: dict[str, Any] | None = None
    best_rows = -1
    for folder, cfg in configs:
        row_count = len(_load_csv_rows(folder.path / "events.csv"))
        if row_count > best_rows:
            best_rows = row_count
            best_cfg = cfg

    if best_cfg is None:
        best_cfg = configs[0][1]

    # Consolidate demographics: first non-empty value wins
    participant_info: dict[str, Any] = {}
    sources: list[str] = []
    missing: list[str] = []
    for demo_field in _DEMOGRAPHIC_FIELDS:
        found_value = ""
        found_folder = ""
        for folder, cfg in configs:
            val = str(cfg.get(demo_field, "")).strip()
            if val:
                found_value = val
                found_folder = folder.path.name
                break
        if found_value:
            participant_info[demo_field] = found_value
            sources.append(f"{demo_field} -> '{found_value}' (from {found_folder})")
        else:
            participant_info[demo_field] = ""
            missing.append(demo_field)
            sources.append(f"{demo_field} -> MISSING (not in any session)")

    # Overwrite demographic fields in the base config
    merged_config = dict(best_cfg)
    for demo_field in _DEMOGRAPHIC_FIELDS:
        merged_config[demo_field] = participant_info[demo_field]

    if missing:
        logger.warning(
            "Demographic fields still missing across all sessions: %s",
            ", ".join(missing),
        )

    return merged_config, participant_info, sources


def merge_task_group(
    task: str,
    folders: list[SessionFolder],
    processed_dir: Path,
    participant_info: dict[str, Any],
    demo_sources: list[str],
    expected_blocks: int = _EXPECTED_BLOCKS,
    dry_run: bool = False,
) -> Path:
    """Merge all session folders for one task into a single merged directory.

    Shared per-participant data (``participant_info`` + ``demo_sources``) is
    consolidated once across all tasks by the caller and passed in here, so it
    is only written to the top-level processed dir, not duplicated per task.

    Returns the path to the merged directory (or the would-be path on dry run).
    """
    merged_dir = processed_dir / f"merged_task-{task}"

    trials, gaps = index_trials(folders, expected_blocks=expected_blocks)

    # Task-specific base config (demographics are shared via participant_info)
    merged_config, _, _ = _consolidate_config(folders)

    participant = folders[0].participant if folders else ""

    # One row per GO segment for pain_ratings.csv (manual pain entry).
    go_rows = _build_pain_rating_rows(folders, participant, task, renamed={})

    # Build report
    report: dict[str, Any] = {
        "task": task,
        "participant": participant,
        "source_folders": [f.path.name for f in folders],
        "merged_dir": str(merged_dir),
        "num_source_folders": len(folders),
        "num_trials_recovered": len(trials),
        "num_gaps": len(gaps),
        "num_go_segments": len(go_rows),
        "participant_info": participant_info,
        "participant_info_sources": demo_sources,
        "trials": [_trial_info_to_dict(t) for t in trials],
        "gaps": gaps,
    }

    if dry_run:
        print(json.dumps(report, indent=2))
        return merged_dir

    merged_dir.mkdir(parents=True, exist_ok=True)

    # Merge CSVs
    events_count = _merge_csv(merged_dir / "events.csv", folders, "events.csv")
    medoc_count = _merge_csv(merged_dir / "medoc_events.csv", folders, "medoc_events.csv")
    trials_count = _merge_csv(merged_dir / "trials.csv", folders, "trials.csv")
    logger.info(
        "Merged CSVs: %d events, %d medoc rows, %d trial records",
        events_count,
        medoc_count,
        trials_count,
    )

    # Copy + rename audio
    renamed = _copy_trial_audio(trials, folders, merged_dir / "audio")
    report["audio_renamed"] = renamed
    logger.info("Copied %d audio files", len(renamed))

    # Write consolidated config.json (participant_info.json + questionnaires.csv
    # are written once at the top-level processed dir by the caller, since they
    # are shared across tasks).
    if merged_config is not None:
        config_path = merged_dir / "config.json"
        config_path.write_text(json.dumps(merged_config, indent=2), encoding="utf-8")
        report["config_source"] = "consolidated from all sessions"
        logger.info("Wrote consolidated config.json with merged demographics")

    # Write pain_ratings_{task}.csv at the top-level processed dir: one row per
    # GO segment with the cross-reference keys + associated temperatures
    # pre-filled and a blank trailing pain_rating column.  Named per task so
    # speech and vowel ratings stay distinct.  Open it in a spreadsheet, click
    # the first pain_rating cell, and arrow straight down the column typing one
    # rating per GO.
    for r in go_rows:
        r["wav_filename"] = renamed.get(r["trial_instance_id"], "")
    pain_path = processed_dir / f"pain_ratings_{task}.csv"
    pcount = write_pain_ratings_csv(pain_path, go_rows)
    report["pain_ratings_path"] = str(pain_path)
    report["pain_rating_rows"] = pcount
    logger.info("Wrote %s (%d GO segments, blank pain_rating)", pain_path.name, pcount)

    # Write report
    report_path = merged_dir / "merge_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote merge report to %s", report_path)

    return merged_dir


def _trial_info_to_dict(info: TrialInfo) -> dict[str, Any]:
    return {
        "trial_instance_id": info.trial_instance_id,
        "source_folder": info.source_folder,
        "session": info.session,
        "block": info.block,
        "trial_num": info.trial_num,
        "has_recording_start": info.has_recording_start,
        "has_recording_end": info.has_recording_end,
        "has_trial_end": info.has_trial_end,
        "expected_go_segments": info.expected_go_segments,
        "actual_go_cues": info.actual_go_cues,
        "wav_filename": info.wav_filename,
        "wav_duration_sec": round(info.wav_duration_sec, 1) if info.wav_duration_sec is not None else None,
        "medoc_present": info.medoc_present,
        "warnings": info.warnings,
    }


# =============================================================================
# Audit-trail copy
# =============================================================================


def copy_originals(
    folders: list[SessionFolder],
    raw_dest: Path,
    dry_run: bool = False,
) -> None:
    """Copy every source session folder into *raw_dest* for the audit trail."""
    if dry_run:
        for f in folders:
            logger.info("[dry-run] would copy %s -> %s", f.path.name, raw_dest / f.path.name)
        return

    raw_dest.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        dst = raw_dest / folder.path.name
        if dst.exists():
            logger.debug("Skipping existing raw copy: %s", dst.name)
            continue
        shutil.copytree(folder.path, dst)
        logger.info("Copied %s -> %s", folder.path.name, dst)


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge multi-session participant folders into a single pipeline input."
    )
    parser.add_argument(
        "participant_dir",
        type=Path,
        help="Participant folder containing session sub-folders (e.g. data/p001).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Only merge this task type (e.g. 'speech' or 'vowel'). Default: merge all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the merge report without copying or writing anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing -processed directory.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    participant_dir = args.participant_dir.resolve()
    if not participant_dir.is_dir():
        logger.error("Not a directory: %s", participant_dir)
        return 1

    # Determine output directory: {participant_dir}-processed
    processed_dir = participant_dir.parent / f"{participant_dir.name}-processed"
    if processed_dir.exists() and not args.force:
        logger.error(
            "Processed directory already exists: %s (use --force to overwrite)",
            processed_dir,
        )
        return 1
    if processed_dir.exists() and args.force:
        logger.info("Removing existing processed directory: %s", processed_dir)
        shutil.rmtree(processed_dir)

    # Discover session folders
    folders = discover_session_folders(participant_dir)
    if not folders:
        logger.error("No session folders found in %s", participant_dir)
        return 1

    logger.info("Found %d session folders for participant %s", len(folders), folders[0].participant)
    for f in folders:
        logger.info("  %s (session=%s, task=%s)", f.path.name, f.session, f.task)

    # Group by task
    groups = group_by_task(folders)
    if args.task:
        groups = {t: v for t, v in groups.items() if t == args.task}
        if not groups:
            logger.error("No folders with task=%s found", args.task)
            return 1

    # Copy originals for audit trail
    raw_dest = processed_dir / "raw"
    all_folders = [f for group in groups.values() for f in group]
    logger.info("Copying %d original folders to %s", len(all_folders), raw_dest)
    copy_originals(all_folders, raw_dest, dry_run=args.dry_run)

    # Shared, per-participant data (demographics + questionnaires) is the same
    # across tasks, so consolidate it once from all folders and write it to the
    # top-level processed dir instead of duplicating it per merged_task-*.
    _, participant_info, demo_sources = _consolidate_config(all_folders)
    if not args.dry_run:
        processed_dir.mkdir(parents=True, exist_ok=True)
        info_path = processed_dir / "participant_info.json"
        info_path.write_text(json.dumps(participant_info, indent=2), encoding="utf-8")
        logger.info("Wrote shared participant_info.json")

        qcount = write_questionnaire_csv(processed_dir / "questionnaires.csv")
        logger.info("Wrote shared questionnaires.csv (%d blank rows)", qcount)

    # Merge each task group (pain_ratings_{task}.csv is written here, named per
    # task so speech and vowel ratings stay distinct).
    for task, task_folders in groups.items():
        logger.info("Merging task=%s (%d source folders)", task, len(task_folders))
        merge_task_group(
            task,
            task_folders,
            processed_dir,
            participant_info,
            demo_sources,
            dry_run=args.dry_run,
        )

    logger.info("Done. Output: %s", processed_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
