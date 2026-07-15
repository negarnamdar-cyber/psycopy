# Vowel Speech Gating Experiment with Medoc Thermal Stimulation

PsychoPy experiment for speech production under thermal stimulation with a Medoc
thermode device. Participants produce speech (sustained vowel or structured
Q&A) while a controlled thermal staircase is applied, with voice activity
detection and acoustic feature extraction.

The central design constraint: the Medoc device changes contact temperature
every 60 s, and the trial schedule is built so that **every temperature step
always lands in the middle of a STOP (no-speak) period**, never inside a GO
(speaking) segment. This decouples temperature transitions from vocal
production and avoids confounding speech acoustics with transient
thermoregulatory artifacts.

For the full, manuscript-ready methods description see
[`METHODOLOGY.md`](./METHODOLOGY.md).

## Experiment Structure

Two task modalities are supported.

### Vowel mode (NORMAL / PRACTICE)

- **4 blocks** x **1 trial per block** = **4 total trials**
- Each trial: **4 minutes (240 seconds)** of alternating STOP/GO segments
  built from four 60-second minute-blocks so the 60/120/180 s Medoc
  temperature steps always land on a STOP, never inside a GO segment
  - 32-44 GO segments per trial, each 1.5-3.5 seconds
  - STOP pauses of 3.5-4.5 seconds between GOs (and straddling each 60s mark)
  - Pattern: STOP -> GO -> STOP -> GO -> ... -> STOP
- **1-minute break** between blocks
- Total experiment time: ~20 minutes (4 trials x 240s + 3 breaks x 60s)

### Speech mode (SPEECH)

- **4 blocks**, each a 4-minute trial; each block contains **8 questions
  (32 total)**
- Each question cycle is a constant **30 seconds**:
  - 13 s READ (STOP, question shown)
  - 12 s ANSWER (GO, screen turns green)
  - 5 s "Rate your pain" prompt (STOP)
- Because the 30 s cycle divides 60 s evenly, the 60/120/180 s temperature
  steps fall between questions, never during a GO speaking period.
- 1-minute break between blocks; ~20 minutes total.
- Questions are configurable via `ExperimentConfig.speech_questions`.

## Architecture

Core modules under `psycopy/*`:

- `psycopy/config.py`: Experiment configuration + startup dialog
- `psycopy/medoc_experiment.py`: Main experiment orchestration
- `psycopy/medoc.py`: Medoc thermode device TCP client
- `psycopy/trial_generator.py`: Trial randomization (vowel + speech schedules)
- `psycopy/schedule.py`: RNG setup + legacy probabilistic scheduling
- `psycopy/runtime.py`: PsychoPy UI primitives
- `psycopy/session.py`: Output paths + batched data loggers
- `psycopy/storage.py`: Atomic CSV/JSON writes
- `psycopy/audio.py`: Audio recording service (crash-safe streaming WAV)
- `psycopy/vad.py`: Voice Activity Detection (WebRTC), real-time service
- `psycopy/features.py`: Post-run openSMILE eGeMAPSv02 extraction
- `psycopy/models.py`: Data models (MedocTrialRecord, enums)

Offline analysis under `scripts/`:

- `scripts/organize_sessions.py`: Merge scattered session folders into a single pipeline input (Phase 0)

Retired scripts live in `scripts/old/`:

- `scripts/old/cnn_analyze.py`: Regression pain CNN evaluation (PyTorch)
- `scripts/old/cnn3_analyze.py`: 3-class pain CNN evaluation (Keras)

## Quick Start

### Prerequisites

1. Medoc thermode device connected to the same network
2. Python 3.10+ installed

### Setup

Linux/macOS:

```bash
bash setup_venv.sh
bash run_experiment.sh
```

Windows:

```cmd
setup_venv.bat
run_experiment.bat
```

## Configuration

The startup dialog will prompt for:

- **Participant ID**: Subject identifier (e.g., "001")
- **Session ID**: Session number (e.g., "01")
- **Age / Sex / Ethnicity / First Language**: demographics
- **Experiment Mode**: Normal, Practice (no Medoc), Practice (with Medoc),
  Practice (short demo), or Speech Q&A
- **Random Seed**: blank = nondeterministic; integer = reproducible
- **Fullscreen Mode**: True/False
- **Enable VAD**: True/False
- **Medoc Device IP**: IP address of the thermode (default: 10.196.94.38)
- **Medoc Device Port**: TCP port (default: 20121)
- **Medoc Timeout (sec)**: socket timeout (default: 5.0)

## Testing Mode

To run without a physical Medoc device (useful for development/testing):

1. Select "Practice (no Medoc device)" mode in the startup dialog
2. The experiment will run but temperature data will not be recorded
3. All trial timing and VAD functionality works normally

## Data Output

Each session creates a directory:
`data/YYYYMMDD_HHMMSS_sub-{participant}_session-{session}_task-{vowel|speech}/`

### Output Files

| File | Description |
|------|-------------|
| `trials.csv` | Per-trial metadata with pain conditions |
| `medoc_events.csv` | Medoc device events (trigger + 5 s temperature polls) |
| `events.csv` | Experiment lifecycle events (cues, recording markers) |
| `config.json` | Configuration snapshot + run metadata |
| `participant_info.json` | Shared consolidated demographics (top-level `pxxx-processed/`, one per participant) |
| `questionnaires.csv` | Shared blank `field,value,note` rows for manual entry of initializing temps + PCS + PANAS (top-level `pxxx-processed/`) |
| `pain_ratings_{task}.csv` | One row per GO segment per task with cross-ref keys + temperatures pre-filled and a blank `pain_rating` column (top-level `pxxx-processed/`) |
| `merge_report.json` | Per-trial merge provenance + warnings (merged sessions) |
| `run.log` | Runtime log |
| `audio/*.wav` | Original audio recordings (44.1 kHz mono) |
| `audio_16k/*.wav` | 16 kHz mono audio for analysis |

### trials.csv Schema

| Column | Description |
|--------|-------------|
| `trial_instance_id` | Unique trial identifier |
| `set_number` | Block index (0-3) |
| `trial_in_set` | Trial index within block (0) |
| `task_type` | "vowel" or "speech" |
| `is_stop_trial` | Always False (stop/go is internal) |
| `trigger_timestamp` | Medoc trigger time |
| `status_timestamp` | Medoc status response time |
| `temperature_celsius` | Recorded temperature |
| `device_state` | Medoc device state code |
| `test_state` | Medoc test state code |
| `response_code` | Medoc response code |

### pain_ratings_{task}.csv Schema

Generated by `organize_sessions.py` into the top-level `pxxx-processed/`
directory, named per task (`pain_ratings_speech.csv`,
`pain_ratings_vowel.csv`). One row per GO segment; everything except
`pain_rating` is pre-filled for ML joins.

| Column | Description |
|--------|-------------|
| `participant` | Participant ID (matches `sub-XXX` audio naming) |
| `task` | "speech" or "vowel" |
| `session` | Session ID (matches `session-XX` audio naming) |
| `trial_instance_id` | Join key to `events.csv` / `medoc_events.csv` / `trials.csv` |
| `block` | Block name (`block0`–`block3`) |
| `segment_index` | 0-based GO segment within the trial |
| `go_id` | Stable unique ID: `{trial_instance_id}_go{segment_index:03d}` |
| `go_duration_sec` | GO segment duration (from `go_cue`) |
| `trial_elapsed_sec` | GO start time relative to trial trigger |
| `temp_at_go_celsius` | Temperature at GO cue (from `go_cue`) |
| `pain_rating` | **Blank — manually entered participant pain rating per GO** |
| `wav_filename` | Renamed merged audio file (join to `audio/`) |

## Voice Activity Detection (VAD)

VAD uses Google's WebRTC VAD (`webrtcvad`) with these defaults:

- **Aggressiveness**: 2 (scale 0-3; 2 is "quality" mode for lab use)
- **Frame duration**: 30 ms
- **Speech onset**: 2 consecutive speech frames
- **Speech offset**: 10 consecutive silent frames (300 ms)
- **Target rate**: 16,000 Hz

A real-time `VADService` (`psycopy/vad.py`) is available for live use. Offline
VAD + acoustic feature extraction is being rebuilt and will be documented here
once complete.

## Offline Post-Processing

### Phase 0: Organize + merge scattered sessions

When a Medoc failure or crash forces a re-run, one logical session ends up
scattered across multiple timestamped folders (each with a different session
ID). `organize_sessions.py` reassembles them into a single merged folder before
the rest of the pipeline runs.

```bash
# Merge all sessions for a participant (speech + vowel)
python scripts/organize_sessions.py data/p001

# Merge only speech sessions
python scripts/organize_sessions.py data/p001 --task speech

# Preview the merge without writing anything
python scripts/organize_sessions.py data/p001 --dry-run

# Overwrite an existing -processed directory
python scripts/organize_sessions.py data/p001 --force
```

Output layout:

```
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
    merged_task-speech/           # organize output -> pipeline input
        audio/
        events.csv
        medoc_events.csv
        trials.csv
        config.json
        merge_report.json
    merged_task-vowel/
```

Key behaviors:

- **Union, not dedup**: every trial that survived is kept — no trial is
  discarded unless it genuinely saved nothing (logged as an accepted gap)
- **Audio renaming**: WAVs are renamed with session IDs
  (`sub-001_session-02_block-0_trial-000.wav`) to prevent collisions across
  sessions
- **Demographic consolidation**: `age`, `sex`, `ethnicity`, `first_language`
  are pulled from whichever session has them and consolidated once into a
  shared top-level `participant_info.json` (demographics only) + per-task
  `config.json`. Because these are per-participant, they live at
  `pxxx-processed/` and are never duplicated per task.
- **Initializing temps + questionnaires**: a shared top-level
  `questionnaires.csv` with `field,value,note` rows (initializing temps, then
  PCS `pcs_1`–`pcs_13`, then PANAS `panas_1`–`panas_20`) is written once per
  participant. The `value` column is blank so you can open it in a spreadsheet,
  click the first empty cell, and arrow straight down the column typing one
  score per row.
- **Per-GO pain ratings**: `pain_ratings_{task}.csv` (e.g.
  `pain_ratings_speech.csv`, `pain_ratings_vowel.csv`) at the top-level
  processed dir — one row per GO segment with the cross-reference keys
  (`trial_instance_id`, `block`, `segment_index`, `go_id`) and the temperature
  at the GO cue (`temp_at_go_celsius`) pre-filled, plus `wav_filename` to join
  to the merged audio. The trailing `pain_rating` column is blank — open it in
  a spreadsheet, click the first cell, and arrow down the column typing one
  rating per GO.
- **`merge_report.json`**: per-trial source folder, WAV duration,
  medoc-present flag, warnings, and accepted gaps

### ML segmentation

ML segmentation (slicing merged audio into per-GO segments) is being rebuilt.
The previous `scripts/ml_segmenter.py` has been retired. This section will be
updated when the new pipeline is complete.

## ML Pain Prediction

```bash
# 1. Organize + merge scattered sessions
python scripts/organize_sessions.py data/p001
# 2. Segment audio + evaluate CNN models (in progress — see scripts/old/ for retired implementations)
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run only Medoc integration tests:

```bash
python -m pytest tests/test_medoc_integration.py -v
```

Run optional mode tests:

```bash
python -m pytest tests/test_medoc_optional.py -v
```

## Troubleshooting

### Medoc Connection Error

If you see "Failed to connect to Medoc device":
1. Check the device IP address matches your network
2. Verify the device is powered on and connected
3. Check firewall settings on port 20121
4. Enable "Practice (no Medoc)" mode for testing

### Audio Recording Issues

- Ensure microphone is connected and not muted
- Check system permissions for audio recording
- Verify `sounddevice` package is installed

## License

MIT License - see LICENSE file
