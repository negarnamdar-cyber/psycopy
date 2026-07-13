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

- `scripts/process.py`: Unified post-processing (VAD + ComParE_2016 features)
- `scripts/ml_segmenter.py`: Slice audio into per-GO segments for ML
- `scripts/cnn_analyze.py`: Regression pain CNN evaluation (PyTorch)
- `scripts/cnn3_analyze.py`: 3-class pain CNN evaluation (Keras)

Pretrained models live in `portable_pain_cnn/`.

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
| `vad_events.csv` | Voice activity detection events (offline) |
| `events.csv` | Experiment lifecycle events (cues, recording markers) |
| `config.json` | Configuration snapshot + run metadata |
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

## Voice Activity Detection (VAD)

VAD uses Google's WebRTC VAD (`webrtcvad`) with these defaults:

- **Aggressiveness**: 2 (scale 0-3; 2 is "quality" mode for lab use)
- **Frame duration**: 30 ms
- **Speech onset**: 2 consecutive speech frames
- **Speech offset**: 10 consecutive silent frames (300 ms)
- **Target rate**: 16,000 Hz

A real-time `VADService` (`psycopy/vad.py`) is available; the production
analysis path runs VAD offline via `scripts/process.py`, computing:

- **GO-onset latency** (first `speech_start` after a GO cue)
- **Speech-cessation latency** (first `speech_end` after a STOP cue)

Both latencies are reported in milliseconds and joined with the temperature
recorded at the cue.

## Offline Post-Processing

```bash
# Run VAD + openSMILE ComParE_2016 features on all unprocessed sessions
python scripts/process.py
# Re-process everything
python scripts/process.py --force
# Process a single session
python scripts/process.py <path>
```

## ML Pain Prediction

```bash
# 1. Slice audio into per-GO segments
python scripts/ml_segmenter.py 001
# 2. Manually fill segments.csv `pain` column with 1-10 ratings
# 3. Evaluate regression CNN (PyTorch)
python scripts/cnn_analyze.py 001
# 4. Evaluate 3-class CNN (Keras)
python scripts/cnn3_analyze.py 001
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
