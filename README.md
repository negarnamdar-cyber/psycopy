# Speech Gating Experiment with Medoc Thermal Stimulation

PsychoPy experiment for speech production under thermal stimulation with Medoc thermode device. The experiment runs 8 sets of 12 trials with varying pain conditions and voice activity detection.

## Experiment Structure

- **8 sets** × **12 trials per set** = **96 total trials**
- Each set contains:
  - 6 vowel trials + 6 sentence trials
  - 3 baseline + 3 low + 3 medium + 3 high pain conditions
- **25% stop trials** globally (24 of 96 total)
- **38-second trial duration** with Medoc thermal stimulation

## Architecture

Core modules under `psycopy/*`:

- `psycopy/config.py`: Experiment configuration + startup dialog
- `psycopy/medoc_experiment.py`: Main experiment orchestration
- `psycopy/medoc.py`: Medoc thermode device TCP client
- `psycopy/trial_generator.py`: Trial randomization (8 sets × 12 trials)
- `psycopy/runtime.py`: PsychoPy UI primitives
- `psycopy/session.py`: Output paths + data loggers
- `psycopy/audio.py`: Audio recording service
- `psycopy/vad.py`: Voice Activity Detection (WebRTC)
- `psycopy/models.py`: Data models (MedocTrialRecord)

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
- **Medoc Device IP**: IP address of the thermode (default: 192.168.1.100)
- **Medoc Device Port**: TCP port (default: 5000)
- **Skip Medoc Connection**: Enable testing mode without hardware

## Testing Mode

To run without a physical Medoc device (useful for development/testing):

1. Check "Skip Medoc Connection (testing mode)" in the startup dialog
2. The experiment will run but temperature data will not be recorded
3. All trial timing and VAD functionality works normally

## Data Output

Each session creates a directory: `data/YYYYMMDD_HHMMSS_sub-{participant}_session-{session}/`

### Output Files

| File | Description |
|------|-------------|
| `trials.csv` | Per-trial metadata with pain conditions |
| `medoc_events.csv` | Medoc device events (trigger, status) |
| `vad_events.csv` | Voice activity detection events |
| `events.csv` | Experiment lifecycle events |
| `config.json` | Configuration snapshot |
| `run.log` | Runtime log |
| `audio/*.wav` | Original audio recordings |
| `audio_16k/*.wav` | 16kHz mono audio for analysis |

### trials.csv Schema

| Column | Description |
|--------|-------------|
| `trial_instance_id` | Unique trial identifier |
| `set_number` | Set index (0-7) |
| `trial_in_set` | Trial index within set (0-11) |
| `task_type` | "vowel" or "sentence" |
| `pain_condition` | "baseline", "low", "medium", "high" |
| `is_stop_trial` | True if stop trial |
| `trigger_timestamp` | Medoc trigger time |
| `status_timestamp` | Medoc status response time |
| `temperature_celsius` | Recorded temperature |
| `device_state` | Medoc device state code |

## Voice Activity Detection (VAD)

VAD is enabled by default and measures:
- Speech onset latency
- Speech cessation latency (time from STOP cue to speech end)
- Stop cue index for each trial

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
3. Check firewall settings on port 5000
4. Enable "Skip Medoc Connection" for testing

### Audio Recording Issues

- Ensure microphone is connected and not muted
- Check system permissions for audio recording
- Verify `sounddevice` package is installed

## License

MIT License - see LICENSE file