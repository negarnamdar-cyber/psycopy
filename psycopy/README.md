# psycopy Experiment Runner

This repository implements a PsychoPy-based speech experiment with optional Medoc thermode integration, audio recording, voice activity detection, and session logging.

## Run the experiment

From the repository root:

```powershell
python run_experiment.py
```

By default, this shows the PsychoPy startup dialog for experiment configuration.

### Run with CLI options

```powershell
python run_experiment.py --no-dialog --practice-no-medoc --participant-id 001 --session-id 01 --fullscreen false
```

### CLI options

- `--normal` : full experiment with Medoc required
- `--practice-no-medoc` : practice mode without Medoc
- `--practice-with-medoc` : practice mode that attempts Medoc if available
- `--participant-id` : participant ID string
- `--session-id` : session ID string
- `--random-seed` : optional random seed
- `--fullscreen true|false` : PsychoPy fullscreen mode
- `--vad-enabled true|false` : enable voice activity detection
- `--medoc-ip` : Medoc device IP address
- `--medoc-port` : Medoc device TCP port
- `--medoc-timeout` : Medoc socket timeout in seconds

## Dependencies

Install the required Python packages:

```powershell
python -m pip install -r requirements.txt
```

If you plan to use VAD on Windows, the `webrtcvad-wheels` package is included.

If you want post-run feature extraction, install `opensmile` separately.
