# psycopy package

PsychoPy-based speech-gating experiment runner with Medoc thermode integration,
audio recording, voice activity detection (VAD), and session logging.

## Run the experiment

The experiment is launched from the repository root (not this directory):

```powershell
python main.py
```

This opens the PsychoPy startup dialog for experiment configuration
(participant/session ID, mode, Medoc connection, random seed, fullscreen,
VAD enable). There is no command-line flag interface; all setup is done
through the dialog.

See the repository-root [`README.md`](../README.md) for the full quick-start,
configuration reference, and data-output schema. See
[`METHODOLOGY.md`](../METHODOLOGY.md) for the manuscript-ready methods
description.

## Modes

The startup dialog's **Mode** field selects:

- **Normal (Full experiment)** — 4 vowel blocks, Medoc connection required
- **Practice (no Medoc device)** — 1 block, no thermal stimulation, no
  temperature data (useful for development/testing)
- **Practice (with Medoc device)** — 1 block, attempts Medoc but does not
  require it
- **Practice (short demo)** — on-screen demo of both tasks (no Medoc, no audio)
- **Speech Q&A** — structured pain-focused Q&A with thermal stimulation

## Module map

- `config.py` — configuration model + startup dialog
- `medoc_experiment.py` — main experiment orchestration (trial loop, cues)
- `medoc.py` — Medoc thermode TCP client (MMS framed protocol)
- `trial_generator.py` — constrained GO/STOP schedule generation
- `runtime.py` — PsychoPy UI primitives (window, cues, waits, abort)
- `session.py` — output paths + batched loggers
- `storage.py` — atomic CSV/JSON writes
- `audio.py` — crash-safe streaming WAV recorder
- `vad.py` — WebRTC VAD real-time service
- `features.py` — post-run openSMILE eGeMAPSv02 extraction
- `models.py` — typed runtime models and enums
- `practice_demo.py` — short on-screen demo of STOP/GO mechanics

## Dependencies

Install from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Core runtime: `psychopy`, `sounddevice`, `scipy`, `numpy`.
Post-processing: `webrtcvad-wheels`, `opensmile`, `pandas`.
Pain CNNs: `torch` + `librosa` (regression) or `tensorflow` + `librosa` (3-class).
