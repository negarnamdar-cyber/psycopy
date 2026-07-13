# Methodology

Manuscript-ready reference for the speech-gating experiment with Medoc thermal
stimulation. Each section below maps to a typical Methods subsection. Code
references use `file:line` notation so claims can be verified against the
implementation (version 0.3.0).

---

## 1. Overview and paradigm

The paradigm is a **GO/STOP speech-gating task** in which participants produce
speech (a sustained vowel or structured Q&A) while receiving controlled thermal
stimulation from a Medoc contact-heat thermode. The central design constraint
is that the Medoc device changes contact temperature every 60 s; the trial
schedule is constructed so that **every temperature step lands in the middle
of a STOP (no-speak) period**, never inside a GO (speaking) segment. This
decouples temperature transitions from vocal production and avoids
confounding speech acoustics with transient thermoregulatory artifacts.

Two task modalities are implemented (`psycopy/config.py:18`,
`psycopy/medoc_experiment.py:1`):

- **Vowel mode** — sustained "Ahh" production.
- **Speech mode** — structured, present-moment pain-focused Q&A.

## 2. Apparatus

### 2.1 Thermal stimulation — Medoc thermode

- **Device:** Medoc contact-heat thermode, controlled over **TCP**
  (`psycopy/medoc.py`).
- **Default endpoint:** IP `10.196.94.38`, TCP port `20121`, connection and
  command timeout 5.0 s (`psycopy/config.py:39`).
- **Wire protocol:** MMS-style framed binary:
  `[4-byte big-endian length] + [body]`, where
  `body = [4-byte BE timestamp] + [1-byte command_id] + parameters`
  (`psycopy/medoc.py:1`).
- **Commands used:** `GET_STATUS=0`, `SELECT_TP=1`, `START=2`, `STOP=5`.
- **Stimulation program:** the device's unified program `11000000` (decimal
  code 192), selected via `SELECT_TP` then started via `START`
  (`psycopy/medoc.py:78,404`). The temperature staircase (baseline 30 C ->
  stepped increases every 60 s) is configured on-device within this unified
  program; the host software selects/starts the program and polls
  temperature rather than sending each setpoint.
- **Temperature readback:** parsed from `GET_STATUS` responses. Temperature
  is a **signed little-endian 16-bit integer divided by 100**, giving a
  resolution of **0.01 C** (`psycopy/medoc.py:493`). Note the mixed
  endianness: timestamp, response code, and `tms` are big-endian; temperature
  is little-endian.
- **Device/test state codes:** `device_state` (IDLE=0, READY=1,
  TEST_IN_PROGRESS=2); `test_state` (IDLE=0, RUNNING=1, PAUSED=2, READY=3)
  (`psycopy/models.py:103`).
- **`tms` field:** a device-side elapsed-time clock (milliseconds since the
  unified program started), independent of the host `monotonic()` clock —
  useful for cross-validating host/device synchronization
  (`psycopy/medoc.py:490`).

#### Connection-per-command (stateless polling)

Each Medoc command opens a **fresh TCP connection** via
`socket.create_connection`, sends one framed request, reads the response, and
closes (`psycopy/medoc.py:222`). The device is queried statelessly, not held
on a persistent session. `connect()` is effectively a reachability handshake
(it connects then immediately closes the socket). Polling every 5 s is
therefore cheap and safe — each poll is an independent connect->send->read->
close cycle.

#### Dual byte-order program selection

`_select_unified_program` first sends the unified program code (192)
**network-byte-order-encoded** (`socket.htonl(192)`, attempt "A"); if the
device does not return OK, it retries with the **raw** code 192 (attempt "B")
after a 0.5 s delay (`psycopy/medoc.py:285`). Both are packed big-endian into
the frame. This dual attempt is a robustness workaround for byte-order
ambiguity in the device's `SELECT_TP` parameter.

#### STOP-safety pre-check

Before sending `STOP`, the client polls status; if the device is already
`READY`/`IDLE` (`test_state in {0,3}` and `device_state in {0,1}`), it
**skips the STOP command entirely** (`psycopy/medoc.py:346`). Sending STOP to
an already-idle device returns `ILLEGAL_STATE`/`NOT_PROPER_STATE` and can
push it into an unresponsive error state. After a STOP, the client polls up to
10 times (0.5 s apart, ~5 s max) waiting for READY before proceeding.

#### Polling cadence and resilience

The host polls `GET_STATUS` **every 5 s** during trials for higher-resolution
temperature data (`psycopy/medoc_experiment.py:255`). Polling uses
`allow_incomplete=True`, so a truncated response returns partial data rather
than raising — the experiment never aborts due to a single lost poll. After
**3 consecutive poll failures**, polling is disabled for the remainder of the
trial to avoid freezing the stimulus loop
(`psycopy/medoc_experiment.py:620`).

#### Per-block lifecycle

`connect -> send_unified_program (STOP-to-ready, SELECT_TP, START) -> run
trial -> stop_unified_program -> disconnect`, with a 1-min break between
blocks (`psycopy/medoc_experiment.py:714`).

#### Empirically observed staircase

Captured device traffic confirms the staircase stepping up across minute
boundaries (host polls at 5 s cadence capture the plateau, not just the
transition): ~34.1 C at t~0 s, ~40.8 C at t~5-25 s, ~43.1 C later — consistent
with a 60 s step interval.

### 2.2 Audio capture

- **Microphone** via `sounddevice` `InputStream`; mono, float32, **44.1 kHz**
  by default (`psycopy/audio.py:31,153`).
- A dedicated writer thread streams PCM to a WAV file with a 256-frame
  buffer; the WAV header's length fields are **rewritten every 1.0 s**
  (`_HEADER_PATCH_INTERVAL_SEC`) so recordings remain playable if the process
  is killed mid-trial (`psycopy/audio.py:103`). Tests verify a patched,
  never-closed file is byte-identical to a properly closed one
  (`tests/test_audio_crash_safety.py:61`).
- **Dropped audio chunks** (when the queue is full) are counted and logged as
  a warning at stop time — a data-integrity indicator (`psycopy/audio.py:189`).
- Preflight device check + up to 2 retry attempts on stream start
  (`psycopy/audio.py:46,150`).

### 2.3 Stimulus presentation

- **PsychoPy 2025.2.4** drives a fullscreen (1920x1080, "height" units)
  window (`psycopy/runtime.py:50`).
- **GO cue** = teal-green screen with "GO"; **STOP cue** = coral/red screen
  with "STOP" (WCAG-AA-oriented palette, `psycopy/runtime.py:21`). A fixation
  cross, stimulus text, and a state indicator are composited each flip.
- A coded graceful-shutdown sequence (Q + 12345) is required to abort; plain
  ESC is disabled to prevent accidental termination (`psycopy/runtime.py:172`).

## 3. Software environment

- **Python 3.10-3.12**; core runtime deps: `psychopy==2025.2.4`,
  `sounddevice`, `scipy`, `numpy`, `opensmile`, `webrtcvad-wheels`
  (`pyproject.toml:28`, `requirements.txt`).
- Entry point `main.py` -> startup GUI dialog (`psycopy/config.py:177`).
- App version stamped into `config.json` metadata as `0.3.0`
  (`psycopy/medoc_experiment.py:95`).

## 4. Trial structure and randomization

### 4.1 Vowel mode (`psycopy/trial_generator.py:136`)

- **4 blocks x 1 trial = 4 trials** (NORMAL); 1 block in practice variants.
  Each trial = **240 s** built from four 60-s "minute-blocks"
  (`psycopy/medoc_experiment.py:57`).
- A single STOP-pause duration `stop_per` is drawn per trial from
  **[3.5, 4.5] s** (snapped to an even hundredth so the whole trial sums to
  exactly 240 s).
- **GO segments:** 1.5-3.5 s, generated via a constrained
  integer-hundredths sampler so that the GO segments within each minute sum to
  an exact target (`_constrained_go_durations`, `trial_generator.py:94`).
- **Schedule layout:** trial begins with a STOP at t = 0; STOP pauses are
  placed at the trial start, straddling the 60/120/180 s temperature-step
  boundaries, and between every pair of GOs. Edge minutes (1st, 4th) lose
  1.5 x `stop_per`; middle minutes (2nd, 3rd) lose 1 x `stop_per` to the
  straddling boundary STOP. Feasible GO count per minute `k` is bounded by
  `k_min` (from wider middle windows) and `k_max` (from narrower edge windows)
  so every GO stays in [1.5, 3.5] s (`trial_generator.py:164`).
- **Net cadence:** a rapid-fire STOP->GO->STOP->GO...->STOP sequence; ~32-44
  GO segments per trial, never crossing a minute boundary.
- **1-min (60 s) break** between blocks; total session ~20 min.
- Stimulus text displayed: "Ahh" (`medoc_experiment.py:60`).

#### Exact-duration scheduling

GO durations are generated in **integer hundredths of a second**
(`_constrained_go_durations`, `trial_generator.py:117`) and the STOP duration
is snapped to an even hundredth (`trial_generator.py:157`), so every trial
sums to **exactly 240.00 s** with no floating-point drift. Tests assert
`abs(total - 240.0) < 1e-6` (`tests/test_medoc_integration.py:423`).

#### Edge vs. middle minute asymmetry

Because the boundary STOP *straddles* each 60/120/180 s mark, the available GO
window differs per minute:

- **Edge minutes** (1st, 4th): `60 - 1.5 x stop_per` (lose the
  trial-start/trial-end STOP plus half of the shared boundary STOP).
- **Middle minutes** (2nd, 3rd): `60 - 1.0 x stop_per` (lose only the shared
  boundary STOP).

The feasible GO count `k` is bounded separately for edge (`k_max`) and middle
(`k_min`) minutes so every GO stays within [1.5, 3.5] s regardless of the
drawn `stop_per` (`trial_generator.py:164`), guaranteeing 32-44 GO segments
per trial (validated in tests, `tests/test_e2e_medoc.py:373`).

### 4.2 Speech mode (`psycopy/trial_generator.py:238`)

- **4 blocks**, each a 240-s trial; each block contains **8 questions
  (32 total)**.
- Each question cycle is a constant **30 s** = **13 s READ** (STOP, question
  shown) + **12 s ANSWER** (GO, screen turns green) + **5 s "Rate your pain"**
  prompt (STOP). STOP totals 18 s; GO speaking is 12 s
  (`medoc_experiment.py:67`).
- Because the 30-s cycle divides 60 s evenly, the 60/120/180 s temperature
  steps fall on cycle boundaries (between questions), never during a GO
  speaking period. An explicit validator rejects any cycle length that does not
  divide 60 s (`trial_generator.py:291`).
- **Stimuli:** 40 open-ended, present-moment pain questions drawn from
  validated instruments — McGill Pain Questionnaire (Melzack, 1975), Brief
  Pain Inventory (Cleeland & Ryan, 1994), PROMIS Pain Interference, and
  semi-structured chronic-pain interview protocols (`config.py:49`). All
  require narrative answers (no yes/no). Questions are shuffled once and
  truncated to 32 (no recycling); fewer are spread evenly across blocks
  (`trial_generator.py:301`).
- Final rest is capped at 30 s for speech blocks (`medoc_experiment.py:59`).

### 4.3 Randomization and reproducibility

- A single `random.Random` instance is seeded from `random_seed` (blank ->
  nondeterministic) and used for all schedule generation and question
  shuffling (`psycopy/schedule.py:19`). This makes any session exactly
  reproducible from the seed stored in `config.json`.
- Tests verify that identical seeds reproduce identical `num_go_segments` and
  `go_segment_durations` (`tests/test_medoc_integration.py:483`).

## 5. Data acquisition and synchronization

### 5.1 Timestamps

- All timing uses `time.monotonic()`. The `EventLogger` records timestamps
  **relative to session start**; within-trial cue events carry
  `trial_elapsed_sec` relative to the trial's `trigger_timestamp`
  (`psycopy/session.py:99`, `medoc_experiment.py:304`).
- Temperature is sampled every 5 s and the most recent reading is attached to
  each GO/STOP cue event, enabling per-segment temperature matching offline
  (`medoc_experiment.py:305,359`).

### 5.2 Per-trial flow (`medoc_experiment.py:169`)

1. Generate `trial_instance_id` = `{participant}_{session}_block{n}_{trial:03d}`.
2. Log `trial_start` (with num_go_segments, go_durations).
3. Send Medoc trigger (logged) and **start audio recording** for the full
   trial -> `sub-{pid}_block-{n}_trial-{nnn}.wav`.
4. For each segment index: optional poll -> log STOP cue (with temperature)
   -> display STOP -> wait -> optional poll -> log GO cue (with temperature)
   -> display GO -> wait. (Speech adds a `rate_pain_prompt` event + screen
   after each GO.)
5. Final STOP (rest) fills remaining time to 240 s; final poll.
6. Stop audio; log `trial_end` (with actual_duration_sec); append a
   `MedocTrialRecord`.

### 5.3 Output directory

`data/{YYYYMMDD_HHMMSS}_sub-{pid}_session-{sid}_task-{vowel|speech}/`
(`psycopy/session.py:42`)

| File | Contents |
|---|---|
| `events.csv` | Lifecycle events: `block_start/end`, `trial_start/end`, `recording_start/end`, `go_cue`, `stop_cue`, `rate_pain_prompt`, `trial_error`, `experiment_start/complete/abort/error` — each with `trial_instance_id`, `block`, `timestamp`, JSON `event_data` |
| `trials.csv` | One row per trial (`MedocTrialRecord`): `trial_instance_id, set_number, trial_in_set, task_type, is_stop_trial, trigger_timestamp, status_timestamp, temperature_raw, temperature_celsius, device_state, test_state, response_code` |
| `medoc_events.csv` | Trigger + every 5-s poll: temperature (raw hex + C), device/test state, response code |
| `config.json` | Full config snapshot + run metadata (schema/app/python version, platform, UTC timestamp) |
| `run.log` | Structured runtime log |
| `audio/*.wav` | Original 44.1 kHz mono recordings (one per trial) |
| `audio_16k/*.wav` | 16 kHz mono copies (created during offline processing) |

### 5.4 Crash-safe logging

All loggers use **batched atomic writes** (temp-file + `os.replace`) and are
incrementally flushed after each trial and at segment boundaries so
accumulated data survives a crash (`psycopy/storage.py:22`,
`medoc_experiment.py:633`). A crash mid-session still yields complete data up
to the last flushed boundary.

## 6. Timing precision

### 6.1 Stimulus cue jitter floor

During every STOP/GO segment, `ui.wait()` loops on `core.wait(0.016)` while
checking for the abort key (`psycopy/runtime.py:227`). This means:

- **Cue onset/offset timing precision is bounded by ~16 ms** (62.5 Hz poll
  rate). This is the jitter floor for GO-onset and STOP-cue timestamps.
- The abort key (Q) is polled **continuously** during all waits, so a graceful
  shutdown request is never missed mid-segment.

## 7. Voice activity detection (VAD)

### 7.1 WebRTC VAD parameters (real-time and offline)

- **Aggressiveness = 2** (scale 0-3; "quality" mode for lab environments)
- **Frame duration = 30 ms** (WebRTC accepts only 10/20/30 ms)
- **Onset criterion:** 2 consecutive speech frames -> `speech_start`
- **Offset criterion:** 10 consecutive silence frames -> `speech_end`
  (i.e., **300 ms of silence**)
- **Target rate:** 16,000 Hz

The 300 ms silence threshold (`silence_frames=10 x 30 ms`) is a
**latency-vs.-false-offset trade-off**: it biases cessation latency upward by
up to ~300 ms but prevents brief pauses within a vowel from being logged as
speech ends.

### 7.2 Real-time component

A thread-safe `VADService` wrapping Google's WebRTC VAD (`webrtcvad`). It
resamples the 44.1 kHz stream to 16 kHz (polyphase `resample_poly`), runs
30-ms frames, and computes speech-cessation latency relative to the STOP cue
(`psycopy/vad.py`). The resampling GCD is precomputed per session
(`psycopy/vad.py:92`). This service is implemented and configurable but is
exercised offline in the production analysis path.

### 7.3 Offline VAD (analysis path)

`scripts/process.py:422` runs WebRTC VAD over each vowel WAV. For every **GO
cue** it finds the first `speech_start` after the cue -> **GO-onset latency**
(`go_latency_ms`); for every **STOP cue** it finds the first `speech_end`
after the cue -> **speech-cessation latency** (`stop_latency_ms`). Results are
written to `vad_events.csv`, joined with the cue's recorded
`temperature_celsius`.

### 7.4 Latency computation

- **GO-onset latency** = first `speech_start` timestamp - GO cue timestamp.
- **Speech-cessation latency** = first `speech_end` timestamp - STOP cue
  timestamp.

Both are reported in **milliseconds** and joined with the temperature recorded
at the cue (`scripts/process.py:510`).

## 8. Offline post-processing pipeline (`scripts/process.py`)

A unified batch processor scans `data/` for unprocessed sessions (tracked via
`processed.json`) and runs, per session:

1. **Discovery** — parses `events.csv` `recording_start` rows, resolves each
   `trial_instance_id` to its WAV, infers `audio_type` (vowel/speech), and
   loads temperature series from `medoc_events.csv`.
2. **VAD** (vowel trials) -> `vad_events.csv` (above).
3. **openSMILE ComParE_2016 features — vowel** (`process.py:709`):
   - Standardize each WAV to 16 kHz mono (`standardize_16k_mono`).
   - Build GO intervals from `go_cue` events, **expand each by +/-0.35 s
     context**, then apply **sliding windows of 10 s with 1-s hop** (with a
     tail window).
   - Extract ComParE_2016 at the **Functionals** level per window ->
     `vowel_features_ComParE.csv`, annotated with `trial_instance_id`,
     `temperature_celsius`, `block`, window/interval bounds.
4. **openSMILE ComParE_2016 features — speech** (`process.py:804`):
   whole-recording ComParE_2016 Functionals per speech WAV ->
   `speech_features_ComParE.csv`.
5. **`summary.csv`** — per-session counts.
6. Marks `processed.json` complete.

An older in-package extractor (`psycopy/features.py`) implements the same
GO-context/sliding-window scheme but with the **eGeMAPSv02** feature set (88
Functionals). Both are at the `FeatureLevel.Functionals` (utterance-level
summary) tier.

### 8.1 Temperature matching (limitation)

`match_temperature_for_segment` returns the **mean of all valid readings for a
trial**, not a timestamp-matched/interpolated value (`process.py:184`). The
code comments flag this as a future improvement: per-segment temperature is
approximated by the trial-level mean rather than the instantaneous reading at
each GO's onset. The raw 5 s poll series *is* preserved in
`medoc_events.csv`, so exact timestamp matching is possible in post-hoc
analysis.

### 8.2 Backward-compatible cue reconstruction

`find_cues` supports two event formats: new sessions with explicit
`go_cue`/`stop_cue` events (carrying per-segment `temperature_celsius`), and
legacy sessions where only `trial_start` + `go_durations` were logged — in
which case cue times are reconstructed assuming even STOP spacing
(`process.py:310`). Earlier-collected data remains analyzable.

### 8.3 GO acoustic context windows

Feature extraction expands each GO segment by **+/-0.35 s** of acoustic
context (to capture onset/frication transients), merges overlapping expanded
intervals, then applies **10 s sliding windows with 1 s hop** plus a tail
window (`features.py:17`, `process.py:652`). This balances temporal resolution
against the minimum duration openSMILE needs for stable Functionals.

## 9. ML segmentation (`scripts/ml_segmenter.py`)

To prepare per-segment clips for the pain CNN:

- Slices each trial WAV into individual **GO segments** using `go_cue`
  timestamps from `events.csv`, resampled to 16 kHz.
- Writes `segment_XXXX.wav` clips plus `segments.csv` with columns:
  `source_file, trial_instance_id, audio_type, segment_index,
  segment_filename, start_sec, end_sec, duration_sec, temperature_celsius,
  pain`.
- `temperature_celsius` is auto-filled from the GO-cue event data; `pain` is
  left **blank for manual 1-10 rating** before CNN analysis.

## 10. Pain prediction CNNs

Two pretrained models evaluate the GO segments against manually entered 1-10
pain ratings.

### 10.1 Regression CNN — `scripts/cnn_analyze.py` (PyTorch `SimpleCNN`)

- **Architecture** (617,921 params, `portable_pain_cnn/model.py`):
  Conv2d(1->32,3x3)+BN+ReLU+MaxPool(2,2) -> Conv2d(32->64,3x3)+BN+ReLU+
  MaxPool(2,2) -> Conv2d(64->128,3x3)+BN+ReLU+AdaptiveAvgPool(4,4) ->
  FC(2048->256)+ReLU+Dropout(0.5) -> FC(256->1).
- **Input:** `(1, 1, 128, 300)` log-mel; `sr=16000, n_mels=128, n_fft=2048,
  hop_length=512, duration=10 s`; `librosa.power_to_db(ref=np.max)`
  per-utterance self-normalization; front-truncate or pad to 300 frames (pad
  value -80 dB); normalize `(spec - mean)/std` from `spec_norm_stats.npz`.
- **Output:** scalar pain score clipped to **[1, 10]**.
- **Metrics:** MAE, RMSE, R^2, Pearson r, Lin's CCC, % within +/-1 and +/-2,
  Bland-Altman (mean diff + 95% LoA), residual summary, binned
  classification confusion matrix; **per-modality** (vowel vs speech)
  breakdown; predicted-vs-true and Bland-Altman plots.

### 10.2 3-class CNN — `scripts/cnn3_analyze.py` (Keras `pain_cnn_model.h5`)

- **Input:** `(1, 64, 63, 1)` log-mel; `sr=16000, n_mels=64, n_fft=2048,
  hop_length=1024, duration=4.0 s` (padded/truncated to exactly 64,000
  samples); `power_to_db(ref=np.max)`; `StandardScaler` normalization
  (mean/scale) from `spec_norm_stats.npz`.
- **Pain->class mapping:** Low 1-3 -> 0, Medium 4-7 -> 1, High 8-10 -> 2.
- **Metrics:** accuracy, macro precision/recall/F1, Cohen's kappa, confusion
  matrix, per-class precision/recall/F1/support, per-modality breakdown,
  top-k most-confidently-wrong predictions; confusion heatmap, per-class bar,
  and probability-distribution plots.

### 10.3 Per-utterance self-normalization

Both CNNs use `librosa.power_to_db(ref=np.max)` for **per-utterance
self-normalization** — the absolute loudness reference is each clip's own
maximum, not a fixed reference. This affects cross-utterance comparability and
should be disclosed.

### 10.4 Small-N caveat

Both analyzers print a warning when <10 segments are evaluated and note that
metrics may be unstable (`cnn3_analyze.py:764`, `cnn_analyze.py:789`). If the
validation set is small, this should be disclosed.

## 11. End-to-end workflow

```
setup_venv.{sh,bat} -> run_experiment.{sh,bat} -> main.py
   startup dialog -> MedocExperiment(config).run()
        4 blocks x 240-s trial: Medoc thermal staircase + GO/STOP speech + 44.1 kHz audio
        -> data/<timestamp>_sub-..._task-{vowel|speech}/
            events.csv, trials.csv, medoc_events.csv, config.json, run.log, audio/*.wav

python scripts/process.py            -> VAD + ComParE_2016 features (vowel & speech) + summary
python scripts/ml_segmenter.py 001  -> ..._segments/segment_XXXX.wav + segments.csv
   (manually fill segments.csv `pain` 1-10)
python scripts/cnn_analyze.py  001  -> regression pain scores + agreement metrics
python scripts/cnn3_analyze.py  001  -> 3-class (low/medium/high) metrics + confusion matrices
```

## 12. Reproducibility and integrity

- **Seeded RNG governs everything.** A single `random.Random(seed)` instance
  drives GO/STOP duration sampling, GO count per minute, and (in speech mode)
  question shuffling. The seed is stored in `config.json`, and tests verify
  identical seeds reproduce identical schedules
  (`tests/test_medoc_integration.py:483`).
- **Config + environment snapshot.** `config.json` embeds a `metadata` block:
  schema version, app version (0.3.0), Python version, platform string, and
  UTC creation timestamp (`runtime_logging.py:13`). This pins the exact
  software environment used for each session.
- **Atomic, incrementally-flushed logs.** All CSV/JSON writes go through
  `tempfile + os.replace` (`storage.py:48`), and loggers are flushed after
  each trial and at each segment boundary (`_flush_loggers`,
  `medoc_experiment.py:633`).
- **Graceful-shutdown protection.** Plain `ESC` is **disabled**; aborting
  requires `Q` then typing the code `12345` (`runtime.py:172`). This prevents
  a participant from accidentally terminating a trial while still allowing the
  experimenter a deterministic, logged (`experiment_abort`) exit.

## 13. Testing

The test suite uses a mock Medoc TCP server (`tests/fixtures/mock_medoc.py`)
so all protocol, timing, and data-output assertions run without real hardware:

```bash
python -m pytest tests/ -v
```

Key validated invariants:

- 4 blocks x 1 trial = 4 trials, all vowel (`tests/test_e2e_medoc.py:284`).
- 32-44 GO segments per trial, each 1.5-3.5 s, total GO < 240 s
  (`tests/test_e2e_medoc.py:373`).
- No GO crosses a 60/120/180 s boundary
  (`tests/test_medoc_integration.py:451`).
- Trial sums to exactly 240.00 s (`tests/test_medoc_integration.py:423`).
- Identical seeds reproduce identical schedules
  (`tests/test_medoc_integration.py:483`).
- Patched (never-closed) WAV header is byte-identical to a closed file
  (`tests/test_audio_crash_safety.py:61`).
