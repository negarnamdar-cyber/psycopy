"""Tests for the unified offline post-processing script."""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np
import pytest

import sys
import types

# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing scripts.process
# ---------------------------------------------------------------------------


class _MockVad:
    def __init__(self, aggressiveness: int = 2):
        pass

    def is_speech(self, frame: bytes, rate: int) -> bool:
        return True


_mock_vad_mod = types.ModuleType("webrtcvad")
_mock_vad_mod.Vad = _MockVad
sys.modules["webrtcvad"] = _mock_vad_mod

# Mock opensmile
_mock_smile_mod = types.ModuleType("opensmile")
_mock_smile_mod.FeatureSet = types.SimpleNamespace(ComParE_2016="ComParE_2016")
_mock_smile_mod.FeatureLevel = types.SimpleNamespace(Functionals="Functionals")


class _MockSmile:
    def __init__(self, **kwargs) -> None:
        pass

    def process_signal(self, signal, sampling_rate):
        import pandas as pd
        return pd.DataFrame({"feat1": [1.0], "feat2": [2.0]})


_mock_smile_mod.Smile = _MockSmile
sys.modules["opensmile"] = _mock_smile_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_vowel_session(tmp_path: Path):
    """Create a minimal vowel session directory with audio and events."""
    session_dir = tmp_path / "20250529_120000_sub-001_session-01"
    audio_dir = session_dir / "audio"
    session_dir.mkdir()
    audio_dir.mkdir()

    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = (np.sin(2 * np.pi * 440 * t) * 0.9 * 32767).astype(np.int16)

    wav_path = audio_dir / "sub-001_block-0_trial-000.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    events_csv = session_dir / "events.csv"
    with events_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["event_type", "timestamp", "trial_instance_id", "block", "event_data"])
        writer.writerow([
            "recording_start", "0.0", "001_01_block0_000", "block0",
            json.dumps({"audio_type": "vowel"}),
        ])
        writer.writerow([
            "stop_cue", "0.1", "001_01_block0_000", "block0",
            json.dumps({"segment_index": 0, "trial_elapsed_sec": 0.1}),
        ])
        writer.writerow([
            "go_cue", "0.3", "001_01_block0_000", "block0",
            json.dumps({"segment_index": 0, "trial_elapsed_sec": 0.3}),
        ])
        writer.writerow([
            "recording_end", "1.0", "001_01_block0_000", "block0",
            json.dumps({}),
        ])

    return session_dir


@pytest.fixture
def tmp_speech_session(tmp_path: Path):
    """Create a minimal free-speech session directory with audio and events."""
    session_dir = tmp_path / "20250529_130000_sub-002_session-01"
    audio_dir = session_dir / "audio"
    session_dir.mkdir()
    audio_dir.mkdir()

    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = (np.sin(2 * np.pi * 300 * t) * 0.8 * 32767).astype(np.int16)

    wav_path = audio_dir / "sub-002_speech.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    events_csv = session_dir / "events.csv"
    with events_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["event_type", "timestamp", "trial_instance_id", "block", "event_data"])
        writer.writerow([
            "recording_start", "0.0", "", "speech",
            json.dumps({"audio_type": "speech"}),
        ])
        writer.writerow([
            "recording_end", "2.0", "", "speech",
            json.dumps({}),
        ])

    return session_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessSession:
    """Tests for scripts/process.py pipeline."""

    def test_is_processed_mark_processed(self, tmp_vowel_session: Path):
        from scripts.process import is_processed, mark_processed

        assert not is_processed(tmp_vowel_session)
        meta = {"status": "completed", "session": tmp_vowel_session.name}
        mark_processed(tmp_vowel_session, meta)
        assert is_processed(tmp_vowel_session)
        status_file = tmp_vowel_session / "processed.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["status"] == "completed"

    def test_read_wav(self, tmp_vowel_session: Path):
        from scripts.process import read_wav

        wav_path = tmp_vowel_session / "audio" / "sub-001_block-0_trial-000.wav"
        samples, rate = read_wav(wav_path)
        assert rate == 44100
        assert samples.dtype == np.float32
        assert len(samples) == 44100
        assert -1.0 <= samples.max() <= 1.0

    def test_find_cues_new_format(self, tmp_vowel_session: Path):
        from scripts.process import find_cues

        cues = find_cues(tmp_vowel_session / "events.csv", "001_01_block0_000")
        assert len(cues) == 2  # one stop + one go
        stop_cues = [c for c in cues if c["cue_type"] == "stop"]
        go_cues = [c for c in cues if c["cue_type"] == "go"]
        assert len(stop_cues) == 1
        assert len(go_cues) == 1
        assert stop_cues[0]["timestamp_sec"] == 0.1
        assert go_cues[0]["timestamp_sec"] == 0.3

    def test_find_cues_old_format(self, tmp_vowel_session: Path):
        """Old sessions have trial_start with go_durations, not explicit cues."""
        from scripts.process import find_cues

        # Build a fake old-format events.csv
        events_csv = tmp_vowel_session / "old_events.csv"
        with events_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["event_type", "timestamp", "trial_instance_id", "block", "event_data"])
            writer.writerow([
                "trial_start", "0.0", "old_01_block0_000", "block0",
                json.dumps({"task_type": "vowel", "num_go_segments": 2, "go_durations": [3.0, 4.0]}),
            ])

        cues = find_cues(events_csv, "old_01_block0_000")
        # With 2 GO segments (3s + 4s = 7s total go) in a 60s trial:
        # stop_duration = (60 - 7) / 3 = 17.666...
        # Expected: STOP[0] -> GO[0] -> STOP[1] -> GO[1] -> STOP[2]
        assert len(cues) == 5
        assert cues[0]["cue_type"] == "stop"
        assert cues[0]["segment_index"] == 0
        assert cues[1]["cue_type"] == "go"
        assert cues[1]["segment_index"] == 0
        assert cues[2]["cue_type"] == "stop"
        assert cues[2]["segment_index"] == 1
        assert cues[3]["cue_type"] == "go"
        assert cues[3]["segment_index"] == 1
        assert cues[4]["cue_type"] == "stop"
        assert cues[4]["segment_index"] == 2

    def test_discover_recordings_vowel(self, tmp_vowel_session: Path):
        from scripts.process import discover_recordings

        recs = discover_recordings(tmp_vowel_session)
        assert len(recs) == 1
        assert recs[0].audio_type == "vowel"
        assert recs[0].trial_instance_id == "001_01_block0_000"

    def test_discover_recordings_speech(self, tmp_speech_session: Path):
        from scripts.process import discover_recordings

        recs = discover_recordings(tmp_speech_session)
        assert len(recs) == 1
        assert recs[0].audio_type == "speech"

    def test_process_vowel_session_creates_all_outputs(self, tmp_vowel_session: Path):
        from scripts.process import process_session

        meta = process_session(tmp_vowel_session, force=True)
        assert meta["status"] == "completed"

        assert (tmp_vowel_session / "vad_events.csv").exists()
        assert (tmp_vowel_session / "vowel_features_ComParE.csv").exists()
        assert (tmp_vowel_session / "summary.csv").exists()
        assert (tmp_vowel_session / "processed.json").exists()

        # VAD rows should include speech_start / speech_end with latency cols
        with (tmp_vowel_session / "vad_events.csv").open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert "speech_start" in [r["event_type"] for r in rows]
        assert "cue_type" in rows[0]
        assert "go_latency_ms" in rows[0]
        assert "stop_latency_ms" in rows[0]

    def test_process_speech_session_creates_outputs(self, tmp_speech_session: Path):
        from scripts.process import process_session

        meta = process_session(tmp_speech_session, force=True)
        assert meta["status"] == "completed"

        assert (tmp_speech_session / "speech_features_ComParE.csv").exists()
        assert (tmp_speech_session / "speech_diarization.csv").exists()
        assert (tmp_speech_session / "summary.csv").exists()

    def test_process_session_skips_already_processed(self, tmp_vowel_session: Path):
        from scripts.process import is_processed, mark_processed, process_session

        mark_processed(tmp_vowel_session, {"status": "completed"})
        meta = process_session(tmp_vowel_session, force=False)
        assert meta["status"] == "skipped"

    def test_summary_content(self, tmp_vowel_session: Path):
        from scripts.process import process_session

        process_session(tmp_vowel_session, force=True)
        with (tmp_vowel_session / "summary.csv").open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["session_id"] == tmp_vowel_session.name
        assert rows[0]["num_vowel_trials"] == "1"

    def test_temperature_in_vad_events(self, tmp_vowel_session: Path):
        from scripts.process import find_cues, run_vad

        cues = find_cues(tmp_vowel_session / "events.csv", "001_01_block0_000")
        # New format should have temperature_celsius key
        assert "temperature_celsius" in cues[0]

        wav_path = tmp_vowel_session / "audio" / "sub-001_block-0_trial-000.wav"
        events = run_vad(wav_path, cues)
        # VAD rows should carry temperature
        for ev in events:
            assert "temperature_celsius" in ev
