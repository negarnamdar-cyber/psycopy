"""Crash-safety tests for audio recording.

The WAV writer thread refreshes the on-disk header length fields so a recording
stays playable if the process is killed before the wave file is closed (the raw
PCM is streamed continuously; only the header sizes would otherwise be stale).

To simulate a crash we patch the header and then read the file back WITHOUT
ever calling wave.close() -- proving the on-disk file is already valid.
"""

import struct
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# sounddevice is optional for these unit tests; stub it so psycopy.audio imports.
sys.modules.setdefault("sounddevice", MagicMock())

from psycopy.audio import AudioService  # noqa: E402


def _make_writer(path: Path) -> wave.Wave_write:
    wf = wave.open(str(path), "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(44100)
    return wf


def _pcm_chunk(num_frames: int = 1000) -> bytes:
    chunk = np.ones(num_frames, dtype=np.float32) * 0.5
    return (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def test_patched_header_survives_unclosed_writer(tmp_path: Path) -> None:
    """A recording whose header was patched (but never wave-close()d) stays playable."""
    path = tmp_path / "rec.wav"
    svc = AudioService(sample_rate=44100)
    pcm = _pcm_chunk(1000)

    wf = _make_writer(path)
    for _ in range(5):  # 5 chunks * 1000 frames = 5000 frames
        wf.writeframes(pcm)
    svc._patch_wave_header(wf)  # refreshes header + flushes to OS; no close() called

    # Read the file back while the writer is still open (simulates a crash:
    # the process never got to finalize the file, yet it is already valid).
    with wave.open(str(path), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 44100
        n_frames = reader.getnframes()

    assert n_frames == 5000, f"Expected 5000 frames on disk, got {n_frames}"
    wf.close()


def test_patched_header_bytes_match_closed_file(tmp_path: Path) -> None:
    """The patched header is byte-identical to a properly closed file's header."""
    patched_path = tmp_path / "patched.wav"
    closed_path = tmp_path / "closed.wav"
    svc = AudioService(sample_rate=44100)
    pcm = _pcm_chunk(1000)

    # Patched version: write 7 chunks, patch header, read WITHOUT closing.
    wf_a = _make_writer(patched_path)
    for _ in range(7):
        wf_a.writeframes(pcm)
    svc._patch_wave_header(wf_a)
    a = patched_path.read_bytes()
    wf_a.close()

    # Reference version: same content, closed normally.
    wf_b = _make_writer(closed_path)
    for _ in range(7):
        wf_b.writeframes(pcm)
    wf_b.close()
    b = closed_path.read_bytes()

    assert a[:44] == b[:44], "Patched WAV header differs from a closed file's header"
    assert a[44:] == b[44:], "Patched WAV payload differs from a closed file's payload"

    # Sanity check the size fields directly.
    data_size = struct.unpack("<I", a[40:44])[0]
    riff_size = struct.unpack("<I", a[4:8])[0]
    assert data_size == 7 * 1000 * 2
    assert riff_size == 36 + data_size


def test_patch_header_is_throttled(tmp_path: Path) -> None:
    """_maybe_patch_header only patches once per interval."""
    path = tmp_path / "rec.wav"
    svc = AudioService(sample_rate=44100)
    pcm = _pcm_chunk(1000)

    wf = _make_writer(path)
    svc._last_header_patch = 0.0  # force the first patch to fire
    wf.writeframes(pcm)
    svc._maybe_patch_header(wf)
    first_patch = svc._last_header_patch
    assert first_patch > 0.0

    # Immediately call again; should be a no-op (throttled).
    wf.writeframes(pcm)
    svc._maybe_patch_header(wf)
    assert svc._last_header_patch == first_patch
    wf.close()
