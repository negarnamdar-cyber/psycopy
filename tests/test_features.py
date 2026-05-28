from pathlib import Path

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")
from scipy.io import wavfile

from psycopy.features import TARGET_SAMPLE_RATE, standardize_wav_16k_mono


def test_standardize_wav_16k_mono(tmp_path: Path) -> None:
    input_wav = tmp_path / "input.wav"
    output_wav = tmp_path / "output.wav"

    sample_rate = 44100
    duration_sec = 0.25
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    left = 0.2 * np.sin(2 * np.pi * 220 * t)
    right = 0.1 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    wavfile.write(input_wav, sample_rate, stereo)

    standardize_wav_16k_mono(input_wav, output_wav)

    out_rate, out_audio = wavfile.read(output_wav)
    assert out_rate == TARGET_SAMPLE_RATE
    assert out_audio.ndim == 1
    assert out_audio.dtype == np.int16
    assert out_audio.size > 0
