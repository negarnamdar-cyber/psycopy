"""Robust audio capture service."""

from __future__ import annotations

import logging
import queue
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd


class AudioServiceError(RuntimeError):
    pass


# How often the writer thread refreshes the on-disk WAV header sizes so a
# recording stays playable if the process is killed mid-trial.  The raw PCM is
# streamed to disk continuously; this only keeps the length fields current.
_HEADER_PATCH_INTERVAL_SEC = 1.0


class AudioService:
    """Audio recorder with preflight checks, retry behavior, and optional VAD."""

    def __init__(self, sample_rate: int = 44100, retries: int = 2):
        self.sample_rate = sample_rate
        self.retries = retries
        self._stream: sd.InputStream | None = None
        self._audio_queue: queue.Queue[np.ndarray | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._wave_file: Any = None
        self._dropped_audio_chunks = 0
        self._lock = threading.Lock()
        self.is_recording = False
        self.filename: str | None = None
        self._monotonic_start: float = 0.0
        self._last_header_patch: float = 0.0
        self.logger = logging.getLogger("psycopy.audio")

    def preflight(self) -> None:
        devices = sd.query_devices()
        input_devices = [device for device in devices if device.get("max_input_channels", 0) > 0]
        if not input_devices:
            raise AudioServiceError("No audio input device detected.")

    def _audio_callback(self, indata, frames, callback_time, status) -> None:
        if status:
            self.logger.warning("Audio callback status: %s", status)
        frame = indata.copy()

        audio_queue = self._audio_queue
        if audio_queue is not None:
            try:
                audio_queue.put_nowait(frame)
            except queue.Full:
                self._dropped_audio_chunks += 1

    def _start_writer(self, filename: Path) -> None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        self._audio_queue = queue.Queue(maxsize=256)
        self._wave_file = wave.open(str(filename), "wb")
        self._wave_file.setnchannels(1)
        self._wave_file.setsampwidth(2)
        self._wave_file.setframerate(self.sample_rate)
        self._last_header_patch = time.monotonic()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        audio_queue = self._audio_queue
        wave_file = self._wave_file
        if audio_queue is None or wave_file is None:
            return

        while True:
            chunk = audio_queue.get()
            try:
                if chunk is None:
                    return
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                wave_file.writeframes(pcm)
                self._maybe_patch_header(wave_file)
            finally:
                audio_queue.task_done()

    def _maybe_patch_header(self, wave_file: Any) -> None:
        """Throttled, best-effort refresh of the WAV header length fields."""
        now = time.monotonic()
        if now - self._last_header_patch < _HEADER_PATCH_INTERVAL_SEC:
            return
        self._last_header_patch = now
        try:
            self._patch_wave_header(wave_file)
        except Exception as exc:  # pragma: no cover - best effort, never fatal
            self.logger.warning("WAV header patch failed: %s", exc)

    def _patch_wave_header(self, wave_file: Any) -> None:
        """Rewrite the RIFF/data chunk sizes to match bytes written so far.

        Keeps the on-disk file playable if the process is killed before the
        wave file is closed.  Single-threaded (called only from the writer
        thread), so there is no concurrent seek/write race.
        """
        fileobj = getattr(wave_file, "_file", None)
        written = getattr(wave_file, "_datawritten", 0)
        if fileobj is None or not written:
            return
        end = fileobj.tell()
        fileobj.seek(4)
        fileobj.write(struct.pack("<I", 36 + written))
        fileobj.seek(40)
        fileobj.write(struct.pack("<I", written))
        fileobj.seek(end)
        fileobj.flush()

    def _finish_writer(self) -> None:
        audio_queue = self._audio_queue
        writer_thread = self._writer_thread

        if audio_queue is not None:
            try:
                audio_queue.put(None, timeout=2.0)
            except queue.Full:
                self.logger.warning("Audio writer queue full during shutdown")
        if writer_thread is not None:
            writer_thread.join(timeout=10.0)
            if writer_thread.is_alive():
                self.logger.warning("Audio writer did not finish before timeout")

        if self._wave_file is not None:
            try:
                self._wave_file.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                self.logger.warning("Error closing audio file: %s", exc)

        self._audio_queue = None
        self._writer_thread = None
        self._wave_file = None

    def start(self, filename: Path) -> None:
        self.filename = str(filename)
        self._dropped_audio_chunks = 0
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                self._start_writer(filename)
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=self._audio_callback,
                )
                self._monotonic_start = time.monotonic()
                self._stream.start()
                self.is_recording = True
                self.logger.info("Audio recording started: %s", self.filename)
                return
            except Exception as exc:  # pragma: no cover - hardware dependent
                last_error = exc
                self.logger.warning("Audio start attempt %s failed: %s", attempt, exc)
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                self._finish_writer()
                time.sleep(0.1)
        raise AudioServiceError(f"Unable to start audio recording: {last_error}")

    def stop(self) -> None:
        if not self.is_recording or self._stream is None or self.filename is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.logger.warning("Error stopping audio stream: %s", exc)

        self._finish_writer()
        self.is_recording = False
        if self._dropped_audio_chunks:
            self.logger.warning(
                "Dropped %d audio chunks while recording", self._dropped_audio_chunks
            )
        self.logger.info("Audio recording stopped: %s", self.filename)

    def abort(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                self.logger.warning("Audio cleanup failed during abort: %s", exc)
            self._stream = None
        with self._lock:
            self.is_recording = False
        self._finish_writer()
