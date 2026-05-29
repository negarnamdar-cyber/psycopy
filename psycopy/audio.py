"""Robust audio capture service."""

from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd


class AudioServiceError(RuntimeError):
    pass


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
        self.logger = logging.getLogger("psycopy.audio")

        # VAD-related attributes
        self._vad: Any = None  # VADService if available
        self._vad_enabled: bool = False
        self._monotonic_start: float = 0.0
        self._vad_config: Any = None  # Store config reference for frame duration etc.

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

        # VAD processing (parallel to normal recording)
        if self._vad_enabled and self._vad is not None:
            # Calculate timestamp relative to recording start
            timestamp = time.monotonic() - self._monotonic_start
            self._vad.process_audio_chunk(frame, timestamp)

    def _start_writer(self, filename: Path) -> None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        self._audio_queue = queue.Queue(maxsize=256)
        self._wave_file = wave.open(str(filename), "wb")
        self._wave_file.setnchannels(1)
        self._wave_file.setsampwidth(2)
        self._wave_file.setframerate(self.sample_rate)
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
            finally:
                audio_queue.task_done()

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
            self.logger.warning("Dropped %d audio chunks while recording", self._dropped_audio_chunks)
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

    # ==========================================================================
    # VAD Methods
    # ==========================================================================

    def enable_vad(self, config) -> None:
        """Initialize VADService if vad_enabled in config.

        Args:
            config: ExperimentConfig with VAD settings.
        """
        if not getattr(config, "vad_enabled", False):
            self._vad_enabled = False
            self._vad = None
            return

        try:
            # Import VADService (pattern from features.py for optional dependency)
            from psycopy.vad import VADService
        except ImportError:
            self._vad_enabled = False
            self._vad = None
            self.logger.warning(
                "VAD requested but webrtcvad package not installed. "
                "Install with: pip install webrtcvad"
            )
            return

        try:
            vad_instance = VADService(
                aggressiveness=getattr(config, "vad_aggressiveness", 2),
                frame_duration_ms=getattr(config, "vad_frame_duration_ms", 30),
                silence_frames=getattr(config, "vad_silence_frames", 10),
                source_rate=self.sample_rate,
            )
            self._vad = vad_instance
            self._vad_enabled = True
            self._vad_config = config
            self.logger.info(
                "VAD enabled: aggressiveness=%d, frame_duration=%dms, silence_frames=%d",
                vad_instance.aggressiveness,
                vad_instance.frame_duration_ms,
                vad_instance.silence_frames,
            )
        except Exception as exc:
            self._vad_enabled = False
            self._vad = None
            self.logger.error("Failed to initialize VAD: %s", exc)

    def start_vad_monitoring(self) -> None:
        """Start tracking speech for a new trial.

        Resets VAD state and prepares for speech detection.
        Should be called at the start of each trial.
        """
        if not self._vad_enabled or self._vad is None:
            return

        self._vad.reset()
        self._monotonic_start = time.monotonic()
        self.logger.debug("VAD monitoring started")

    def stop_vad_monitoring(self) -> list[dict[str, Any]]:
        """Stop tracking speech and get accumulated VAD events.

        Returns:
            List of VAD events (speech_start, speech_end) with timestamps.
        """
        if not self._vad_enabled or self._vad is None:
            return []

        events = self._vad.get_events()
        self.logger.debug("VAD monitoring stopped, %d events recorded", len(events))
        return events

    def set_stop_cue_time(self) -> float | None:
        """Record the time when STOP state appeared on screen.

        Called when STOP state appears to measure speech cessation latency.

        Returns:
            Relative timestamp (seconds from recording start) or None if VAD disabled.
        """
        if not self._vad_enabled or self._vad is None:
            return None

        # Calculate timestamp relative to recording start
        timestamp = time.monotonic() - self._monotonic_start
        self._vad.set_stop_cue_time(timestamp)
        self.logger.debug("Stop cue time recorded: %.3f", timestamp)
        return timestamp

    def get_vad_events(self) -> list[dict[str, Any]]:
        """Return list of VAD events for current trial.

        Returns:
            List of VAD event dictionaries, or empty list if VAD not enabled.
        """
        if not self._vad_enabled or self._vad is None:
            return []
        return self._vad.get_events()

    def get_speech_cessation_latency(self) -> float | None:
        """Return latency from STOP cue to silence.

        Returns:
            Time in seconds from stop_cue_time to speech_end_time,
            or None if speech hasn't stopped or stop cue wasn't set.
        """
        if not self._vad_enabled or self._vad is None:
            return None
        return self._vad.get_speech_cessation_latency()

    @property
    def vad_enabled(self) -> bool:
        """Check if VAD is currently enabled."""
        return self._vad_enabled and self._vad is not None

    @property
    def vad_is_speaking(self) -> bool:
        """Check if speech is currently detected.

        Returns:
            True if speech is detected, False otherwise or if VAD not enabled.
        """
        if not self._vad_enabled or self._vad is None:
            return False
        return self._vad.is_speaking
