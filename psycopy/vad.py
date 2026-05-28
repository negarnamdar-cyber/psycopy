"""WebRTC Voice Activity Detection service."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import resample_poly

try:
    import webrtcvad
except ImportError:
    webrtcvad = None  # type: ignore[assignment]

HAS_WEBRTC_VAD = webrtcvad is not None


class VADServiceError(RuntimeError):
    """Raised when VAD service encounters an error."""
    pass


class VADService:
    """Real-time voice activity detection using WebRTC VAD.

    This service processes audio chunks from the sounddevice callback in real-time,
    performing voice activity detection to track speech start/end events and measure
    speech cessation latency.

    Thread Safety:
        This service is designed to be called from the audio callback thread.
        All state changes use atomic operations on simple variables to ensure
        thread safety without locks.
    """

    # Frame duration options in milliseconds
    VALID_FRAME_DURATIONS_MS = (10, 20, 30)

    # VAD aggressiveness: 0=permissive, 3=aggressive
    VALID_AGGRESSIVENESS = (0, 1, 2, 3)

    def __init__(
        self,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        silence_frames: int = 10,
        source_rate: int = 44100,
    ) -> None:
        """Initialize the VAD service.

        Args:
            aggressiveness: VAD sensitivity (0=permissive, 3=aggressive). Default 2 for lab.
            frame_duration_ms: Frame duration in ms (10, 20, or 30). Default 30ms.
            silence_frames: Number of consecutive silent frames to trigger speech end.
            source_rate: Source audio sample rate in Hz. Default 44100.

        Raises:
            VADServiceError: If webrtcvad is not installed or parameters are invalid.
        """
        if not HAS_WEBRTC_VAD:
            raise VADServiceError(
                "webrtcvad package not installed. On Windows, use: pip install webrtcvad-wheels"
            )

        if aggressiveness not in self.VALID_AGGRESSIVENESS:
            raise VADServiceError(
                f"Invalid aggressiveness {aggressiveness}. Must be one of {self.VALID_AGGRESSIVENESS}"
            )

        if frame_duration_ms not in self.VALID_FRAME_DURATIONS_MS:
            raise VADServiceError(
                f"Invalid frame_duration_ms {frame_duration_ms}. Must be one of {self.VALID_FRAME_DURATIONS_MS}"
            )

        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.silence_frames = silence_frames
        self.source_rate = source_rate
        self.target_rate = 16000

        # Frame size in samples at 16kHz
        self.frame_size = int(self.target_rate * frame_duration_ms / 1000)

        # Initialize WebRTC VAD
        self._vad = webrtcvad.Vad(aggressiveness)  # type: ignore[union-attr]

        # Buffer for resampling (stores float32 samples from source rate)
        self._buffer: np.ndarray = np.array([], dtype=np.float32)

        # GCD for efficient resampling
        self._gcd = np.gcd(source_rate, self.target_rate)
        self._up = self.target_rate // self._gcd
        self._down = source_rate // self._gcd

        # State tracking (atomic variables for thread safety)
        self._is_speaking: bool = False
        self._consecutive_silence: int = 0
        self._consecutive_speech: int = 0
        self._speech_start_time: float | None = None
        self._speech_end_time: float | None = None
        self._stop_cue_time: float | None = None

        # Events list for current trial
        self._events: list[dict[str, Any]] = []

        # Logger
        self.logger = logging.getLogger("psycopy.vad")

    def process_audio_chunk(self, indata: np.ndarray, timestamp: float) -> list[dict[str, Any]]:
        """Process an audio chunk and detect voice activity.

        Args:
            indata: float32 array from sounddevice callback (shape: [samples, 1] or [samples]).
            timestamp: Monotonic timestamp when chunk was captured.

        Returns:
            List of VAD events (speech_start, speech_end) with timestamps.
            Each event is a dict with 'type', 'timestamp', and optional 'latency'.
        """
        events: list[dict[str, Any]] = []

        if not HAS_WEBRTC_VAD:
            return events

        # Flatten if needed (mono)
        if indata.ndim > 1:
            audio = indata.flatten()
        else:
            audio = indata.astype(np.float32)

        # Append to buffer
        self._buffer = np.concatenate([self._buffer, audio])

        # Calculate minimum samples needed for resampling to produce at least one frame
        # We need enough samples that after resampling we get frame_size samples
        min_source_samples = int(np.ceil(self.frame_size * self._down / self._up))

        # Process while we have enough samples
        while len(self._buffer) >= min_source_samples:
            # Calculate how many source samples we need for an integer number of frames
            # After resampling: target_samples = source_samples * up / down
            # We want target_samples to be a multiple of frame_size
            # So: source_samples * up / down >= frame_size
            # source_samples >= frame_size * down / up

            # Get the exact number of source samples for resampling
            # Resampling ratio is up/down, so for every `down` source samples we get `up` target samples
            # For frame_size target samples, we need: frame_size * down / up source samples
            # But resample_poly works better with multiples

            # Calculate samples to use for clean resampling
            # Use enough for one or more complete frames at 16kHz
            source_samples_per_frame = int(np.ceil(self.frame_size * self._down / self._up))

            if len(self._buffer) < source_samples_per_frame:
                break

            # Extract samples for resampling
            samples_to_process = source_samples_per_frame
            chunk = self._buffer[:samples_to_process]
            self._buffer = self._buffer[samples_to_process:]

            # Resample to 16kHz
            resampled = resample_poly(chunk, self._up, self._down).astype(np.float32)

            # Convert to int16 PCM
            pcm = (resampled * 32767.0).astype(np.int16)

            # Process complete frames
            frame_bytes = self.frame_size * 2  # int16 = 2 bytes
            i = 0
            while i + self.frame_size <= len(pcm):
                frame = pcm[i:i + self.frame_size]
                frame_bytes = frame.tobytes()

                # Calculate precise timestamp for this frame
                frame_time = timestamp + (i / self.target_rate)

                # Detect speech
                try:
                    is_speech = self._vad.is_speech(frame_bytes, self.target_rate)
                except Exception as e:
                    self.logger.warning("VAD error: %s", e)
                    i += self.frame_size
                    continue

                event = self._update_state(is_speech, frame_time)
                if event:
                    events.append(event)

                i += self.frame_size

        return events

    def _update_state(self, is_speech: bool, timestamp: float) -> dict[str, Any] | None:
        """Update VAD state and emit events.

        Args:
            is_speech: Whether current frame contains speech.
            timestamp: Timestamp of the current frame.

        Returns:
            Event dict if a state transition occurred, None otherwise.
        """
        event: dict[str, Any] | None = None

        if is_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0

            # Speech start: require some consecutive speech frames to avoid noise
            if not self._is_speaking and self._consecutive_speech >= 2:
                self._is_speaking = True
                self._speech_start_time = timestamp
                event = {
                    "type": "speech_start",
                    "timestamp": timestamp,
                }
                self._events.append(event)
                self.logger.debug("Speech started at %.3f", timestamp)

            # Reset speech end time if speech continues
            self._speech_end_time = None

        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0

            # Speech end: require consecutive silence frames
            if self._is_speaking and self._consecutive_silence >= self.silence_frames:
                self._is_speaking = False
                self._speech_end_time = timestamp

                # Calculate cessation latency if stop cue was set
                latency = None
                if self._stop_cue_time is not None:
                    latency = timestamp - self._stop_cue_time

                event = {
                    "type": "speech_end",
                    "timestamp": timestamp,
                    "speech_duration": (
                        timestamp - self._speech_start_time
                        if self._speech_start_time is not None
                        else None
                    ),
                    "latency_from_stop_cue": latency,
                }
                self._events.append(event)

                self.logger.debug(
                    "Speech ended at %.3f, duration=%.3f, latency=%.3f",
                    timestamp,
                    event.get("speech_duration", 0) or 0,
                    latency if latency is not None else 0,
                )

        return event

    def get_speech_cessation_latency(self) -> float | None:
        """Get the time from stop cue to speech end.

        Returns:
            Time in seconds from stop_cue_time to speech_end_time,
            or None if speech hasn't stopped yet or stop cue wasn't set.
        """
        if self._speech_end_time is None:
            return None
        if self._stop_cue_time is None:
            return None
        return self._speech_end_time - self._stop_cue_time

    def set_stop_cue_time(self, timestamp: float) -> None:
        """Record the time when STOP state appeared on screen.

        Args:
            timestamp: Monotonic timestamp of STOP cue appearance.
        """
        self._stop_cue_time = timestamp
        self.logger.debug("Stop cue recorded at %.3f", timestamp)

    def get_events(self) -> list[dict[str, Any]]:
        """Get all VAD events for the current trial.

        Returns:
            Copy of the events list for the current trial.
        """
        return list(self._events)

    def reset(self) -> None:
        """Clear buffers and state for a new trial."""
        self._buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._consecutive_silence = 0
        self._consecutive_speech = 0
        self._speech_start_time = None
        self._speech_end_time = None
        self._stop_cue_time = None
        self._events = []
        self.logger.debug("VAD state reset")

    @property
    def is_speaking(self) -> bool:
        """Current speech state."""
        return self._is_speaking

    @property
    def is_available(self) -> bool:
        """Check if WebRTC VAD is available."""
        return HAS_WEBRTC_VAD

    def get_statistics(self) -> dict[str, Any]:
        """Get current VAD statistics.

        Returns:
            Dict with current state statistics.
        """
        return {
            "is_speaking": self._is_speaking,
            "consecutive_speech": self._consecutive_speech,
            "consecutive_silence": self._consecutive_silence,
            "speech_start_time": self._speech_start_time,
            "speech_end_time": self._speech_end_time,
            "stop_cue_time": self._stop_cue_time,
            "buffer_length": len(self._buffer),
            "event_count": len(self._events),
        }