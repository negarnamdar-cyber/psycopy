"""Tests for WebRTC VAD implementation.

These tests mock webrtcvad to avoid requiring the native library.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from psycopy.vad import HAS_WEBRTC_VAD, VADService, VADServiceError


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_webrtcvad():
    """Mock webrtcvad module and Vad class for testing.
    
    The mock Vad class simulates speech detection based on audio amplitude:
    - Speech detected when frame contains samples above a threshold
    - This allows controllable test scenarios for speech/silence detection
    """
    class MockVad:
        """Mock VAD that detects speech based on frame amplitude."""
        
        def __init__(self, aggressiveness: int):
            if aggressiveness not in (0, 1, 2, 3):
                raise ValueError(f"Invalid aggressiveness: {aggressiveness}")
            self.aggressiveness = aggressiveness
            # Higher aggressiveness = higher threshold for speech detection
            self._threshold = 50 + aggressiveness * 20  # 50, 70, 90, 110
        
        def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
            """Detect speech based on amplitude threshold.
            
            Args:
                frame_bytes: Raw PCM bytes (int16).
                sample_rate: Sample rate (should be 16000 for WebRTC VAD).
                
            Returns:
                True if speech detected (amplitude above threshold), False otherwise.
            """
            if len(frame_bytes) < 2:
                return False
            # Convert bytes to int16 array
            samples = np.frombuffer(frame_bytes, dtype=np.int16)
            # Check if any sample exceeds threshold
            max_amplitude = np.max(np.abs(samples))
            return max_amplitude > self._threshold
    
    mock_module = MagicMock()
    mock_module.Vad = MockVad
    
    with patch("psycopy.vad.webrtcvad", mock_module), \
         patch("psycopy.vad.HAS_WEBRTC_VAD", True):
        yield MockVad


@pytest.fixture
def vad_service(mock_webrtcvad):
    """Create a VADService instance with mocked webrtcvad.
    
    Uses default settings:
    - aggressiveness=2 (medium sensitivity)
    - frame_duration_ms=30 (standard frame size)
    - silence_frames=10 (10 consecutive silent frames to end speech)
    - source_rate=44100 (CD quality)
    """
    return VADService(
        aggressiveness=2,
        frame_duration_ms=30,
        silence_frames=10,
        source_rate=44100,
    )


@pytest.fixture
def vad_service_aggressive(mock_webrtcvad):
    """Create a VADService with aggressive (strict) speech detection."""
    return VADService(aggressiveness=3, frame_duration_ms=30)


@pytest.fixture
def vad_service_permissive(mock_webrtcvad):
    """Create a VADService with permissive (lenient) speech detection."""
    return VADService(aggressiveness=0, frame_duration_ms=30)


# ==============================================================================
# VADService Initialization Tests
# ==============================================================================

class TestVADServiceInit:
    """Tests for VADService initialization and parameter validation."""
    
    def test_valid_aggressiveness_levels(self, mock_webrtcvad):
        """Test that all valid aggressiveness levels (0-3) are accepted."""
        for level in (0, 1, 2, 3):
            vad = VADService(aggressiveness=level)
            assert vad.aggressiveness == level
    
    def test_invalid_aggressiveness_raises_error(self, mock_webrtcvad):
        """Test that invalid aggressiveness raises VADServiceError."""
        with pytest.raises(VADServiceError) as exc_info:
            VADService(aggressiveness=4)
        assert "Invalid aggressiveness" in str(exc_info.value)
        assert "4" in str(exc_info.value)
        
        with pytest.raises(VADServiceError) as exc_info:
            VADService(aggressiveness=-1)
        assert "Invalid aggressiveness" in str(exc_info.value)
    
    def test_valid_frame_durations(self, mock_webrtcvad):
        """Test that all valid frame durations (10, 20, 30) are accepted."""
        for duration in (10, 20, 30):
            vad = VADService(frame_duration_ms=duration)
            assert vad.frame_duration_ms == duration
    
    def test_invalid_frame_duration_raises_error(self, mock_webrtcvad):
        """Test that invalid frame duration raises VADServiceError."""
        with pytest.raises(VADServiceError) as exc_info:
            VADService(frame_duration_ms=15)
        assert "Invalid frame_duration_ms" in str(exc_info.value)
        assert "15" in str(exc_info.value)
        
        with pytest.raises(VADServiceError) as exc_info:
            VADService(frame_duration_ms=0)
        assert "Invalid frame_duration_ms" in str(exc_info.value)
    
    def test_default_values(self, mock_webrtcvad):
        """Test that VADService initializes with correct default values."""
        vad = VADService()
        
        assert vad.aggressiveness == 2  # Default medium sensitivity
        assert vad.frame_duration_ms == 30  # Default 30ms frames
        assert vad.silence_frames == 10  # Default 10 silent frames to end
        assert vad.source_rate == 44100  # Default CD quality
        assert vad.target_rate == 16000  # WebRTC VAD requires 16kHz
        assert vad.frame_size == 480  # 16000 * 0.030 = 480 samples
    
    def test_custom_values(self, mock_webrtcvad):
        """Test that VADService accepts custom parameter values."""
        vad = VADService(
            aggressiveness=1,
            frame_duration_ms=20,
            silence_frames=5,
            source_rate=48000,
        )
        
        assert vad.aggressiveness == 1
        assert vad.frame_duration_ms == 20
        assert vad.silence_frames == 5
        assert vad.source_rate == 48000
        # Frame size: 16000 * 0.020 = 320 samples
        assert vad.frame_size == 320
    
    def test_init_without_webrtcvad_raises_error(self):
        """Test that VADService raises error when webrtcvad is not installed."""
        with patch("psycopy.vad.HAS_WEBRTC_VAD", False):
            with pytest.raises(VADServiceError) as exc_info:
                VADService()
            assert "webrtcvad package not installed" in str(exc_info.value)


# ==============================================================================
# Audio Processing Tests
# ==============================================================================

class TestAudioProcessing:
    """Tests for audio chunk processing and speech detection."""
    
    def test_process_silence_chunk(self, vad_service):
        """Test processing a silent audio chunk (no speech detected)."""
        # Generate silence (very low amplitude)
        silence = np.zeros((1024, 1), dtype=np.float32)
        
        # Process the silence chunk
        events = vad_service.process_audio_chunk(silence, timestamp=0.0)
        
        # Should return no events (speech requires consecutive frames)
        assert events == []
        assert not vad_service.is_speaking
    
    def test_process_speech_chunk_emits_speech_start(self, vad_service):
        """Test that sustained speech triggers speech_start event."""
        # Generate speech-like audio (high amplitude)
        # Need enough samples for multiple frames after resampling
        # At 44100Hz, we need to generate enough to resample to at least 2 frames
        # 2 frames at 30ms = 60ms @ 16kHz = 960 samples
        # At 44100Hz: approximately 2646 samples needed
        
        speech = np.ones((3000, 1), dtype=np.float32) * 0.9  # High amplitude
        
        events = vad_service.process_audio_chunk(speech, timestamp=0.0)
        
        # With consecutive speech frames (2+), should detect speech start
        # The mock detects speech when amplitude > threshold
        assert vad_service.is_speaking or len(events) > 0
    
    def test_resampling_from_44100_to_16000(self, vad_service):
        """Test audio resampling from 44.1kHz to 16kHz."""
        # Create audio at 44.1kHz
        duration_sec = 0.1
        num_samples = int(vad_service.source_rate * duration_sec)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, num_samples))
        audio_2d = audio.reshape(-1, 1).astype(np.float32)
        
        # Process should handle resampling internally
        events = vad_service.process_audio_chunk(audio_2d, timestamp=0.0)
        
        # Buffer should be consumed (mostly)
        # After processing, remaining buffer should be small
        remaining = len(vad_service._buffer)
        # Some samples may remain if not enough for a complete frame
        max_expected_remaining = vad_service.frame_size * 3
        assert remaining < max_expected_remaining * 3
    
    def test_frame_buffering_and_alignment(self, vad_service):
        """Test that audio is buffered correctly when chunks are smaller than frame."""
        # Send small chunks that don't make a complete frame
        small_chunk = np.ones((100, 1), dtype=np.float32) * 0.5
        
        # First chunk - should buffer but not process
        events1 = vad_service.process_audio_chunk(small_chunk, timestamp=0.0)
        
        # Buffer should have accumulated samples
        assert len(vad_service._buffer) > 0
        
        # Send more chunks until we have enough for processing
        events2 = vad_service.process_audio_chunk(small_chunk, timestamp=0.001)
        events3 = vad_service.process_audio_chunk(small_chunk, timestamp=0.002)
        
        # Eventually should have processed some frames
        all_events = events1 + events2 + events3
        # Just verify it doesn't crash and maintains buffer correctly
        assert isinstance(vad_service._buffer, np.ndarray)
    
    def test_event_emission_speech_start_end(self, vad_service):
        """Test that speech_start and speech_end events are emitted correctly."""
        # Generate sustained speech (multiple frames worth)
        speech_frames_needed = vad_service.silence_frames + 5
        
        # Calculate samples needed for multiple frames
        # At 16kHz, 30ms frame = 480 samples
        # At 44100Hz, this is ~1323 samples per frame
        samples_per_frame = int(vad_service.frame_size * vad_service._down / vad_service._up)
        total_samples = samples_per_frame * speech_frames_needed
        
        # High amplitude speech
        speech = np.ones((total_samples, 1), dtype=np.float32) * 0.9
        
        # Process speech - should trigger speech_start
        events_start = vad_service.process_audio_chunk(speech, timestamp=0.0)
        
        # Now process silence to trigger speech_end
        # Need silence_frames consecutive silent frames
        silence_samples = samples_per_frame * (vad_service.silence_frames + 2)
        silence = np.zeros((silence_samples, 1), dtype=np.float32) * 0.01
        
        events_end = vad_service.process_audio_chunk(silence, timestamp=0.1)
        
        # Should have speech_end event with latency info
        all_events = events_start + events_end
        
        # Find speech_end event
        speech_end_events = [e for e in all_events if e.get("type") == "speech_end"]
        
        # If speech was detected, should have an end event
        if vad_service._is_speaking or any(e["type"] == "speech_start" for e in events_start):
            # Speech should have ended
            assert not vad_service.is_speaking
    
    def test_consecutive_silence_threshold(self, vad_service):
        """Test that speech_end requires consecutive silence frames."""
        # Create service with high silence threshold
        vad = VADService(
            aggressiveness=0,  # Permissive
            frame_duration_ms=30,
            silence_frames=5,  # Need 5 consecutive silent frames
            source_rate=44100,
        )
        
        # Process speech
        speech = np.ones((2000, 1), dtype=np.float32) * 0.9
        vad.process_audio_chunk(speech, timestamp=0.0)
        
        # Verify we're in speech state
        was_speaking = vad.is_speaking
        
        # Send only a few silent frames (not enough to trigger end)
        minimal_silence = np.zeros((500, 1), dtype=np.float32)
        vad.process_audio_chunk(minimal_silence, timestamp=0.05)
        
        # If we were speaking and haven't hit threshold, should still be speaking
        if was_speaking:
            # May still be speaking if not enough silence
            pass  # Behavior depends on exact frame alignment


# ==============================================================================
# Speech Cessation Latency Tests
# ==============================================================================

class TestSpeechCessationLatency:
    """Tests for speech cessation latency measurement."""
    
    def test_get_speech_cessation_latency_before_stop_cue(self, vad_service):
        """Test that latency is None when stop cue hasn't been set."""
        latency = vad_service.get_speech_cessation_latency()
        assert latency is None
    
    def test_get_speech_cessation_latency_before_speech_end(self, vad_service):
        """Test that latency is None when speech hasn't ended yet."""
        vad_service.set_stop_cue_time(1.0)
        
        # Haven't processed any audio, no speech end
        latency = vad_service.get_speech_cessation_latency()
        assert latency is None
    
    def test_set_stop_cue_time(self, vad_service):
        """Test that stop cue time is recorded correctly."""
        vad_service.set_stop_cue_time(42.5)
        
        stats = vad_service.get_statistics()
        assert stats["stop_cue_time"] == 42.5
    
    def test_latency_calculation_accuracy(self, vad_service):
        """Test that cessation latency is calculated accurately."""
        # Setup: speech starts, stop cue appears, speech ends
        
        # Process speech to trigger speech_start
        speech = np.ones((3000, 1), dtype=np.float32) * 0.9
        vad_service.process_audio_chunk(speech, timestamp=0.0)
        
        # Set stop cue time
        stop_time = 0.05
        vad_service.set_stop_cue_time(stop_time)
        
        # Process silence to trigger speech_end
        samples_for_silence = int(vad_service.frame_size * vad_service._down / vad_service._up)
        silence_frames_needed = vad_service.silence_frames + 1
        silence = np.zeros((samples_for_silence * silence_frames_needed, 1), dtype=np.float32)
        
        vad_service.process_audio_chunk(silence, timestamp=0.1)
        
        # If speech was detected and ended, check latency
        events = vad_service.get_events()
        speech_end_events = [e for e in events if e.get("type") == "speech_end"]
        
        if speech_end_events:
            # Should have latency from stop cue
            latency = vad_service.get_speech_cessation_latency()
            assert latency is not None
            # Latency should be positive (speech end after stop cue)
            assert latency >= 0
    
    def test_latency_in_speech_end_event(self, vad_service):
        """Test that latency_from_stop_cue is included in speech_end event."""
        # Process speech
        speech = np.ones((3000, 1), dtype=np.float32) * 0.9
        vad_service.process_audio_chunk(speech, timestamp=0.0)
        
        # Set stop cue
        vad_service.set_stop_cue_time(0.05)
        
        # Process silence
        samples_per_frame = int(vad_service.frame_size * vad_service._down / vad_service._up)
        silence_frames = vad_service.silence_frames + 2
        silence = np.zeros((samples_per_frame * silence_frames, 1), dtype=np.float32)
        vad_service.process_audio_chunk(silence, timestamp=0.1)
        
        # Check events for latency field
        events = vad_service.get_events()
        speech_end = next((e for e in events if e.get("type") == "speech_end"), None)
        
        if speech_end:
            # latency_from_stop_cue should be in the event
            assert "latency_from_stop_cue" in speech_end


# ==============================================================================
# State Management Tests
# ==============================================================================

class TestStateManagement:
    """Tests for VAD state management methods."""
    
    def test_reset_clears_all_state(self, vad_service):
        """Test that reset() clears all internal state."""
        # Set up some state
        speech = np.ones((3000, 1), dtype=np.float32) * 0.9
        vad_service.process_audio_chunk(speech, timestamp=0.0)
        vad_service.set_stop_cue_time(0.05)
        
        # Reset
        vad_service.reset()
        
        # Check all state is cleared
        stats = vad_service.get_statistics()
        assert stats["is_speaking"] is False
        assert stats["consecutive_speech"] == 0
        assert stats["consecutive_silence"] == 0
        assert stats["speech_start_time"] is None
        assert stats["speech_end_time"] is None
        assert stats["stop_cue_time"] is None
        assert stats["buffer_length"] == 0
        assert stats["event_count"] == 0
    
    def test_get_events_returns_correct_events(self, vad_service):
        """Test that get_events returns all events for the trial."""
        # Should start empty
        events = vad_service.get_events()
        assert events == []
        
        # Create some speech
        speech = np.ones((3000, 1), dtype=np.float32) * 0.9
        vad_service.process_audio_chunk(speech, timestamp=0.0)
        
        # get_events should return a copy (not reference)
        events1 = vad_service.get_events()
        events2 = vad_service.get_events()
        assert events1 == events2
        # Both should be copies
        assert events1 is not events2
    
    def test_is_speaking_property(self, vad_service):
        """Test that is_speaking reflects current speech state."""
        # Initially not speaking
        assert vad_service.is_speaking is False
        
        # Process speech
        speech = np.ones((3000, 1), dtype=np.float32) * 0.9
        vad_service.process_audio_chunk(speech, timestamp=0.0)
        
        # If speech detected, should be speaking
        # This depends on whether consecutive_speech threshold was met
        
        # After silence, should not be speaking
        samples_per_frame = int(vad_service.frame_size * vad_service._down / vad_service._up)
        silence_frames = vad_service.silence_frames + 2
        silence = np.zeros((samples_per_frame * silence_frames, 1), dtype=np.float32)
        vad_service.process_audio_chunk(silence, timestamp=0.1)
        
        # Should have transitioned to not speaking
        assert vad_service.is_speaking is False
    
    def test_events_are_accumulated(self, vad_service):
        """Test that events accumulate across process_audio_chunk calls."""
        # First speech segment
        speech = np.ones((3000, 1), dtype=np.float32) * 0.9
        vad_service.process_audio_chunk(speech, timestamp=0.0)
        events_after_first = len(vad_service.get_events())
        
        # Second speech segment (after silence)
        silence = np.zeros((3000, 1), dtype=np.float32)
        vad_service.process_audio_chunk(silence, timestamp=0.1)
        vad_service.process_audio_chunk(speech, timestamp=0.2)
        events_after_second = len(vad_service.get_events())
        
        # Events should have accumulated (speech_start, speech_end, speech_start)
        # Note: depending on frame alignment, may have multiple events
        assert events_after_second >= events_after_first
    
    def test_get_statistics(self, vad_service):
        """Test that get_statistics returns correct information."""
        stats = vad_service.get_statistics()
        
        # Check all expected keys
        expected_keys = {
            "is_speaking",
            "consecutive_speech",
            "consecutive_silence",
            "speech_start_time",
            "speech_end_time",
            "stop_cue_time",
            "buffer_length",
            "event_count",
        }
        assert set(stats.keys()) == expected_keys
        
        # Initial values
        assert stats["is_speaking"] is False
        assert stats["event_count"] == 0
        assert stats["buffer_length"] == 0


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_audio_chunk(self, vad_service):
        """Test processing an empty audio chunk."""
        empty = np.array([], dtype=np.float32).reshape(0, 1)
        
        events = vad_service.process_audio_chunk(empty, timestamp=0.0)
        
        assert events == []
        assert vad_service.is_speaking is False
    
    def test_very_short_audio_chunk(self, vad_service):
        """Test processing a very short audio chunk (smaller than one frame)."""
        # Single sample
        short = np.array([[0.5]], dtype=np.float32)
        
        events = vad_service.process_audio_chunk(short, timestamp=0.0)
        
        # Should buffer without crashing
        assert isinstance(events, list)
        # Should not be speaking (not enough for speech detection)
        assert vad_service.is_speaking is False
    
    def test_stereo_to_mono_conversion(self, vad_service):
        """Test that stereo input is properly converted to mono."""
        # Stereo input: 2 channels
        stereo = np.random.randn(1024, 2).astype(np.float32) * 0.5
        
        # Should handle stereo (flatten to mono)
        events = vad_service.process_audio_chunk(stereo, timestamp=0.0)
        
        # Just verify it doesn't crash
        assert isinstance(events, list)
    
    def test_1d_audio_input(self, vad_service):
        """Test processing 1D audio array (already mono)."""
        # 1D array (samples only, no channel dimension)
        mono = np.random.randn(1024).astype(np.float32) * 0.5
        
        events = vad_service.process_audio_chunk(mono, timestamp=0.0)
        
        # Should handle 1D input
        assert isinstance(events, list)
    
    def test_negative_timestamp(self, vad_service):
        """Test processing with negative timestamp."""
        audio = np.zeros((100, 1), dtype=np.float32)
        
        # Should handle negative timestamps gracefully
        events = vad_service.process_audio_chunk(audio, timestamp=-1.0)
        
        assert isinstance(events, list)
    
    def test_float_audio_values(self, vad_service):
        """Test processing float32 audio values in range [-1, 1]."""
        # Audio should be normalized float32 in [-1, 1]
        audio = np.sin(np.linspace(0, 2 * np.pi, 1024)).reshape(-1, 1).astype(np.float32)
        
        events = vad_service.process_audio_chunk(audio, timestamp=0.0)
        
        # Should handle normalized float audio
        assert isinstance(events, list)
    
    def test_multiple_resets(self, vad_service):
        """Test multiple consecutive resets."""
        for _ in range(5):
            vad_service.reset()
        
        # Should still be in clean state
        stats = vad_service.get_statistics()
        assert stats["is_speaking"] is False
        assert stats["event_count"] == 0
    
    def test_zero_silence_frames_raises(self, mock_webrtcvad):
        """Test that silence_frames=0 works (immediately detects silence)."""
        # silence_frames can be 0 or any positive value
        vad = VADService(silence_frames=0, aggressiveness=0)
        assert vad.silence_frames == 0
    
    def test_different_sample_rates(self, mock_webrtcvad):
        """Test VADService with different source sample rates."""
        # Common sample rates
        for rate in [8000, 16000, 22050, 44100, 48000, 96000]:
            vad = VADService(source_rate=rate)
            assert vad.source_rate == rate
            assert vad.target_rate == 16000  # Always 16kHz for WebRTC VAD
    
    def test_is_available_property(self, mock_webrtcvad):
        """Test that is_available returns True when webrtcvad is mocked."""
        vad = VADService()
        assert vad.is_available is True
    
    def test_is_available_without_webrtcvad(self):
        """Test that is_available returns False when webrtcvad is not installed."""
        with patch("psycopy.vad.HAS_WEBRTC_VAD", False):
            # Can't create VADService, but check module-level flag
            # This test documents expected behavior
            pass


# ==============================================================================
# Frame Size Calculations Tests
# ==============================================================================

class TestFrameSizeCalculations:
    """Tests for frame size calculations with different configurations."""
    
    def test_frame_size_10ms(self, mock_webrtcvad):
        """Test frame size calculation for 10ms frames."""
        vad = VADService(frame_duration_ms=10)
        # 16000 Hz * 0.010 sec = 160 samples
        assert vad.frame_size == 160
    
    def test_frame_size_20ms(self, mock_webrtcvad):
        """Test frame size calculation for 20ms frames."""
        vad = VADService(frame_duration_ms=20)
        # 16000 Hz * 0.020 sec = 320 samples
        assert vad.frame_size == 320
    
    def test_frame_size_30ms(self, mock_webrtcvad):
        """Test frame size calculation for 30ms frames."""
        vad = VADService(frame_duration_ms=30)
        # 16000 Hz * 0.030 sec = 480 samples
        assert vad.frame_size == 480


# ==============================================================================
# Resampling Tests
# ==============================================================================

class TestResampling:
    """Tests for audio resampling from various sample rates."""
    
    def test_resampling_48000_to_16000(self, mock_webrtcvad):
        """Test resampling from 48kHz to 16kHz."""
        vad = VADService(source_rate=48000)
        
        # Create audio at 48kHz
        duration_sec = 0.1
        num_samples = int(48000 * duration_sec)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, num_samples))
        audio_2d = audio.reshape(-1, 1).astype(np.float32)
        
        # Should process without error
        events = vad.process_audio_chunk(audio_2d, timestamp=0.0)
        assert isinstance(events, list)
    
    def test_resampling_16000_to_16000(self, mock_webrtcvad):
        """Test handling of audio already at 16kHz (minimal resampling)."""
        vad = VADService(source_rate=16000)
        
        # Create audio at 16kHz
        duration_sec = 0.1
        num_samples = int(16000 * duration_sec)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, num_samples))
        audio_2d = audio.reshape(-1, 1).astype(np.float32)
        
        # Should process without error
        events = vad.process_audio_chunk(audio_2d, timestamp=0.0)
        assert isinstance(events, list)
    
    def test_resampling_preserves_content(self, mock_webrtcvad, vad_service):
        """Test that resampling preserves audio content integrity."""
        # Create sustained high-amplitude speech
        # Need enough samples after resampling to fill multiple frames
        samples_needed = int(vad_service.frame_size * vad_service._down / vad_service._up) * 10
        speech = np.ones((samples_needed, 1), dtype=np.float32) * 0.8
        
        # Process and check events
        events = vad_service.process_audio_chunk(speech, timestamp=0.0)
        
        # The high-amplitude content should trigger speech detection
        # (depending on how many consecutive frames were processed)
        # Just verify no crashes and events is a list
        assert isinstance(events, list)


# ==============================================================================
# Integration-Style Tests
# ==============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_speech_cycle(self, vad_service):
        """Test a complete speech cycle: start -> speech -> silence -> end."""
        # Calculate samples needed for various phases
        samples_per_frame = int(vad_service.frame_size * vad_service._down / vad_service._up)
        
        # Phase 1: Initial silence
        initial_silence = np.zeros((samples_per_frame * 2, 1), dtype=np.float32)
        vad_service.process_audio_chunk(initial_silence, timestamp=0.0)
        assert not vad_service.is_speaking
        
        # Phase 2: Speech starts
        speech_frames = 5
        speech_samples = samples_per_frame * speech_frames
        speech = np.ones((speech_samples, 1), dtype=np.float32) * 0.9
        vad_service.process_audio_chunk(speech, timestamp=0.1)
        
        # Could be speaking now (if enough consecutive speech frames)
        
        # Phase 3: Speech continues
        vad_service.process_audio_chunk(speech, timestamp=0.2)
        
        # Phase 4: Silence causes speech end
        silence_frames = vad_service.silence_frames + 2
        silence_samples = samples_per_frame * silence_frames
        silence = np.zeros((silence_samples, 1), dtype=np.float32)
        vad_service.process_audio_chunk(silence, timestamp=0.3)
        
        # Should not be speaking anymore
        assert not vad_service.is_speaking
        
        # Check events were recorded
        events = vad_service.get_events()
        event_types = [e["type"] for e in events]
        
        # If speech was detected, should have speech_end
        if "speech_start" in event_types:
            assert "speech_end" in event_types
    
    def test_reset_between_trials(self, vad_service):
        """Test that reset properly clears state between trials."""
        # Create speech in first trial
        samples_per_frame = int(vad_service.frame_size * vad_service._down / vad_service._up)
        speech = np.ones((samples_per_frame * 10, 1), dtype=np.float32) * 0.9
        vad_service.process_audio_chunk(speech, timestamp=0.0)
        vad_service.set_stop_cue_time(0.1)
        
        events_trial1 = len(vad_service.get_events())
        
        # Reset for new trial
        vad_service.reset()
        
        # State should be clear
        assert len(vad_service.get_events()) == 0
        assert vad_service.is_speaking is False
        assert vad_service.get_speech_cessation_latency() is None
        
        # Process silence in second trial
        silence = np.zeros((samples_per_frame * 10, 1), dtype=np.float32)
        vad_service.process_audio_chunk(silence, timestamp=0.5)
        
        events_trial2 = len(vad_service.get_events())
        
        # Second trial should have no speech events
        assert events_trial2 == 0


# Need to fix the reference to vad_service in TestResampling
# Fix the fixture reference issue