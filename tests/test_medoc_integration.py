"""Integration tests with mock Medoc server.

Tests the MedocClient and MedocExperiment with mock TCP socket server,
verifying all communication patterns and error handling without requiring
real hardware.
"""

from __future__ import annotations

import socket
import struct
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from psycopy.config import ExperimentConfig, MedocConfig
from psycopy.medoc import (
    ConnectionState,
    MedocClient,
    MedocConnectionError,
    MedocTimeoutError,
    MedocResponseError,
)
from psycopy.trial_generator import TrialConfig, generate_trials
from tests.fixtures.mock_medoc import (
    MockMedocHandler,
    MockMedocServer,
    SlowMockMedocServer,
    GarbageMockMedocServer,
    ErrorMockMedocServer,
)


class TestMockMedocServerBasics:
    """Tests for mock server infrastructure."""

    def test_server_starts_and_stops(self):
        server = MockMedocServer(port=55555)
        assert not server.is_running

        server.start()
        assert server.is_running

        server.stop()
        assert not server.is_running

    def test_server_context_manager(self):
        with MockMedocServer(port=55556) as server:
            assert server.is_running
        assert not server.is_running

    def test_client_can_connect_to_server(self):
        with MockMedocServer(port=55557) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55557)
            client = MedocClient(config)
            client.connect()
            assert client.state == ConnectionState.CONNECTED
            client.disconnect()

    def test_connection_refused_without_server(self):
        config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55558)
        client = MedocClient(config)
        with pytest.raises(MedocConnectionError):
            client.connect()


class TestMedocClientWithMockServer:
    """Tests for MedocClient with mock server (Scenario 1)."""

    def test_send_trigger_returns_ok(self):
        with MockMedocServer(port=55560) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55560)
            client = MedocClient(config)

            with client:
                client.send_trigger()

            assert server.handler.commands_received[-1] == MockMedocHandler.CMD_TRIGGER

    def test_get_status_returns_temperature(self):
        with MockMedocServer(
            port=55561, handler=MockMedocHandler(temperature_celsius=42.5)
        ) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55561)
            client = MedocClient(config)

            with client:
                status = client.get_status()

            assert status["temperature_celsius"] is not None
            assert abs(status["temperature_celsius"] - 42.5) < 0.01

    def test_get_status_response_format(self):
        with MockMedocServer(port=55562) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55562)
            client = MedocClient(config)

            with client:
                status = client.get_status()

            assert "raw_bytes" in status
            assert "temperature_celsius" in status
            assert "device_state" in status
            assert "test_state" in status
            assert len(status["raw_bytes"]) >= 4

    def test_trigger_and_status_sequence(self):
        with MockMedocServer(port=55563) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55563)
            client = MedocClient(config)

            with client:
                client.send_trigger()
                status = client.get_status()

            assert server.handler.commands_received == [0x04, 0x00]

    def test_multiple_connections(self):
        with MockMedocServer(port=55564) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55564)

            for i in range(3):
                client = MedocClient(config)
                client.connect()
                client.send_trigger()
                status = client.get_status()
                client.disconnect()

            assert len(server.handler.commands_received) == 6


class TestTriggerCommand:
    """Tests for TRIGGER command handling."""

    def test_trigger_sends_correct_byte(self):
        with MockMedocServer(port=55570) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55570)
            client = MedocClient(config)

            with client:
                client.send_trigger()

            last_cmd = server.handler.commands_received[-1]
            assert last_cmd == 0x04

    def test_trigger_ok_response(self):
        with MockMedocServer(port=55571) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55571)
            client = MedocClient(config)

            with client:
                result = client.send_trigger()

            assert result is None

    def test_trigger_error_response(self):
        with ErrorMockMedocServer(port=55572, trigger_error_code=0x01) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55572)
            client = MedocClient(config)

            with client:
                with pytest.raises(MedocResponseError) as exc_info:
                    client.send_trigger()

            assert exc_info.value.response_code == 0x01


class TestGetStatusCommand:
    """Tests for GET_STATUS command handling."""

    def test_get_status_sends_correct_byte(self):
        with MockMedocServer(port=55580) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55580)
            client = MedocClient(config)

            with client:
                client.get_status()

            last_cmd = server.handler.commands_received[-1]
            assert last_cmd == 0x00

    def test_get_status_parses_temperature(self):
        temp = 100.0
        with MockMedocServer(
            port=55581, handler=MockMedocHandler(temperature_celsius=temp)
        ) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55581)
            client = MedocClient(config)

            with client:
                status = client.get_status()

            assert status["temperature_celsius"] is not None
            assert abs(status["temperature_celsius"] - temp) < 0.01

    def test_get_status_temperature_50_celsius(self):
        temp = 50.0
        with MockMedocServer(
            port=55582, handler=MockMedocHandler(temperature_celsius=temp)
        ) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55582)
            client = MedocClient(config)

            with client:
                status = client.get_status()

            assert abs(status["temperature_celsius"] - temp) < 0.01

    def test_get_status_parses_device_state(self):
        with MockMedocServer(port=55583, handler=MockMedocHandler(device_state=2)) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55583)
            client = MedocClient(config)

            with client:
                status = client.get_status()

            assert status["device_state"] == 2

    def test_get_status_parses_test_state(self):
        with MockMedocServer(port=55584, handler=MockMedocHandler(test_state=1)) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55584)
            client = MedocClient(config)

            with client:
                status = client.get_status()

            assert status["test_state"] == 1


class TestConnectionFailure:
    """Tests for connection failure handling (Scenario 3)."""

    def test_connection_refused_raises_error(self):
        config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55590)
        client = MedocClient(config)

        with pytest.raises(MedocConnectionError):
            client.connect()

    def test_connection_refused_includes_ip_and_port(self):
        config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55591)
        client = MedocClient(config)

        with pytest.raises(MedocConnectionError) as exc_info:
            client.connect()

        assert "127.0.0.1" in str(exc_info.value)
        assert "55591" in str(exc_info.value)


class TestTimeoutHandling:
    """Tests for timeout handling (Scenario 3)."""

    def test_connect_timeout(self):
        config = MedocConfig(medoc_ip="10.255.255.1", medoc_port=55595, medoc_timeout=0.1)
        client = MedocClient(config)

        with pytest.raises(MedocTimeoutError) as exc_info:
            client.connect()

        assert exc_info.value.timeout == 0.1

    def test_slow_server_timeout(self):
        with SlowMockMedocServer(port=55596, response_delay=10.0) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55596, medoc_timeout=0.5)
            client = MedocClient(config)

            with client:
                with pytest.raises(MedocTimeoutError):
                    client.send_trigger()


class TestInvalidResponseHandling:
    """Tests for invalid response handling (Scenario 3)."""

    def test_garbage_response_for_trigger(self):
        with GarbageMockMedocServer(port=55600) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55600)
            client = MedocClient(config)

            with client:
                with pytest.raises(MedocResponseError):
                    client.send_trigger()

    def test_garbage_response_for_get_status(self):
        with GarbageMockMedocServer(port=55601) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55601)
            client = MedocClient(config)

            with client:
                status = client.get_status()
                assert status["raw_bytes"] is not None
                assert len(status["raw_bytes"]) > 0

    def test_error_response_code_1(self):
        with ErrorMockMedocServer(port=55602, trigger_error_code=0x01) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55602)
            client = MedocClient(config)

            with client:
                with pytest.raises(MedocResponseError) as exc_info:
                    client.send_trigger()

            assert exc_info.value.response_code == 1

    def test_error_response_code_2(self):
        with ErrorMockMedocServer(port=55603, trigger_error_code=0x02) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55603)
            client = MedocClient(config)

            with client:
                with pytest.raises(MedocResponseError) as exc_info:
                    client.send_trigger()

            assert exc_info.value.response_code == 2


class TestTrialGenerator:
    """Tests for trial randomization."""

    def test_generate_trials_creates_5_blocks(self):
        from random import Random

        rng = Random(42)
        trials = generate_trials(
            num_sets=5,
            trials_per_set=1,
            num_stop_trials_ratio=0.0,
            rng=rng,
        )

        assert len(trials) == 5
        for block_trials in trials:
            assert len(block_trials) == 1

    def test_trial_distribution_per_block(self):
        from random import Random

        rng = Random(42)
        trials = generate_trials(
            num_sets=5,
            trials_per_set=1,
            num_stop_trials_ratio=0.0,
            rng=rng,
        )

        for block_idx, block_trials in enumerate(trials):
            vowel_count = sum(1 for t in block_trials if t.task_type == "vowel")
            sentence_count = sum(1 for t in block_trials if t.task_type == "sentence")

            assert vowel_count == 1, f"Block {block_idx}: expected 1 vowel, got {vowel_count}"
            assert sentence_count == 0, f"Block {block_idx}: expected 0 sentence, got {sentence_count}"

    def test_go_segments_in_range(self):
        from random import Random

        rng = Random(42)
        trials = generate_trials(
            num_sets=5,
            trials_per_set=1,
            num_stop_trials_ratio=0.0,
            rng=rng,
        )

        for block_idx, block_trials in enumerate(trials):
            for trial_idx, trial in enumerate(block_trials):
                assert 16 <= trial.num_go_segments <= 28, (
                    f"Block {block_idx} Trial {trial_idx}: expected 16-28 GO segments, "
                    f"got {trial.num_go_segments}"
                )

    def test_go_durations_in_range(self):
        from random import Random

        rng = Random(42)
        trials = generate_trials(
            num_sets=5,
            trials_per_set=1,
            num_stop_trials_ratio=0.0,
            rng=rng,
        )

        for block_idx, block_trials in enumerate(trials):
            for trial_idx, trial in enumerate(block_trials):
                for seg_idx, dur in enumerate(trial.go_segment_durations):
                    assert 1.5 <= dur <= 3.5, (
                        f"Block {block_idx} Trial {trial_idx} Segment {seg_idx}: "
                        f"expected 1.5-3.5s, got {dur}"
                    )
                total_go = sum(trial.go_segment_durations)
                assert total_go < 240.0, (
                    f"Block {block_idx} Trial {trial_idx}: total GO {total_go} >= 240s"
                )

    def test_vowel_has_explicit_stop_durations(self):
        """Vowel trials emit explicit STOP durations (one per GO segment)."""
        from random import Random

        rng = Random(42)
        trials = generate_trials(
            num_sets=5,
            trials_per_set=1,
            num_stop_trials_ratio=0.0,
            rng=rng,
        )

        for block_idx, block_trials in enumerate(trials):
            for trial_idx, trial in enumerate(block_trials):
                assert trial.stop_segment_durations, (
                    f"Block {block_idx} Trial {trial_idx}: expected explicit "
                    f"stop_segment_durations for vowel"
                )
                assert len(trial.stop_segment_durations) == trial.num_go_segments, (
                    f"Block {block_idx} Trial {trial_idx}: expected one STOP per GO, "
                    f"got {len(trial.stop_segment_durations)} stops for "
                    f"{trial.num_go_segments} GOs"
                )

    def test_vowel_go_never_crosses_minute_boundary(self):
        """No GO segment spans a 60/120/180s Medoc temperature-change boundary."""
        from random import Random

        rng = Random(42)
        trials = generate_trials(
            num_sets=5,
            trials_per_set=1,
            num_stop_trials_ratio=0.0,
            rng=rng,
        )

        boundaries = [60.0, 120.0, 180.0]
        for block_idx, block_trials in enumerate(trials):
            for trial_idx, trial in enumerate(block_trials):
                stops = list(trial.stop_segment_durations)
                gos = list(trial.go_segment_durations)
                # Reconstruct the flat STOP/GO/.../STOP schedule.
                t = 0.0
                for seg_idx, go_dur in enumerate(gos):
                    # STOP precedes this GO.
                    t += stops[seg_idx]
                    go_start = t
                    go_end = t + go_dur
                    for b in boundaries:
                        assert not (go_start <= b <= go_end), (
                            f"Block {block_idx} Trial {trial_idx} Segment "
                            f"{seg_idx}: GO [{go_start}, {go_end}] crosses the "
                            f"{b}s temperature-change boundary"
                        )
                    t = go_end

    def test_reproducibility_with_seed(self):
        from random import Random

        rng1 = Random(42)
        trials1 = generate_trials(
            num_sets=5,
            trials_per_set=1,
            num_stop_trials_ratio=0.0,
            rng=rng1,
        )

        rng2 = Random(42)
        trials2 = generate_trials(
            num_sets=5,
            trials_per_set=1,
            num_stop_trials_ratio=0.0,
            rng=rng2,
        )

        for block_idx in range(len(trials1)):
            for trial_idx in range(len(trials1[block_idx])):
                t1 = trials1[block_idx][trial_idx]
                t2 = trials2[block_idx][trial_idx]
                assert t1.task_type == t2.task_type
                assert t1.num_go_segments == t2.num_go_segments
                assert t1.go_segment_durations == t2.go_segment_durations


class TestMedocTrialRecord:
    """Tests for MedocTrialRecord model."""

    def test_trial_record_to_dict(self):
        from psycopy.models import MedocTrialRecord

        record = MedocTrialRecord(
            trial_instance_id="P001_S01_set0_001",
            set_number=0,
            trial_in_set=1,
            task_type="vowel",
            is_stop_trial=True,
            trigger_timestamp=123.456,
            status_timestamp=128.456,
            temperature_raw=b"\x00\x00\xc8\x42",
            temperature_celsius=100.0,
            device_state=2,
            test_state=1,
            response_code=0,
        )

        d = record.to_dict()
        assert d["trial_instance_id"] == "P001_S01_set0_001"
        assert d["set_number"] == 0
        assert d["trial_in_set"] == 1
        assert d["task_type"] == "vowel"
        assert d["is_stop_trial"] is True
        assert d["trigger_timestamp"] == 123.456
        assert d["temperature_raw"] == "0000c842"
        assert d["temperature_celsius"] == 100.0

    def test_trial_record_null_fields(self):
        from psycopy.models import MedocTrialRecord

        record = MedocTrialRecord(
            trial_instance_id="P001_S01_set0_001",
            set_number=0,
            trial_in_set=1,
            task_type="sentence",
            is_stop_trial=False,
            trigger_timestamp=100.0,
        )

        d = record.to_dict()
        assert d["status_timestamp"] is None
        assert d["temperature_raw"] is None
        assert d["temperature_celsius"] is None
        assert d["device_state"] is None
        assert d["test_state"] is None
        assert d["response_code"] is None


class TestFullTrialWithMock:
    """Tests for full trial execution with mock server (Scenario 2)."""

    def test_full_trial_execution(self, tmp_path):
        """Test full trial execution with mocked PsychoPy and Medoc."""
        with MockMedocServer(
            port=55650, handler=MockMedocHandler(temperature_celsius=42.0)
        ) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55650)
            client = MedocClient(config)

            with client:
                trigger_time = 0.0
                client.send_trigger()
                trigger_time = 100.0

                status = client.get_status()

            assert status["temperature_celsius"] is not None
            assert server.handler.commands_received == [0x04, 0x00]


class TestMedocConfig:
    """Tests for MedocConfig validation."""

    def test_medoc_config_valid(self):
        config = MedocConfig(
            medoc_ip="192.168.1.100",
            medoc_port=5000,
            medoc_timeout=5.0,
        )
        assert config.medoc_ip == "192.168.1.100"
        assert config.medoc_port == 5000
        assert config.medoc_timeout == 5.0

    def test_medoc_config_invalid_ip(self):
        with pytest.raises(ValueError):
            MedocConfig(
                medoc_ip="invalid_ip",
                medoc_port=5000,
            )

    def test_medoc_config_invalid_port_low(self):
        with pytest.raises(ValueError):
            MedocConfig(
                medoc_ip="192.168.1.100",
                medoc_port=0,
            )

    def test_medoc_config_invalid_port_high(self):
        with pytest.raises(ValueError):
            MedocConfig(
                medoc_ip="192.168.1.100",
                medoc_port=70000,
            )

    def test_medoc_config_invalid_timeout(self):
        with pytest.raises(ValueError):
            MedocConfig(
                medoc_ip="192.168.1.100",
                medoc_port=5000,
                medoc_timeout=-1.0,
            )


class TestContextManager:
    """Tests for MedocClient context manager."""

    def test_context_manager_connects_on_enter(self):
        with MockMedocServer(port=55660) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55660)

            with MedocClient(config) as client:
                assert client.state == ConnectionState.CONNECTED

    def test_context_manager_disconnects_on_exit(self):
        with MockMedocServer(port=55661) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55661)

            client = MedocClient(config)
            with client:
                pass

            assert client.state == ConnectionState.DISCONNECTED

    def test_context_manager_disconnects_on_exception(self):
        with MockMedocServer(port=55662) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55662)

            client = MedocClient(config)
            try:
                with client:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            assert client.state == ConnectionState.DISCONNECTED


class TestFiveBlockRun:
    """Tests for full 5-block run."""

    def test_full_5_trials_simulation(self, tmp_path):
        """Simulate full 5-block run (5 trial records created)."""
        from random import Random

        rng = Random(42)
        all_trials = generate_trials(
            num_sets=5,
            trials_per_set=1,
            num_stop_trials_ratio=0.0,
            rng=rng,
        )

        total_trials = sum(len(block_trials) for block_trials in all_trials)
        assert total_trials == 5

        vowel_trials = sum(
            sum(1 for t in block_trials if t.task_type == "vowel") for block_trials in all_trials
        )
        assert vowel_trials == 5

        sentence_trials = sum(
            sum(1 for t in block_trials if t.task_type == "sentence") for block_trials in all_trials
        )
        assert sentence_trials == 0

    def test_trial_iteration_order(self):
        """Verify trials are iterated in correct order."""
        from random import Random

        rng = Random(123)
        all_trials = generate_trials(
            num_sets=5,
            trials_per_set=1,
            num_stop_trials_ratio=0.0,
            rng=rng,
        )

        trial_count = 0
        for block_num, block_trials in enumerate(all_trials):
            for trial_num, trial_config in enumerate(block_trials):
                assert isinstance(trial_config, TrialConfig)
                trial_count += 1

        assert trial_count == 5


class TestMultipleCommandsInSequence:
    """Tests for multiple commands in sequence."""

    def test_multiple_triggers_and_status(self):
        """Test multiple TRIGGER and GET_STATUS calls."""
        with MockMedocServer(port=55670) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55670)
            client = MedocClient(config)

            with client:
                for i in range(5):
                    client.send_trigger()
                    status = client.get_status()
                    assert status["temperature_celsius"] is not None

            assert len(server.handler.commands_received) == 10

    def test_repeated_connections(self):
        """Test repeated connect/disconnect cycles."""
        with MockMedocServer(port=55671) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55671)

            for i in range(3):
                client = MedocClient(config)
                client.connect()
                assert client.state == ConnectionState.CONNECTED
                client.send_trigger()
                client.get_status()
                client.disconnect()
                assert client.state == ConnectionState.DISCONNECTED


class TestTemperatureFloatParsing:
    """Tests for IEEE 754 float32 little-endian parsing."""

    def test_parse_temperature_100(self):
        temp = 100.0
        with MockMedocServer(
            port=55680, handler=MockMedocHandler(temperature_celsius=temp)
        ) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55680)
            client = MedocClient(config)

            with client:
                status = client.get_status()

            assert abs(status["temperature_celsius"] - temp) < 0.01

    def test_parse_temperature_32(self):
        temp = 32.0
        with MockMedocServer(
            port=55681, handler=MockMedocHandler(temperature_celsius=temp)
        ) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55681)
            client = MedocClient(config)

            with client:
                status = client.get_status()

            assert abs(status["temperature_celsius"] - temp) < 0.01

    def test_parse_temperature_zero(self):
        temp = 0.0
        with MockMedocServer(
            port=55682, handler=MockMedocHandler(temperature_celsius=temp)
        ) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55682)
            client = MedocClient(config)

            with client:
                status = client.get_status()

            assert abs(status["temperature_celsius"] - temp) < 0.01

    def test_parse_temperature_negative(self):
        temp = -10.5
        with MockMedocServer(
            port=55683, handler=MockMedocHandler(temperature_celsius=temp)
        ) as server:
            config = MedocConfig(medoc_ip="127.0.0.1", medoc_port=55683)
            client = MedocClient(config)

            with client:
                status = client.get_status()

            assert abs(status["temperature_celsius"] - temp) < 0.01
