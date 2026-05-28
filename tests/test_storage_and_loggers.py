from pathlib import Path

from psycopy.models import TrialRecord, TrialStatus
from psycopy.storage import atomic_write_json
from psycopy.utils import TrialLogger


def test_atomic_write_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    atomic_write_json(path, {"a": 1})
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip().startswith("{")


def test_incremental_trial_logging(tmp_path: Path) -> None:
    trials = TrialLogger(tmp_path / "trials.csv")
    trials.log_trial(
        TrialRecord(
            participant_id="001",
            session_id="01",
            timestamp="2026-01-01T00:00:00",
            block="baseline",
            block_index=1,
            trial_number=1,
            trial_id="1",
            sentence_text="hello",
            trial_duration_sec=12.0,
            actual_duration_sec=12.0,
            audio_filename="x.wav",
            num_switches=6,
            start_state="GO",
            status=TrialStatus.SUCCESS,
        )
    )
    trials.save()
    assert (tmp_path / "trials.csv").exists()
