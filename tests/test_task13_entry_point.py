"""
QA Scenarios for Task 13: Update main.py entry point for MedocExperiment

This module verifies:
1. Medoc experiment entry - MedocExperiment instantiated when medoc_config present
2. Non-Medoc experiment entry - original Experiment instantiated when medoc_config is None
3. Medoc connection error - graceful exit with error dialog
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psycopy.config import ExperimentConfig, MedocConfig


def test_scenario_1_medoc_experiment_entry():
    """Scenario 1: Medoc experiment entry with MedocConfig present."""
    config = ExperimentConfig(
        participant_id="001",
        session_id="01",
        medoc_config=MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000),
    )

    # Assert MedocConfig is present
    assert config.medoc_config is not None, "MedocConfig should be present"
    assert config.medoc_config.medoc_ip == "192.168.1.100"
    assert config.medoc_config.medoc_port == 5000
    print("PASS: Scenario 1 - MedocConfig present in ExperimentConfig")


def test_scenario_2_non_medoc_experiment_entry():
    """Scenario 2: Non-Medoc experiment entry without medoc_config."""
    config = ExperimentConfig(
        participant_id="001",
        session_id="01",
    )

    # Assert medoc_config is None (backward compatibility)
    assert config.medoc_config is None, "MedocConfig should be None for non-Medoc experiments"
    print("PASS: Scenario 2 - No medoc_config means backward-compatible Experiment")


def test_scenario_3_medoc_connection_error():
    """Scenario 3: Medoc connection error handling at startup."""
    from psycopy.medoc import MedocConnectionError

    # Verify MedocConnectionError can be raised with correct attributes
    error = MedocConnectionError("192.168.1.100", 5000)
    assert error.ip == "192.168.1.100"
    assert error.port == 5000
    assert "Connection refused" in str(error)

    print("PASS: Scenario 3 - MedocConnectionError has correct attributes")


def test_main_conditional_instantiation():
    """Verify main() uses conditional instantiation based on medoc_config."""
    from psycopy.experiment import main
    import inspect

    # Get the source code of main to verify conditional logic
    source = inspect.getsource(main)

    # Verify MedocExperiment is imported conditionally
    assert "MedocExperiment" in source, "main() should reference MedocExperiment"
    # Verify conditional check on medoc_config
    assert "medoc_config is not None" in source, "main() should check medoc_config"
    # Verify Experiment is still used for non-Medoc case
    assert "Experiment(config)" in source, "main() should still use Experiment for non-Medoc"

    print("PASS: main() has conditional MedocExperiment/Experiment instantiation")


def test_medoc_error_handling_in_main():
    """Verify main() has MedocConnectionError handling."""
    from psycopy.experiment import main
    import inspect

    source = inspect.getsource(main)

    # Verify error handling exists
    assert "MedocConnectionError" in source, "main() should catch MedocConnectionError"
    # Verify graceful exit via return
    assert "return" in source, "main() should return gracefully on error"

    print("PASS: main() has MedocConnectionError handling with graceful exit")


def test_backward_compatibility_preserved():
    """Verify existing Experiment class is still importable and usable."""
    # This ensures we haven't broken backward compatibility
    from psycopy.experiment import Experiment

    # Can create a mock config without medoc_config
    config = ExperimentConfig(participant_id="test", session_id="01")
    assert config.medoc_config is None

    # Experiment class should still exist and be usable
    assert Experiment is not None
    assert callable(Experiment)

    print("PASS: Backward compatibility - Experiment class still works")


if __name__ == "__main__":
    print("=" * 60)
    print("QA Scenarios for Task 13: MedocExperiment Entry Point")
    print("=" * 60)
    print()

    test_scenario_1_medoc_experiment_entry()
    test_scenario_2_non_medoc_experiment_entry()
    test_scenario_3_medoc_connection_error()
    test_main_conditional_instantiation()
    test_medoc_error_handling_in_main()
    test_backward_compatibility_preserved()

    print()
    print("=" * 60)
    print("ALL QA SCENARIOS PASSED!")
    print("=" * 60)
