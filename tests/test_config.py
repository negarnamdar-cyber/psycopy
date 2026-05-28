from psycopy.config import MedocConfig, ExperimentConfig


def test_valid_medoc_config():
    config = MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000, medoc_timeout=5.0)
    assert config.medoc_ip == "192.168.1.100"
    assert config.medoc_port == 5000
    assert config.medoc_timeout == 5.0
    assert config.baseline_temp == 30.0
    print("PASS: test_valid_medoc_config")


def test_invalid_ip_address():
    try:
        MedocConfig(medoc_ip="invalid", medoc_port=5000)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid IP" in str(e)
        print("PASS: test_invalid_ip_address")


def test_invalid_port():
    try:
        MedocConfig(medoc_ip="192.168.1.100", medoc_port=70000)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "port" in str(e).lower()
        print("PASS: test_invalid_port")


def test_experiment_config_composition():
    medoc = MedocConfig(medoc_ip="192.168.1.100", medoc_port=5000)
    config = ExperimentConfig(participant_id="001", medoc_config=medoc)
    assert config.medoc_config is not None
    assert config.medoc_config.medoc_ip == "192.168.1.100"
    print("PASS: test_experiment_config_composition")


if __name__ == "__main__":
    test_valid_medoc_config()
    test_invalid_ip_address()
    test_invalid_port()
    test_experiment_config_composition()
    print("\nAll QA scenarios passed!")
