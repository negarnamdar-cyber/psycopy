from pathlib import Path

from psycopy.features import FeatureExtractor


def test_extract_run_matrix_writes_disabled_manifest(tmp_path: Path) -> None:
    extractor = FeatureExtractor()
    extractor.smile = None
    extractor.disabled_reason = "test-disabled"

    manifest = tmp_path / "features_manifest.json"
    extractor.extract_run_matrix(
        trials=[],
        audio_dir=tmp_path / "audio",
        standardized_audio_dir=tmp_path / "audio_16k",
        features_csv_path=tmp_path / "features.csv",
        manifest_path=manifest,
        metadata={"app_version": "test"},
    )

    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert '"status": "disabled"' in text
    assert '"reason": "test-disabled"' in text
