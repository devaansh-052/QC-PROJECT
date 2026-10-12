import json
from pathlib import Path
import numpy as np
import pytest
from quantum_fraud.evaluation.metrics import binary_metrics, select_f1_threshold
from quantum_fraud.evaluation.registry import model_settings, normalize, run_timestamp, sha256, validate_registry, verify_metrics, write_new


def test_immutable_json_and_hash(tmp_path):
    path = tmp_path / "config.json"
    write_new(path, {"seed": 42})
    before = sha256(path)
    with pytest.raises(FileExistsError):
        write_new(path, {"seed": 43})
    assert sha256(path) == before


def test_original_timestamp():
    assert run_timestamp("vqc_20260915T144414") == "2026-09-15T14:44:14+00:00"
    with pytest.raises(ValueError):
        run_timestamp("unknown")


def metrics_fixture(tmp_path):
    y = np.array([0, 0, 1, 1])
    validation = np.array([.1, .7, .6, .9])
    test = np.array([.2, .8, .3, .95])
    np.save(tmp_path / "labels.npy", y)
    np.savez(tmp_path / "scores.npz", validation=validation, test=test)
    threshold = select_f1_threshold(y, validation)
    return {"experiment_id": "fixture", "source_scores": "scores.npz", "source_labels": {s: "labels.npy" for s in ("validation", "test")},
            "score_keys": {s: s for s in ("validation", "test")},
            "metrics": {"validation": binary_metrics(y, validation, threshold), "test": binary_metrics(y, test, threshold)}}


def test_metric_verification_detects_corruption(tmp_path):
    result = metrics_fixture(tmp_path)
    verify_metrics(result, tmp_path)
    result["metrics"]["test"]["recall"] = .123
    with pytest.raises(AssertionError):
        verify_metrics(result, tmp_path)


def test_registry_detects_artifact_tampering(tmp_path):
    result = metrics_fixture(tmp_path)
    write_new(tmp_path / "config.json", {"experiment_id": "fixture"})
    result["config_sha256"] = sha256(tmp_path / "config.json")
    write_new(tmp_path / "result.json", result)
    write_new(tmp_path / "registry.json", {"status": "complete", "experiments": [{"config": "config.json", "result": "result.json"}],
        "artifacts": [{"path": "scores.npz", "sha256": sha256(tmp_path / "scores.npz")}]})
    assert validate_registry(tmp_path, tmp_path)["experiments_verified"] == 1
    (tmp_path / "scores.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_registry(tmp_path, tmp_path)


def test_normalization_records_resample_and_missing_timing(tmp_path, monkeypatch):
    import quantum_fraud.evaluation.registry as registry
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(registry, "PREPARED", manifest_path)
    manifest = {"status": "complete", "dataset": {"sha256": "a" * 64}, "feature_policy": {"target": "isFraud"}, "input_features": ["step", "amount"],
                "config": {"seed": 42}, "splits": {s: {"rows": n, "fraud": n // 2, "step_min": i, "step_max": i} for i, (s, n) in enumerate((("train", 128), ("validation", 20), ("test", 20)), 1)}}
    write_new(manifest_path, manifest)
    path = tmp_path / "ablation_20260919T082721_fixture"
    path.mkdir()
    report = {"subset_manifest": str(manifest_path), "config": {"training_rows": 96}}
    write_new(path / "results.json", report)
    record = {"model": "rbf", "features": 2, "seed": 43, "training_seconds": 1.2, "validation": {}, "test": {}}
    config, result = normalize(path, report, record, manifest)
    assert config["sample_size"]["train"] == {"rows": 96, "fraud": 48}
    assert config["random_seed"] == 43
    assert result["inference_duration"]["seconds_by_partition"] is None
    assert result["inference_duration"]["missing_reason"]
    assert config["hyperparameters"]["gamma"] == "scale"


def test_quantum_hyperparameters_are_explicit():
    assert model_settings("qsvc", {"circuit": {"qubits": 2}}, {}, {}) == {"C": 1, "kernel": "fidelity_statevector"}
