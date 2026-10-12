"""Handwritten software fixtures only; these rows are never research datasets."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from quantum_fraud.data import DatasetError, load_paysim, validate_paysim
from quantum_fraud.data.paysim import REQUIRED_COLUMNS
from quantum_fraud.data.policy import BALANCE_COLUMNS, PRIMARY_FEATURE_COLUMNS


@pytest.fixture
def frame():
    """Tiny explicit schema examples for testing, not empirical evidence."""
    return pd.DataFrame([
        [1, "PAYMENT", 10.0, "C1", 20.0, 10.0, "M1", 0.0, 0.0, 0, 0],
        [2, "TRANSFER", 30.0, "C2", 30.0, 0.0, "C3", 0.0, 30.0, 1, 0],
    ], columns=REQUIRED_COLUMNS)


def test_report_and_provenance(frame, tmp_path):
    path = tmp_path / "fixture.csv"
    frame.to_csv(path, index=False)
    loaded, report = load_paysim(path)
    pd.testing.assert_frame_equal(loaded, frame)
    assert report["shape"] == {"rows": 2, "columns": 11}
    assert report["target_distribution"] == {"0": 1, "1": 1}
    assert report["fraud_percentage"] == 50.0
    assert report["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["dtypes"]["amount"] == "float64"
    assert report["step_range"] == {"min": 1, "max": 2}
    assert report["target_column"] == "isFraud"
    json.dumps(report, allow_nan=False)


def test_primary_policy_excludes_leakage(frame):
    policy = validate_paysim(frame)["feature_policy"]
    assert policy["target"] == "isFraud"
    assert set(policy["candidate_features"]) == {"step", "type", "amount"}
    assert set(BALANCE_COLUMNS).issubset(policy["excluded_features"])
    assert {"nameOrig", "nameDest", "isFlaggedFraud"}.issubset(policy["excluded_features"])
    assert not set(PRIMARY_FEATURE_COLUMNS).intersection((*BALANCE_COLUMNS, "isFraud", "isFlaggedFraud"))
    assert policy["exclude_balance_derived_features"] is True


def test_load_reads_once_and_preserves_raw_bytes(frame, tmp_path, monkeypatch):
    path = tmp_path / "fixture.csv"
    frame.to_csv(path, index=False)
    original_bytes = path.read_bytes()
    original_stat = path.stat()
    real_read_csv = pd.read_csv
    calls = []

    def tracked_read_csv(*args, **kwargs):
        calls.append(args[0])
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", tracked_read_csv)
    load_paysim(path)
    assert len(calls) == 1
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_cli_rejects_report_output_in_raw(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(root / "scripts/prepare_data.py"),
        "--data", str(tmp_path / "unused.csv"),
        "--output-dir", str(root / "data/raw/reports"),
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "outside data/raw" in result.stderr
    assert not (root / "data/raw/reports").exists()


def test_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="--data PATH"):
        load_paysim(tmp_path / "missing.csv")


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_required_columns(frame, column):
    with pytest.raises(DatasetError, match="Missing required"):
        validate_paysim(frame.drop(columns=[column]))


@pytest.mark.parametrize("column,value", [("isFraud", -1), ("isFraud", 2), ("isFraud", np.nan), ("isFraud", "yes"), ("isFlaggedFraud", 2), ("amount", -1), ("amount", np.inf), ("amount", "bad"), ("step", 1.5), ("step", 0), ("type", "UNKNOWN"), ("nameOrig", " ")])
def test_invalid_values(frame, column, value):
    frame[column] = frame[column].astype(object) if isinstance(value, str) else frame[column]
    if isinstance(value, float):
        frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(DatasetError) as error:
        validate_paysim(frame)
    assert error.value.report["errors"]


def test_quality_warnings_do_not_clean(frame):
    frame.loc[0, "amount"] = np.nan
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    original = frame.copy(deep=True)
    report = validate_paysim(frame)
    assert report["missing_values"]["amount"] == 2
    assert report["duplicate_rows"] == 1
    pd.testing.assert_frame_equal(frame, original)


def test_single_class_warning(frame):
    assert "Only one target class" in validate_paysim(frame.iloc[[0]])["warnings"][0]


@pytest.mark.parametrize("content,match", [("", "empty"), ("step,step\n1,1", "Duplicate"), ("wrong\n1", "Missing required")])
def test_bad_files(tmp_path, content, match):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(DatasetError, match=match):
        load_paysim(path)


def test_empty_and_duplicate_frame(frame):
    with pytest.raises(DatasetError, match="no transaction"):
        validate_paysim(frame.iloc[:0])
    with pytest.raises(DatasetError, match="Duplicate column"):
        validate_paysim(pd.concat([frame, frame[["amount"]]], axis=1))


def test_extra_fields_in_csv_fail(frame, tmp_path):
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    lines = path.read_text().splitlines()
    lines[1] += ",unexpected"
    path.write_text("\n".join(lines))
    with pytest.raises(DatasetError, match="Cannot read"):
        load_paysim(path)


def test_cli_success_unique_outputs_and_failure(frame, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
    path = tmp_path / "fixture.csv"
    frame.to_csv(path, index=False)
    outputs = tmp_path / "reports"
    command = [sys.executable, str(script), "--data", str(path), "--output-dir", str(outputs)]
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        report = json.loads(Path(result.stdout.strip()).read_text())
        assert report["shape"]["rows"] == 2
        assert report["scope"] == "full_file_validation_only_no_cleaning"
    assert len(list(outputs.glob("*.json"))) == 2
    result = subprocess.run([sys.executable, str(script), "--data", str(tmp_path / "missing.csv")], capture_output=True, text=True)
    assert result.returncode == 2
    assert "CSV not found" in result.stderr
    assert "Traceback" not in result.stderr
