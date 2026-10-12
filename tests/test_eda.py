"""Verify EDA arithmetic on handwritten cases, not research observations."""

import json

import numpy as np
import pandas as pd
import pytest

from quantum_fraud.data.eda import rates_by, run_eda, summarize_eda
from quantum_fraud.data.paysim import REQUIRED_COLUMNS


@pytest.fixture
def frame():
    return pd.DataFrame([
        [1, "PAYMENT", 0., "C1", 10., 10., "M1", 0., 0., 0, 0],
        [1, "TRANSFER", 100., "C2", 100., 0., "C3", 0., 100., 1, 0],
        [2, "TRANSFER", 1000., "C4", 2000., 1000., "C5", 0., 1000., 0, 0],
        [3, "CASH_OUT", 10000., "C6", 10000., 0., "C7", 0., 10000., 1, 0],
    ], columns=REQUIRED_COLUMNS)


def test_group_denominators_and_amount_edges(frame):
    before = frame.copy(deep=True)
    tables = summarize_eda(frame)
    transfer = tables["transaction_types"].loc["TRANSFER"]
    assert transfer["rows"] == 2 and transfer["fraud"] == 1
    assert transfer["fraud_percentage"] == 50
    assert tables["time_steps"].loc[1, "fraud_percentage"] == 50
    assert tables["amount_bands"].loc["100–<1k", "fraud"] == 1
    assert tables["amount_bands"].loc["1k–<10k", "fraud"] == 0
    assert tables["amount_bands"].rows.sum() == len(frame)
    hist = tables["amount_histogram"].groupby("isFraud").sum(numeric_only=True)
    assert hist["count"].to_list() == [2, 2]
    np.testing.assert_allclose(hist.within_class_fraction, [1, 1])
    pd.testing.assert_frame_equal(frame, before)


def test_missing_groups_are_not_dropped(frame):
    frame.loc[0, "type"] = None
    assert rates_by(frame, frame.type).rows.sum() == 4
    frame.loc[0, "amount"] = np.nan
    tables = summarize_eda(frame)
    assert tables["amount_bands"].rows.sum() == 4
    assert tables["amount_histogram"]["count"].sum() == 3


def test_diagnostic_balances_and_correlations(frame):
    tables = summarize_eda(frame)
    balances = tables["balance_diagnostics"]
    row = balances[(balances.field == "newbalanceOrig") & balances.isFraud.eq(1)].iloc[0]
    assert row.zero_count == 2 and row.zero_percentage == 100
    assert tables["primary_numeric_correlations"].loc["isFraud", "isFraud"] == pytest.approx(1)


def test_single_class_constant_amounts(frame):
    frame["isFraud"] = 0
    frame["amount"] = 0.
    tables = summarize_eda(frame)
    assert tables["primary_numeric_correlations"].isna().loc["isFraud"].all()
    assert tables["amount_histogram"]["count"].sum() == 4


def test_run_artifacts_preserve_source_and_do_not_overwrite(frame, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "fixture.csv"
    frame.to_csv(source, index=False)
    original = source.read_bytes()
    first = run_eda(source, tmp_path / "results")
    report = json.loads(first.read_text())
    assert report["status"] == "complete"
    assert len(report["figures"]) == 12
    for name in report["figures"]:
        assert (tmp_path / "results/figures" / report["run_id"] / name).stat().st_size > 1000
    for name in report["tables"]:
        assert (first.parent / name).is_file()
    second = run_eda(source, tmp_path / "results")
    assert first != second and first.exists()
    assert source.read_bytes() == original


def test_missing_amounts_single_class_run(frame, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "edge_cases.csv"
    frame["isFraud"] = 0
    frame["amount"] = np.nan
    frame.to_csv(source, index=False, float_format="%.17g")
    report = json.loads(run_eda(source, tmp_path / "results").read_text())
    assert report["validation"]["missing_values"]["amount"] == 4
    assert report["status"] == "complete"


def test_outputs_cannot_enter_data_directory(tmp_path):
    with pytest.raises(ValueError, match="outside the dataset"):
        run_eda(tmp_path / "data/raw/file.csv", tmp_path / "data/raw/reports")
