import json
import numpy as np
import pytest
import requests
from streamlit.testing.v1 import AppTest
from quantum_fraud.dashboard.research import ROOT, curves, read_run, score_data


def test_unreviewed_run_rejected():
    with pytest.raises(ValueError, match="reviewed"):
        read_run("../../unreviewed")


def test_scores_require_aligned_labels(tmp_path):
    np.save(tmp_path / "test_y.npy", np.array([0, 1]))
    np.savez(tmp_path / "example_scores.npz", test=np.array([.3]))
    with pytest.raises(ValueError, match="inconsistent"):
        score_data(tmp_path, {"subset_manifest": str(tmp_path / "manifest.json")}, {"model": "example"})
    with pytest.raises(ValueError, match="evaluation"):
        score_data(tmp_path, {}, {}, "train")


def test_curve_endpoints():
    roc, pr = curves(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
    assert roc.iloc[-1].tolist() == [1, 1]
    assert pr.iloc[0]["precision"] == .5


def test_dashboard_offline(monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr("quantum_fraud.dashboard.app.api_request", offline)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert "API is offline" in app.info[0].value


def test_dashboard_prediction_and_duplicate(monkeypatch):
    history = []
    def service(method, route, **kwargs):
        if route == "/health":
            return {"model": "explicit_test_fixture", "threshold": .5}
        if method == "POST":
            if history:
                raise requests.HTTPError("409 duplicate ID")
            history.append({**kwargs["json"], "fraud_probability": .8, "predicted_class": 1,
                            "risk_level": "HIGH", "decision": "REVIEW", "model": "explicit_test_fixture", "processing_ms": 1.2})
            return history[-1]
        return {"transactions": history}
    monkeypatch.setattr("quantum_fraud.dashboard.app.api_request", service)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[1].value == "1"
    assert "Fraud alert" in app.warning[0].value
    assert len(app.dataframe[0].value) == 1
    app.button[0].click().run()
    assert "409" in app.error[0].value
