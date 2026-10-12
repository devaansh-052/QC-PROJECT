"""Classical model and metric contract checks on explicit software fixtures."""

import joblib
import json
import numpy as np
import pytest
import yaml

from quantum_fraud.classical.models import build_models, prediction_scores
from quantum_fraud.classical.train import train_classical
from quantum_fraud.evaluation.metrics import binary_metrics, select_f1_threshold, select_fpr_threshold


def test_known_metrics_and_validation_thresholds():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([.1, .4, .35, .8])
    report = binary_metrics(labels, scores, .5)
    assert report["confusion_matrix"] == [[2, 0], [1, 1]]
    assert report["precision"] == 1 and report["recall"] == .5
    assert report["f1"] == pytest.approx(2/3)
    assert report["pr_auc_average_precision"] == pytest.approx(5/6)
    assert report["roc_auc"] == .75
    assert select_f1_threshold(labels, scores) == .35
    assert select_fpr_threshold(labels, scores, 0) == .8


def test_tied_scores_do_not_invent_attainable_fpr():
    labels = np.array([0, 0, 1, 1])
    scores = np.ones(4)
    threshold = select_fpr_threshold(labels, scores, .01)
    assert threshold > 1
    assert binary_metrics(labels, scores, threshold)["false_positive_rate"] == 0


def test_invalid_score_shapes_and_values():
    for scores in ([.1], [.1, np.nan], [[.1], [.2]]):
        with pytest.raises(ValueError):
            binary_metrics([0, 1], scores, .5)


def test_all_four_models_train_predict_and_reload(tmp_path):
    settings = {
        "logistic_regression": {"max_iter": 1000},
        "random_forest": {"n_estimators": 5, "max_depth": 3},
        "linear_svm": {"max_iter": 10000},
        "xgboost": {"n_estimators": 5, "max_depth": 2},
    }
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]], dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    for name, model in build_models(settings, 42, {0: 1., 1: 1.}).items():
        model.fit(X, y)
        scores = prediction_scores(model, X)
        assert scores.shape == (8,) and np.isfinite(scores).all()
        path = tmp_path / f"{name}.joblib"
        joblib.dump(model, path)
        np.testing.assert_allclose(scores, prediction_scores(joblib.load(path), X))
        with pytest.raises(ValueError):
            prediction_scores(model, np.zeros((2, 3)))


def test_training_runner_never_uses_test_labels_to_select_threshold(tmp_path):
    """Flip only test labels: all selected thresholds must remain identical."""
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    X = np.column_stack([np.arange(20, dtype=float), np.tile([0., 1.], 10)])
    y = np.array([0]*10 + [1]*10, dtype=np.int8)
    for split in ("train", "validation", "test"):
        np.save(prepared / f"{split}_X.npy", X)
        np.save(prepared / f"{split}_y.npy", y)
    (prepared / "manifest.json").write_text(json.dumps({"status": "complete", "seed": 42, "effective_training_class_weights": {"0": 1, "1": 1}}))
    config = tmp_path / "classical.yaml"
    config.write_text(yaml.safe_dump({
        "logistic_regression": {"max_iter": 1000}, "random_forest": {"n_estimators": 3},
        "linear_svm": {"max_iter": 10000}, "xgboost": {"n_estimators": 3, "max_depth": 2},
    }))
    first = json.loads(train_classical(prepared, config, tmp_path / "results").read_text())
    np.save(prepared / "test_y.npy", 1-y)
    second = json.loads(train_classical(prepared, config, tmp_path / "results").read_text())
    for a, b in zip(first["models"], second["models"]):
        assert a["test"]["threshold"] == b["test"]["threshold"]
        for target in ("0.01", "0.005"):
            assert a["validation_selected_fpr_operating_points"][target]["test"]["threshold"] == b["validation_selected_fpr_operating_points"][target]["test"]["threshold"]
