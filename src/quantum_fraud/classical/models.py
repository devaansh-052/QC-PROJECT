"""Reproducible class-weighted baseline factories."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def build_models(settings: dict, seed: int, class_weights: dict):
    """Use a linear SVM to make full-data optimization tractable.

    Random forest draws bootstrap samples per tree as its configured algorithm;
    all models receive the same complete training partition.
    """
    expected = {"logistic_regression", "random_forest", "linear_svm", "xgboost"}
    if set(settings) != expected:
        raise ValueError(f"Classical config needs exactly {sorted(expected)}")
    weights = {int(key): value for key, value in class_weights.items()}
    return {
        "logistic_regression": LogisticRegression(**settings["logistic_regression"], class_weight=weights, random_state=seed),
        "random_forest": RandomForestClassifier(**settings["random_forest"], class_weight=weights, n_jobs=2, random_state=seed),
        "linear_svm": LinearSVC(**settings["linear_svm"], class_weight=weights, dual=False, random_state=seed),
        "xgboost": XGBClassifier(**settings["xgboost"], scale_pos_weight=weights[1] / weights[0], objective="binary:logistic", tree_method="hist", n_jobs=2, random_state=seed),
    }


def prediction_scores(model, X: np.ndarray) -> np.ndarray:
    """Return positive-class scores; linear SVM margins are not probabilities."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Invalid probability outputs")
        return scores
    return model.decision_function(X)
