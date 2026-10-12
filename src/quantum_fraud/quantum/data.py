"""Reproducible supervised subsets and training-only angle scaling."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from quantum_fraud.features.reduction import PCAProjection
from quantum_fraud.preprocessing.pipeline import validate_labels


class BoundedAngleScaler(TransformerMixin, BaseEstimator):
    """Train-fitted standardization with smooth bounded angles, without hard clipping."""

    def __init__(self, angle_range=(-1.0, 1.0)):
        self.angle_range = angle_range

    def fit(self, X, y=None):
        self.standard_ = StandardScaler().fit(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        lower, upper = self.angle_range
        return (lower + upper) / 2 + (upper - lower) / 2 * np.tanh(self.standard_.transform(X))


def sample_positions(labels, size: int, seed: int, balanced: bool = False) -> np.ndarray:
    """Select without replacement; evaluation subsets retain approximate prevalence."""
    labels = validate_labels(labels)
    if type(size) is not int or not 2 <= size <= len(labels):
        raise ValueError("Subset size must be between 2 and the partition size")
    rng = np.random.default_rng(seed)
    n1 = size // 2 if balanced else int(round(size * np.mean(labels)))
    n1 = min(max(n1, 1), size - 1)
    counts = {1: n1, 0: size - n1}
    selected = []
    for label, count in counts.items():
        available = np.flatnonzero(labels == label)
        if count > len(available):
            raise ValueError("Insufficient class representation for requested subset")
        selected.append(rng.choice(available, size=count, replace=False))
    return np.sort(np.concatenate(selected))


def prepare_quantum_data(prepared: Path, feature_run: Path, output: Path, config: dict) -> Path:
    """Reuse train-fitted PCA, fit angle scalers on the supervised subset only."""
    parent = json.loads((prepared / "manifest.json").read_text())
    features = json.loads((feature_run / "results.json").read_text())
    if parent["status"] != "complete" or features["status"] != "complete" or Path(features["prepared_manifest"]).resolve() != (prepared / "manifest.json").resolve():
        raise ValueError("Require matched completed preprocessing and reduction runs")
    pca = joblib.load(feature_run / "pca_full.joblib")
    if any(type(k) is not int or k < 1 or k > len(pca.components_) for k in config["dimensions"]):
        raise ValueError("Unsupported quantum feature dimension")
    if len(config["angle_range"]) != 2 or not np.isfinite(config["angle_range"]).all() or config["angle_range"][0] >= config["angle_range"][1]:
        raise ValueError("Invalid angle range")
    if config["angle_encoding"] != "standard_tanh_v1":
        raise ValueError("Unsupported angle encoding")
    output.mkdir(parents=True, exist_ok=False)
    sampled, report = {}, {}
    for offset, split in enumerate(("train", "validation", "test")):
        X = np.load(prepared / f"{split}_X.npy", mmap_mode="r")
        y = np.load(prepared / f"{split}_y.npy", mmap_mode="r")
        size = 2 * config["train_per_class"] if split == "train" else config[f"{split}_rows"]
        indices = sample_positions(y, size, config["seed"] + offset, balanced=split == "train")
        sampled[split] = np.asarray(X[indices])
        original = np.load(prepared / f"{split}_positions.npy", mmap_mode="r")[indices]
        np.save(output / f"{split}_positions.npy", original, allow_pickle=False)
        np.save(output / f"{split}_partition_positions.npy", indices, allow_pickle=False)
        np.save(output / f"{split}_full_X.npy", sampled[split], allow_pickle=False)
        np.save(output / f"{split}_y.npy", y[indices], allow_pickle=False)
        report[split] = {"rows": size, "fraud": int(y[indices].sum()), "parent_rows": len(y), "parent_fraud": int(y.sum())}
    outside_range, saturation = {}, {}
    for dimension in config["dimensions"]:
        projection = PCAProjection(pca, dimension)
        scaler = BoundedAngleScaler(angle_range=tuple(config["angle_range"]))
        scaler.fit(projection.transform(sampled["train"]))
        joblib.dump({"projection": projection, "angle_scaler": scaler}, output / f"transform_{dimension}.joblib")
        outside_range[str(dimension)] = {}
        saturation[str(dimension)] = {}
        for split, X in sampled.items():
            reduced = projection.transform(X)
            outside_range[str(dimension)][split] = int(np.any((reduced < scaler.data_min_) | (reduced > scaler.data_max_), axis=1).sum())
            angles = scaler.transform(reduced)
            lower, upper = config["angle_range"]
            saturation[str(dimension)][split] = int(np.any((angles <= lower + 1e-7) | (angles >= upper - 1e-7), axis=1).sum())
            np.save(output / f"{split}_{dimension}q_X.npy", angles, allow_pickle=False)
    result = {"status": "complete", "dataset": parent["dataset"], "preprocessing_manifest": str((prepared / "manifest.json").resolve()),
              "feature_results": str((feature_run / "results.json").resolve()), "config": config, "splits": report,
              "outside_training_minmax_rows": outside_range, "near_saturated_rows": saturation, "input_features": parent["feature_names"],
              "limitations": ["Training is intentionally balanced and tiny; probabilities need calibration at target prevalence", "Validation/test are class-proportional subsets, not balanced; few positives limit inference", "PCA was fitted on full original training data; angle scaling fits only this supervised subset", "No raw balances or balance-derived fields", "All compared classifiers use identical saved indices and features"]}
    path = output / "manifest.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return path


def load_quantum_arrays(data: Path, dimension: int | None = None) -> dict:
    """Load the small, fixed experiment arrays; None selects all seven inputs."""
    if json.loads((data / "manifest.json").read_text())["status"] != "complete":
        raise ValueError("Quantum subset is incomplete")
    suffix = "full" if dimension is None else f"{dimension}q"
    return {split: (np.load(data / f"{split}_{suffix}_X.npy"), np.load(data / f"{split}_y.npy")) for split in ("train", "validation", "test")}
