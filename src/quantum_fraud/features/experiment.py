"""Feature reduction grid with matched full-training linear-SVM controls."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import warnings

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import yaml

from quantum_fraud.classical.models import prediction_scores
from quantum_fraud.evaluation.metrics import binary_metrics, select_f1_threshold
from quantum_fraud.features.reduction import ColumnSelection, PCAProjection, estimate_mutual_information, ranked_indices

LOGGER = logging.getLogger(__name__)


def run_feature_experiments(prepared: Path, classical_run: Path, config: Path, output_root: Path) -> Path:
    """Fit reducers exclusively on training and compare identical saved rows."""
    manifest = json.loads((prepared / "manifest.json").read_text())
    baseline = json.loads((classical_run / "results.json").read_text())
    if manifest["status"] != "complete" or baseline["status"] != "complete" or Path(baseline["preprocessing_manifest"]).resolve() != (prepared / "manifest.json").resolve():
        raise ValueError("Feature experiments require completed, exactly matched preprocessing and classical runs")
    settings = yaml.safe_load(config.read_text())
    if not isinstance(settings, dict) or set(settings) != {"dimensions", "mutual_information_sample_limit"}:
        raise ValueError("Feature config requires dimensions and mutual_information_sample_limit")
    dimensions = settings["dimensions"]
    if not isinstance(dimensions, list) or not dimensions or any(type(k) is not int or k < 1 for k in dimensions) or len(set(dimensions)) != len(dimensions):
        raise ValueError("dimensions must be unique positive integers")
    arrays = {name: (np.load(prepared / f"{name}_X.npy", mmap_mode="r"), np.load(prepared / f"{name}_y.npy", mmap_mode="r")) for name in ("train", "validation", "test")}
    names = manifest["feature_names"]
    count = len(names)
    valid = [k for k in dimensions if k <= min(arrays["train"][0].shape)]
    if not valid:
        raise ValueError("No requested dimension is supported")
    run_id = datetime.now(timezone.utc).strftime("features_%Y%m%dT%H%M%S_") + uuid4().hex[:12]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        LOGGER.info("Estimating mutual information using training data only")
        mi_scores, mi_positions = estimate_mutual_information(*arrays["train"], [name.startswith("categorical__") for name in names], settings["mutual_information_sample_limit"], manifest["seed"])
        np.save(output / "mi_training_array_positions.npy", mi_positions, allow_pickle=False)
        forest = joblib.load(classical_run / "random_forest.joblib")
        if forest.n_features_in_ != count:
            raise ValueError("Forest importance dimensions do not match")
        forest_scores = forest.feature_importances_
        pca = PCA(n_components=max(valid), svd_solver="full")
        LOGGER.info("Fitting PCA on all %d training rows", len(arrays["train"][1]))
        pca.fit(arrays["train"][0])
        joblib.dump(pca, output / "pca_full.joblib")
        weights = {int(k): v for k, v in manifest["effective_training_class_weights"].items()}
        results = []
        for method in ("mutual_information", "random_forest_importance", "pca"):
            for dimension in valid:
                LOGGER.info("Training matched linear SVM: %s, %d features", method, dimension)
                if method == "pca":
                    reducer = PCAProjection(pca, dimension)
                    selected_names = [f"PC{i+1}" for i in range(dimension)]
                else:
                    scores = mi_scores if method == "mutual_information" else forest_scores
                    indices = ranked_indices(scores, dimension)
                    reducer = ColumnSelection(indices, count).fit(arrays["train"][0])
                    selected_names = [names[i] for i in indices]
                model = LinearSVC(C=1.0, max_iter=10000, dual=False, class_weight=weights, random_state=manifest["seed"])
                start = perf_counter()
                training_X = reducer.transform(arrays["train"][0])
                transform_seconds = perf_counter() - start
                start = perf_counter()
                model.fit(training_X, arrays["train"][1])
                training_seconds = perf_counter() - start
                del training_X
                validation_scores = prediction_scores(model, reducer.transform(arrays["validation"][0]))
                test_scores = prediction_scores(model, reducer.transform(arrays["test"][0]))
                threshold = select_f1_threshold(arrays["validation"][1], validation_scores)
                record = {"method": method, "dimension": dimension, "features": selected_names,
                    "training_transform_seconds": transform_seconds, "control_training_seconds": training_seconds,
                    "validation": binary_metrics(arrays["validation"][1], validation_scores, threshold),
                    "test": binary_metrics(arrays["test"][1], test_scores, threshold),
                    "explained_variance_ratio": reducer.explained_variance_ratio_.tolist() if method == "pca" else None}
                stem = f"{method}_{dimension}"
                joblib.dump({"reducer": reducer, "control": model}, output / f"{stem}.joblib")
                restored = joblib.load(output / f"{stem}.joblib")
                np.testing.assert_allclose(prediction_scores(restored["control"], restored["reducer"].transform(arrays["test"][0][:20])), test_scores[:20])
                np.savez(output / f"{stem}_scores.npz", validation=validation_scores, test=test_scores)
                results.append(record)
    report = {
        "status": "complete", "run_id": run_id, "prepared_manifest": str((prepared / "manifest.json").resolve()),
        "classical_results": str((classical_run / "results.json").resolve()), "settings": settings,
        "dataset": manifest["dataset"], "seed": manifest["seed"], "input_features": names,
        "unsupported_dimensions": [k for k in dimensions if k not in valid],
        "mutual_information_scores": dict(zip(names, mi_scores.tolist())),
        "forest_importance_scores": dict(zip(names, forest_scores.tolist())),
        "mi_estimation_rows": len(mi_positions), "mi_estimation_fraud_rows": int(arrays["train"][1][mi_positions].sum()),
        "pca_components": pca.components_.tolist(), "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "controls": results,
        "limitations": ["Only seven eligible encoded columns; eight-feature request cannot be fulfilled without changing approved representation", "MI uses a deterministic class-proportional training subset to bound nearest-neighbor cost", "Forest importances can be biased; PCA maximizes variance, not fraud signal", "All controls train on the same complete training partition; MI estimation subset is not the classifier training set", "No quantum models or matched RBF kernel controls yet", "Single seed, prespecified grid, no selection based on test results"],
    }
    path = output / "results.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path
