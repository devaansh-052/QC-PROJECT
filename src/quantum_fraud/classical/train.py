"""Train and evaluate full-data baselines with shared temporal partitions."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import yaml

from quantum_fraud.classical.models import build_models, prediction_scores
from quantum_fraud.evaluation.metrics import binary_metrics, select_f1_threshold, select_fpr_threshold

LOGGER = logging.getLogger(__name__)


def train_classical(prepared: Path, config_path: Path, output_root: Path) -> Path:
    """Run each baseline sequentially; failures block the final completion marker."""
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Preprocessing run is incomplete")
    settings = yaml.safe_load(config_path.read_text())
    models = build_models(settings, manifest["seed"], manifest["effective_training_class_weights"])
    partitions = {name: (np.load(prepared / f"{name}_X.npy", mmap_mode="r"), np.load(prepared / f"{name}_y.npy", mmap_mode="r")) for name in ("train", "validation", "test")}
    run_id = datetime.now(timezone.utc).strftime("classical_%Y%m%dT%H%M%S_") + uuid4().hex[:12]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    records = []
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name, model in models.items():
            LOGGER.info("Training %s on %d rows", name, len(partitions["train"][1]))
            start = perf_counter()
            model.fit(*partitions["train"])
            training_seconds = perf_counter() - start
            scores, inference = {}, {}
            for split in ("validation", "test"):
                start = perf_counter()
                scores[split] = prediction_scores(model, partitions[split][0])
                inference[split] = perf_counter() - start
            threshold = select_f1_threshold(partitions["validation"][1], scores["validation"])
            fixed_fpr = {}
            for requested in (.01, .005):
                fixed_threshold = select_fpr_threshold(partitions["validation"][1], scores["validation"], requested)
                fixed_fpr[str(requested)] = {
                    "selected_on": "validation", "requested_validation_fpr": requested,
                    "validation": binary_metrics(partitions["validation"][1], scores["validation"], fixed_threshold),
                    "test": binary_metrics(partitions["test"][1], scores["test"], fixed_threshold),
                }
            record = {
                "model": name, "threshold_selected_on": "validation_f1",
                "score_kind": "uncalibrated_model_probability" if hasattr(model, "predict_proba") else "svm_margin_not_probability",
                "training_seconds": training_seconds,
                "prediction_batch_seconds": inference,
                "prediction_amortized_ms_per_row": {split: 1000 * inference[split] / len(scores[split]) for split in inference},
                "validation": binary_metrics(partitions["validation"][1], scores["validation"], threshold),
                "test": binary_metrics(partitions["test"][1], scores["test"], threshold),
                "validation_selected_fpr_operating_points": fixed_fpr,
                "hyperparameters": settings[name],
            }
            joblib.dump(model, output / f"{name}.joblib")
            restored = joblib.load(output / f"{name}.joblib")
            np.testing.assert_allclose(prediction_scores(restored, partitions["test"][0][:20]), scores["test"][:20], rtol=1e-6, atol=1e-8)
            np.savez(output / f"{name}_scores.npz", validation_scores=scores["validation"], test_scores=scores["test"])
            (output / f"{name}_metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            records.append(record)
            LOGGER.info("Finished %s: %.2fs training, test AP=%.6f", name, training_seconds, record["test"]["pr_auc_average_precision"])
    result = {
        "status": "complete", "run_id": run_id, "preprocessing_manifest": str((prepared / "manifest.json").resolve()),
        "preprocessing": manifest, "models": records,
        "versions": {name: version(name) for name in ("scikit-learn", "xgboost", "numpy", "joblib")},
        "limitations": ["Single seed; no statistical superiority claim", "Linear SVM is not an RBF-kernel control for future QSVC", "Probability outputs are class-weighted and uncalibrated; not deployment fraud probabilities", "Batch-amortized timing excludes preprocessing, API overhead, serialization and model loading", "Test FPR can exceed validation-selected FPR under temporal shift", "No hyperparameter search; these are fixed initial baselines"],
    }
    path = output / "results.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return path
