"""Phase 6 experiments and reusable frozen-threshold score recording."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
import yaml

from quantum_fraud.evaluation.metrics import binary_metrics, select_f1_threshold, select_fpr_threshold
from quantum_fraud.quantum.data import load_quantum_arrays, prepare_quantum_data
from quantum_fraud.quantum.kernel import build_qsvc, circuit_metadata

LOGGER = logging.getLogger(__name__)


def runtime_versions() -> dict:
    return {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "qiskit", "qiskit-machine-learning", "qiskit-aer")}


def evaluate_scores(arrays: dict, validation_scores, test_scores) -> dict:
    """Select all thresholds on validation, then freeze them for test evaluation."""
    threshold = select_f1_threshold(arrays["validation"][1], validation_scores)
    points = {}
    for fpr in (.01, .005):
        selected = select_fpr_threshold(arrays["validation"][1], validation_scores, fpr)
        points[str(fpr)] = {"validation": binary_metrics(arrays["validation"][1], validation_scores, selected), "test": binary_metrics(arrays["test"][1], test_scores, selected)}
    return {"threshold_selected_on": "validation_f1", "validation": binary_metrics(arrays["validation"][1], validation_scores, threshold), "test": binary_metrics(arrays["test"][1], test_scores, threshold), "validation_selected_fpr_points": points}


def run_qsvc(prepared: Path, feature_run: Path, config_path: Path, output_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    expected = {"seed", "train_per_class", "validation_rows", "test_rows", "dimensions", "feature_maps", "repetitions", "angle_range", "angle_encoding"}
    if not isinstance(config, dict) or set(config) != expected or not config["dimensions"] or not config["feature_maps"] or not config["repetitions"]:
        raise ValueError("Invalid quantum configuration")
    run_id = datetime.now(timezone.utc).strftime("qsvc_%Y%m%dT%H%M%S_") + uuid4().hex[:12]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    prepare_quantum_data(prepared, feature_run, output / "data", config)
    records = []
    with threadpool_limits(limits=2):
        for dimension in config["dimensions"]:
            arrays = load_quantum_arrays(output / "data", dimension)
            variants = [(f"qsvc_{name}_r{reps}", name, reps) for name in config["feature_maps"] for reps in config["repetitions"]]
            variants.append(("rbf_svm", None, None))
            for name, map_name, reps in variants:
                LOGGER.info("Training %s with %dq on identical %d rows", name, dimension, len(arrays["train"][1]))
                model, circuit = build_qsvc(dimension, map_name, reps) if map_name else (SVC(C=1.0, kernel="rbf", gamma="scale"), None)
                start = perf_counter()
                model.fit(*arrays["train"])
                train_seconds = perf_counter() - start
                prediction_seconds, scores = {}, {}
                for split in ("validation", "test"):
                    start = perf_counter()
                    scores[split] = model.decision_function(arrays[split][0])
                    prediction_seconds[split] = perf_counter() - start
                record = {"model": name, "features": dimension, "simulation": "ideal_exact_statevector" if map_name else "classical",
                          "feature_map": map_name, "repetitions": reps, "circuit": circuit_metadata(circuit) if circuit else None,
                          "training_seconds": train_seconds, "prediction_batch_seconds": prediction_seconds, "score_kind": "svm_margin_not_probability",
                          "kernel_timings": list(model.quantum_kernel.timings) if map_name else None,
                          **evaluate_scores(arrays, scores["validation"], scores["test"])}
                stem = f"{name}_{dimension}q"
                if map_name:
                    model.to_dill(str(output / f"{stem}.model"))
                    restored = type(model).from_dill(str(output / f"{stem}.model"))
                else:
                    joblib.dump(model, output / f"{stem}.model")
                    restored = joblib.load(output / f"{stem}.model")
                np.testing.assert_allclose(restored.decision_function(arrays["test"][0][:5]), scores["test"][:5], atol=1e-10)
                np.savez(output / f"{stem}_scores.npz", **scores)
                records.append(record)
                (output / f"{stem}_metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    report = {"status": "complete", "run_id": run_id, "config": config, "versions": runtime_versions(),
              "subset_manifest": str((output / "data/manifest.json").resolve()), "models": records,
              "limitations": ["Simulator runs on classical hardware; no quantum speedup or physical hardware claim", "Tiny balanced training set; evaluation prevalence differs and positive counts are small", "Same samples and PCA/angles for quantum and RBF SVM controls", "Statevector construction includes simulator work; remaining kernel time includes binding, cache and overlap computation", "Single seed and prespecified grid; no model selection on test results"]}
    path = output / "results.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path
