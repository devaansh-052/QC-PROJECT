"""Validation-calibrated, validation-selected probability ensembles."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from quantum_fraud.quantum.data import load_quantum_arrays
from quantum_fraud.quantum.experiment import evaluate_scores, runtime_versions


def blend(classical, quantum, alpha):
    a, b = np.asarray(classical), np.asarray(quantum)
    if a.shape != b.shape or a.ndim != 1 or not 0 <= alpha <= 1:
        raise ValueError("Aligned probability vectors and alpha in [0,1] required")
    if not np.isfinite([a, b]).all() or (a < 0).any() or (a > 1).any() or (b < 0).any() or (b > 1).any():
        raise ValueError("Invalid probabilities")
    return alpha * a + (1 - alpha) * b


def validation_partition(y, seed=42):
    calibration, selection = train_test_split(np.arange(len(y)), test_size=.5, stratify=y, random_state=seed)
    return np.sort(calibration), np.sort(selection)


def run_ensemble(qsvc_run: Path, vqc_run: Path, root: Path):
    data = qsvc_run / "data"
    vqc_report = json.loads((vqc_run / "results.json").read_text())
    if vqc_report["status"] != "complete" or Path(vqc_report["subset_manifest"]).resolve() != (data / "manifest.json").resolve():
        raise ValueError("VQC and QSVC must use the identical completed subset")
    arrays = load_quantum_arrays(data, 2)
    output = root / (datetime.now(timezone.utc).strftime("ensemble_%Y%m%dT%H%M%S_") + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / "data").mkdir()
    calibration, selection = validation_partition(arrays["validation"][1])
    np.savez(output / "validation_partition.npz", calibration=calibration, selection=selection)
    evaluation = dict(arrays)
    evaluation["validation"] = tuple(x[selection] for x in arrays["validation"])
    for split in ("validation", "test"):
        np.save(output / "data" / f"{split}_y.npy", evaluation[split][1])
        positions = np.load(data / f"{split}_positions.npy")
        np.save(output / "data" / f"{split}_positions.npy", positions[selection] if split == "validation" else positions)
    parent = json.loads((data / "manifest.json").read_text())
    manifest = {**parent, "parent_manifest": str((data / "manifest.json").resolve()),
                "validation_role": "selection half only; disjoint calibration half retained separately",
                "splits": {s: {"rows": len(evaluation[s][1]), "fraud": int(evaluation[s][1].sum())} for s in evaluation}}
    (output / "data/manifest.json").write_text(json.dumps(manifest, indent=2))
    config = {"seed": 42, "features": 2, "alphas": [.25, .5, .75], "alpha_selection": "maximum selection-validation average precision; first grid value wins ties",
              "qsvc": "qsvc_zz_r1_2q", "vqc": "vqc_linear_r1_2q", "calibration": "sigmoid on disjoint half of parent validation",
              "xgboost": {"n_estimators": 100, "max_depth": 3, "learning_rate": .05, "n_jobs": 2, "random_state": 42}}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    model = XGBClassifier(**config["xgboost"], objective="binary:logistic", tree_method="hist")
    start = perf_counter()
    model.fit(*arrays["train"])
    xgb_training = perf_counter() - start
    joblib.dump(model, output / "xgboost.joblib")
    raw = {"xgboost": {s: model.predict_proba(arrays[s][0])[:, 1] for s in ("validation", "test")}}
    for name, run, stem in (("qsvc", qsvc_run, config["qsvc"]), ("vqc", vqc_run, config["vqc"])):
        with np.load(run / f"{stem}_scores.npz") as saved:
            raw[name] = {s: saved[s].copy() for s in ("validation", "test")}
    probabilities, timings = {}, {}
    for name, scores in raw.items():
        start = perf_counter()
        calibrator = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        calibrator.fit(scores["validation"][calibration, None], arrays["validation"][1][calibration])
        timings[name] = perf_counter() - start
        joblib.dump(calibrator, output / f"{name}_calibrator.joblib")
        probabilities[name] = {s: calibrator.predict_proba(scores[s][:, None])[:, 1] for s in scores}
        probabilities[name]["validation"] = probabilities[name]["validation"][selection]
    records = []
    def record(name, scores, seconds, alpha=None):
        item = {"model": name, "artifact_stem": name, "features": 2, "alpha": alpha,
                "training_seconds": seconds, "score_kind": "validation_sigmoid_calibrated_probability",
                **evaluate_scores(evaluation, scores["validation"], scores["test"])}
        np.savez(output / f"{name}_scores.npz", **scores)
        records.append(item)
        return item
    for name, scores in probabilities.items():
        record(name, scores, timings[name] + (xgb_training if name == "xgboost" else 0))
    selected = {}
    for quantum in ("qsvc", "vqc"):
        candidates = []
        for alpha in config["alphas"]:
            start = perf_counter()
            scores = {s: blend(probabilities["xgboost"][s], probabilities[quantum][s], alpha) for s in ("validation", "test")}
            item = record(f"xgboost_{quantum}_a{alpha}", scores, 0, alpha)
            item["blend_and_evaluation_seconds"] = perf_counter() - start
            candidates.append(item)
        chosen = max(candidates, key=lambda r: r["validation"]["pr_auc_average_precision"])
        selected[quantum] = {"model": chosen["model"], "alpha": chosen["alpha"]}
    report = {"status": "complete", "run_id": output.name, "config": config, "versions": runtime_versions(),
              "subset_manifest": str((output / "data/manifest.json").resolve()), "models": records, "selected": selected,
              "parents": {"qsvc": str(qsvc_run.resolve()), "vqc": str(vqc_run.resolve())},
              "limitations": ["Predeclared two-qubit base models, not selected by test scores",
                              "Only 13 parent validation positives split between calibration and alpha/threshold selection",
                              "Quantum models reused; zero reported training time excludes historical fit cost",
                              "Calibration and selection share the parent temporal window; tiny samples limit calibration reliability",
                              "No superiority or production-calibration claim"]}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return output
