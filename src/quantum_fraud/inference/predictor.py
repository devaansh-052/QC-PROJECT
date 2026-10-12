"""Frozen preprocessing and validation-threshold inference."""
import json
from pathlib import Path
from threading import Lock
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def serving_config(path=None):
    config = yaml.safe_load(Path(path or PROJECT_ROOT / "configs/serving.yaml").read_text())
    if set(config) != {"preprocessing_run", "model_run", "model_name", "database"} or config["model_name"] != "xgboost":
        raise ValueError("Unsupported serving configuration")
    return config


class SavedPredictor:
    def __init__(self, config):
        prepared = PROJECT_ROOT / config["preprocessing_run"]
        models = PROJECT_ROOT / config["model_run"]
        manifest = json.loads((prepared / "manifest.json").read_text())
        model_report = json.loads((models / "results.json").read_text())
        if model_report["status"] != "complete" or Path(model_report["preprocessing_manifest"]).resolve() != (prepared / "manifest.json").resolve():
            raise ValueError("Model and preprocessing provenance do not match")
        if manifest["status"] != "complete":
            raise ValueError("Preprocessing run is incomplete")
        self.preprocessor = joblib.load(prepared / "preprocessor.joblib")
        self.model = joblib.load(models / "xgboost.joblib")
        metrics = json.loads((models / "xgboost_metrics.json").read_text())
        if metrics["threshold_selected_on"] != "validation_f1":
            raise ValueError("Serving requires a validation-selected threshold")
        self.threshold = metrics["validation"]["threshold"]
        self.name = "xgboost_full_training"
        self.run_id = models.name
        self.dataset_sha256 = manifest["dataset"]["sha256"]
        self._lock = Lock()

    def predict(self, transaction):
        start = perf_counter()
        with self._lock:
            transform_start = perf_counter()
            X = self.preprocessor.transform(pd.DataFrame([{k: transaction[k] for k in ("step", "type", "amount")}]))
            preprocessing_ms = (perf_counter() - transform_start) * 1000
            infer_start = perf_counter()
            probability = float(self.model.predict_proba(X)[0, 1])
            model_ms = (perf_counter() - infer_start) * 1000
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid model output")
        positive = probability >= self.threshold
        risk = "HIGH" if positive else "MEDIUM" if probability >= self.threshold / 2 else "LOW"
        return {"fraud_probability": probability, "probability_status": "uncalibrated_research_score",
                "predicted_class": int(positive), "risk_level": risk, "decision": "REVIEW" if positive else "ALLOW",
                "threshold": self.threshold, "model": self.name, "model_run": self.run_id,
                "preprocessing_ms": preprocessing_ms, "model_ms": model_ms,
                "processing_ms": (perf_counter() - start) * 1000, "simulation": True}
