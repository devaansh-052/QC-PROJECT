"""Prepare reproducible on-disk partitions with train-fitted preprocessing."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import yaml

from quantum_fraud.data.paysim import load_paysim
from quantum_fraud.data.policy import PRIMARY_FEATURE_COLUMNS, primary_feature_policy
from quantum_fraud.preprocessing.pipeline import balanced_class_weights, build_preprocessor, temporal_split, undersample_training


def load_preprocessing_config(path: Path) -> dict:
    """Read strict phase-specific settings, rejecting misspelled keys."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"train_fraction", "validation_fraction", "sampling"}:
        raise ValueError("Preprocessing config needs train_fraction, validation_fraction, sampling")
    a, b = raw["train_fraction"], raw["validation_fraction"]
    if type(a) not in (float, int) or type(b) not in (float, int) or not (0 < a < 1 and 0 < b < 1 and a + b < 1):
        raise ValueError("Invalid preprocessing split fractions")
    if raw["sampling"] not in ("none", "random_undersample"):
        raise ValueError("sampling must be none or random_undersample")
    return raw


def prepare_partitions(source: Path, output_root: Path, settings: dict, seed: int = 42) -> Path:
    """Load raw once and save each transformed split sequentially.

    A completed manifest is written last. Outputs must be outside raw data.
    Original positional indices make row/sample comparisons reproducible.
    """
    source, output_root = source.resolve(), output_root.resolve()
    project_raw = Path(__file__).resolve().parents[3] / "data/raw"
    if output_root.is_relative_to(project_raw) or output_root == source.parent or (source.parent.name.lower() == "raw" and output_root.is_relative_to(source.parent)):
        raise ValueError("Processed outputs must be outside the raw/source directory")
    if settings.get("sampling") not in ("none", "random_undersample"):
        raise ValueError("Unsupported sampling strategy")
    frame, validation = load_paysim(source)
    labels = frame["isFraud"].to_numpy(dtype=np.int8, copy=True)
    steps = frame["step"].to_numpy(copy=True)
    positions = temporal_split(steps, labels, settings["train_fraction"], settings["validation_fraction"])
    features = frame.loc[:, list(PRIMARY_FEATURE_COLUMNS)].copy()
    del frame
    train_positions = positions["train"]
    original_weights = balanced_class_weights(labels[train_positions])
    pipeline = build_preprocessor()
    # Fit before optional sampling, using only original training observations.
    pipeline.fit(features.iloc[train_positions])
    if settings["sampling"] == "random_undersample":
        train_positions = undersample_training(train_positions, labels[train_positions], seed)
    run_id = datetime.now(timezone.utc).strftime("preprocessing_%Y%m%dT%H%M%S_") + uuid4().hex[:12]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump({**settings, "seed": seed}), encoding="utf-8")
    joblib.dump(pipeline, output / "preprocessor.joblib")
    split_reports = {}
    for name, original_positions in positions.items():
        selected = train_positions if name == "train" else original_positions
        transformed = pipeline.transform(features.iloc[selected])
        if not np.isfinite(transformed).all():
            raise ValueError(f"Nonfinite transformed values in {name}")
        np.save(output / f"{name}_X.npy", transformed, allow_pickle=False)
        np.save(output / f"{name}_y.npy", labels[selected], allow_pickle=False)
        np.save(output / f"{name}_positions.npy", selected, allow_pickle=False)
        split_reports[name] = {
            "rows_before_sampling": len(original_positions), "rows": len(selected),
            "features": transformed.shape[1],
            "step_min": int(steps[original_positions].min()), "step_max": int(steps[original_positions].max()),
            "class_counts_before_sampling": {str(label): int(np.sum(labels[original_positions] == label)) for label in (0, 1)},
            "class_counts": {str(label): int(np.sum(labels[selected] == label)) for label in (0, 1)},
        }
        del transformed
    reloaded = joblib.load(output / "preprocessor.joblib")
    probe = features.iloc[positions["validation"][:10]]
    np.testing.assert_allclose(reloaded.transform(probe), pipeline.transform(probe))
    report = {
        "status": "complete", "run_id": run_id, "dataset": validation["source"],
        "settings": settings, "seed": seed, "feature_policy": primary_feature_policy(),
        "feature_names": pipeline.get_feature_names_out().tolist(),
        "splits": split_reports,
        "original_training_class_weights": original_weights,
        "effective_training_class_weights": balanced_class_weights(labels[train_positions]),
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
        "notes": ["Fractions refer to distinct ordered time steps, not row counts", "No raw balance fields or balance derivatives", "Numeric medians, scales, category vocabulary and variance selector fit only on original train", "Select either effective class weights or an unweighted loss; do not multiply class weights by identical sample weights", "Full-file EDA was performed; holdout was not untouched during exploration", "No SMOTE; optional undersampling affects train only"],
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
