"""Read completed, explicitly selected research artifacts without model loading."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[3]


def catalog():
    return json.loads((ROOT / "configs/research_runs.json").read_text())


def read_run(run_id):
    allowed = {item for group in catalog().values() for item in group}
    if run_id not in allowed:
        raise ValueError("Run is not in the reviewed research catalog")
    path = ROOT / "results/metrics" / run_id
    report = json.loads((path / "results.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Incomplete research run")
    return path, report


def artifact_stem(record):
    return record.get("artifact_stem", record["model"] if "features" not in record else f'{record["model"]}_{record["features"]}q')


def score_data(path, report, record, split="test"):
    if split not in ("validation", "test"):
        raise ValueError("Only evaluation partitions are available")
    classical = "preprocessing_manifest" in report
    manifest = Path(report["preprocessing_manifest"] if classical else report["subset_manifest"])
    labels = np.load(manifest.parent / f"{split}_y.npy", mmap_mode="r", allow_pickle=False)
    with np.load(path / f"{artifact_stem(record)}_scores.npz", allow_pickle=False) as archive:
        scores = archive[f"{split}_scores" if classical else split]
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Saved labels and scores are inconsistent")
    return labels, scores


def summary_table(report):
    rows = []
    for record in report["models"]:
        test = record["test"]
        rows.append({"model": artifact_stem(record), "features": record.get("features", 7),
                     "AP": test["pr_auc_average_precision"], "ROC AUC": test["roc_auc"],
                     "F1": test["f1"], "recall": test["recall"], "precision": test["precision"],
                     "training seconds": record.get("training_seconds"),
                     "test batch seconds": record.get("prediction_batch_seconds", {}).get("test")})
    return pd.DataFrame(rows)


def curves(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return (pd.DataFrame({"false positive rate": fpr, "true positive rate": tpr}),
            pd.DataFrame({"recall": recall, "precision": precision}))
