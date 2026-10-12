"""Independently verify stored simulator scores and export a comparison CSV."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from quantum_fraud.evaluation.metrics import binary_metrics, select_f1_threshold, select_fpr_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "results.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Experiment has no completed result")
    data = Path(report["subset_manifest"]).parent
    labels = {split: np.load(data / f"{split}_y.npy") for split in ("validation", "test")}
    rows = []
    for record in report["models"]:
        stem = record.get("artifact_stem", f"{record['model']}_{record['features']}q")
        with np.load(args.run / f"{stem}_scores.npz") as scores:
            threshold = select_f1_threshold(labels["validation"], scores["validation"])
            for split in ("validation", "test"):
                recomputed = binary_metrics(labels[split], scores[split], threshold)
                for key, value in recomputed.items():
                    np.testing.assert_allclose(value, record[split][key], rtol=1e-10, atol=1e-12)
            for target_fpr in (.01, .005):
                operating_threshold = select_fpr_threshold(labels["validation"], scores["validation"], target_fpr)
                for split in ("validation", "test"):
                    recomputed = binary_metrics(labels[split], scores[split], operating_threshold)
                    for key, value in recomputed.items():
                        np.testing.assert_allclose(value, record["validation_selected_fpr_points"][str(target_fpr)][split][key], rtol=1e-10, atol=1e-12)
        rows.append({"model": record["model"], "features": record["features"],
                     **{key: value for key, value in record["test"].items() if key != "confusion_matrix"},
                     "training_seconds": record["training_seconds"]})
    with (args.run / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified all validation/test metrics for {len(rows)} models; comparison.csv exported.")


if __name__ == "__main__":
    main()
