"""Recompute saved classical scores and export a comparison CSV without training."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from quantum_fraud.evaluation.metrics import binary_metrics, select_f1_threshold


def main() -> None:
    """Verify stored scores/metrics and write a reviewable comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed classical run directory")
    args = parser.parse_args()
    result = json.loads((args.run / "results.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Classical run is incomplete")
    prepared = Path(result["preprocessing_manifest"]).parent
    validation_labels = np.load(prepared / "validation_y.npy", mmap_mode="r")
    test_labels = np.load(prepared / "test_y.npy", mmap_mode="r")
    rows = []
    for record in result["models"]:
        with np.load(args.run / f"{record['model']}_scores.npz") as scores:
            threshold = select_f1_threshold(validation_labels, scores["validation_scores"])
            recomputed = binary_metrics(test_labels, scores["test_scores"], threshold)
        for key, value in recomputed.items():
            np.testing.assert_allclose(value, record["test"][key], rtol=1e-12, atol=1e-12)
        row = {"model": record["model"], **{key: value for key, value in recomputed.items() if key != "confusion_matrix"}, "training_seconds": record["training_seconds"], "test_batch_prediction_seconds": record["prediction_batch_seconds"]["test"]}
        rows.append(row)
    output = args.run / "comparison.csv"
    # Idempotent derived export; immutable per-model score/results files remain intact.
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Recomputed all stored test metrics successfully: {output}")


if __name__ == "__main__":
    main()
