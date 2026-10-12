"""Verify feature-control scores and export a comparison table and figure."""

import argparse
import csv
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from quantum_fraud.evaluation.metrics import binary_metrics, select_f1_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "results.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Feature run is incomplete")
    prepared = Path(report["prepared_manifest"]).parent
    validation_y = np.load(prepared / "validation_y.npy", mmap_mode="r")
    test_y = np.load(prepared / "test_y.npy", mmap_mode="r")
    rows = []
    for control in report["controls"]:
        stem = f"{control['method']}_{control['dimension']}"
        with np.load(args.run / f"{stem}_scores.npz") as scores:
            threshold = select_f1_threshold(validation_y, scores["validation"])
            metrics = binary_metrics(test_y, scores["test"], threshold)
        for key, value in metrics.items():
            np.testing.assert_allclose(value, control["test"][key], rtol=1e-12, atol=1e-12)
        rows.append({"method": control["method"], "dimension": control["dimension"],
                     "features": ";".join(control["features"]),
                     **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                     "control_training_seconds": control["control_training_seconds"]})
    with (args.run / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)
    for method, label in [("mutual_information", "Mutual information"), ("random_forest_importance", "Forest importance"), ("pca", "PCA")]:
        selected = [row for row in rows if row["method"] == method]
        left.plot([row["dimension"] for row in selected], [row["pr_auc_average_precision"] for row in selected], marker="o", label=label)
    left.set(xlabel="Reduced feature count", ylabel="Test average precision", title="Matched linear SVMs — one seed")
    left.legend()
    left.set_xticks([2, 4, 6])
    variance = np.cumsum(report["pca_explained_variance_ratio"])
    right.plot(np.arange(1,len(variance)+1), variance, marker="o")
    right.set(xlabel="PCA components", ylabel="Cumulative training variance fraction", title="Train-fitted PCA")
    right.set_ylim(0, 1.05)
    fig.suptitle("Prespecified feature grid; no selection based on test results")
    root = Path(__file__).resolve().parents[1]
    output = root / "results/figures" / report["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"feature_comparison.{extension}", dpi=160)
    print(f"All nine control metric records verified; comparison saved in {args.run} and {output}")


if __name__ == "__main__":
    main()
