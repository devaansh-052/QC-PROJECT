"""Reproduce every saved Phase 8 score from tensor-only checkpoints."""

import argparse
import json
from pathlib import Path

import quantum_fraud  # Initialize Windows native libraries before sklearn.
import numpy as np
import torch

from quantum_fraud.hybrid.models import configure_determinism, load_network, predict_logits
from quantum_fraud.hybrid.train import tensor_arrays
from quantum_fraud.quantum.data import load_quantum_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "results.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Run is incomplete")
    configure_determinism(report["config"]["seed"])
    arrays = tensor_arrays(load_quantum_arrays(Path(report["subset_manifest"]).parent))
    for record in report["models"]:
        stem = record["artifact_stem"]
        model = load_network(args.run / f"{stem}.pt")
        assert model.parameter_counts() == record["parameters"]
        best = min(record["history"], key=lambda row: row["validation_bce"])
        assert record["selected_epoch"] == best["epoch"]
        with np.load(args.run / f"{stem}_scores.npz") as saved:
            for split in ("validation", "test"):
                logits = predict_logits(model, arrays[split][0])
                scores = torch.sigmoid(logits).numpy()
                assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
                np.testing.assert_allclose(scores, saved[split], rtol=1e-6, atol=1e-7)
                if split == "validation":
                    bce = float(torch.nn.BCEWithLogitsLoss()(logits, arrays[split][1]))
                    np.testing.assert_allclose(bce, best["validation_bce"], atol=1e-7)
        print(f"{stem}: all validation/test scores, parameter counts and selected checkpoint verified")


if __name__ == "__main__":
    main()
