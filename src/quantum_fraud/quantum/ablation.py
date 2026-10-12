"""Controlled one-factor-at-a-time quantum kernel ablations."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit_machine_learning.algorithms import QSVC
from sklearn.svm import SVC

from quantum_fraud.quantum.data import load_quantum_arrays, sample_positions
from quantum_fraud.quantum.experiment import evaluate_scores, runtime_versions
from quantum_fraud.quantum.kernel import TimedKernel, circuit_metadata


def ablation_circuit(qubits, repetitions, entanglement, map_name):
    if qubits not in (2, 4, 6) or repetitions not in (1, 2, 3) or entanglement not in ("none", "linear", "full") or map_name not in ("z", "zz"):
        raise ValueError("Unsupported ablation configuration")
    if map_name == "z" or entanglement == "none":
        return z_feature_map(qubits, reps=repetitions)
    return zz_feature_map(qubits, reps=repetitions, entanglement=entanglement)


def ablation_grid():
    return [(4, 1, "linear", "zz"), (2, 1, "linear", "zz"), (6, 1, "linear", "zz"),
            (4, 2, "linear", "zz"), (4, 3, "linear", "zz"), (4, 1, "none", "zz"),
            (4, 1, "full", "zz"), (4, 1, "none", "z")]


def run_ablations(data: Path, root: Path):
    output = root / (datetime.now(timezone.utc).strftime("ablation_%Y%m%dT%H%M%S_") + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    config = {"seeds": [42, 43, 44], "grid": ablation_grid(), "training_rows": 96,
              "protocol": "48 per class without replacement from fixed 128-row training pool; unchanged validation/test",
              "representation": "parent train-fitted PCA and angle scales, common within each comparison",
              "unsupported": {"8_qubits": "Only seven eligible encoded inputs; maximum PCA rank six"}}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    records = []
    for seed in config["seeds"]:
        positions = sample_positions(load_quantum_arrays(data, 2)["train"][1], 96, seed, balanced=True)
        np.save(output / f"training_positions_seed{seed}.npy", positions)
        variants = [(q, reps, ent, fmap, "qsvc") for q, reps, ent, fmap in ablation_grid()]
        variants += [(q, 0, "none", "none", "rbf") for q in (2, 4, 6)]
        for q, reps, ent, fmap, kind in variants:
            arrays = load_quantum_arrays(data, q)
            name = f"{kind}_{q}q_r{reps}_{ent}_{fmap}"
            stem = f"{name}_seed{seed}"
            print(stem, flush=True)
            circuit = ablation_circuit(q, reps, ent, fmap) if kind == "qsvc" else None
            model = QSVC(quantum_kernel=TimedKernel(circuit), C=1) if circuit else SVC(C=1, gamma="scale")
            start = perf_counter()
            model.fit(arrays["train"][0][positions], arrays["train"][1][positions])
            fit_seconds = perf_counter() - start
            scores, timings = {}, {}
            for split in ("validation", "test"):
                start = perf_counter()
                scores[split] = model.decision_function(arrays[split][0])
                timings[split] = perf_counter() - start
            np.savez(output / f"{stem}_scores.npz", **scores)
            if circuit:
                model.to_dill(str(output / f"{stem}.model"))
                restored = QSVC.from_dill(str(output / f"{stem}.model"))
            else:
                joblib.dump(model, output / f"{stem}.model")
                restored = joblib.load(output / f"{stem}.model")
            np.testing.assert_allclose(restored.decision_function(arrays["test"][0][:5]), scores["test"][:5], atol=1e-10)
            records.append({"model": name, "artifact_stem": stem, "features": q, "seed": seed,
                            "encoding_repetitions": reps, "entanglement": ent, "feature_map": fmap,
                            "circuit": circuit_metadata(circuit) if circuit else None,
                            "training_seconds": fit_seconds, "prediction_batch_seconds": timings,
                            "kernel_timings": model.quantum_kernel.timings if circuit else None,
                            **evaluate_scores(arrays, scores["validation"], scores["test"])})
    report = {"status": "complete", "run_id": output.name, "config": config, "versions": runtime_versions(),
              "subset_manifest": str((data / "manifest.json").resolve()), "models": records,
              "limitations": ["Three resamples of one small training pool; not three independent datasets",
                              "One-factor-at-a-time design does not measure factor interactions",
                              "Encoding repetitions and actual logical depth are distinct",
                              "ZZ without pair interactions equals Z; duplicate is an intentional sanity check",
                              "No significance or quantum advantage claim; only 29 test positives"]}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    rows = [{"model": r["model"], "seed": r["seed"], "qubits": r["features"],
             "logical_depth": r["circuit"]["logical_depth"] if r["circuit"] else 0,
             **{k: r["test"][k] for k in ("precision", "recall", "f1", "pr_auc_average_precision", "mcc", "false_positive_rate")},
             "training_seconds": r["training_seconds"], "test_batch_seconds": r["prediction_batch_seconds"]["test"]} for r in records]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "study.csv", index=False)
    frame.groupby("model").agg({k: ["mean", "std"] for k in ("pr_auc_average_precision", "recall", "f1", "mcc", "training_seconds", "test_batch_seconds")}).to_csv(output / "mean_std.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grouped = frame.groupby("model")["pr_auc_average_precision"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(grouped.index, grouped["mean"], xerr=grouped["std"])
    ax.set(xlabel="Test average precision: mean ± sample SD (3 training resamples)", title="Controlled kernel ablations; 29 test fraud cases")
    fig.savefig(output / "ablation.png", dpi=140)
    fig.savefig(output / "ablation.pdf")
    plt.close(fig)
    return output
