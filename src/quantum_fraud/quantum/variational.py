"""Small exact-statevector VQC experiments with explicit optimization budgets."""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.primitives import QMLSampler

from quantum_fraud.quantum.data import load_quantum_arrays
from quantum_fraud.quantum.experiment import evaluate_scores, runtime_versions
from quantum_fraud.quantum.kernel import feature_map, circuit_metadata


class LossTrace:
    def __init__(self):
        self.values = []

    def __call__(self, weights, loss):
        self.values.append(float(loss))


def build_vqc(qubits, repetitions, entanglement, seed=42, maxiter=60):
    if repetitions not in (1, 2, 3) or entanglement not in ("linear", "circular") or maxiter < 1:
        raise ValueError("Invalid VQC configuration")
    encoding = feature_map(qubits, "zz", 1)
    ansatz = real_amplitudes(qubits, reps=repetitions, entanglement=entanglement)
    trace = LossTrace()
    model = VQC(feature_map=encoding, ansatz=ansatz, sampler=QMLSampler(shots=None),
                optimizer=COBYLA(maxiter=maxiter, tol=1e-4), callback=trace,
                initial_point=np.random.default_rng(seed).uniform(-np.pi, np.pi, ansatz.num_parameters))
    return model, trace, encoding.compose(ansatz), ansatz.num_parameters


def run_vqc(data: Path, output_root: Path):
    run_id = datetime.now(timezone.utc).strftime("vqc_%Y%m%dT%H%M%S")
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for qubits in (2, 4, 6):
        arrays = load_quantum_arrays(data, qubits)
        for entanglement in ("linear", "circular"):
            for reps in (1, 2, 3):
                stem = f"vqc_{entanglement}_r{reps}_{qubits}q"
                print(f"Training {stem}", flush=True)
                model, trace, circuit, parameters = build_vqc(qubits, reps, entanglement)
                start = perf_counter()
                model.fit(*arrays["train"])
                elapsed = perf_counter() - start
                scores, timings = {}, {}
                for split in ("validation", "test"):
                    start = perf_counter()
                    scores[split] = model.predict_proba(arrays[split][0])[:, 1]
                    timings[split] = perf_counter() - start
                model.to_dill(str(output / f"{stem}.model"))
                restored = VQC.from_dill(str(output / f"{stem}.model"))
                np.testing.assert_allclose(restored.predict_proba(arrays["test"][0][:5])[:, 1], scores["test"][:5], atol=1e-10)
                np.savez(output / f"{stem}_scores.npz", **scores)
                record = {"model": stem, "artifact_stem": stem, "features": qubits,
                          "simulation": "ideal_exact_statevector", "seed": 42, "ansatz_repetitions": reps,
                          "entanglement": entanglement, "circuit": circuit_metadata(circuit),
                          "trainable_parameters": parameters, "optimizer": "COBYLA", "max_evaluations": 60,
                          "optimizer_evaluations": int(model.fit_result.nfev),
                          "optimizer_iterations": getattr(model.fit_result, "nit", None),
                          "budget_exhausted": int(model.fit_result.nfev) >= 60,
                          "convergence_claim": "No convergence guarantee; fixed evaluation budget",
                          "objective_loss": trace.values, "training_seconds": elapsed,
                          "prediction_batch_seconds": timings, "score_kind": "fraud_probability",
                          **evaluate_scores(arrays, scores["validation"], scores["test"])}
                records.append(record)
                (output / f"{stem}_metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    report = {"status": "complete", "run_id": run_id, "versions": runtime_versions(),
              "subset_manifest": str((data / "manifest.json").resolve()), "models": records,
              "matched_controls": str((data.parent / "results.json").resolve()),
              "limitations": ["Single seed, 128 training rows, 13 validation positives, 29 test positives",
                              "Classical simulator; no speedup claim", "Two-qubit linear/circular circuits may coincide",
                              "Compare only with Phase 6 RBF controls on identical angles and rows"]}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return output


def plot_convergence(run: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads((run / "results.json").read_text())
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
    for record in report["models"]:
        ax = axes[(record["features"] // 2) - 1, int(record["entanglement"] == "circular")]
        losses = record["objective_loss"]
        ax.plot(np.arange(1, len(losses) + 1), losses, label=f"reps={record['ansatz_repetitions']}")
        ax.set(title=f"{record['features']} qubits, {record['entanglement']}", xlabel="Objective evaluation", ylabel="Cross entropy")
        ax.legend()
    fig.savefig(run / "convergence.png", dpi=140)
    fig.savefig(run / "convergence.pdf")
    plt.close(fig)
