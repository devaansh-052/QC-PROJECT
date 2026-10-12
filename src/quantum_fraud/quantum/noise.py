"""Matched exact, finite-shot and noisy VQC inference; no hardware claims."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_machine_learning.algorithms import VQC

from quantum_fraud.evaluation.metrics import binary_metrics, select_f1_threshold
from quantum_fraud.quantum.data import load_quantum_arrays
from quantum_fraud.quantum.experiment import evaluate_scores, runtime_versions
from quantum_fraud.quantum.kernel import circuit_metadata


def illustrative_noise():
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(.01, 1), ["sx", "x"])
    noise.add_all_qubit_quantum_error(depolarizing_error(.03, 2), ["cx"])
    noise.add_all_qubit_readout_error(ReadoutError([[.98, .02], [.03, .97]]))
    return noise


def measurement_probability(counts, interpret=None):
    total = sum(counts.values())
    if total <= 0 or any(v < 0 for v in counts.values()):
        raise ValueError("Positive measurement counts required")
    interpret = interpret or (lambda outcome: outcome % 2)
    return sum(value for key, value in counts.items() if interpret(int(key.replace(" ", ""), 2)) == 1) / total


def bound_vqc_circuits(model, X):
    circuit = model.neural_network.circuit.remove_final_measurements(inplace=False)
    weight_bindings = dict(zip(model.ansatz.parameters, model.weights))
    return [circuit.assign_parameters({**weight_bindings, **dict(zip(model.feature_map.parameters, row))}) for row in X]


def exact_probabilities(circuits, interpret=None):
    interpret = interpret or (lambda outcome: outcome % 2)
    return np.array([sum(p for i, p in enumerate(Statevector(c).probabilities()) if interpret(i) == 1) for c in circuits])


def sampled_probabilities(circuits, shots=256, seed=42, noisy=False, interpret=None):
    if type(shots) is not int or shots <= 0:
        raise ValueError("Positive integer shots required")
    backend = AerSimulator(method="density_matrix", noise_model=illustrative_noise() if noisy else None, max_parallel_threads=2)
    scores, backend_seconds = [], 0.0
    for start in range(0, len(circuits), 128):
        batch = [c.copy() for c in circuits[start:start + 128]]
        for circuit in batch:
            circuit.measure_all()
        result = backend.run(batch, shots=shots, seed_simulator=seed + start).result()
        if not result.success:
            raise RuntimeError("Aer inference failed")
        backend_seconds += float(result.time_taken)
        scores.extend(measurement_probability(result.get_counts(i), interpret) for i in range(len(batch)))
    return np.asarray(scores), backend_seconds


def run_noise(data: Path, vqc_run: Path, root: Path):
    source = json.loads((vqc_run / "results.json").read_text())
    if Path(source["subset_manifest"]).resolve() != (data / "manifest.json").resolve():
        raise ValueError("Mismatched VQC data")
    model = VQC.from_dill(str(vqc_run / "vqc_linear_r1_2q.model"))
    arrays = load_quantum_arrays(data, 2)
    output = root / (datetime.now(timezone.utc).strftime("noise_%Y%m%dT%H%M%S_") + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    config = {"shots": 256, "seeds": [42, 43, 44], "model": "vqc_linear_r1_2q", "basis": ["rz", "sx", "x", "cx"],
              "noise": {"one_qubit_depolarizing": .01, "two_qubit_depolarizing": .03, "readout": [[.98, .02], [.03, .97]]},
              "interpretation": "trained VQC interpretation: integer outcome modulo 2", "training": "reuse frozen Phase 7 weights"}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    circuits, exact, exact_times = {}, {}, {}
    start = perf_counter()
    # Transpile the parameterized template once, preserving parameter identities.
    template = transpile(model.neural_network.circuit.remove_final_measurements(inplace=False), basis_gates=config["basis"], optimization_level=0, seed_transpiler=42)
    transpilation_seconds = perf_counter() - start
    weights = dict(zip(model.ansatz.parameters, model.weights))
    for split in ("validation", "test"):
        circuits[split] = [template.assign_parameters({**weights, **dict(zip(model.feature_map.parameters, row))}) for row in arrays[split][0]]
        start = perf_counter()
        exact[split] = exact_probabilities(circuits[split], model.neural_network.interpret)
        exact_times[split] = perf_counter() - start
    with np.load(vqc_run / "vqc_linear_r1_2q_scores.npz") as prior:
        for split in exact:
            np.testing.assert_allclose(exact[split], prior[split], atol=1e-10)
    threshold = select_f1_threshold(arrays["validation"][1], exact["validation"])
    reference = binary_metrics(arrays["test"][1], exact["test"], threshold)
    records, repeated = [], {"shot": [], "noisy": []}
    variants = [("exact", None)] + [(mode, seed) for mode in repeated for seed in config["seeds"]]
    for mode, seed in variants:
        print(f"noise study: {mode} seed={seed}", flush=True)
        scores, durations, backend_durations = {}, {}, {}
        for split in ("validation", "test"):
            if mode == "exact":
                scores[split], durations[split] = exact[split], exact_times[split]
            else:
                start = perf_counter()
                scores[split], backend_durations[split] = sampled_probabilities(circuits[split], config["shots"], seed, mode == "noisy", model.neural_network.interpret)
                durations[split] = perf_counter() - start
        if mode != "exact":
            repeated[mode].append(scores["test"])
        stem = f"{mode}_seed{seed}"
        np.savez(output / f"{stem}_scores.npz", **scores)
        fixed = binary_metrics(arrays["test"][1], scores["test"], threshold)
        records.append({"model": mode, "artifact_stem": stem, "features": 2, "seed": seed,
                        "shots": None if mode == "exact" else config["shots"], "training_seconds": 0,
                        "prediction_batch_seconds": durations, "aer_backend_seconds": backend_durations,
                        "fixed_ideal_threshold_test": fixed,
                        "change_from_exact_fixed_threshold": {key: fixed[key] - reference[key] for key in ("precision", "recall", "f1", "pr_auc_average_precision")},
                        **evaluate_scores(arrays, scores["validation"], scores["test"])})
    stability = {}
    for mode, values in repeated.items():
        deviations = np.std(values, axis=0, ddof=1)
        stability[mode] = {"mean_per_transaction_sd": float(deviations.mean()), "max_sd": float(deviations.max()),
                           "mean_absolute_error_from_exact": float(np.abs(np.asarray(values) - exact["test"]).mean())}
    report = {"status": "complete", "run_id": output.name, "config": config, "versions": runtime_versions(),
              "subset_manifest": str((data / "manifest.json").resolve()), "source_model": str((vqc_run / "vqc_linear_r1_2q.model").resolve()),
              "circuit": circuit_metadata(template), "transpilation_seconds": transpilation_seconds,
              "models": records, "stability": stability,
              "limitations": ["Illustrative noise rates, not a hardware calibration", "One frozen small VQC; no noisy retraining",
                              "Aer backend time includes simulator work and is nested within prediction wall time",
                              "Signed metric changes can improve or degrade; no forced degradation claim", "No physical hardware results"]}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    frame = pd.DataFrame([{"mode": r["model"], "seed": r["seed"], **r["change_from_exact_fixed_threshold"]} for r in records])
    frame.to_csv(output / "degradation.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode in repeated:
        axes[0].scatter(exact["test"], np.mean(repeated[mode], axis=0), s=4, alpha=.2, label=mode)
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="Exact probability", ylabel="Mean sampled probability")
    axes[0].legend()
    axes[1].bar(list(stability), [v["mean_per_transaction_sd"] for v in stability.values()])
    axes[1].set(ylabel="Mean per-transaction SD across 3 simulator seeds")
    fig.savefig(output / "noise.png", dpi=160)
    fig.savefig(output / "noise.pdf")
    plt.close(fig)
    return output
