"""Paired neural experiments with validation-only checkpoint selection."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
from torch import nn
import yaml

from quantum_fraud.hybrid.models import FraudNetwork, configure_determinism, load_network, predict_logits
from quantum_fraud.quantum.data import load_quantum_arrays
from quantum_fraud.quantum.experiment import evaluate_scores, runtime_versions


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != {"seed", "epochs", "batch_size", "learning_rate", "architecture", "checkpoint_selection"}:
        raise ValueError("Invalid hybrid configuration")
    if any(type(config[k]) is not int or config[k] < 1 for k in ("epochs", "batch_size")):
        raise ValueError("Positive integer epochs and batch size required")
    if type(config["seed"]) is not int or config["seed"] < 0 or not 0 < config["learning_rate"] < 1:
        raise ValueError("Invalid seed or learning rate")
    if config["architecture"] != "eligible7_dense16_latent2_v1" or config["checkpoint_selection"] != "minimum_validation_bce":
        raise ValueError("Unsupported architecture or checkpoint protocol")
    return config


def tensor_arrays(arrays):
    result = {}
    for split, (X, y) in arrays.items():
        if X.ndim != 2 or X.shape != (len(y), 7) or not np.isfinite(X).all() or set(np.unique(y)) != {0, 1}:
            raise ValueError("Expected finite seven-feature arrays with both target classes")
        result[split] = (torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return result


def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def fit_network(model, train, validation, config):
    """The fitting function deliberately receives no test data."""
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"])
    initial = snapshot(model)
    history, best_state, best_loss, best_epoch = [], None, float("inf"), None
    first_gradients = None
    training_seconds = validation_seconds = 0.0
    start_all = perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        start = perf_counter()
        order = torch.randperm(len(train[1]), generator=generator)
        total_loss = 0.0
        for offset in range(0, len(order), config["batch_size"]):
            indices = order[offset:offset + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train[0][indices]), train[1][indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            norms = {}
            for name in ("encoder", "bottleneck", "head"):
                grads = [p.grad for p in getattr(model, name).parameters() if p.requires_grad]
                if any(g is None or not torch.isfinite(g).all() for g in grads):
                    raise ValueError(f"Missing or nonfinite {name} gradient")
                norms[name] = float(torch.sqrt(sum(g.detach().square().sum() for g in grads)))
            if first_gradients is None:
                first_gradients = norms
                if any(value == 0 for value in norms.values()):
                    raise ValueError("Initial gradient did not reach every model component")
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        training_seconds += perf_counter() - start
        start = perf_counter()
        logits = predict_logits(model, validation[0])
        val_loss = float(loss_fn(logits, validation[1]))
        validation_seconds += perf_counter() - start
        if not np.isfinite(val_loss):
            raise ValueError("Nonfinite validation loss")
        history.append({"epoch": epoch, "training_batch_mean_bce": total_loss / len(order), "validation_bce": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, snapshot(model)
        print(f"{'hybrid' if model.quantum else 'classical'} epoch {epoch}/{config['epochs']}: train={total_loss / len(order):.6f} val={val_loss:.6f}", flush=True)
    model.load_state_dict(best_state)
    changes = {name: float(torch.sqrt(sum((p.detach() - initial[key]).square().sum()
                                          for key, p in model.named_parameters() if key.startswith(name + "."))))
               for name in ("encoder", "bottleneck", "head")}
    if any(value == 0 for value in changes.values()):
        raise ValueError("Selected checkpoint did not update every model component")
    return {"history": history, "selected_epoch": best_epoch, "selected_validation_bce": best_loss,
            "first_batch_gradient_l2": first_gradients, "selected_parameter_change_l2": changes,
            "training_seconds": perf_counter() - start_all, "optimizer_training_seconds": training_seconds,
            "validation_monitor_seconds": validation_seconds,
            "optimizer_steps": config["epochs"] * ((len(train[1]) + config["batch_size"] - 1) // config["batch_size"])}


def run_hybrid(data: Path, config_path: Path, output_root: Path):
    config = read_config(config_path)
    arrays = load_quantum_arrays(data)
    tensors = tensor_arrays(arrays)
    run_id = datetime.now(timezone.utc).strftime("hybrid_%Y%m%dT%H%M%S_") + uuid4().hex[:8]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    records = []
    for quantum in (True, False):
        configure_determinism(config["seed"])
        model = FraudNetwork(quantum, config["seed"])
        stem = "hybrid_qnn" if quantum else "classical_neural_control"
        training = fit_network(model, tensors["train"], tensors["validation"], config)
        scores, durations = {}, {}
        for split in ("validation", "test"):
            start = perf_counter()
            scores[split] = torch.sigmoid(predict_logits(model, tensors[split][0])).numpy().astype(np.float64)
            durations[split] = perf_counter() - start
        torch.save({"architecture": config["architecture"], "quantum": quantum,
                    "seed": config["seed"], "state_dict": snapshot(model), "selected_epoch": training["selected_epoch"]}, output / f"{stem}.pt")
        restored = load_network(output / f"{stem}.pt")
        probe = torch.sigmoid(predict_logits(restored, tensors["test"][0][:16])).numpy()
        np.testing.assert_allclose(probe, scores["test"][:16], rtol=1e-6, atol=1e-7)
        np.savez(output / f"{stem}_scores.npz", **scores)
        record = {"model": stem, "artifact_stem": stem, "features": 7, "latent_features": 2,
                  "simulation": "ideal_exact_statevector" if quantum else "classical", "qubits": 2 if quantum else 0,
                  "circuit": model.circuit_info, "parameters": model.parameter_counts(),
                  "score_kind": "uncalibrated_fraud_probability", "prediction_batch_seconds": durations,
                  **training, **evaluate_scores(arrays, scores["validation"], scores["test"])}
        records.append(record)
        (output / f"{stem}_metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    report = {"status": "complete", "run_id": run_id, "config": config,
              "versions": {**runtime_versions(), "torch": version("torch")},
              "subset_manifest": str((data / "manifest.json").resolve()), "models": records,
              "limitations": ["One paired seed and tiny balanced training subset; no statistical advantage claim",
                              "Identical seven-feature inputs, samples, common-layer initialization, optimizer and batch order",
                              "Checkpoint selection and thresholds use validation only; test scored after selection",
                              "20-epoch training budget does not guarantee convergence",
                              "No hardware or quantum speedup claim; uncalibrated scores under prevalence shift"]}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return output


def plot_training(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads((run / "results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, record in zip(axes, report["models"]):
        history = record["history"]
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["training_batch_mean_bce"] for row in history], label="Training batch mean")
        ax.plot(epochs, [row["validation_bce"] for row in history], label="Validation")
        ax.axvline(record["selected_epoch"], color="gray", linestyle="--", label="Selected checkpoint")
        ax.set(title=f"{record['model']} ({record['parameters']['total']} parameters)", xlabel="Epoch", ylabel="Binary cross entropy")
        ax.legend()
    fig.savefig(run / "training.png", dpi=150)
    fig.savefig(run / "training.pdf")
    plt.close(fig)
