import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from quantum_fraud.hybrid.models import FraudNetwork, configure_determinism, load_network, predict_logits
from quantum_fraud.hybrid.train import fit_network, read_config, tensor_arrays


def test_capacity_and_paired_initialization():
    hybrid, classical = FraudNetwork(True), FraudNetwork(False)
    assert hybrid.parameter_counts()["total"] == 169
    assert classical.parameter_counts()["total"] == 171
    for left, right in zip(hybrid.encoder.parameters(), classical.encoder.parameters()):
        torch.testing.assert_close(left, right)
    for left, right in zip(hybrid.head.parameters(), classical.head.parameters()):
        torch.testing.assert_close(left, right)
    assert hybrid.bottleneck.neural_network.input_gradients


def test_quantum_and_encoder_gradients_match_finite_difference():
    configure_determinism(42)
    model = FraudNetwork(True)
    X = torch.tensor([[.2, -.1, .3, .5, -.4, .2, .1]], requires_grad=True)
    model(X).sum().backward()
    assert X.grad is not None and torch.linalg.vector_norm(X.grad) > 0
    assert torch.linalg.vector_norm(model.encoder[0].weight.grad) > 0
    assert torch.linalg.vector_norm(model.bottleneck.weight.grad) > 0
    # Independently perturb a quantum weight and an encoder weight through the full circuit.
    for parameter, index in ((model.bottleneck.weight, (0,)), (model.encoder[0].weight, (0, 0))):
        analytic = float(parameter.grad[index])
        with torch.no_grad():
            original = parameter[index].item()
            parameter[index] = original + .001
            plus = float(model(X.detach()).sum())
            parameter[index] = original - .001
            minus = float(model(X.detach()).sum())
            parameter[index] = original
        np.testing.assert_allclose(analytic, (plus - minus) / .002, atol=1e-4, rtol=.03)


@pytest.mark.parametrize("quantum", [True, False])
def test_actual_training_checkpoint_reload(quantum, tmp_path):
    configure_determinism(42)
    X = np.random.default_rng(3).normal(size=(8, 7)).astype(np.float32)
    y = np.array([0, 1] * 4)
    arrays = tensor_arrays({"train": (X, y), "validation": (X * .9, y)})
    config = read_config(Path("configs/hybrid.yaml"))
    config.update(epochs=2, batch_size=4)
    model = FraudNetwork(quantum)
    result = fit_network(model, arrays["train"], arrays["validation"], config)
    assert result["optimizer_steps"] == 4
    assert result["selected_epoch"] == min(result["history"], key=lambda r: r["validation_bce"])["epoch"]
    assert all(value > 0 for value in result["first_batch_gradient_l2"].values())
    assert all(value > 0 for value in result["selected_parameter_change_l2"].values())
    probabilities = torch.sigmoid(predict_logits(model, arrays["validation"][0]))
    assert torch.isfinite(probabilities).all() and ((probabilities > 0) & (probabilities < 1)).all()
    path = tmp_path / "model.pt"
    torch.save({"architecture": config["architecture"], "quantum": quantum, "seed": 42, "state_dict": model.state_dict()}, path)
    restored = load_network(path)
    torch.testing.assert_close(torch.sigmoid(predict_logits(restored, arrays["validation"][0])), probabilities)
    measured_loss = float(torch.nn.BCEWithLogitsLoss()(predict_logits(restored, arrays["validation"][0]), arrays["validation"][1]))
    assert measured_loss == pytest.approx(result["selected_validation_bce"], abs=1e-7)


def test_invalid_hybrid_inputs_and_config(tmp_path):
    with pytest.raises(ValueError, match="seven"):
        FraudNetwork(False)(torch.zeros(2, 6))
    with pytest.raises(ValueError):
        tensor_arrays({"train": (np.full((2, 7), np.nan), np.array([0, 1]))})
    config = read_config("configs/hybrid.yaml")
    config["epochs"] = 0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        read_config(path)


def test_classical_then_hybrid_import_in_fresh_process():
    # A fresh process is essential: the test runner already imported Torch.
    result = subprocess.run([sys.executable, "-c",
                             "from quantum_fraud.classical.models import build_models; "
                             "from quantum_fraud.hybrid.models import FraudNetwork; "
                             "assert FraudNetwork(True).parameter_counts()['total'] == 169; print('imports-ok')"],
                            capture_output=True, text=True, timeout=60, check=True)
    assert "imports-ok" in result.stdout
