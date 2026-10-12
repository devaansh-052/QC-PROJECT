"""A small TorchConnector quantum bottleneck with genuine input gradients."""

import numpy as np
import torch
from torch import nn
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN

from quantum_fraud.quantum.kernel import feature_map, circuit_metadata


class FraudNetwork(nn.Module):
    """Seven eligible features -> 16 -> 2 -> bottleneck -> one fraud logit."""

    def __init__(self, quantum: bool, seed: int = 42):
        super().__init__()
        self.quantum = quantum
        self.seed = seed
        torch.manual_seed(seed)
        # Construct common layers first so paired models share their initialization.
        self.encoder = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, 2), nn.Tanh())
        self.head = nn.Linear(2, 1)
        self.circuit_info = None
        if quantum:
            encoding = feature_map(2, "zz", 1)
            ansatz = real_amplitudes(2, reps=1, entanglement="linear")
            circuit = encoding.compose(ansatz)
            estimator = StatevectorEstimator(default_precision=0.0, seed=seed)
            qnn = EstimatorQNN(circuit=circuit, estimator=estimator,
                               observables=[SparsePauliOp("ZI"), SparsePauliOp("IZ")],
                               input_params=encoding.parameters, weight_params=ansatz.parameters,
                               gradient=ParamShiftEstimatorGradient(estimator),
                               input_gradients=True, default_precision=0.0)
            self.bottleneck = TorchConnector(qnn, initial_weights=np.random.default_rng(seed).uniform(-1, 1, ansatz.num_parameters))
            self.circuit_info = circuit_metadata(circuit)
        else:
            self.bottleneck = nn.Sequential(nn.Linear(2, 2), nn.Tanh())

    def forward(self, X):
        if X.ndim != 2 or X.shape[1] != 7:
            raise ValueError("Expected a batch with seven eligible input features")
        return self.head(self.bottleneck(self.encoder(X))).squeeze(-1)

    def parameter_counts(self):
        counts = {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
                  for name, module in (("encoder", self.encoder), ("bottleneck", self.bottleneck), ("head", self.head))}
        counts["total"] = sum(counts.values())
        return counts


def configure_determinism(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def predict_logits(model, X, batch_size=128):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[start:start + batch_size]) for start in range(0, len(X), batch_size)])


def load_network(path):
    """Reconstruct the circuit from explicit metadata and load tensor weights only."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != "eligible7_dense16_latent2_v1":
        raise ValueError("Unsupported checkpoint architecture")
    model = FraudNetwork(quantum=checkpoint["quantum"], seed=checkpoint["seed"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model
