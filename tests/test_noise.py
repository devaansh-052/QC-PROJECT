import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from quantum_fraud.quantum.noise import measurement_probability, exact_probabilities, sampled_probabilities, bound_vqc_circuits
from quantum_fraud.quantum.variational import build_vqc


def test_parity_and_exact_probabilities():
    assert measurement_probability({"00": 5, "01": 2, "10": 3, "11": 10}) == .6
    with pytest.raises(ValueError):
        measurement_probability({})
    circuit = QuantumCircuit(2)
    circuit.x(0)
    np.testing.assert_allclose(exact_probabilities([circuit]), [1])


def test_real_aer_shots_noise_and_reproducibility():
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"])
    exact, _ = sampled_probabilities([circuit], shots=512, seed=42)
    noisy, _ = sampled_probabilities([circuit], shots=512, seed=42, noisy=True)
    again, _ = sampled_probabilities([circuit], shots=512, seed=42, noisy=True)
    assert exact[0] == 1 and 0 < noisy[0] < 1
    np.testing.assert_array_equal(noisy, again)


def test_trained_vqc_measurement_mapping_matches_exact_reconstruction():
    model, _, _, _ = build_vqc(2, 1, "linear", maxiter=12)
    X = np.array([[-.8, -.2], [-.5, .2], [.5, -.2], [.8, .2]])
    model.fit(X, np.array([0, 0, 1, 1]))
    reconstructed = exact_probabilities(bound_vqc_circuits(model, X), model.neural_network.interpret)
    np.testing.assert_allclose(reconstructed, model.predict_proba(X)[:, 1], atol=1e-10)
