import numpy as np
import pytest
from qiskit_machine_learning.algorithms import VQC
from quantum_fraud.quantum.variational import build_vqc


@pytest.mark.parametrize("qubits", [2, 4, 6])
@pytest.mark.parametrize("reps", [1, 2, 3])
@pytest.mark.parametrize("entanglement", ["linear", "circular"])
def test_vqc_grid(qubits, reps, entanglement):
    model, trace, circuit, parameters = build_vqc(qubits, reps, entanglement)
    assert circuit.num_qubits == qubits
    assert parameters == qubits * (reps + 1)
    assert model.neural_network.output_shape == (2,)


def test_vqc_real_fit_reload(tmp_path):
    X = np.array([[-.8, -.2], [-.5, .2], [.5, -.2], [.8, .2]])
    model, trace, _, _ = build_vqc(2, 1, "linear", maxiter=12)
    initial = model.initial_point.copy()
    model.fit(X, np.array([0, 0, 1, 1]))
    scores = model.predict_proba(X)
    assert trace.values and np.isfinite(trace.values).all()
    assert not np.allclose(initial, model.weights)
    np.testing.assert_allclose(scores.sum(axis=1), 1)
    assert np.all((scores >= 0) & (scores <= 1))
    model.to_dill(str(tmp_path / "vqc.model"))
    restored = VQC.from_dill(str(tmp_path / "vqc.model"))
    np.testing.assert_allclose(restored.predict_proba(X), scores)
    with pytest.raises(Exception):
        model.predict_proba(np.ones((2, 3)))


def test_vqc_invalid_configuration():
    with pytest.raises(ValueError):
        build_vqc(2, 4, "linear")
