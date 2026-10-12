import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from quantum_fraud.quantum.ablation import ablation_circuit, ablation_grid


def test_ablation_factors_and_equivalence():
    grid = ablation_grid()
    assert {row[0] for row in grid} == {2, 4, 6}
    assert {row[1] for row in grid} == {1, 2, 3}
    assert {row[2] for row in grid} == {"none", "linear", "full"}
    a = ablation_circuit(4, 1, "none", "zz")
    b = ablation_circuit(4, 1, "none", "z")
    values = [.1, .2, -.3, .4]
    np.testing.assert_allclose(Statevector(a.assign_parameters(values)).data, Statevector(b.assign_parameters(values)).data)
    assert ablation_circuit(4, 1, "full", "zz").count_ops()["cx"] > ablation_circuit(4, 1, "linear", "zz").count_ops()["cx"]
    with pytest.raises(ValueError):
        ablation_circuit(8, 1, "linear", "zz")
