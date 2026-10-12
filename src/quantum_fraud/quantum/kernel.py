"""Exact statevector QSVC with separate state-construction/kernel timing."""

from contextvars import ContextVar
from time import perf_counter

import numpy as np
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

_timing = ContextVar("statevector_timing", default=None)


class TimedStatevector(Statevector):
    """Time the public statevector-construction hook used by the QML kernel."""

    def __init__(self, *args, **kwargs):
        start = perf_counter()
        super().__init__(*args, **kwargs)
        ledger = _timing.get()
        if ledger is not None:
            ledger["statevector_construction_seconds"] += perf_counter() - start
            ledger["statevectors_constructed"] += 1


class TimedKernel(FidelityStatevectorKernel):
    """Record per-evaluate wall time and its statevector-construction component."""

    def __init__(self, feature_map):
        super().__init__(feature_map=feature_map, statevector_type=TimedStatevector, shots=None, auto_clear_cache=True)
        self.timings = []

    def evaluate(self, x_vec, y_vec=None):
        ledger = {"statevector_construction_seconds": 0.0, "statevectors_constructed": 0}
        token = _timing.set(ledger)
        start = perf_counter()
        try:
            result = super().evaluate(x_vec, y_vec)
        finally:
            _timing.reset(token)
        ledger["kernel_total_seconds"] = perf_counter() - start
        ledger["kernel_other_seconds"] = ledger["kernel_total_seconds"] - ledger["statevector_construction_seconds"]
        self.timings.append(ledger)
        return result


def feature_map(dimension: int, name: str, repetitions: int = 1):
    """Create modern circuit-library functions, avoiding deprecated blueprint classes."""
    if dimension not in (2, 4, 6) or type(repetitions) is not int or repetitions < 1:
        raise ValueError("Supported qubit counts: 2, 4, 6; repetitions must be positive")
    if name == "z":
        return z_feature_map(dimension, reps=repetitions)
    if name == "zz":
        return zz_feature_map(dimension, reps=repetitions, entanglement="linear")
    raise ValueError("feature_map must be z or zz")


def build_qsvc(dimension: int, map_name: str, repetitions: int = 1):
    """Return an exact ideal-simulator QSVC; scores are SVM margins."""
    circuit = feature_map(dimension, map_name, repetitions)
    return QSVC(quantum_kernel=TimedKernel(circuit), C=1.0, probability=False), circuit


def circuit_metadata(circuit) -> dict:
    """Measure the constructed logical circuit, not a hardware-transpiled depth."""
    return {"qubits": circuit.num_qubits, "logical_depth": circuit.depth(), "gate_counts": dict(circuit.count_ops()), "parameters": circuit.num_parameters}
