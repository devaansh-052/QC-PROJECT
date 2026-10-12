import json
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

from quantum_fraud.quantum.hardware import runtime_service, write_new, collect_hardware, submit_hardware


def test_missing_credentials_are_actionable_and_not_disclosed(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_API_KEY", raising=False)
    monkeypatch.delenv("IBM_QUANTUM_INSTANCE", raising=False)
    with pytest.raises(ValueError, match="Set IBM_QUANTUM_API_KEY"):
        runtime_service()


def test_installed_runtime_sdk_local_execution():
    backend = AerSimulator(max_parallel_threads=2)
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.measure_all()
    compiled = transpile(circuit, backend)
    result = SamplerV2(mode=backend).run([compiled], shots=32).result()
    assert result[0].data.meas.get_counts() == {"01": 32}


def test_immutable_submission_and_nonblocking_collection(tmp_path, monkeypatch):
    write_new(tmp_path / "submission.json", {"job_id": "test-only-job"})
    with pytest.raises(FileExistsError):
        write_new(tmp_path / "submission.json", {})
    with pytest.raises(FileExistsError):
        submit_hardware(tmp_path, "test-backend")
    fake_job = SimpleNamespace(status=lambda: "QUEUED")
    monkeypatch.setattr("quantum_fraud.quantum.hardware.runtime_service", lambda: SimpleNamespace(job=lambda _: fake_job))
    assert collect_hardware(tmp_path)["hardware_metrics_available"] is False
    assert not (tmp_path / "hardware_result.json").exists()
