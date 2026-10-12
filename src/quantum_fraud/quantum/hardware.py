"""Small, explicit IBM submission workflow; preparation never contacts hardware."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from qiskit import qpy, transpile
from qiskit_machine_learning.algorithms import VQC

from quantum_fraud.evaluation.metrics import binary_metrics
from quantum_fraud.quantum.data import load_quantum_arrays
from quantum_fraud.quantum.noise import bound_vqc_circuits, exact_probabilities, measurement_probability


def write_new(path, content):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, default=str, allow_nan=False)


def runtime_service():
    token, instance = os.getenv("IBM_QUANTUM_API_KEY"), os.getenv("IBM_QUANTUM_INSTANCE")
    if not token or not instance:
        raise ValueError("Set IBM_QUANTUM_API_KEY and IBM_QUANTUM_INSTANCE locally; credentials must not be sent in chat or stored in results")
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)


def prepare_hardware(data: Path, vqc_run: Path, root: Path):
    report = json.loads((vqc_run / "results.json").read_text())
    if Path(report["subset_manifest"]).resolve() != (data / "manifest.json").resolve():
        raise ValueError("Mismatched source data")
    arrays = load_quantum_arrays(data, 2)
    X, y = arrays["test"]
    positions = np.sort(np.r_[np.flatnonzero(y == 0)[:4], np.flatnonzero(y == 1)[:4]])
    if len(positions) != 8:
        raise ValueError("Need four examples of each class")
    model = VQC.from_dill(str(vqc_run / "vqc_linear_r1_2q.model"))
    circuits = bound_vqc_circuits(model, X[positions])
    ideal = exact_probabilities(circuits, model.neural_network.interpret)
    for circuit in circuits:
        circuit.measure_all()
    output = root / (datetime.now(timezone.utc).strftime("hardware_%Y%m%dT%H%M%S_") + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "circuits.qpy").open("wb") as stream:
        qpy.dump(circuits, stream)
    local_compiled = transpile(circuits, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=42)
    source_record = next(r for r in report["models"] if r["model"] == "vqc_linear_r1_2q")
    threshold = source_record["validation"]["threshold"]
    request = {"run_id": output.name, "status": "hardware_execution_pending", "hardware_executed": False,
               "sdk_version": version("qiskit-ibm-runtime"), "logical_qubits": 2, "shots": 128,
               "source_model": str((vqc_run / "vqc_linear_r1_2q.model").resolve()),
               "subset_manifest": str((data / "manifest.json").resolve()),
               "test_subset_positions": positions.tolist(), "csv_positions": np.load(data / "test_positions.npy")[positions].tolist(),
               "labels": y[positions].tolist(), "threshold": threshold,
               "ideal_reference_probabilities": ideal.tolist(), "ideal_reference_metrics": binary_metrics(y[positions], ideal, threshold),
               "local_compilation_depths": [c.depth() for c in local_compiled],
               "limitations": ["Eight deliberately balanced diagnostic rows, not prevalence-representative evaluation",
                               "Local compilation is not physical hardware execution", "Device noise and calibration drift may change results"]}
    write_new(output / "request.json", request)
    return output


def submit_hardware(bundle: Path, backend_name: str):
    if not backend_name:
        raise ValueError("Explicit IBM backend name required")
    if (bundle / "submission.json").exists() or (bundle / "submission.lock").exists():
        raise FileExistsError("Bundle already submitted or submission is uncertain; do not duplicate a job")
    request = json.loads((bundle / "request.json").read_text())
    service = runtime_service()
    backend = service.backend(backend_name)
    if backend.configuration().simulator:
        raise ValueError("Hardware submission requires a physical backend")
    with (bundle / "circuits.qpy").open("rb") as stream:
        circuits = qpy.load(stream)
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    compiled = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=42).run(circuits)
    # Retain this reservation on an uncertain network failure: manual recovery
    # must check IBM's jobs view before any resubmission.
    write_new(bundle / "submission.lock", {"started_utc": datetime.now(timezone.utc).isoformat(), "backend": backend_name})
    sampler = SamplerV2(mode=backend, options={"max_execution_time": 120})
    job = sampler.run(compiled, shots=request["shots"])
    submission = {"job_id": job.job_id(), "backend": backend_name, "logical_qubits": 2,
                  "shots": request["shots"], "compiled_depths": [c.depth() for c in compiled],
                  "compiled_widths": [c.num_qubits for c in compiled], "submitted_utc": datetime.now(timezone.utc).isoformat()}
    write_new(bundle / "submission.json", submission)
    with (bundle / "hardware_circuits.qpy").open("xb") as stream:
        qpy.dump(compiled, stream)
    return submission


def collect_hardware(bundle: Path):
    submission = json.loads((bundle / "submission.json").read_text())
    if (bundle / "hardware_result.json").exists():
        return json.loads((bundle / "hardware_result.json").read_text())
    job = runtime_service().job(submission["job_id"])
    status = str(job.status()).upper()
    if status not in ("DONE", "COMPLETED"):
        return {"job_id": submission["job_id"], "status": status, "hardware_metrics_available": False}
    request = json.loads((bundle / "request.json").read_text())
    result = job.result()
    counts = [pub.data.meas.get_counts() for pub in result]
    if len(counts) != len(request["labels"]) or any(sum(c.values()) != request["shots"] for c in counts):
        raise ValueError("Hardware result shape or shot count does not match request")
    scores = [measurement_probability(c) for c in counts]
    report = {"status": "complete", "hardware_executed": True, **submission,
              "counts": counts, "scores": scores, "job_metrics": job.metrics(),
              "primitive_metadata": result.metadata,
              "metrics": binary_metrics(request["labels"], scores, request["threshold"]),
              "limitations": request["limitations"]}
    write_new(bundle / "hardware_result.json", report)
    return report
