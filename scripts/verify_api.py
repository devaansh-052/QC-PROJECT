"""Launch an isolated local HTTP service, exercise real models, then stop it."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from datetime import datetime, timezone
from uuid import uuid4

from quantum_fraud.inference.simulation import simulate
import requests


def main():
    output = Path("results/simulation") / (datetime.now(timezone.utc).strftime("api_check_%Y%m%dT%H%M%S_") + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = dict(os.environ, QUANTUM_FRAUD_DATABASE=str((output / "transactions.sqlite3").resolve()))
    url = f"http://127.0.0.1:{port}"
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "quantum_fraud.api.app:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port), "--no-access-log"],
                                   env=env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"API startup failed; see {output / 'server.log'}")
                try:
                    health = requests.get(url + "/health", timeout=2)
                    health.raise_for_status()
                    break
                except requests.RequestException:
                    if monotonic() >= deadline:
                        raise TimeoutError("API did not become ready")
                    sleep(.2)
            run = simulate(url, count=20, root=output)
            report = json.loads((run / "results.json").read_text())
            recent = requests.get(url + "/transactions", timeout=10).json()["transactions"]
            assert len(recent) == 20
            payload = {k: recent[0][k] for k in ("transaction_id", "step", "type", "amount")}
            assert requests.post(url + "/predict", json=payload, timeout=10).status_code == 409
            assert requests.post(url + "/predict", json={**payload, "isFraud": 1}, timeout=10).status_code == 422
            result = {"status": "complete", "health": health.json(), "transactions_verified": 20,
                      "simulation_results": str((run / "results.json").resolve()),
                      "http_roundtrip_ms_mean": report["http_roundtrip_ms_mean"],
                      "processing_ms_mean": report["processing_ms_mean"], "duplicate_status": 409, "leakage_field_status": 422}
            (output / "verification.json").write_text(json.dumps(result, indent=2))
            print(output, flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
