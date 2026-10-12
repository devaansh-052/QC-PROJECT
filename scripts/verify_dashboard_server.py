"""Check the actual local Streamlit HTTP server and stop it afterward."""
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from datetime import datetime, timezone
from uuid import uuid4
import json
import requests


def main():
    output = Path("results/simulation") / (datetime.now(timezone.utc).strftime("dashboard_http_%Y%m%dT%H%M%S_") + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.headless=true", "--server.address=127.0.0.1", f"--server.port={port}", "--browser.gatherUsageStats=false"], stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Streamlit exited; see {output}")
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=2)
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    if monotonic() >= deadline:
                        raise TimeoutError("Streamlit startup timed out")
                    sleep(.2)
            index = requests.get(f"http://127.0.0.1:{port}/", timeout=10)
            assert index.status_code == 200 and "streamlit" in index.text.lower()
            (output / "verification.json").write_text(json.dumps({"status": "complete", "health": response.text, "index_http_status": index.status_code}))
            print(output)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
