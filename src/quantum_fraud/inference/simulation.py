"""Deterministic synthetic demonstration traffic, not research evaluation data."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import requests


def transactions(count, seed=42, prefix="demo"):
    if not 1 <= count <= 10000:
        raise ValueError("count must be between 1 and 10000")
    rng = np.random.default_rng(seed)
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    return [{"transaction_id": f"{prefix}-{i:05d}", "step": int(rng.integers(1, 744)),
             "type": str(rng.choice(types)), "amount": round(float(rng.lognormal(9, 2)), 2)} for i in range(count)]


def simulate(base_url, count=20, seed=42, root=Path("results/simulation")):
    run_id = datetime.now(timezone.utc).strftime("simulation_%Y%m%dT%H%M%S_") + uuid4().hex[:8]
    output = root / run_id
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with requests.Session() as session:
        for payload in transactions(count, seed, run_id):
            start = perf_counter()
            response = session.post(base_url.rstrip("/") + "/predict", json=payload, timeout=30)
            response.raise_for_status()
            rows.append({**response.json(), "http_roundtrip_ms": (perf_counter() - start) * 1000})
    report = {"status": "complete", "run_id": run_id, "seed": seed, "count": count,
              "source": "synthetic_demo_generator_not_research_evidence", "transactions": rows,
              "http_roundtrip_ms_mean": float(np.mean([r["http_roundtrip_ms"] for r in rows])),
              "processing_ms_mean": float(np.mean([r["processing_ms"] for r in rows]))}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return output
