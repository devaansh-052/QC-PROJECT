from pathlib import Path
from quantum_fraud.hybrid.ensemble import run_ensemble
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9 validation-only ensemble selection")
    parser.add_argument("--qsvc-run", type=Path, required=True)
    parser.add_argument("--vqc-run", type=Path, required=True)
    args = parser.parse_args()
    print(run_ensemble(args.qsvc_run, args.vqc_run, Path("results/metrics")))
