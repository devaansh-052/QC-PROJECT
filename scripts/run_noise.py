from pathlib import Path
import argparse
from quantum_fraud.quantum.noise import run_noise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 11 matched noisy inference")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--vqc-run", type=Path, required=True)
    args = parser.parse_args()
    print(run_noise(args.data, args.vqc_run, Path("results/metrics")))
