from pathlib import Path
import argparse
from quantum_fraud.quantum.ablation import run_ablations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 10 controlled kernel ablations")
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    print(run_ablations(args.data, Path("results/metrics")))
