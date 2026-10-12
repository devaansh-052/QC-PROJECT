"""Run Phase 8 paired hybrid/classical neural training on the fixed PaySim subset."""
import argparse
from pathlib import Path
from quantum_fraud.hybrid.train import run_hybrid, plot_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid.yaml"))
    args = parser.parse_args()
    output = run_hybrid(args.data, args.config, Path("results/metrics"))
    plot_training(output)
    print(output)
