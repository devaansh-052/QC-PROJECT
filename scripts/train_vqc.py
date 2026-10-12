"""Train the Phase 7 prespecified VQC grid on a completed Phase 6 subset."""
import argparse
from pathlib import Path
from quantum_fraud.quantum.variational import run_vqc, plot_convergence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    output = run_vqc(args.data, Path("results/metrics"))
    plot_convergence(output)
    print(output)
