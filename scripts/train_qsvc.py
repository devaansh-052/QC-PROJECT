"""Phase 6 ideal-simulator QSVC and matched RBF SVM experiments."""

import argparse
from pathlib import Path

from quantum_fraud.quantum.experiment import run_qsvc
from quantum_fraud.utils.logging import configure_logging


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--feature-run", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=root / "configs/quantum.yaml")
    args = parser.parse_args()
    configure_logging()
    print(run_qsvc(args.prepared, args.feature_run, args.config, root / "results/metrics"))


if __name__ == "__main__":
    main()
