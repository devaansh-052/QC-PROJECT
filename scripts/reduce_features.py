"""Phase 5: run the fixed feature-reduction grid with matched controls."""

import argparse
from pathlib import Path

from quantum_fraud.features.experiment import run_feature_experiments
from quantum_fraud.utils.logging import configure_logging


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--classical-run", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=root / "configs/features.yaml")
    args = parser.parse_args()
    configure_logging()
    print(run_feature_experiments(args.prepared, args.classical_run, args.config, root / "results/metrics"))


if __name__ == "__main__":
    main()
