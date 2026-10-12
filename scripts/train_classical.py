"""Train Phase 4 baselines on an explicit completed preprocessing run."""

import argparse
from pathlib import Path

from quantum_fraud.classical.train import train_classical
from quantum_fraud.utils.logging import configure_logging


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=root / "configs/classical.yaml")
    args = parser.parse_args()
    configure_logging()
    print(train_classical(args.prepared, args.config, root / "results/metrics"))


if __name__ == "__main__":
    main()
