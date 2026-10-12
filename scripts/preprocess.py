"""Phase 3: generate temporal partitions and a saved training-only pipeline."""

import argparse
from pathlib import Path

from quantum_fraud.preprocessing.prepare import load_preprocessing_config, prepare_partitions
from quantum_fraud.utils.config import load_config
from quantum_fraud.utils.logging import configure_logging


def main() -> int:
    """Execute the configured preprocessing workflow."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/base.yaml")
    parser.add_argument("--preprocessing-config", type=Path, default=root / "configs/preprocessing.yaml")
    args = parser.parse_args()
    logger = configure_logging()
    try:
        base = load_config(args.config)
        settings = load_preprocessing_config(args.preprocessing_config)
        print(prepare_partitions(base.data_path, root / "data/processed", settings, base.seed))
        return 0
    except (ValueError, OSError, MemoryError) as exc:
        logger.error("Preprocessing failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
