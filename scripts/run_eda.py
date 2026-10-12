"""Run Phase 2 full-file EDA and save aggregate tables and six figures."""

import argparse
from pathlib import Path

from quantum_fraud.data.eda import run_eda
from quantum_fraud.utils.config import load_config
from quantum_fraud.utils.logging import configure_logging


def main() -> int:
    """Use the configured source and output root; never write to raw data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/base.yaml")
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    logger = configure_logging()
    try:
        config = load_config(args.config)
        configure_logging(config.log_level)
        report = run_eda(args.data or config.data_path, config.results_dir, config.seed)
        logger.info("Completed EDA: %s", report)
        print(report)
        return 0
    except (ValueError, OSError, MemoryError) as exc:
        logger.error("EDA failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
