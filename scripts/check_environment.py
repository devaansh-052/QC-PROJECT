"""Run the Phase 0 import/configuration/reproducibility smoke check."""

from pathlib import Path

import quantum_fraud
from quantum_fraud.utils.config import load_config
from quantum_fraud.utils.logging import configure_logging
from quantum_fraud.utils.seeds import seed_everything


def main() -> None:
    """Check installed package and report one deterministic random draw."""
    config = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")
    logger = configure_logging(config.log_level)
    rng = seed_everything(config.seed)
    logger.info("quantum_fraud=%s seed=%s smoke_draw=%s", quantum_fraud.__version__, config.seed, rng.random())


if __name__ == "__main__":
    main()
