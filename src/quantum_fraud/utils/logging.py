"""Logging scoped to this package without altering the application's root logger."""

import logging


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure one stream handler, including across repeated calls."""
    logger = logging.getLogger("quantum_fraud")
    logger.setLevel(level)
    if not any(getattr(handler, "_quantum_fraud", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler._quantum_fraud = True
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
