"""Explicit random seeds; deterministic results remain library/backend dependent."""

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and legacy NumPy; return a separately seeded modern generator.

    Pass the same seed explicitly to future estimators and quantum primitives.
    PYTHONHASHSEED must be set before starting Python; changing it here is ineffective.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
