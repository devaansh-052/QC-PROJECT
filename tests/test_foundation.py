"""Phase 0 acceptance checks."""

import logging
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from quantum_fraud.utils.config import load_config
from quantum_fraud.utils.logging import configure_logging
from quantum_fraud.utils.seeds import seed_everything


def test_config_paths_independent_of_working_directory(tmp_path, monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "configs/base.yaml"
    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    assert config.seed == 42
    assert config.data_path == config_path.parent.parent / "data/PS_20174392719_1491204439457_log.csv"


@pytest.mark.parametrize("content", ["[]", "", "seed: 42", "!!python/object:bad {}"])
def test_invalid_config(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("key,value", [("seed", True), ("seed", -1), ("log_level", "TRACE"), ("data_path", ""), ("results_dir", 3)])
def test_invalid_config_values(tmp_path, key, value):
    raw = {"seed": 42, "log_level": "INFO", "data_path": "data/raw/paysim.csv", "results_dir": "results"}
    raw[key] = value
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_config(tmp_path):
    with pytest.raises(ValueError, match="Cannot load configuration"):
        load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        seed_everything(seed)


def test_reproducible_streams():
    rng = seed_everything(42)
    first = (random.random(), np.random.random(), rng.random())
    rng = seed_everything(42)
    assert first == (random.random(), np.random.random(), rng.random())
    assert seed_everything(43).random() != first[2]


def test_logging_idempotent():
    root_handlers = list(logging.getLogger().handlers)
    logger = configure_logging()
    count = len(logger.handlers)
    assert configure_logging("DEBUG") is logger
    assert len(logger.handlers) == count
    assert logging.getLogger().handlers == root_handlers
