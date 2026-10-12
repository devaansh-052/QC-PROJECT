"""Behavioral tests for leakage boundaries and saved transformations."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from quantum_fraud.preprocessing.pipeline import balanced_class_weights, build_preprocessor, temporal_split, undersample_training
from quantum_fraud.preprocessing.prepare import load_preprocessing_config, prepare_partitions
from quantum_fraud.data.paysim import REQUIRED_COLUMNS


def test_whole_step_split_is_disjoint_ordered_and_complete():
    steps = np.repeat(np.arange(1, 21), 2)
    labels = np.tile([0, 1], 20)
    permutation = np.random.default_rng(42).permutation(len(steps))
    steps, labels = steps[permutation], labels[permutation]
    splits = temporal_split(steps, labels)
    assert sorted(np.concatenate(list(splits.values()))) == list(range(40))
    assert steps[splits["train"]].max() == 14
    assert steps[splits["validation"]].min() == 15
    assert steps[splits["validation"]].max() == 17
    assert steps[splits["test"]].min() == 18


def test_single_class_tail_rejected_without_rebalancing():
    steps = np.arange(1, 21)
    labels = np.tile([0, 1], 10)
    labels[17:] = 1
    with pytest.raises(ValueError, match="test partition lacks both classes"):
        temporal_split(steps, labels)


@pytest.mark.parametrize("steps", [[1, np.nan, 3, 4], [0, 1, 2, 3], [1, 2, 2.5, 3]])
def test_invalid_time_is_not_imputed(steps):
    with pytest.raises(ValueError):
        temporal_split(np.array(steps), np.array([0, 1, 0, 1]))


def test_scaling_imputation_vocabulary_and_selection_use_train_only():
    train = pd.DataFrame({"step": [1., 2., 3., 4.], "amount": [0., 3., np.nan, 15.], "type": ["A", "B", None, "A"]})
    pipeline = build_preprocessor().fit(train)
    numeric = pipeline.named_steps["columns"].named_transformers_["numeric"]
    assert numeric.named_steps["imputer"].statistics_[1] == pytest.approx(np.log(4))
    mean = numeric.named_steps["scaler"].mean_.copy()
    train_output = pipeline.transform(train)
    np.testing.assert_allclose(train_output[:, :2].mean(axis=0), [0, 0], atol=1e-12)
    future = pd.DataFrame({"step": [10000.], "amount": [1e12], "type": ["UNSEEN"], "isFraud": [1], "newbalanceOrig": [1e15]})
    output = pipeline.transform(future)
    assert np.isfinite(output).all()
    assert output.shape[1] == train_output.shape[1]
    np.testing.assert_array_equal(mean, numeric.named_steps["scaler"].mean_)
    assert not any("UNSEEN" in name or "balance" in name or "isFraud" in name for name in pipeline.get_feature_names_out())
    np.testing.assert_allclose(output, pipeline.transform(future.drop(columns=["isFraud", "newbalanceOrig"])))


def test_weights_and_sampling_use_only_given_training_rows():
    labels = np.array([0, 0, 0, 0, 1, 1])
    weights = balanced_class_weights(labels)
    assert weights == {0: .75, 1: 1.5}
    selected = undersample_training(np.arange(6), labels, 42)
    assert len(selected) == 4 and labels[selected].sum() == 2
    assert set(selected).issubset(range(6))
    np.testing.assert_array_equal(selected, undersample_training(np.arange(6), labels, 42))


def test_missing_dimensions_and_invalid_labels_fail():
    with pytest.raises(ValueError, match="Missing eligible"):
        build_preprocessor().fit(pd.DataFrame({"step": [1]}))
    with pytest.raises(ValueError, match="binary"):
        balanced_class_weights(np.array([0, 2]))


@pytest.mark.parametrize("sampling", ["none", "random_undersample"])
def test_saved_partitions_and_pipeline_round_trip(tmp_path, sampling):
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = pd.DataFrame([
        [step, "TRANSFER" if label else "PAYMENT", float(step * 10), f"C{step}{label}", 10., 0., "M1", 0., 0., label, 0]
        for step in range(1, 21) for label in (0, 1)
    ], columns=REQUIRED_COLUMNS)
    source = raw / "fixture.csv"
    frame.to_csv(source, index=False)
    original = source.read_bytes()
    path = prepare_partitions(source, tmp_path / "processed", {"train_fraction": .7, "validation_fraction": .15, "sampling": sampling})
    report = json.loads(path.read_text())
    assert report["status"] == "complete"
    pipeline = joblib.load(path.parent / "preprocessor.joblib")
    numeric = pipeline.named_steps["columns"].named_transformers_["numeric"]
    assert numeric.named_steps["scaler"].mean_[0] == pytest.approx(7.5)
    all_positions = []
    for name in ("train", "validation", "test"):
        indices = np.load(path.parent / f"{name}_positions.npy")
        all_positions.extend(indices)
        actual = np.load(path.parent / f"{name}_X.npy")
        np.testing.assert_allclose(actual, pipeline.transform(frame.iloc[indices]))
        np.testing.assert_array_equal(np.load(path.parent / f"{name}_y.npy"), frame.isFraud.iloc[indices])
    assert sorted(all_positions) == list(range(len(frame)))
    assert source.read_bytes() == original


@pytest.mark.parametrize("text", ["[]", "train_fraction: 0.9\nvalidation_fraction: 0.2\nsampling: none", "train_fraction: 0.7\nvalidation_fraction: 0.15\nsampling: smote", "train_fraction: true\nvalidation_fraction: 0.15\nsampling: none"])
def test_invalid_preprocessing_config(tmp_path, text):
    path = tmp_path / "preprocessing.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_preprocessing_config(path)


def test_preprocessing_output_cannot_be_raw(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        prepare_partitions(tmp_path / "raw/input.csv", tmp_path / "raw/processed", {"sampling": "none"})
