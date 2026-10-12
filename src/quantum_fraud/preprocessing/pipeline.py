"""Leakage-aware features, whole-step partitions and training-only imbalance tools."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from quantum_fraud.data.policy import PRIMARY_FEATURE_COLUMNS


def validate_labels(labels: np.ndarray) -> np.ndarray:
    """Require binary, one-dimensional labels with both classes."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Labels must be a one-dimensional binary array containing both 0 and 1")
    return labels


def temporal_split(steps: np.ndarray, labels: np.ndarray, train_fraction: float = .7, validation_fraction: float = .15) -> dict[str, np.ndarray]:
    """Return original positional row indices, partitioned by distinct step rank.

    Boundaries depend only on step values. Labels only validate class coverage;
    they never move boundaries. Missing/invalid time cannot be safely imputed here.
    """
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than one")
    steps, labels = np.asarray(steps), validate_labels(labels)
    if steps.ndim != 1 or len(steps) != len(labels) or not np.issubdtype(steps.dtype, np.number):
        raise ValueError("steps must be a numeric vector aligned with labels")
    if not np.isfinite(steps).all() or (steps < 1).any() or (steps % 1 != 0).any():
        raise ValueError("Temporal splitting requires nonmissing positive integer steps")
    unique = np.unique(steps)
    train_end = int(len(unique) * train_fraction)
    validation_end = int(len(unique) * (train_fraction + validation_fraction))
    if not 0 < train_end < validation_end < len(unique):
        raise ValueError("Too few distinct steps for three nonempty partitions")
    first, second = unique[train_end - 1], unique[validation_end - 1]
    partitions = {
        "train": np.flatnonzero(steps <= first),
        "validation": np.flatnonzero((steps > first) & (steps <= second)),
        "test": np.flatnonzero(steps > second),
    }
    for name, positions in partitions.items():
        try:
            validate_labels(labels[positions])
        except ValueError as exc:
            raise ValueError(f"{name} partition lacks both classes; boundaries were not adjusted. Review the documented split protocol.") from exc
    return partitions


class EligibleFeatures(TransformerMixin, BaseEstimator):
    """Select the fixed allowlist and replace amount with log1p(amount).

    log1p is a deterministic feature definition, not a fitted scaling operation.
    All further learned transforms are inside the shared sklearn pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None):
        """Validate schema; there are no learned feature-generation parameters."""
        self.transform(X)
        self.n_features_in_ = len(PRIMARY_FEATURE_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ignore excluded/extra columns, validate eligible fields and preserve index."""
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("Expected a dataframe with unique named columns")
        missing = set(PRIMARY_FEATURE_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Missing eligible feature columns: {sorted(missing)}")
        result = X.loc[:, list(PRIMARY_FEATURE_COLUMNS)].copy()
        for name in ("step", "amount"):
            values = result[name]
            if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
                raise ValueError(f"{name} must be numeric")
            observed = values.dropna()
            if not np.isfinite(observed).all() or (observed < 0).any():
                raise ValueError(f"{name} must be finite and nonnegative when present")
        if not result["type"].dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError("type must contain strings or missing values")
        result["type"] = result["type"].astype(object).where(result["type"].notna(), np.nan)
        result["log_amount"] = np.log1p(result.pop("amount"))
        return result

    def get_feature_names_out(self, input_features=None):
        """Names after deterministic feature generation."""
        return np.array(["step", "type", "log_amount"], dtype=object)


def build_preprocessor() -> Pipeline:
    """Build an unfitted pipeline; callers must fit exclusively on training rows."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)),
    ])
    return Pipeline([
        ("eligible", EligibleFeatures()),
        ("columns", ColumnTransformer([
            ("numeric", numeric, ["step", "log_amount"]),
            ("categorical", categorical, ["type"]),
        ], remainder="drop", verbose_feature_names_out=True)),
        ("selector", VarianceThreshold(threshold=0.0)),
    ])


def balanced_class_weights(training_labels: np.ndarray) -> dict[int, float]:
    """Compute sklearn-compatible N/(2*N_class) weights from training labels."""
    labels = validate_labels(training_labels)
    return {label: len(labels) / (2 * int(np.sum(labels == label))) for label in (0, 1)}


def undersample_training(training_positions: np.ndarray, training_labels: np.ndarray, seed: int) -> np.ndarray:
    """Return a balanced subset of supplied training positions, with no replacement.

    Validation/test positions must never be passed here. Sorting preserves source
    row order; the caller records positions for exact reproduction.
    """
    labels = validate_labels(training_labels)
    positions = np.asarray(training_positions)
    if positions.ndim != 1 or len(positions) != len(labels) or len(np.unique(positions)) != len(positions):
        raise ValueError("Training positions must be unique and aligned with training labels")
    count = min(int(np.sum(labels == label)) for label in (0, 1))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(positions[labels == label], size=count, replace=False) for label in (0, 1)]))
