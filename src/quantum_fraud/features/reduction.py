"""Fixed-column selectors and deterministic training-only MI estimation."""

import os

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

from quantum_fraud.preprocessing.pipeline import validate_labels


class PCAProjection(TransformerMixin, BaseEstimator):
    """Use the leading components of one already train-fitted PCA decomposition."""

    def __init__(self, pca, dimension: int):
        self.pca = pca
        self.dimension = dimension

    def transform(self, X):
        if not 1 <= self.dimension <= len(self.pca.components_) or np.ndim(X) != 2 or X.shape[1] != self.pca.n_features_in_:
            raise ValueError("Unsupported PCA projection dimension")
        return (X - self.pca.mean_) @ self.pca.components_[:self.dimension].T

    @property
    def explained_variance_ratio_(self):
        return self.pca.explained_variance_ratio_[:self.dimension]


class ColumnSelection(TransformerMixin, BaseEstimator):
    """Persist an explicit ranked selection without copying every source column."""

    def __init__(self, indices, input_dimension: int):
        self.indices = indices
        self.input_dimension = input_dimension

    def fit(self, X, y=None):
        if np.ndim(X) != 2 or X.shape[1] != self.input_dimension:
            raise ValueError("Input feature dimension mismatch")
        indices = np.asarray(self.indices)
        if indices.ndim != 1 or len(indices) == 0 or not np.issubdtype(indices.dtype, np.integer) or len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= self.input_dimension).any():
            raise ValueError("Selected columns must be unique in-range integer indices")
        self.n_features_in_ = self.input_dimension
        return self

    def transform(self, X):
        if np.ndim(X) != 2 or X.shape[1] != self.input_dimension:
            raise ValueError("Input feature dimension mismatch")
        return X[:, self.indices]


def ranked_indices(scores: np.ndarray, dimension: int) -> np.ndarray:
    """Rank descending; original column order breaks ties reproducibly."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or type(dimension) is not int or not 1 <= dimension <= len(scores):
        raise ValueError("Invalid scores or unsupported reduced dimension")
    return np.argsort(-scores, kind="stable")[:dimension]


def estimate_mutual_information(X, labels, discrete_mask, limit: int, seed: int):
    """Fit MI on a deterministic class-proportional subset of training rows only.

    Return sampled training-array positions so the estimate can be reproduced.
    The cap bounds nearest-neighbor estimator cost; no validation/test data is used.
    """
    labels = validate_labels(labels)
    if type(limit) is not int or limit < 2 or len(X) != len(labels):
        raise ValueError("MI sample limit must be >=2 and arrays aligned")
    if len(discrete_mask) != X.shape[1]:
        raise ValueError("Discrete feature mask does not match input dimension")
    if len(labels) <= limit:
        selected = np.arange(len(labels))
    else:
        rng = np.random.default_rng(seed)
        positives = np.flatnonzero(labels == 1)
        negatives = np.flatnonzero(labels == 0)
        n_positive = min(len(positives), max(1, int(round(limit * len(positives) / len(labels)))))
        n_negative = min(len(negatives), limit - n_positive)
        if n_negative == 0:
            n_positive -= 1
            n_negative = 1
        selected = np.sort(np.concatenate([rng.choice(positives, n_positive, replace=False), rng.choice(negatives, n_negative, replace=False)]))
    # Explicit resource cap avoids platform-dependent physical-core probing on
    # Windows installations without the legacy CPU discovery utility.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
    scores = mutual_info_classif(X[selected], labels[selected], discrete_features=np.asarray(discrete_mask, dtype=bool), random_state=seed)
    return scores, selected
