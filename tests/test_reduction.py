"""Feature selection arithmetic and train-only estimation contracts."""

import joblib
import json
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import yaml

from quantum_fraud.features.reduction import ColumnSelection, PCAProjection, estimate_mutual_information, ranked_indices
from quantum_fraud.features.experiment import run_feature_experiments


def test_pca_projection_matches_fitted_pca_without_learning_future_values(tmp_path):
    train = np.random.default_rng(42).normal(size=(30, 7))
    pca = PCA(n_components=6).fit(train)
    original_mean = pca.mean_.copy()
    future = np.ones((5, 7)) * 1000
    projection = PCAProjection(pca, 2)
    np.testing.assert_allclose(projection.transform(future), pca.transform(future)[:, :2])
    np.testing.assert_array_equal(pca.mean_, original_mean)
    joblib.dump(projection, tmp_path / 'pca.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path / 'pca.joblib').transform(future), projection.transform(future))


def test_ranked_ties_and_dimensions():
    np.testing.assert_array_equal(ranked_indices(np.array([.5, .5, .1]), 2), [0, 1])
    with pytest.raises(ValueError):
        ranked_indices(np.ones(7), 8)


def test_projection_shape_values_and_reload(tmp_path):
    X = np.arange(30).reshape(10, 3)
    selector = ColumnSelection(np.array([2, 0]), 3).fit(X)
    np.testing.assert_array_equal(selector.transform(X), X[:, [2, 0]])
    path = tmp_path / "selector.joblib"
    joblib.dump(selector, path)
    np.testing.assert_array_equal(joblib.load(path).transform(X), X[:, [2, 0]])
    with pytest.raises(ValueError):
        selector.transform(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        ColumnSelection([1, 1], 3).fit(X)


def test_mi_subset_is_reproducible_and_retains_classes():
    labels = np.array([0] * 90 + [1] * 10)
    X = np.column_stack([np.arange(100, dtype=float), labels])
    scores, positions = estimate_mutual_information(X, labels, [False, True], 50, 42)
    again, positions_again = estimate_mutual_information(X, labels, [False, True], 50, 42)
    assert len(positions) == 50 and labels[positions].sum() == 5
    assert len(np.unique(positions)) == 50
    np.testing.assert_array_equal(positions, positions_again)
    np.testing.assert_allclose(scores, again)
    assert scores[1] > 0


def test_feature_grid_artifacts_and_explicit_unsupported_dimension(tmp_path):
    """Handwritten-size software fixture; all grid methods must execute."""
    prepared, classical = tmp_path/'prepared', tmp_path/'classical'
    prepared.mkdir()
    classical.mkdir()
    rng = np.random.default_rng(42)
    X = np.column_stack([rng.normal(size=(100, 2)), np.eye(5)[np.arange(100) % 5]])
    y = np.tile([0, 1], 50)
    for split in ('train', 'validation', 'test'):
        np.save(prepared/f'{split}_X.npy', X)
        np.save(prepared/f'{split}_y.npy', y)
    names = ['numeric__step', 'numeric__log_amount'] + [f'categorical__type_{i}' for i in range(5)]
    manifest = {'status':'complete','seed':42,'effective_training_class_weights':{'0':1,'1':1},'feature_names':names,'dataset':{'kind':'software_fixture'}}
    (prepared/'manifest.json').write_text(json.dumps(manifest))
    (classical/'results.json').write_text(json.dumps({'status':'complete','preprocessing_manifest':str(prepared/'manifest.json')}))
    forest = RandomForestClassifier(n_estimators=3, random_state=42).fit(X,y)
    joblib.dump(forest, classical/'random_forest.joblib')
    config = tmp_path/'features.yaml'
    config.write_text(yaml.safe_dump({'dimensions':[2,4,8], 'mutual_information_sample_limit':50}))
    result_path = run_feature_experiments(prepared, classical, config, tmp_path/'results')
    report = json.loads(result_path.read_text())
    assert report['status'] == 'complete'
    assert report['unsupported_dimensions'] == [8]
    assert len(report['controls']) == 6
    assert report['mi_estimation_rows'] == 50
    for control in report['controls']:
        stem = f"{control['method']}_{control['dimension']}"
        saved = joblib.load(result_path.parent/f'{stem}.joblib')
        assert saved['reducer'].transform(X).shape == (100,control['dimension'])
