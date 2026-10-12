"""Small circuit smoke tests; no synthetic fixture is a research result."""

import numpy as np
import json
import joblib
import pytest
import yaml
from sklearn.decomposition import PCA
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from quantum_fraud.quantum.data import BoundedAngleScaler, sample_positions
from quantum_fraud.quantum.kernel import build_qsvc, feature_map
from quantum_fraud.quantum.experiment import run_qsvc


def test_subset_reproducibility_and_prevalence():
    labels = np.array([0]*900 + [1]*100)
    positions = sample_positions(labels, 100, 42)
    assert len(np.unique(positions)) == 100 and labels[positions].sum() == 10
    np.testing.assert_array_equal(positions, sample_positions(labels, 100, 42))
    assert labels[sample_positions(labels, 100, 42, balanced=True)].sum() == 50
    with pytest.raises(ValueError):
        sample_positions(labels, 400, 42, balanced=True)


def test_smooth_angles_preserve_moderate_out_of_range_differences():
    scaler = BoundedAngleScaler().fit(np.array([[-1.],[1.]]))
    before = scaler.standard_.mean_.copy()
    result = scaler.transform(np.array([[2.],[3.]]))
    assert 0 < result[0,0] < result[1,0] < 1
    np.testing.assert_array_equal(before, scaler.standard_.mean_)


@pytest.mark.parametrize("dimension", [2,4,6])
@pytest.mark.parametrize("name", ["z","zz"])
def test_kernel_is_symmetric_psd_and_has_unit_diagonal(dimension, name):
    model, circuit = build_qsvc(dimension, name)
    X = np.random.default_rng(42).uniform(-1,1,(4,dimension))
    kernel = model.quantum_kernel.evaluate(X)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(kernel), 1, atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-10
    assert circuit.num_qubits == dimension
    timing = model.quantum_kernel.timings[-1]
    assert timing["statevectors_constructed"] == 4
    assert 0 <= timing["statevector_construction_seconds"] <= timing["kernel_total_seconds"]


def test_qsvc_fit_predict_and_saved_reload(tmp_path):
    model, _ = build_qsvc(2,"zz")
    X = np.array([[-1,-1],[-.5,.2],[.2,-.4],[1,1.]])
    model.fit(X, np.array([0,0,1,1]))
    scores = model.decision_function(X)
    assert scores.shape == (4,) and np.isfinite(scores).all()
    model.to_dill(str(tmp_path / "qsvc.model"))
    restored = type(model).from_dill(str(tmp_path / "qsvc.model"))
    np.testing.assert_allclose(scores, restored.decision_function(X))
    with pytest.raises(ValueError):
        restored.decision_function(np.ones((2,3)))


def test_aer_compatible_bell_smoke():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0,1)
    circuit.measure_all()
    simulator = AerSimulator(max_parallel_threads=1)
    counts = simulator.run(transpile(circuit, simulator), shots=128, seed_simulator=42).result().get_counts()
    assert set(counts) == {"00","11"} and sum(counts.values()) == 128


def test_invalid_circuit_configuration():
    with pytest.raises(ValueError):
        feature_map(8,"zz")


def test_qsvc_runner_matches_subsets_and_fits_angles_on_train_only(tmp_path):
    prepared, features = tmp_path/'prepared', tmp_path/'features'
    prepared.mkdir(); features.mkdir()
    X = np.random.default_rng(42).normal(size=(20,7))
    labels = np.array([0]*16+[1]*4)
    for i, split in enumerate(('train','validation','test')):
        np.save(prepared/f'{split}_X.npy', X + i*100)
        np.save(prepared/f'{split}_y.npy', labels)
        np.save(prepared/f'{split}_positions.npy', np.arange(20)+i*20)
    (prepared/'manifest.json').write_text(json.dumps({'status':'complete','dataset':{'kind':'software_fixture'},'feature_names':[str(i) for i in range(7)]}))
    (features/'results.json').write_text(json.dumps({'status':'complete','prepared_manifest':str(prepared/'manifest.json')}))
    joblib.dump(PCA(n_components=6).fit(X), features/'pca_full.joblib')
    config={'seed':42,'train_per_class':2,'validation_rows':4,'test_rows':4,'dimensions':[2],'feature_maps':['z'],'repetitions':[1],'angle_range':[-1.,1.],'angle_encoding':'standard_tanh_v1'}
    config_path=tmp_path/'quantum.yaml'; config_path.write_text(yaml.safe_dump(config))
    result_path=run_qsvc(prepared,features,config_path,tmp_path/'results')
    report=json.loads(result_path.read_text())
    assert len(report['models'])==2
    data=result_path.parent/'data'
    indices=[np.load(data/f'{split}_positions.npy') for split in ('train','validation','test')]
    assert len(np.unique(np.concatenate(indices)))==12
    transform=joblib.load(data/'transform_2.joblib')
    training=np.load(data/'train_full_X.npy')
    np.testing.assert_allclose(transform['angle_scaler'].data_min_,transform['projection'].transform(training).min(axis=0))
    for split in ('train','validation','test'):
        angles=np.load(data/f'{split}_2q_X.npy')
        assert angles.shape==(4,2) and (np.abs(angles)<=1+1e-12).all()
