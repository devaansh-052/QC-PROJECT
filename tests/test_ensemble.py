import numpy as np
import pytest
from quantum_fraud.hybrid.ensemble import blend, validation_partition


def test_convex_probability_blend():
    a, b = np.array([.1, .8]), np.array([.7, .2])
    np.testing.assert_allclose(blend(a, b, .25), [.55, .35])
    np.testing.assert_equal(blend(a, b, 1), a)
    for invalid in (-.1, 1.1):
        with pytest.raises(ValueError):
            blend(a, b, invalid)
    with pytest.raises(ValueError):
        blend(a, [np.nan, 0], .5)


def test_calibration_selection_are_disjoint_and_reproducible():
    y = np.array([0] * 2035 + [1] * 13)
    calibration, selection = validation_partition(y)
    assert not np.intersect1d(calibration, selection).size
    np.testing.assert_array_equal(np.sort(np.r_[calibration, selection]), np.arange(len(y)))
    assert y[calibration].sum() > 0 and y[selection].sum() > 0
    np.testing.assert_array_equal(calibration, validation_partition(y)[0])
