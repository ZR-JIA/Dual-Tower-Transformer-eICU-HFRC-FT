"""Calibration intercept and slope on synthetic data (no patient data)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_hfcs_tuned import calibration_intercept_slope  # noqa: E402


def _synthetic(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(-1.0, 1.0, n)
    p = 1 / (1 + np.exp(-z))
    return rng.binomial(1, p), z, p


def test_perfect_calibration():
    y, _, p = _synthetic()
    a, b = calibration_intercept_slope(y, p)
    assert abs(a) < 0.05 and abs(b - 1) < 0.05


def test_too_extreme_predictions_give_slope_below_one():
    y, z, _ = _synthetic()
    a, b = calibration_intercept_slope(y, 1 / (1 + np.exp(-2 * z)))
    assert abs(b - 0.5) < 0.05


def test_overestimated_risk_gives_negative_intercept():
    y, z, _ = _synthetic()
    a, b = calibration_intercept_slope(y, 1 / (1 + np.exp(-(z + 1))))
    assert abs(a + 1) < 0.07 and abs(b - 1) < 0.05
