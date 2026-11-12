"""Tests for the DeLong AUROC comparison in utils.statistical_tests."""
import importlib.util
import os

import numpy as np
from sklearn.metrics import roc_auc_score

# Load the module by path: importing the utils package pulls in torch.
_path = os.path.join(os.path.dirname(__file__), "..", "utils", "statistical_tests.py")
_spec = importlib.util.spec_from_file_location("statistical_tests", _path)
_st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_st)
_placement_values, delong_test = _st._placement_values, _st.delong_test


def _brute_placements(pos, neg):
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return psi.mean(axis=1), psi.mean(axis=0)


def test_placements_match_pairwise_definition_with_ties():
    rng = np.random.default_rng(0)
    pos = np.round(rng.normal(1, 1, 37), 1)   # rounding creates ties
    neg = np.round(rng.normal(0, 1, 53), 1)
    v10, v01 = _placement_values(pos, neg)
    b10, b01 = _brute_placements(pos, neg)
    assert np.allclose(v10, b10) and np.allclose(v01, b01)
    y = np.r_[np.ones(37), np.zeros(53)]
    assert np.isclose(v10.mean(), roc_auc_score(y, np.r_[pos, neg]))


def test_variance_matches_brute_force_delong():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(60), np.zeros(140)]
    s1 = y * 1.0 + rng.normal(0, 1, 200)
    s2 = y * 0.4 + rng.normal(0, 1, 200)
    z, p, diff = delong_test(y, s1, s2)
    pos, neg = y == 1, y == 0
    a10, a01 = _brute_placements(s1[pos], s1[neg])
    b10, b01 = _brute_placements(s2[pos], s2[neg])
    se = np.sqrt(np.var(a10 - b10, ddof=1) / pos.sum() + np.var(a01 - b01, ddof=1) / neg.sum())
    assert np.isclose(diff, roc_auc_score(y, s1) - roc_auc_score(y, s2))
    assert np.isclose(z, diff / se)


def test_clearly_different_models_are_significant():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(300), np.zeros(700)]
    good = y * 1.5 + rng.normal(0, 1, 1000)
    noise = rng.normal(0, 1, 1000)
    _, p, diff = delong_test(y, good, noise)
    assert diff > 0.3 and p < 1e-6


def test_same_model_is_not_significant():
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(50), np.zeros(150)]
    s = y + rng.normal(0, 1, 200)
    z, p, diff = delong_test(y, s, s + 1e-12 * rng.normal(size=200))
    assert abs(diff) < 1e-6 and (np.isnan(p) or p > 0.5)
