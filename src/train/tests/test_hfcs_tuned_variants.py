"""Tests for the ablation switches used by run_hfcs_tuned.py."""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from architectures.models.dualtower_ft import FTRightTower  # noqa: E402
from run_hfcs_tuned import SHOCK_COLS, VARIANTS, early_stopping_split  # noqa: E402


def test_mean_pooling_keeps_output_shape_and_differs_from_cls():
    torch.manual_seed(0)
    x = torch.randn(8, 12)
    cls = FTRightTower(12, 16, 2, 1, 0.0, pooling="cls").eval()
    mean = FTRightTower(12, 16, 2, 1, 0.0, pooling="mean").eval()
    mean.load_state_dict(cls.state_dict())
    a, b = cls(x), mean(x)
    assert a.shape == b.shape == (8, 16)
    assert not torch.allclose(a, b)


def test_early_stopping_split_is_stratified_and_disjoint():
    df = pd.DataFrame({"patient_id": np.arange(200),
                       "mortality": np.r_[np.ones(54), np.zeros(146)]})
    fit, es = early_stopping_split(df, seed=1)
    assert len(es) == 40 and len(fit) == 160
    assert set(fit.patient_id).isdisjoint(es.patient_id)
    assert abs(es.mortality.mean() - df.mortality.mean()) < 0.03


def test_variants_are_single_changes():
    assert VARIANTS["full"] == {}
    assert all(len(v) == 1 for k, v in VARIANTS.items() if k != "full")
    assert len(SHOCK_COLS) == len(set(SHOCK_COLS)) == 10
