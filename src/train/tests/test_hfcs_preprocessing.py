"""Tests for the fold-wise preprocessing used by run_hfcs_tuned.py.

The pack test needs the cleaned cohorts and packs of the two preprocessing
repositories (patient-level data, not public); it is skipped without them.
Point HFCS_PREPROCESSING at the folder that holds eICU/ and MIMIC/."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hfcs_preprocessing import FoldPreprocessor, load_raw_split  # noqa: E402

ROOT = Path(os.environ.get("HFCS_PREPROCESSING",
                           Path(__file__).resolve().parents[5] / "preprocessing"))
PACKS = ROOT / "eICU/train_ready_hfcs/PACKS/patient_stratified"
RAW_EICU = ROOT / "eICU/outputs/hfcs/obs_cleaned.csv"
RAW_MIMIC = ROOT / "MIMIC/data/hfcs/mimic_hfcs_cleaned.csv"
EXTERNAL = ROOT / "MIMIC/data/hfcs/external"


@pytest.mark.skipif(not RAW_EICU.exists(), reason="patient-level data not available")
@pytest.mark.parametrize("seed", ["seed_42", "seed_860", "seed_5191", "seed_5390", "seed_7270"])
def test_fit_on_outer_training_split_reproduces_the_packs(seed):
    raw, features, _ = load_raw_split(PACKS / seed, RAW_EICU, RAW_MIMIC)
    prep = FoldPreprocessor(features).fit(raw["train"])
    expected = {s: pd.read_csv(PACKS / seed / f"{s}.csv", dtype={"patient_id": str})
                for s in ("train", "val", "test")}
    expected["external"] = pd.read_csv(EXTERNAL / seed / "external_test.csv",
                                       dtype={"patient_id": str})
    for s, want in expected.items():
        got = prep.transform(raw[s])
        assert got.columns.tolist() == want.columns.tolist()
        assert (got["patient_id"] == want["patient_id"]).all()
        assert (got["mortality"] == want["mortality"]).all()
        np.testing.assert_allclose(got[features].to_numpy(float),
                                   want[features].to_numpy(float), rtol=0, atol=1e-9)


def test_single_patient_hospitals_train_the_unknown_code():
    df = pd.DataFrame({"patient_id": [str(i) for i in range(8)],
                       "mortality": [0, 1, 0, 1, 0, 1, 0, 1],
                       "age": np.arange(8, dtype=float),
                       "site_id": ["a", "a", "b", "b", "b", "c", "d", "e"]})
    prep = FoldPreprocessor(["age", "site_id"], min_counts={"site_id": 2}).fit(df)
    unk = list(prep.encoders["site_id"].classes_).index("<UNK>")
    got = prep.transform(df)["site_id"].to_numpy()
    # c, d and e are seen once: their patients carry the unknown code in training
    assert (got[5:] == unk).all() and (got[:5] != unk).all()
    new = df.iloc[:1].assign(site_id="z")
    assert prep.transform(new)["site_id"].iloc[0] == unk
    # without pooling, no fitting patient carries the unknown code
    plain = FoldPreprocessor(["age", "site_id"]).fit(df)
    unk_plain = list(plain.encoders["site_id"].classes_).index("<UNK>")
    assert (plain.transform(df)["site_id"] != unk_plain).all()


def test_statistics_come_only_from_the_fitting_patients():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"patient_id": [str(i) for i in range(40)],
                       "mortality": rng.integers(0, 2, 40),
                       "age": rng.normal(65, 10, 40),
                       "site_id": [str(i % 4) for i in range(40)],
                       "lactate": rng.normal(3, 1, 40)})
    df.loc[::5, "lactate"] = np.nan
    fit, scored = df.iloc[:30], df.iloc[30:].copy()
    a = FoldPreprocessor(["age", "site_id", "lactate"]).fit(fit)
    scored["lactate"], scored["site_id"] = 1e6, "99"
    b = FoldPreprocessor(["age", "site_id", "lactate"]).fit(pd.concat([fit, scored]))
    assert a.imputer.statistics_[1] == fit["lactate"].median()
    assert not np.isclose(a.scaler.mean_[1], b.scaler.mean_[1])
    got = a.transform(scored)
    assert (got["site_id"] == list(a.encoders["site_id"].classes_).index("<UNK>")).all()
    pd.testing.assert_frame_equal(a.transform(fit), FoldPreprocessor(
        ["age", "site_id", "lactate"]).fit(fit).transform(fit))
