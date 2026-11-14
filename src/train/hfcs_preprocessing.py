"""
Fold-wise preprocessing for the HF-CS runs.

The eICU packs (preprocessing/eICU, branch hfcs) hold features that were
imputed and standardised once per outer split, with statistics from the 218
training patients. Inner cross-validation folds mix those patients with
validation patients, so a scored inner fold would contain patients whose
values set the medians, means and standard deviations. This module refits the
same transforms on any set of patients instead:

  * categorical codes: LabelEncoder on the fitting patients plus a reserved
    unknown code; values not seen in them map to that code;
  * numerical features: median imputation, then standardisation.

min_counts pools rare categories into the unknown code: a value seen in fewer
than min_counts[col] fitting patients is coded as unknown, for the fitting
patients too. With {"site_id": 2}, hospitals seen only once are treated like
unseen hospitals, so the unknown code is learned from real patients instead of
staying untrained (no fitting patient would carry it otherwise).

It reproduces _dual_tower_preprocess (preprocessing/eICU,
run_obs_trainReady.py) and apply_eicu_transforms.py (preprocessing/MIMIC,
hfcs/): fitted on an outer training split, it gives back that split's packs
(tests/test_hfcs_preprocessing.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

CAT_COLS = ["gender", "ethnicity", "site_id"]
UNKNOWN_CATEGORY = "<UNK>"


def load_raw_split(pack_dir: Path, raw_eicu: Path, raw_mimic: Path = None):
    """Untransformed frames for one outer split, in the packs' row order.

    Patients and column layout come from the pack (train/val/test.csv); the
    values come from the cleaned cohorts, before any imputation or scaling."""
    eicu = pd.read_csv(raw_eicu)
    eicu["patient_id"] = eicu["uniquepid"].astype(str)
    eicu = eicu.set_index("patient_id")
    frames, layout = {}, None
    for s in ("train", "val", "test"):
        pack = pd.read_csv(pack_dir / f"{s}.csv", usecols=["patient_id"], dtype=str)
        frames[s] = eicu.loc[pack["patient_id"]].reset_index()
        if layout is None:
            layout = pd.read_csv(pack_dir / f"{s}.csv", nrows=0).columns.tolist()
    if raw_mimic is not None:
        mimic = pd.read_csv(raw_mimic)
        mimic["patient_id"] = mimic["uniquepid"].astype(str)
        frames["external"] = mimic
    features = [c for c in layout if c not in ("patient_id", "mortality")]
    apache = None
    if (pack_dir / "baseline_test.csv").exists():
        apache = pd.read_csv(pack_dir / "baseline_test.csv")
    return frames, features, apache


class FoldPreprocessor:
    """Category codes, median imputation and standardisation fitted on one
    set of patients and applied unchanged to any other."""

    def __init__(self, features, min_counts=None):
        self.features = list(features)
        self.cat_cols = [c for c in CAT_COLS if c in self.features]
        self.num_cols = [c for c in self.features if c not in self.cat_cols]
        self.min_counts = dict(min_counts or {})

    @staticmethod
    def _categories(df, col):
        values = df[col] if col in df else pd.Series(index=df.index, dtype=object)
        return values.fillna("Unknown").astype(str)

    def fit(self, raw: pd.DataFrame):
        self.encoders, self.known = {}, {}
        for col in self.cat_cols:
            counts = self._categories(raw, col).value_counts()
            known = counts.index[counts >= self.min_counts.get(col, 1)].tolist()
            le = LabelEncoder()
            le.fit(known + [UNKNOWN_CATEGORY])
            self.encoders[col], self.known[col] = le, set(known)
        x_num = raw.reindex(columns=self.num_cols)
        empty = x_num.columns[x_num.isna().all()].tolist()
        if empty:
            raise RuntimeError(f"no observed values to fit on for {empty}")
        self.imputer = SimpleImputer(strategy="median").fit(x_num)
        self.scaler = StandardScaler().fit(self.imputer.transform(x_num))
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Frame with patient_id, mortality and the features in pack order."""
        ids = pd.DataFrame({"patient_id": raw["patient_id"].astype(str).to_numpy(),
                            "mortality": raw["mortality"].to_numpy()})
        x_cat = pd.DataFrame(index=ids.index)
        for col in self.cat_cols:
            enc = self.encoders[col]
            values = self._categories(raw, col)
            x_cat[col] = enc.transform(values.where(values.isin(self.known[col]),
                                                    UNKNOWN_CATEGORY))
        x_num = self.scaler.transform(self.imputer.transform(raw.reindex(columns=self.num_cols)))
        x_num = pd.DataFrame(x_num.astype(np.float64), columns=self.num_cols, index=ids.index)
        return pd.concat([ids, x_cat, x_num], axis=1)[["patient_id", "mortality"] + self.features]
