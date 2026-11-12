#!/usr/bin/env python3
"""
HF-CS comparison with equal-budget hyperparameter tuning for every learner.

For each outer split (seed_*):
  * development set = train (60%) + validation (20%) = 291 patients; the test
    set (20%) and the MIMIC-IV external cohort are never seen during tuning;
  * every learner (DualTower-FT, DT-Transformer, XGBoost, random forest,
    logistic regression) gets the same Optuna budget (TPE sampler, --trials
    trials, default 40); each trial is scored by the mean AUROC of a
    stratified 3-fold cross-validation on the development set;
  * the best configuration is refitted on the whole development set and scored
    on the test set and on MIMIC-IV.
The scored inner fold is never used for training decisions: neural models
stop early (AUROC, patience 15, at most 100 epochs) on a stratified 20% of the
inner TRAINING folds, and the refit trains for the median best epoch of the
best trial's folds, with the same cosine schedule. XGBoost tunes the number of
trees directly (no early stopping). Class weighting (balanced or none) is a
tuned choice for every learner. APACHE IV is scored on the test set.

--variant runs an ablation of DualTower-FT with the same tuning protocol:
  mean_pool     mean pooling instead of the [CLS] token
  no_safeguard  runtime safeguard switched off (training and inference)
  no_site       hospital (site_id) removed from the inputs
  no_shock      shock-specific inputs removed (vasoactive, ventilation, MCS
                flags and cuff mean arterial pressure summaries)
--compare FULL_RUN VARIANT_RUN tests each model of the variant run against the
same model of the full run on the same patients (paired bootstrap).
DualTower-FT is compared with every other score by a paired, class-stratified
bootstrap of the AUROC difference (2,000 resamples).

The packs are already imputed and standardised with the outer training split
(preprocessing/eICU, branch hfcs), and categorical codes are fixed, so no
preprocessing is refitted inside the inner folds; that preprocessing is
label-free.

Usage (from src/train), one process per split so splits can run in parallel:
    python run_hfcs_tuned.py --packs .../PACKS/patient_stratified \
        --external .../MIMIC/data/hfcs/external --out ../../outputs/hfcs_tuned \
        --run-name tuned --splits seed_42 [--variant no_site --models dualtower_ft]
    python run_hfcs_tuned.py --summarise ../../outputs/hfcs_tuned/tuned
    python run_hfcs_tuned.py --compare ../../outputs/hfcs_tuned/tuned \
        ../../outputs/hfcs_tuned/abl_no_site
"""
import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from architectures.model_factory import build_model
from engine.trainers import NeuralTrainer
from run_hfcs_experiments import CAT_COLS, arrays, evaluate, load_split, loader
from utils.config_manager import ConfigManager
from utils.seed import set_all_seeds
from utils.statistical_tests import paired_bootstrap_auroc

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger("hfcs_tuned")
log.setLevel(logging.INFO)

MODELS = ["dualtower_ft", "dualtower", "xgboost", "random_forest", "logistic_regression"]
NEURAL = {"dualtower_ft", "dualtower"}
REFERENCE = "dualtower_ft"
MAX_EPOCHS, PATIENCE, INNER_FOLDS, ES_FRACTION = 100, 15, 3, 0.2

SHOCK_COLS = ["vasopressor_24h", "inotrope_24h", "n_vasoactive_24h", "mech_vent_24h",
              "mcs_24h", "noninvasivemean_mean", "noninvasivemean_std",
              "noninvasivemean_min", "noninvasivemean_max", "noninvasivemean_slope"]
VARIANTS = {
    "full": {},
    "mean_pool": {"pooling": "mean"},
    "no_safeguard": {"safeguard": False},
    "no_site": {"drop": ["site_id"]},
    "no_shock": {"drop": SHOCK_COLS},
}


# ------------------------------------------------------------ search spaces --
def neural_params(trial):
    return {
        "trans_d_model": trial.suggest_categorical("trans_d_model", [32, 64, 128, 256]),
        "trans_nhead": trial.suggest_categorical("trans_nhead", [2, 4]),
        "trans_num_layers": trial.suggest_int("trans_num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "fusion_hidden_dim": trial.suggest_categorical("fusion_hidden_dim", [32, 64, 128]),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "none"]),
    }


def xgb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "none"]),
    }


def rf_params(trial):
    unlimited = trial.suggest_categorical("unlimited_depth", [True, False])
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": None if unlimited else trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", "none"]),
    }


def lr_params(trial):
    return {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"]),
    }


SPACES = {"dualtower_ft": neural_params, "dualtower": neural_params, "xgboost": xgb_params,
          "random_forest": rf_params, "logistic_regression": lr_params}


# ---------------------------------------------------------------- learners --
def neural_config(cm, name, features, frames, seed):
    """Published configuration with the categorical inputs present in `features`.

    num_classes is raised to the observed cardinality (all sets, including the
    external cohort, so every code has an embedding row)."""
    cfg = copy.deepcopy(cm.load_config(name))
    published = sorted(cfg["model_config"]["architecture"]["cat_features"],
                       key=lambda c: int(c["index"]))
    cats = []
    for spec, col in zip(published, CAT_COLS):
        if col not in features:
            continue
        idx = features.index(col)
        if "site_id" in features and idx != int(spec["index"]):
            raise RuntimeError(f"{col} is at index {idx}, config expects {spec['index']}")
        observed = int(max(df[col].max() for df in frames.values())) + 1
        cats.append({"name": col, "index": idx,
                     "num_classes": max(int(spec["num_classes"]), observed),
                     "emb_dim": int(spec["emb_dim"])})
    cfg["model_config"]["architecture"]["cat_features"] = cats
    cfg.setdefault("common", {}).setdefault("reproducibility", {})["seed"] = seed
    return cfg


class Learner:
    """fit(X_df, y) / predict(X_df) wrapper with one interface for all models."""

    def __init__(self, name, params, cm, features, frames, seed, device, variant="full"):
        self.name, self.params, self.seed, self.device = name, dict(params), seed, device
        self.features, self.cm, self.frames = features, cm, frames
        self.variant = VARIANTS[variant]
        self.cat_cols = [c for c in CAT_COLS if c in features]
        self.best_epoch = None

    # neural ---------------------------------------------------------------
    def _neural_cfg(self, early_stopping):
        cfg = neural_config(self.cm, self.name, self.features, self.frames, self.seed)
        cfg["model_config"].setdefault("loss", {})["pos_weight"] = (
            "auto" if self.params["class_weight"] == "balanced" else 1.0)
        arch = cfg["model_config"]["architecture"]
        if "pooling" in self.variant:
            arch["pooling"] = self.variant["pooling"]
        for k in ("trans_d_model", "trans_nhead", "trans_num_layers", "dropout",
                  "fusion_hidden_dim"):
            arch[k] = self.params[k]
        cfg["model_config"]["optimizer"]["lr"] = self.params["lr"]
        cfg["model_config"]["optimizer"]["weight_decay"] = self.params["weight_decay"]
        cfg["model_config"].setdefault("train_loop", {})["batch_size"] = self.params["batch_size"]
        cfg["train"]["training"]["max_epochs"] = MAX_EPOCHS
        cfg["model_config"]["scheduler"].setdefault("cosine", {})["T_max"] = MAX_EPOCHS
        es = cfg["train"]["early_stopping"]
        es.update({"enabled": early_stopping, "metric": "val/auroc", "mode": "max",
                   "patience": PATIENCE, "restore_best_weights": True})
        cfg["common"].setdefault("promotion", {})["primary_metric"] = "auroc"
        return cfg

    def fit_neural(self, tr, va=None, epochs=None):
        # Splits share one GPU; on a transient out-of-memory error wait and
        # retry the same fit instead of dropping the configuration.
        for attempt in range(5):
            try:
                return self._fit_neural(tr, va, epochs)
            except torch.cuda.OutOfMemoryError:
                self.trainer = None
                torch.cuda.empty_cache()
                log.warning(f"{self.name}: CUDA out of memory, retry {attempt + 1}/5")
                time.sleep(60)
        return self._fit_neural(tr, va, epochs)

    def _fit_neural(self, tr, va=None, epochs=None):
        cfg = self._neural_cfg(early_stopping=va is not None)
        set_all_seeds(self.seed)
        bs = self.params["batch_size"]
        Xtr, ytr = arrays(tr, self.features)
        model = build_model(self.name, cfg, len(self.features))
        if not self.variant.get("safeguard", True):
            model.safeguard_enabled = False
        self.trainer = NeuralTrainer(model, cfg, self.device)
        val_df = va if va is not None else tr
        Xva, yva = arrays(val_df, self.features)
        self.trainer.fit(loader(Xtr, ytr, bs, self.seed, True),
                         loader(Xva, yva, bs, self.seed, False),
                         epochs=epochs if epochs else MAX_EPOCHS)
        self.best_epoch = self.trainer.best_epoch + 1
        return self

    # classical ------------------------------------------------------------
    def _classical(self, ytr):
        p = dict(self.params)
        if self.name == "xgboost":
            balanced = p.pop("class_weight") == "balanced"
            spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1)) if balanced else 1.0
            return XGBClassifier(**p, scale_pos_weight=spw, tree_method="hist",
                                 eval_metric="auc", n_jobs=4, random_state=self.seed,
                                 verbosity=0)
        if self.name == "random_forest":
            q = dict(p)
            q["class_weight"] = None if q["class_weight"] == "none" else q["class_weight"]
            return RandomForestClassifier(**q, n_jobs=4, random_state=self.seed)
        cat_idx = [self.features.index(c) for c in self.cat_cols]
        num_idx = [i for i in range(len(self.features)) if i not in cat_idx]
        return make_pipeline(
            ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx),
                               ("num", "passthrough", num_idx)]),
            LogisticRegression(C=p["C"], penalty=p["penalty"], solver="liblinear",
                               class_weight=None if p["class_weight"] == "none" else "balanced",
                               max_iter=5000, random_state=self.seed))

    def fit(self, tr, va=None, epochs=None):
        if self.name in NEURAL:
            return self.fit_neural(tr, va, epochs)
        X, y = arrays(tr, self.features)
        self.model = self._classical(y).fit(X, y)
        return self

    def predict(self, df):
        X, y = arrays(df, self.features)
        if self.name in NEURAL:
            bs = self.params["batch_size"]
            return self.trainer.predict(loader(X, y, bs, self.seed, False)).ravel()
        return self.model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------- tuning ----
def early_stopping_split(tr, seed):
    """Stratified ES_FRACTION of the inner training folds for early stopping."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=ES_FRACTION, random_state=seed)
    i_fit, i_es = next(sss.split(tr, tr["mortality"]))
    return tr.iloc[i_fit], tr.iloc[i_es]


def tune(name, dev, cm, features, frames, seed, device, n_trials, variant="full"):
    folds = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
                 .split(dev, dev["mortality"]))

    def objective(trial):
        params = SPACES[name](trial)
        aucs, epochs = [], []
        for k, (i_tr, i_va) in enumerate(folds):
            tr, va = dev.iloc[i_tr], dev.iloc[i_va]
            learner = Learner(name, params, cm, features, frames, seed, device, variant)
            if name in NEURAL:
                # the scored fold `va` is never seen during training
                fit, es = early_stopping_split(tr, seed + k)
                learner.fit(fit, es)
                epochs.append(learner.best_epoch)
            else:
                learner.fit(tr)
            aucs.append(roc_auc_score(va["mortality"], learner.predict(va)))
            trial.report(float(np.mean(aucs)), k)
        trial.set_user_attr("fold_auroc", [float(a) for a in aucs])
        if epochs:
            trial.set_user_attr("best_epochs", epochs)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def run_split(pack, external_root, out_root, models, n_trials, device, variant="full"):
    seed = int(pack.name.split("_")[1])
    cm = ConfigManager()
    frames, features, apache = load_split(pack, external_root)
    drop = VARIANTS[variant].get("drop", [])
    missing = [c for c in drop if c not in features]
    if missing:
        raise RuntimeError(f"variant {variant}: columns not found {missing}")
    features = [c for c in features if c not in drop]
    dev = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    sets = [s for s in ("test", "external") if s in frames]
    preds = {s: pd.DataFrame({"patient_id": frames[s]["patient_id"],
                              "mortality": frames[s]["mortality"]}) for s in sets}
    results, tuning = {}, {}
    out = out_root / pack.name
    out.mkdir(parents=True, exist_ok=True)

    for name in models:
        t0 = time.time()
        study = tune(name, dev, cm, features, frames, seed, device, n_trials, variant)
        best = study.best_trial
        params = SPACES[name](optuna.trial.FixedTrial(best.params))
        epochs = int(np.median(best.user_attrs["best_epochs"])) if name in NEURAL else None
        final = Learner(name, params, cm, features, frames, seed, device,
                        variant).fit(dev, epochs=epochs)
        results[name] = {}
        for s in sets:
            p = final.predict(frames[s])
            preds[s][name] = p
            results[name][s] = evaluate(frames[s]["mortality"].to_numpy(), p, 0.5, seed)
        tuning[name] = {"inner_cv_auroc": best.value, "best_params": best.params,
                        "fold_auroc": best.user_attrs["fold_auroc"],
                        "refit_epochs": epochs, "trials": len(study.trials),
                        "minutes": round((time.time() - t0) / 60, 1)}
        study.trials_dataframe().to_csv(out / f"optuna_{name}.csv", index=False)
        log.info(f"{pack.name} {name:20s} inner-CV AUROC {best.value:.3f}  "
                 + "  ".join(f"{s}: AUROC {results[name][s]['auroc']:.3f} "
                             f"AUPRC {results[name][s]['auprc']:.3f}" for s in sets)
                 + f"  ({tuning[name]['minutes']} min)")

    if apache is not None:
        a = preds["test"][["patient_id"]].merge(apache[["patient_id", "apache_iv_pred"]],
                                                on="patient_id", how="left")
        preds["test"]["apache_iv"] = a["apache_iv_pred"].to_numpy()
        ok = preds["test"]["apache_iv"].notna().to_numpy()
        results["apache_iv"] = {"test": evaluate(preds["test"]["mortality"].to_numpy()[ok],
                                                 preds["test"]["apache_iv"].to_numpy()[ok],
                                                 0.5, seed)}
        results["apache_iv"]["test"]["n_missing_score"] = int((~ok).sum())

    comparisons = {}
    if REFERENCE in models:
        for s in sets:
            df = preds[s]
            for other in [c for c in df.columns if c not in ("patient_id", "mortality", REFERENCE)]:
                m = df[other].notna().to_numpy()
                comparisons[f"{s}:{REFERENCE}_vs_{other}"] = paired_bootstrap_auroc(
                    df["mortality"].to_numpy()[m], df[REFERENCE].to_numpy()[m],
                    df[other].to_numpy()[m], n_bootstrap=2000, random_state=seed) | {
                    "n": int(m.sum())}

    for s in sets:
        preds[s].to_csv(out / f"predictions_{s}.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump({"seed": seed, "variant": variant, "n_features": len(features),
                   "n_trials": n_trials, "inner_folds": INNER_FOLDS,
                   "results": results, "tuning": tuning, "bootstrap": comparisons}, f, indent=2)


def summarise(run_dir):
    run_dir = Path(run_dir)
    rows, comps = [], []
    for f in sorted(run_dir.glob("seed_*/metrics.json")):
        d = json.load(open(f))
        for model, by_set in d["results"].items():
            for s, m in by_set.items():
                rows.append({"split": f.parent.name, "model": model, "set": s,
                             **{k: m.get(k) for k in ("auroc", "auprc", "brier", "ece")}})
        for key, c in d["bootstrap"].items():
            s, pair = key.split(":")
            comps.append({"split": f.parent.name, "set": s, "comparison": pair, **c})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics_long.csv", index=False)
    agg = df.groupby(["set", "model"])[["auroc", "auprc", "brier", "ece"]].agg(["mean", "std"]).round(4)
    agg.to_csv(run_dir / "metrics_summary.csv")
    c = pd.DataFrame(comps)
    c.to_csv(run_dir / "bootstrap_long.csv", index=False)
    cs = c.groupby(["set", "comparison"]).agg(
        mean_diff=("auroc_diff", "mean"), significant=("p_value", lambda p: int((p < 0.05).sum())),
        splits=("p_value", "size")).round(4)
    cs.to_csv(run_dir / "bootstrap_summary.csv")
    pd.set_option("display.width", 160)
    print(agg.to_string())
    print()
    print(cs.to_string())


def compare(full_dir, variant_dir):
    """Paired bootstrap of each variant model against the same model of the full run."""
    full_dir, variant_dir = Path(full_dir), Path(variant_dir)
    rows = []
    for vf in sorted(variant_dir.glob("seed_*/predictions_*.csv")):
        split, s = vf.parent.name, vf.stem.replace("predictions_", "")
        v, f = pd.read_csv(vf), pd.read_csv(full_dir / split / vf.name)
        m = f.merge(v, on=["patient_id", "mortality"], suffixes=("_full", "_var"))
        if len(m) != len(v):
            raise RuntimeError(f"{split}/{s}: patients differ between runs")
        seed = int(split.split("_")[1])
        for model in [c for c in v.columns if c in MODELS]:
            y = m["mortality"].to_numpy()
            r = paired_bootstrap_auroc(y, m[f"{model}_var"].to_numpy(),
                                       m[f"{model}_full"].to_numpy(),
                                       n_bootstrap=2000, random_state=seed)
            rows.append({"split": split, "set": s, "model": model,
                         "auroc_full": roc_auc_score(y, m[f"{model}_full"]),
                         "auroc_variant": roc_auc_score(y, m[f"{model}_var"]), **r})
    df = pd.DataFrame(rows)
    df.to_csv(variant_dir / "compare_long.csv", index=False)
    agg = df.groupby(["set", "model"]).agg(
        auroc_full=("auroc_full", "mean"), auroc_variant=("auroc_variant", "mean"),
        sd_variant=("auroc_variant", "std"), mean_diff=("auroc_diff", "mean"),
        significant=("p_value", lambda p: int((p < 0.05).sum())),
        splits=("p_value", "size")).round(4)
    agg.to_csv(variant_dir / "compare_summary.csv")
    print(agg.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs")
    ap.add_argument("--external", default=None)
    ap.add_argument("--out", default="../../outputs/hfcs_tuned")
    ap.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--splits", default=None)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    ap.add_argument("--summarise", default=None, help="run directory to summarise")
    ap.add_argument("--compare", nargs=2, metavar=("FULL_RUN", "VARIANT_RUN"), default=None)
    args = ap.parse_args()
    if args.summarise:
        summarise(args.summarise)
        return
    if args.compare:
        compare(*args.compare)
        return

    device = torch.device(args.device)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    packs = sorted(Path(args.packs).glob("seed_*"))
    if args.splits:
        packs = [p for p in packs if p.name in set(args.splits.split(","))]
    out_root = Path(args.out) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / f"run_config_{'_'.join(p.name for p in packs)}.json", "w") as f:
        json.dump(vars(args) | {"models": models, "torch": torch.__version__,
                                "optuna": optuna.__version__}, f, indent=2)
    external_root = Path(args.external) if args.external else None
    for pack in packs:
        run_split(pack, external_root, out_root, models, args.trials, device, args.variant)


if __name__ == "__main__":
    main()
