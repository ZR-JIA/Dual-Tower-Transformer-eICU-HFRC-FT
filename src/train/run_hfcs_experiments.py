#!/usr/bin/env python3
"""
HF-CS experiments: train on the eICU HF-CS packs, evaluate on each split's
eICU test set and on the MIMIC-IV external cohort.

The packs are already imputed, standardised (numeric) and label-encoded
(categorical) with transforms fitted on each split's eICU TRAINING data
(preprocessing/eICU and preprocessing/MIMIC, branch hfcs). This driver
therefore does NOT run TabularPreprocessor: doing so would re-scale the
categorical codes and clamp the z-scored age column to [0, 149] through the
substring rule in _validate_categoricals.

Models
    dualtower_ft    PRICAI 2026 configuration (config/models/dualtower_ft.yaml)
    dualtower       CCAI 2026 configuration (config/models/dualtower.yaml)
    xgboost, random_forest   repository configurations
    logistic_regression      sklearn, L2, one-hot categoricals
    apache_iv       eICU APACHE IV predicted mortality (no training; eICU only)

Changes to the published configurations, all forced by the new cohort:
    * cat_features keep their positions (indices 1-3) and embedding sizes;
      num_classes is raised to the observed cardinality where the published
      value is smaller, so no category is clamped into another.
    * pos_weight = "auto" (computed from each training split) instead of the
      stroke ratio 6.13; XGBoost scale_pos_weight likewise.
    * neural models train on --device (default: CUDA when available) with
      seeded, deterministic cuDNN (set_all_seeds); tree models and logistic
      regression run on CPU. Probabilities are reported uncalibrated for every
      model so that Brier/ECE compare like with like.

Usage (from src/train):
    python run_hfcs_experiments.py \
        --packs ../../../../preprocessing/eICU/train_ready_hfcs/PACKS/patient_stratified \
        --external ../../../../preprocessing/MIMIC/data/hfcs/external \
        --out ../../outputs/hfcs
"""
import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from architectures.model_factory import build_model
from engine.metrics import (compute_bootstrap_ci, compute_classification_metrics,
                            compute_ece, select_best_threshold_youden)
from engine.trainers import NeuralTrainer, TreeTrainer
from utils.config_manager import ConfigManager
from utils.seed import get_generator, seed_worker, set_all_seeds
from utils.statistical_tests import delong_test

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hfcs")
log.setLevel(logging.INFO)

CAT_COLS = ["gender", "ethnicity", "site_id"]
NEURAL = ["dualtower_ft", "dualtower"]
TREES = ["xgboost", "random_forest"]
ALL_MODELS = NEURAL + TREES + ["logistic_regression"]
REFERENCE = "dualtower_ft"
CI_METRICS = ["auroc", "auprc", "brier"]


# ----------------------------------------------------------------- data ----
def load_split(pack_dir: Path, external_root: Path):
    frames = {s: pd.read_csv(pack_dir / f"{s}.csv") for s in ("train", "val", "test")}
    ext_path = external_root / pack_dir.name / "external_test.csv" if external_root else None
    if ext_path is not None and ext_path.exists():
        frames["external"] = pd.read_csv(ext_path)
    features = [c for c in frames["train"].columns if c not in ("patient_id", "mortality")]
    for name, df in frames.items():
        if [c for c in df.columns if c not in ("patient_id", "mortality")] != features:
            raise RuntimeError(f"{pack_dir.name}/{name}: feature columns differ from train")
    apache = None
    if (pack_dir / "baseline_test.csv").exists():
        apache = pd.read_csv(pack_dir / "baseline_test.csv")
    return frames, features, apache


def arrays(df, features):
    return df[features].to_numpy(dtype=np.float32), df["mortality"].to_numpy(dtype=np.float32)


def loader(X, y, batch_size, seed, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      worker_init_fn=seed_worker, generator=get_generator(seed))


# --------------------------------------------------------------- models ----
def neural_config(cm, name, features, frames, seed):
    cfg = copy.deepcopy(cm.load_config(name))
    published = sorted(cfg["model_config"]["architecture"]["cat_features"],
                       key=lambda c: int(c["index"]))
    cats = []
    for spec, col in zip(published, CAT_COLS):
        idx = features.index(col)
        if idx != int(spec["index"]):
            raise RuntimeError(f"{col} is at index {idx}, config expects {spec['index']}")
        observed = int(max(df[col].max() for df in frames.values())) + 1
        cats.append({"name": col, "index": idx,
                     "num_classes": max(int(spec["num_classes"]), observed),
                     "emb_dim": int(spec["emb_dim"])})
    cfg["model_config"]["architecture"]["cat_features"] = cats
    cfg["model_config"].setdefault("loss", {})["pos_weight"] = "auto"
    cfg.setdefault("common", {}).setdefault("reproducibility", {})["seed"] = seed
    return cfg


def fit_neural(cm, name, features, frames, seed, device):
    cfg = neural_config(cm, name, features, frames, seed)
    set_all_seeds(seed)
    Xtr, ytr = arrays(frames["train"], features)
    Xva, yva = arrays(frames["val"], features)
    bs = int(cfg["model_config"].get("train_loop", {}).get("batch_size", 512))
    model = build_model(name, cfg, len(features))
    trainer = NeuralTrainer(model, cfg, device)
    trainer.fit(loader(Xtr, ytr, bs, seed, True), loader(Xva, yva, bs, seed, False))

    def predict(df):
        X, y = arrays(df, features)
        return trainer.predict(loader(X, y, bs, seed, False)).ravel()
    return predict, cfg["model_config"]["architecture"]["cat_features"]


def fit_tree(cm, name, features, frames, seed):
    cfg = copy.deepcopy(cm.load_config(name))
    params = cfg["model_config"]["params"]
    params["random_state"] = seed
    Xtr, ytr = arrays(frames["train"], features)
    Xva, yva = arrays(frames["val"], features)
    if name == "xgboost":
        params["scale_pos_weight"] = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    model = build_model(name, cfg, len(features))
    trainer = TreeTrainer(model, cfg)
    trainer.fit(Xtr, ytr, Xva, yva)
    return lambda df: trainer.predict(arrays(df, features)[0]).ravel()


def fit_logistic(features, frames, seed):
    cat_idx = [features.index(c) for c in CAT_COLS]
    num_idx = [i for i in range(len(features)) if i not in cat_idx]
    pipe = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx),
                           ("num", "passthrough", num_idx)]),
        LogisticRegression(C=1.0, max_iter=5000, random_state=seed))
    Xtr, ytr = arrays(frames["train"], features)
    pipe.fit(Xtr, ytr)
    return lambda df: pipe.predict_proba(arrays(df, features)[0])[:, 1]


# ------------------------------------------------------------- metrics ----
def evaluate(y, p, threshold, seed):
    m = compute_classification_metrics(y, p, threshold=threshold)
    m["ece"] = compute_ece(y, p)
    ci = compute_bootstrap_ci(y, p, metric_names=CI_METRICS, threshold=threshold,
                              n_bootstrap=1000, random_state=seed)
    for k in CI_METRICS:
        m[f"{k}_ci_lower"], m[f"{k}_ci_upper"] = ci[k]["lower"], ci[k]["upper"]
    m["n"], m["events"], m["threshold"] = int(len(y)), int(y.sum()), float(threshold)
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in m.items()}


def run_split(cm, pack_dir, external_root, models, out_dir, device):
    seed = int(pack_dir.name.split("_")[1])
    frames, features, apache = load_split(pack_dir, external_root)
    sets = [s for s in ("test", "external") if s in frames]
    preds = {s: pd.DataFrame({"patient_id": frames[s]["patient_id"],
                              "mortality": frames[s]["mortality"]}) for s in sets}
    results, notes = {}, {}

    for name in models:
        t0 = time.time()
        if name in NEURAL:
            predict, cats = fit_neural(cm, name, features, frames, seed, device)
            notes[name] = {"cat_features": cats}
        elif name in TREES:
            predict = fit_tree(cm, name, features, frames, seed)
        else:
            predict = fit_logistic(features, frames, seed)
        yva = frames["val"]["mortality"].to_numpy()
        thr = select_best_threshold_youden(yva, predict(frames["val"]))
        thr = float(thr[0] if isinstance(thr, (tuple, list)) else thr)
        results[name] = {}
        for s in sets:
            p = predict(frames[s])
            preds[s][name] = p
            results[name][s] = evaluate(frames[s]["mortality"].to_numpy(), p, thr, seed)
        log.info(f"{pack_dir.name} {name:20s} {time.time() - t0:6.1f}s  "
                 + "  ".join(f"{s}: AUROC {results[name][s]['auroc']:.3f} "
                             f"AUPRC {results[name][s]['auprc']:.3f}" for s in sets))

    # APACHE IV: eICU test only, on stays where the score exists.
    if apache is not None:
        a = preds["test"][["patient_id"]].merge(apache[["patient_id", "apache_iv_pred"]],
                                                on="patient_id", how="left")
        preds["test"]["apache_iv"] = a["apache_iv_pred"].to_numpy()
        ok = preds["test"]["apache_iv"].notna().to_numpy()
        y = preds["test"]["mortality"].to_numpy()[ok]
        results["apache_iv"] = {"test": evaluate(y, preds["test"]["apache_iv"].to_numpy()[ok],
                                                 0.5, seed)}
        results["apache_iv"]["test"]["n_missing_score"] = int((~ok).sum())

    # DeLong: reference model against every other score, per evaluation set.
    delong = {}
    for s in sets:
        df = preds[s]
        for other in [c for c in df.columns if c not in ("patient_id", "mortality", REFERENCE)]:
            mask = df[other].notna().to_numpy()
            z, pval, diff = delong_test(df["mortality"].to_numpy()[mask],
                                        df[REFERENCE].to_numpy()[mask],
                                        df[other].to_numpy()[mask])
            delong[f"{s}:{REFERENCE}_vs_{other}"] = {"z": float(z), "p": float(pval),
                                                     "auroc_diff": float(diff),
                                                     "n": int(mask.sum())}

    split_out = out_dir / pack_dir.name
    split_out.mkdir(parents=True, exist_ok=True)
    for s in sets:
        preds[s].to_csv(split_out / f"predictions_{s}.csv", index=False)
    with open(split_out / "metrics.json", "w") as f:
        json.dump({"seed": seed, "n_features": len(features), "results": results,
                   "delong": delong, "notes": notes}, f, indent=2)
    return results, delong


def summarise(all_results, out_dir):
    rows = []
    for split, res in all_results.items():
        for model, by_set in res.items():
            for s, m in by_set.items():
                rows.append({"split": split, "model": model, "set": s,
                             **{k: m.get(k) for k in ("auroc", "auprc", "brier", "ece",
                                                      "f1", "n", "events")}})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_long.csv", index=False)
    agg = (df.groupby(["set", "model"])[["auroc", "auprc", "brier", "ece"]]
             .agg(["mean", "std"]).round(4))
    agg.to_csv(out_dir / "metrics_summary.csv")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--external", default=None)
    ap.add_argument("--out", default="../../outputs/hfcs")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--splits", default=None, help="comma-separated seed_* names (default all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="device for the neural models (cuda or cpu)")
    args = ap.parse_args()

    device = torch.device(args.device)
    log.info(f"Neural models on {device}"
             + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    cm = ConfigManager()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    packs = sorted(Path(args.packs).glob("seed_*"))
    if args.splits:
        keep = set(args.splits.split(","))
        packs = [p for p in packs if p.name in keep]
    out_dir = Path(args.out) / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    external_root = Path(args.external) if args.external else None

    all_results, all_delong = {}, {}
    for pack in packs:
        all_results[pack.name], all_delong[pack.name] = run_split(
            cm, pack, external_root, models, out_dir, device)
    agg = summarise(all_results, out_dir)
    with open(out_dir / "delong_all.json", "w") as f:
        json.dump(all_delong, f, indent=2)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args) | {"models": models, "splits": [p.name for p in packs],
                                "torch": torch.__version__,
                                "cuda_device": torch.cuda.get_device_name(0)
                                if device.type == "cuda" else None}, f, indent=2)
    pd.set_option("display.width", 160)
    print("\n" + agg.to_string())
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
