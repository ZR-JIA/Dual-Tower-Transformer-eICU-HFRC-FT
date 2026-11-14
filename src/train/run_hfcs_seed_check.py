#!/usr/bin/env python3
"""
Training-seed robustness of a tuned HF-CS run.

For each outer split, every model is refitted on the whole development set with
the hyperparameters and refit epochs already selected in RUN_DIR (no new
tuning), once per training seed, and scored on the test set and MIMIC-IV. This
separates the variation caused by training randomness (initialisation, batch
order, tree sampling) from the variation between splits. Training seed k = 0 is
the main run's own seed, so it must reproduce the main run.

Only aggregates are written to RUN_DIR:
    seed_check_long.csv     AUROC per split, model, training seed and set
    seed_check_summary.csv  per model and set: mean AUROC over seeds and splits,
                            SD between splits, mean SD across seeds within a
                            split, and the AUROC of the seeds' mean prediction

Usage (from src/train):
    python run_hfcs_seed_check.py --run ../../outputs/hfcs_tuned_v3/tuned40v3 \
        --packs .../PACKS/patient_stratified --raw-eicu .../obs_cleaned.csv \
        --raw-mimic .../mimic_hfcs_cleaned.csv --seeds 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from hfcs_preprocessing import FoldPreprocessor, load_raw_split
from run_hfcs_tuned import MODELS, SITE_MIN_COUNT, SPACES, Learner
from utils.config_manager import ConfigManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="tuned run directory (seed_*/metrics.json)")
    ap.add_argument("--packs", required=True)
    ap.add_argument("--raw-eicu", required=True)
    ap.add_argument("--raw-mimic", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run, cm = Path(args.run), ConfigManager()
    rows, ens, repro = [], [], []
    for pack in sorted(Path(args.packs).glob("seed_*")):
        split_seed = int(pack.name.split("_")[1])
        main_run = json.load(open(run / pack.name / "metrics.json"))
        raw, layout, _ = load_raw_split(pack, Path(args.raw_eicu), Path(args.raw_mimic))
        dev_raw = pd.concat([raw["train"], raw["val"]], ignore_index=True)
        prep = FoldPreprocessor(layout, SITE_MIN_COUNT).fit(dev_raw)
        frames = {"dev": prep.transform(dev_raw), "test": prep.transform(raw["test"]),
                  "external": prep.transform(raw["external"])}
        for name in args.models.split(","):
            tuned = main_run["tuning"][name]
            params = SPACES[name](optuna.trial.FixedTrial(tuned["best_params"]))
            preds = {"test": [], "external": []}
            for k in range(args.seeds):
                seed = split_seed + 1000 * k
                learner = Learner(name, params, cm, layout, frames, seed, args.device)
                learner.fit(frames["dev"], epochs=tuned["refit_epochs"])
                for s in preds:
                    p = learner.predict(frames[s])
                    preds[s].append(p)
                    auc = roc_auc_score(frames[s]["mortality"], p)
                    rows.append({"split": pack.name, "model": name, "train_seed": seed,
                                 "set": s, "auroc": auc})
                    if k == 0:
                        repro.append(abs(auc - main_run["results"][name][s]["auroc"]))
            for s in preds:
                ens.append({"split": pack.name, "model": name, "set": s,
                            "auroc_seed_ensemble": roc_auc_score(frames[s]["mortality"],
                                                                 np.mean(preds[s], axis=0))})
            print(pack.name, name, "done", flush=True)
    long = pd.DataFrame(rows)
    long.to_csv(run / "seed_check_long.csv", index=False)
    per_split = long.groupby(["model", "set", "split"])["auroc"].agg(["mean", "std"]).reset_index()
    summ = per_split.groupby(["model", "set"]).agg(
        auroc_mean=("mean", "mean"), sd_between_splits=("mean", "std"),
        sd_across_seeds=("std", "mean"))
    summ = summ.join(pd.DataFrame(ens).groupby(["model", "set"])["auroc_seed_ensemble"].mean())
    summ.round(4).to_csv(run / "seed_check_summary.csv")
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string())
    print(f"largest difference between seed k=0 and the main run: {max(repro):.2e}")


if __name__ == "__main__":
    main()
