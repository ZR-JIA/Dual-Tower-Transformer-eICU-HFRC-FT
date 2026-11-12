#!/usr/bin/env python3
"""
Aggregate analyses of a tuned HF-CS run (see run_hfcs_tuned.py).

Patient-level prediction files are read where they are and never copied; only
aggregates are written to RUN_DIR:
    apache_subset_summary.csv  every model on the test patients with an APACHE IV score
    optimism_summary.csv       best inner-CV AUROC versus test AUROC, per model
    selected_params.csv        best hyperparameters per model and split
    roc_cal_external.pdf       mean ROC curve and decile calibration on the external cohort

Usage (from src/train):
    python analyze_hfcs_tuned.py ../../outputs/hfcs_tuned/tuned40v2
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

MODELS = ["dualtower_ft", "dualtower", "xgboost", "random_forest", "logistic_regression"]
LABELS = {"dualtower_ft": "DualTower-FT", "dualtower": "DT-Transformer", "xgboost": "XGBoost",
          "random_forest": "Random forest", "logistic_regression": "Logistic regression"}
# Validated categorical slots 1-5 (light surface); line styles give a second encoding.
COLORS = {"dualtower_ft": "#2a78d6", "dualtower": "#eb6834", "xgboost": "#1baf7a",
          "random_forest": "#eda100", "logistic_regression": "#e87ba4"}
STYLES = {"dualtower_ft": "-", "dualtower": "--", "xgboost": "-.",
          "random_forest": (0, (1, 1)), "logistic_regression": (0, (5, 1, 1, 1, 1, 1))}


def splits(run):
    return sorted(p for p in run.glob("seed_*") if (p / "metrics.json").exists())


def apache_subset(run):
    rows = []
    for s in splits(run):
        p = pd.read_csv(s / "predictions_test.csv")
        sub = p[p["apache_iv"].notna()]
        for m in [c for c in MODELS + ["apache_iv"] if c in sub]:
            rows.append({"split": s.name, "model": m, "n": len(sub),
                         "auroc": roc_auc_score(sub["mortality"], sub[m]),
                         "auprc": average_precision_score(sub["mortality"], sub[m])})
    df = pd.DataFrame(rows)
    out = df.groupby("model").agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                                  auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"),
                                  n_min=("n", "min"), n_max=("n", "max")).round(4)
    out.to_csv(run / "apache_subset_summary.csv")
    return out


def optimism_and_params(run):
    opt, params = [], []
    for s in splits(run):
        d = json.load(open(s / "metrics.json"))
        for m, t in d["tuning"].items():
            opt.append({"split": s.name, "model": m, "inner_cv": t["inner_cv_auroc"],
                        "test": d["results"][m]["test"]["auroc"],
                        "external": d["results"][m].get("external", {}).get("auroc")})
            params.append({"split": s.name, "model": m, **t["best_params"]})
    o = pd.DataFrame(opt)
    o["optimism"] = o["inner_cv"] - o["test"]
    summ = o.groupby("model")[["inner_cv", "test", "external", "optimism"]].mean().round(4)
    summ.to_csv(run / "optimism_summary.csv")
    pd.DataFrame(params).to_csv(run / "selected_params.csv", index=False)
    return summ


def roc_cal_figure(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "pdf.fonttype": 42})
    grid = np.linspace(0, 1, 201)
    tprs = {m: [] for m in MODELS}
    cal = {m: [] for m in MODELS}
    for s in splits(run):
        p = pd.read_csv(s / "predictions_external.csv")
        y = p["mortality"].to_numpy()
        for m in MODELS:
            fpr, tpr, _ = roc_curve(y, p[m])
            tprs[m].append(np.interp(grid, fpr, tpr))
            bins = pd.qcut(p[m].rank(method="first"), 10, labels=False)
            g = pd.DataFrame({"pred": p[m], "obs": y, "bin": bins}).groupby("bin").mean()
            cal[m].append(g[["pred", "obs"]].to_numpy())
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.85))
    for ax in (a, b):
        ax.plot([0, 1], [0, 1], color="#b5b4ad", lw=0.6, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(color="#e8e7e2", lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for m in MODELS:
        mean_tpr = np.mean(tprs[m], axis=0)
        mean_tpr[0] = 0.0
        a.plot(grid, mean_tpr, color=COLORS[m], ls=STYLES[m], lw=1.1, label=LABELS[m])
        c = np.mean(cal[m], axis=0)
        b.plot(c[:, 0], c[:, 1], color=COLORS[m], ls=STYLES[m], lw=1.0, marker="o",
               ms=2.2, mec="white", mew=0.3)
    a.set_xlabel("False positive rate")
    a.set_ylabel("True positive rate")
    b.set_xlabel("Predicted risk")
    b.set_ylabel("Observed mortality")
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=6, handlelength=2.6,
               columnspacing=1.0, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.98, bottom=0.36, wspace=0.42)
    fig.savefig(run / "roc_cal_external.pdf")
    plt.close(fig)


def main():
    run = Path(sys.argv[1])
    pd.set_option("display.width", 160)
    print("== same APACHE subset\n", apache_subset(run))
    print("== optimism\n", optimism_and_params(run))
    roc_cal_figure(run)
    print("wrote", run / "roc_cal_external.pdf")


if __name__ == "__main__":
    main()
