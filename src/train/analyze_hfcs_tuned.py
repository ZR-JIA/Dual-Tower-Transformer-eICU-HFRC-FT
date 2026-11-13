#!/usr/bin/env python3
"""
Aggregate analyses of a tuned HF-CS run (see run_hfcs_tuned.py).

Patient-level prediction files are read where they are and never copied; only
aggregates are written to RUN_DIR:
    apache_subset_summary.csv  every model on the test patients with an APACHE IV score
    optimism_summary.csv       best inner-CV AUROC versus test AUROC, per model
    selected_params.csv        best hyperparameters per model and split
    roc_cal_external.pdf       mean ROC curve and decile calibration on the external cohort
    external_ci_calibration.csv  external AUROC with a patient-bootstrap 95% CI, and
                               calibration intercept and slope, per model

Usage (from src/train):
    python analyze_hfcs_tuned.py ../../outputs/hfcs_tuned/tuned40v2
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
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


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y, p, iters=50):
    """Calibration-in-the-large (intercept with the slope fixed at 1) and the
    calibration slope (unpenalised logistic regression of y on logit p)."""
    y = np.asarray(y, dtype=float)
    z = _logit(p)
    x = np.column_stack([np.ones_like(z), z])
    b = np.zeros(2)
    for _ in range(iters):
        mu = expit(x @ b)
        h = x.T @ (x * (mu * (1 - mu))[:, None])
        b += np.linalg.solve(h, x.T @ (y - mu))
    # The intercept solves sum(y - expit(a + z)) = 0, which decreases in a, so
    # bisection is used: Newton steps diverge for very extreme predictions.
    lo, hi = -30.0, 30.0
    for _ in range(100):
        a = (lo + hi) / 2
        if (y - expit(a + z)).sum() > 0:
            lo = a
        else:
            hi = a
    return (lo + hi) / 2, b[1]


def external_ci_and_calibration(run, n_boot=2000, seed=2026):
    """Mean external AUROC over the split models with a 95% CI from a
    class-stratified bootstrap of the external patients, and the mean (range)
    calibration intercept and slope over the split models."""
    preds = [pd.read_csv(s / "predictions_external.csv") for s in splits(run)]
    y = preds[0]["mortality"].to_numpy()
    for p in preds[1:]:
        assert np.array_equal(p["mortality"].to_numpy(), y), "external rows differ between splits"
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
             for _ in range(n_boot)]
    rows = []
    for m in MODELS:
        scores = np.array([p[m].to_numpy() for p in preds])
        point = np.mean([roc_auc_score(y, s) for s in scores])
        dist = [np.mean([roc_auc_score(y[i], s[i]) for s in scores]) for i in boots]
        cal = np.array([calibration_intercept_slope(y, s) for s in scores])
        rows.append({"model": m, "auroc": point,
                     "auroc_ci_low": np.percentile(dist, 2.5),
                     "auroc_ci_high": np.percentile(dist, 97.5),
                     "intercept_mean": cal[:, 0].mean(), "intercept_min": cal[:, 0].min(),
                     "intercept_max": cal[:, 0].max(),
                     "slope_mean": cal[:, 1].mean(), "slope_min": cal[:, 1].min(),
                     "slope_max": cal[:, 1].max()})
    out = pd.DataFrame(rows).set_index("model").round(4)
    out.to_csv(run / "external_ci_calibration.csv")
    return out


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
    # JATI template: figure text in 8 pt Times New Roman (Liberation Serif has the same metrics
    # and embeds as TrueType).
    plt.rcParams.update({"font.size": 8, "font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                                        "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "legend.fontsize": 8, "pdf.fonttype": 42})
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
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 2.45))
    for ax, tag in ((a, "a"), (b, "b")):
        ax.plot([0, 1], [0, 1], color="#b5b4ad", lw=0.6, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1.0"])
        ax.set_yticklabels(["0", "0.5", "1.0"])
        ax.set_aspect("equal")
        ax.grid(color="#e8e7e2", lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(-0.30, 1.02, tag, transform=ax.transAxes, fontweight="bold",
                va="bottom", ha="left")
    # The adapted model is drawn last and thicker so it stays in front.
    for m in MODELS[1:] + MODELS[:1]:
        lw, z = (1.6, 5) if m == "dualtower_ft" else (0.9, 3)
        mean_tpr = np.mean(tprs[m], axis=0)
        mean_tpr[0] = 0.0
        a.plot(grid, mean_tpr, color=COLORS[m], ls=STYLES[m], lw=lw, zorder=z, label=LABELS[m])
        c = np.mean(cal[m], axis=0)
        b.plot(c[:, 0], c[:, 1], color=COLORS[m], ls=STYLES[m], lw=lw, zorder=z, marker="o",
               ms=2.4, mec="white", mew=0.3)
    a.set_xlabel("False positive rate")
    a.set_ylabel("True positive rate")
    b.set_xlabel("Predicted risk")
    b.set_ylabel("Observed mortality")
    handles, labels = a.get_legend_handles_labels()
    order = [labels.index(LABELS[m]) for m in MODELS]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               ncol=3, frameon=False, handlelength=1.8, handletextpad=0.4, columnspacing=0.8,
               borderaxespad=0.1, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.30, wspace=0.45)
    fig.savefig(run / "roc_cal_external.pdf")
    plt.close(fig)


def main():
    run = Path(sys.argv[1])
    pd.set_option("display.width", 160)
    print("== same APACHE subset\n", apache_subset(run))
    print("== optimism\n", optimism_and_params(run))
    print("== external AUROC CI and calibration\n", external_ci_and_calibration(run))
    roc_cal_figure(run)
    print("wrote", run / "roc_cal_external.pdf")


if __name__ == "__main__":
    main()
