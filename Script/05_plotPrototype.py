#!/usr/bin/env python3
"""
05_plot_prototype.py
Genera i plot finali leggendo results_prototype.json.

Output (in RESULTS_DIR/figures/):
  fig1_roc_curve.pdf
  fig2_score_distribution.pdf
  fig3_selftraining.pdf
  fig4_confusion.pdf

Uso:
  export RESULTS_DIR=/home/jovyan/work/shared/results/v1_prototype
  python3 05_plot_prototype.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

RESULTS_DIR = os.environ.get("RESULTS_DIR",
              "/home/jovyan/work/shared/results/v1_prototype")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Palette Wong 2011 (accessibile)
C_NR   = "#D55E00"
C_R    = "#0072B2"
C_BEST = "#009E73"
C_GREY = "#999999"
C_DARK = "#222222"
C_AMB  = "#E69F00"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})


def bootstrap_roc_band(y_true, y_score, n_bootstrap=2000, seed=42):
    rng      = np.random.RandomState(seed)
    fpr_grid = np.linspace(0, 1, 200)
    tpr_boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        ys, yp = np.array(y_true)[idx], np.array(y_score)[idx]
        if len(np.unique(ys)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(ys, yp)
        tpr_boot.append(np.interp(fpr_grid, fpr_b, tpr_b))
    tpr_boot = np.array(tpr_boot)
    return fpr_grid, tpr_boot.mean(0), \
           np.percentile(tpr_boot, 2.5, 0), np.percentile(tpr_boot, 97.5, 0)


# ── FIG 1 — ROC CURVE ────────────────────────────────────────────────────────

def plot_roc(res):
    y_true  = np.array(res["labels_test"])
    y_score = np.array(res["scores_test"])
    auc     = res["auc"]
    ci_low  = res["ci_low"]
    ci_high = res["ci_high"]
    thr_y   = res["threshold_youden"]["threshold"]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr_g, tpr_mean, tpr_lo, tpr_hi = bootstrap_roc_band(y_true, y_score)

    closest    = np.argmin(np.abs(thresholds - thr_y))
    youden_fpr = fpr[closest]
    youden_tpr = tpr[closest]

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.fill_between(fpr_g, tpr_lo, tpr_hi, color=C_R, alpha=0.15,
                    label="95% CI bootstrap")
    ax.plot(fpr, tpr, color=C_R, lw=2,
            label=f"ProtoDANN (AUC = {auc:.3f}; 95% CI {ci_low:.3f}–{ci_high:.3f})")
    ax.plot([0,1],[0,1], color=C_GREY, lw=1, ls="--", label="No skill (AUC = 0.50)")
    ax.scatter([youden_fpr],[youden_tpr], color=C_BEST, s=60, zorder=5,
               label=f"Soglia Youden ({thr_y:.4f})\nSens={youden_tpr:.2f}, Spec={1-youden_fpr:.2f}")

    ax.set_xlabel("1 − Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("ROC Curve — Test set (GSE14671, n = 23)\n"
                 "ProtoDANN: RNA-seq → microarray (cross-platform)", pad=8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", frameon=False)

    n_nr = int((y_true==0).sum()); n_r = int((y_true==1).sum())
    ax.text(0.98, 0.08, f"n = {len(y_true)}  ({n_nr} NR, {n_r} R)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=C_GREY)

    out = os.path.join(FIG_DIR, "fig1_roc_curve.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"  Salvato: {out}")


# ── FIG 2 — SCORE DISTRIBUTION ───────────────────────────────────────────────

def jitter(n, width=0.08, seed=0):
    return np.random.RandomState(seed).uniform(-width, width, n)

def plot_score_distribution(res):
    y_true  = np.array(res["labels_test"])
    y_score = np.array(res["scores_test"])
    thr_y   = res["threshold_youden"]["threshold"]

    nr_scores = y_score[y_true == 0]
    r_scores  = y_score[y_true == 1]

    fig, ax = plt.subplots(figsize=(3.8, 4.2))

    bp = ax.boxplot([nr_scores, r_scores], positions=[0,1], widths=0.35,
                    patch_artist=True,
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(color=C_DARK, lw=1),
                    capprops=dict(color=C_DARK, lw=1),
                    flierprops=dict(marker="", linestyle="none"),
                    boxprops=dict(lw=0))
    bp["boxes"][0].set_facecolor(C_NR + "55")
    bp["boxes"][1].set_facecolor(C_R  + "55")

    ax.scatter(0 + jitter(len(nr_scores), seed=1), nr_scores,
               color=C_NR, s=40, zorder=3, alpha=0.85, lw=0.4,
               edgecolors="white", label="Non-Responder")
    ax.scatter(1 + jitter(len(r_scores),  seed=2), r_scores,
               color=C_R,  s=40, zorder=3, alpha=0.85, lw=0.4,
               edgecolors="white", label="Responder")

    ax.axhline(thr_y, color=C_BEST, ls=":", lw=1.5,
               label=f"Soglia Youden ({thr_y:.4f})")

    ax.set_xticks([0,1])
    ax.set_xticklabels([f"Non-Responder\n(n={len(nr_scores)})",
                         f"Responder\n(n={len(r_scores)})"])
    ax.set_ylabel("Prototype score (cosine similarity)")
    ax.set_title("Score Distribution — Test set (n = 23)\n"
                 "Score = cosine similarity con centroide responder", pad=8)
    ax.set_xlim(-0.6, 1.6)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    # Nota sul range compresso
    ax.text(0.5, 0.02,
            "Nota: score compressi in [0.97, 1.00] — separazione reale ma piccola",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color=C_GREY, style="italic")

    out = os.path.join(FIG_DIR, "fig2_score_distribution.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"  Salvato: {out}")


# ── FIG 3 — SELF-TRAINING ────────────────────────────────────────────────────

def plot_selftraining(res):
    history    = res.get("round_history", [])
    if not history:
        print("  Nessun round_history — skip fig3")
        return

    rounds     = [h["round"]      for h in history]
    aucs       = [h["train_auc"]  for h in history]
    ranges     = [h.get("score_range", None) for h in history]
    degenerate = [h.get("is_degenerate", False) for h in history]
    best_round = res.get("best_round", 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.0),
                                    sharex=True, gridspec_kw={"hspace": 0.35})

    # Panel 1: AUC
    colors = [C_AMB if d else C_R for d in degenerate]
    ax1.plot(rounds, aucs, color=C_GREY, lw=1, zorder=1)
    for r, a, c in zip(rounds, aucs, colors):
        ax1.scatter([r], [a], color=c, s=45, zorder=3)

    ax1.scatter([best_round], [aucs[best_round]], color=C_BEST, s=90, zorder=5,
                label=f"Best round = {best_round}  (AUC = {aucs[best_round]:.3f})")
    ax1.axvline(best_round, color=C_BEST, ls=":", lw=1.2, alpha=0.7)
    ax1.set_ylabel("AUC su mic_train_set")
    ax1.set_title("Self-training Progression\nBest round selezionato su training set", pad=6)
    ax1.legend(frameon=False, fontsize=8)

    # Legenda colori
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        plt.scatter([],[],color=C_R,    s=40, label="Round valido"),
        plt.scatter([],[],color=C_AMB,  s=40, label="Round DEGENERE (ignorato)"),
        plt.scatter([],[],color=C_BEST, s=60, label=f"Best round = {best_round}"),
    ], frameon=False, fontsize=8)

    # Panel 2: score_range
    valid_r = [(r, rng) for r, rng, d in zip(rounds, ranges, degenerate)
               if rng is not None and not d]
    degen_r = [(r, rng) for r, rng, d in zip(rounds, ranges, degenerate)
               if rng is not None and d]

    if valid_r:
        vr, va = zip(*valid_r)
        ax2.scatter(vr, va, color=C_R, s=45, zorder=3)
    if degen_r:
        dr, da = zip(*degen_r)
        ax2.scatter(dr, da, color=C_AMB, s=45, zorder=3)

    ax2.axhline(0.01, color=C_DARK, ls="--", lw=1,
                label="Soglia anti-degenerazione (0.01)")
    ax2.set_xlabel("Self-training round  (0 = ProtoDANN iniziale)")
    ax2.set_ylabel("Score range (max − min)")
    ax2.set_title("Score range per round\n(< 0.01 = modello collassato)", pad=6)
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_xticks(rounds)

    out = os.path.join(FIG_DIR, "fig3_selftraining.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"  Salvato: {out}")


# ── FIG 4 — CONFUSION MATRICES ───────────────────────────────────────────────

def _draw_cm(ax, cm, title, thr_label, cmap):
    n  = cm.sum()
    ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=n)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["NR (pred)", "R (pred)"])
    ax.set_yticklabels(["NR (true)", "R (true)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}\n(threshold = {thr_label})", pad=6)
    for i in range(2):
        for j in range(2):
            val = cm[i,j]; pct = val/n*100
            col = "white" if val > n*0.4 else C_DARK
            ax.text(j, i, f"{val}\n({pct:.0f}%)",
                    ha="center", va="center", color=col,
                    fontsize=11, fontweight="bold")
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp+fn, 1); spec = tn / max(tn+fp, 1)
    ax.text(0.5, -0.22, f"Sensitivity = {sens:.2f}   Specificity = {spec:.2f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=C_DARK)


def plot_confusion(res):
    y_true   = np.array(res["labels_test"])
    preds_05 = np.array(res["threshold_05"]["preds"])
    preds_y  = np.array(res["threshold_youden"]["preds"])
    thr_y    = res["threshold_youden"]["threshold"]

    cm_05 = confusion_matrix(y_true, preds_05, labels=[0,1])
    cm_y  = confusion_matrix(y_true, preds_y,  labels=[0,1])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.suptitle("Confusion Matrices — Test set (GSE14671, n = 23)\n"
                 "ProtoDANN: RNA-seq → microarray",
                 fontsize=10, y=1.05)

    _draw_cm(axes[0], cm_05, "Soglia fissa", "0.50",       plt.cm.Blues)
    _draw_cm(axes[1], cm_y,  "Soglia Youden", f"{thr_y:.4f}\n(calibrated on train set)",
             plt.cm.Greens)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(FIG_DIR, "fig4_confusion.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"  Salvato: {out}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    results_path = os.path.join(RESULTS_DIR, "results_prototype.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results_prototype.json non trovato in {RESULTS_DIR}")

    with open(results_path) as f:
        res = json.load(f)

    print(f"Caricato: {results_path}")
    print(f"  AUC: {res['auc']:.4f}  CI [{res['ci_low']:.4f}, {res['ci_high']:.4f}]")
    print(f"  Best round: {res['best_round']}  (train AUC: {res['best_train_auc']:.4f})")
    print()

    plot_roc(res)
    plot_score_distribution(res)
    plot_selftraining(res)
    plot_confusion(res)

    print(f"\nTutti i plot in: {FIG_DIR}")


if __name__ == "__main__":
    main()