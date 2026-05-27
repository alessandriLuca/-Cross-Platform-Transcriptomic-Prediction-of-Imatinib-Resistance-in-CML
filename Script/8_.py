#!/usr/bin/env python3
"""
08_permutation_test.py
Test di permutazione sull'AUC del ProtoDANN sul test set.

Procedura:
  1. Carica i scores del ProtoDANN (dal results_prototype.json)
  2. Permuta le label 10.000 volte
  3. Calcola l'AUC su ogni permutazione
  4. p-value = proporzione di AUC permutati >= AUC osservato

Se p < 0.05: l'AUC osservato è significativamente diverso dal caso.
Se p >= 0.05: non si può escludere che sia dovuto al caso dato N=23.

Output:
  RESULTS_DIR/permutation_test.json
  RESULTS_DIR/figures/fig_permutation.pdf

Uso:
  export RESULTS_DIR=/home/jovyan/work/shared/results/v1_prototype
  python3 08_permutation_test.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = os.environ.get("RESULTS_DIR",
              "/home/jovyan/work/shared/results/v1_prototype")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_PERMUTATIONS = 10000
SEED           = 42

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def main():
    # Carica scores e label del ProtoDANN
    proto_path = os.path.join(RESULTS_DIR, "results_prototype.json")
    with open(proto_path) as f:
        proto = json.load(f)

    y_true  = np.array(proto["labels_test"])
    y_score = np.array(proto["scores_test"])
    auc_obs = roc_auc_score(y_true, y_score)

    log.info(f"AUC osservato: {auc_obs:.4f}")
    log.info(f"N campioni: {len(y_true)}  ({(y_true==0).sum()} NR, {(y_true==1).sum()} R)")
    log.info(f"Esecuzione {N_PERMUTATIONS} permutazioni...")

    # Permutazione
    rng      = np.random.RandomState(SEED)
    auc_perm = []
    for _ in range(N_PERMUTATIONS):
        y_shuffled = rng.permutation(y_true)
        try:
            auc_perm.append(roc_auc_score(y_shuffled, y_score))
        except ValueError:
            continue
    auc_perm = np.array(auc_perm)

    p_value = float((auc_perm >= auc_obs).mean())

    log.info(f"\nRisultato permutation test:")
    log.info(f"  AUC osservato:        {auc_obs:.4f}")
    log.info(f"  AUC medio permutato:  {auc_perm.mean():.4f} ± {auc_perm.std():.4f}")
    log.info(f"  p-value:              {p_value:.4f}")
    if p_value < 0.05:
        log.info(f"  Conclusione: AUC significativamente > caso (p < 0.05)")
    elif p_value < 0.1:
        log.info(f"  Conclusione: tendenza (0.05 <= p < 0.10) — non significativo ma suggestivo")
    else:
        log.info(f"  Conclusione: AUC NON significativamente diverso dal caso (p >= 0.05)")
        log.info(f"  Con N=23 e 6 NR, questo era il rischio atteso.")

    # Salva risultati
    result = {
        "auc_observed":      float(auc_obs),
        "auc_perm_mean":     float(auc_perm.mean()),
        "auc_perm_std":      float(auc_perm.std()),
        "auc_perm_95pct":    float(np.percentile(auc_perm, 95)),
        "p_value":           p_value,
        "n_permutations":    N_PERMUTATIONS,
        "n_test":            int(len(y_true)),
        "n_NR":              int((y_true==0).sum()),
        "n_R":               int((y_true==1).sum()),
        "significant_005":   bool(p_value < 0.05),
    }
    out_json = os.path.join(RESULTS_DIR, "permutation_test.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Salvato: {out_json}")

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.hist(auc_perm, bins=60, color="#0072B2", alpha=0.7, edgecolor="white",
            label=f"AUC permutati (n={N_PERMUTATIONS})")
    ax.axvline(auc_obs, color="#D55E00", lw=2.5,
               label=f"AUC osservato = {auc_obs:.3f}")
    ax.axvline(np.percentile(auc_perm, 95), color="#999999", lw=1.5, ls="--",
               label=f"95° percentile permutato = {np.percentile(auc_perm, 95):.3f}")

    ax.set_xlabel("AUC")
    ax.set_ylabel("Frequenza")
    ax.set_title(f"Permutation Test — ProtoDANN\n"
                 f"p-value = {p_value:.4f}  (N permutazioni = {N_PERMUTATIONS})", pad=8)
    ax.legend(frameon=False, fontsize=9)

    # Annotazione p-value
    ax.text(0.97, 0.92, f"p = {p_value:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#D55E00", fontweight="bold")

    out_fig = os.path.join(FIG_DIR, "fig_permutation.pdf")
    fig.savefig(out_fig); plt.close(fig)
    log.info(f"Salvato: {out_fig}")


if __name__ == "__main__":
    main()