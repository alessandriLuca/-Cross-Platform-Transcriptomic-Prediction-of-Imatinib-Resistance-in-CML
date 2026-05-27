#!/usr/bin/env python3
"""
07_baseline.py
Confronto tra ProtoDANN e due baseline classiche:
  - Regressione logistica con penalità L1 (LASSO)
  - Random Forest

Entrambe addestrate su RNA-seq (GSE130404) e testate su microarray (GSE14671).
Stesso split train/test del ProtoDANN — test set mai toccato durante lo sviluppo.

Output:
  RESULTS_DIR/baseline_results.json   — AUC + CI per tutti i modelli
  RESULTS_DIR/baseline_comparison.csv — tabella riassuntiva

Uso:
  export RESULTS_DIR=/home/jovyan/work/shared/results/v1_prototype
  export PROC_DIR=/home/jovyan/work/shared/processed
  export RAW_DIR=/home/jovyan/work/shared/raw
  python3 07_baseline.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROC_DIR    = os.environ.get("PROC_DIR",    "/home/jovyan/work/shared/processed")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/home/jovyan/work/shared/results/v1_prototype")
RAW_DIR     = os.environ.get("RAW_DIR",     "/home/jovyan/work/shared/raw")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_BOOTSTRAP = 2000
SEED        = 42
np.random.seed(SEED)


# ── DATA ──────────────────────────────────────────────────────────────────────

def load_data():
    rna_X = np.load(os.path.join(PROC_DIR, "rnaseq_expr.npy"),       allow_pickle=False)
    rna_y = np.load(os.path.join(PROC_DIR, "rnaseq_labels.npy"),     allow_pickle=False)
    mic_X = np.load(os.path.join(PROC_DIR, "microarray_expr.npy"),   allow_pickle=False)
    mic_y = np.load(os.path.join(PROC_DIR, "microarray_labels.npy"), allow_pickle=False)

    meta        = pd.read_csv(os.path.join(RAW_DIR, "GSE14671", "metadata.csv"))
    sample_info = pd.read_csv(os.path.join(PROC_DIR, "sample_info.csv"))
    mic_samples = sample_info[sample_info["dataset"] == "GSE14671"]["sample_id"].values

    is_train = []
    for sid in mic_samples:
        row   = meta[meta["sample_id"] == sid]
        title = str(row["title"].values[0]).lower() if len(row) > 0 else ""
        is_train.append("training" in title)
    is_train      = np.array(is_train)
    mic_test_idx  = np.where(~is_train)[0]

    mic_X_test = mic_X[mic_test_idx]
    mic_y_test = mic_y[mic_test_idx]

    log.info(f"RNA-seq train: {rna_X.shape}  labels: {np.bincount(rna_y)}")
    log.info(f"Microarray test: {mic_X_test.shape}  labels: {np.bincount(mic_y_test)}")
    return rna_X, rna_y, mic_X_test, mic_y_test


def bootstrap_auc_ci(y_true, y_score, n=N_BOOTSTRAP, seed=SEED):
    rng  = np.random.RandomState(seed)
    auc  = roc_auc_score(y_true, y_score)
    boot = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        ys, yp = y_true[idx], y_score[idx]
        if len(np.unique(ys)) < 2:
            continue
        boot.append(roc_auc_score(ys, yp))
    boot = np.array(boot)
    return float(auc), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ── MODELLI ───────────────────────────────────────────────────────────────────

def run_lasso(rna_X, rna_y, mic_X_test, mic_y_test):
    """
    Regressione logistica L1 (LASSO).
    LogisticRegressionCV seleziona C ottimale via 5-fold CV su training set.
    Normalizzazione z-score separata per train e test (no data leakage).
    """
    log.info("\n--- Baseline 1: Logistic Regression LASSO ---")

    # Scaler fittato SOLO su RNA-seq training
    scaler  = StandardScaler()
    rna_X_s = scaler.fit_transform(rna_X)
    mic_X_s = scaler.transform(mic_X_test)   # stessa trasformazione

    # Classe minority weight
    n_nr = (rna_y == 0).sum(); n_r = (rna_y == 1).sum()
    cw   = {0: n_r / n_nr, 1: 1.0}   # bilancia NR

    model = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="l1",
        solver="liblinear",
        class_weight=cw,
        max_iter=1000,
        random_state=SEED,
        scoring="roc_auc",
        n_jobs=-1,
    )
    model.fit(rna_X_s, rna_y)

    n_nonzero = (model.coef_[0] != 0).sum()
    log.info(f"  C ottimale: {model.C_[0]:.4f}")
    log.info(f"  Geni selezionati (coef != 0): {n_nonzero} / {rna_X.shape[1]}")

    scores = model.predict_proba(mic_X_s)[:, 1]
    auc, ci_low, ci_high = bootstrap_auc_ci(mic_y_test, scores)
    log.info(f"  AUC test: {auc:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    return {
        "model": "Logistic Regression LASSO",
        "auc": auc, "ci_low": ci_low, "ci_high": ci_high,
        "n_genes_selected": int(n_nonzero),
        "best_C": float(model.C_[0]),
        "scores": scores.tolist(),
    }


def run_random_forest(rna_X, rna_y, mic_X_test, mic_y_test):
    """
    Random Forest.
    Iperparametri conservativi per dataset piccolo.
    No normalizzazione necessaria.
    """
    log.info("\n--- Baseline 2: Random Forest ---")

    n_nr = (rna_y == 0).sum(); n_r = (rna_y == 1).sum()
    cw   = {0: n_r / n_nr, 1: 1.0}

    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        max_depth=5,           # limitato per evitare overfitting su dataset piccolo
        min_samples_leaf=3,
        class_weight=cw,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(rna_X, rna_y)

    scores = model.predict_proba(mic_X_test)[:, 1]
    auc, ci_low, ci_high = bootstrap_auc_ci(mic_y_test, scores)
    log.info(f"  AUC test: {auc:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    # Top feature importance
    top_idx   = np.argsort(model.feature_importances_)[-10:][::-1]
    log.info(f"  Top feature indices per importanza: {top_idx.tolist()}")

    return {
        "model": "Random Forest",
        "auc": auc, "ci_low": ci_low, "ci_high": ci_high,
        "scores": scores.tolist(),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    rna_X, rna_y, mic_X_test, mic_y_test = load_data()

    # Carica risultato ProtoDANN per confronto
    proto_path = os.path.join(RESULTS_DIR, "results_prototype.json")
    with open(proto_path) as f:
        proto = json.load(f)
    proto_result = {
        "model":   "ProtoDANN (nostro modello)",
        "auc":     proto["auc"],
        "ci_low":  proto["ci_low"],
        "ci_high": proto["ci_high"],
    }
    log.info(f"\nProtoDANN (riferimento): AUC={proto['auc']:.4f}  "
             f"CI [{proto['ci_low']:.4f}, {proto['ci_high']:.4f}]")

    lasso_result = run_lasso(rna_X, rna_y, mic_X_test, mic_y_test)
    rf_result    = run_random_forest(rna_X, rna_y, mic_X_test, mic_y_test)

    results = [proto_result, lasso_result, rf_result]

    # Salva JSON
    out_json = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Tabella CSV
    df = pd.DataFrame([{
        "Modello":   r["model"],
        "AUC":       f"{r['auc']:.4f}",
        "CI 2.5%":   f"{r['ci_low']:.4f}",
        "CI 97.5%":  f"{r['ci_high']:.4f}",
    } for r in results])
    out_csv = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
    df.to_csv(out_csv, index=False)

    log.info(f"\n{'='*60}")
    log.info("CONFRONTO MODELLI — Test set (23 campioni, GSE14671)")
    log.info(f"{'='*60}")
    log.info(df.to_string(index=False))
    log.info(f"\nSalvati: {out_json}  e  {out_csv}")


if __name__ == "__main__":
    main()