#!/usr/bin/env python3
"""
03_preprocess.py
Steps:
  1. Carica expression_matrix_mapped.csv per entrambi i dataset
  2. Parsa le label cliniche
  3. Trova geni comuni
  4. Normalizza: log2(CPM+1) per RNA-seq, già log2 per microarray
  5. Z-score per gene (within dataset)
  6. Salva .npy + sample_info.csv in PROC_DIR
"""

import os, re, numpy as np, pandas as pd, logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR  = os.environ.get("RAW_DIR",  "/home/jovyan/work/shared/raw")
PROC_DIR = os.environ.get("PROC_DIR", "/home/jovyan/work/shared/processed")
os.makedirs(PROC_DIR, exist_ok=True)


def load_expression(gse_id):
    path = os.path.join(RAW_DIR, gse_id, "expression_matrix_mapped.csv")
    df = pd.read_csv(path, index_col=0)
    log.info(f"  {gse_id} expression: {df.shape} (genes x samples)")
    return df

def load_metadata(gse_id):
    return pd.read_csv(os.path.join(RAW_DIR, gse_id, "metadata.csv"))

def parse_rnaseq_labels(meta_df):
    labels = {}
    for _, row in meta_df.iterrows():
        sid = row["sample_id"]
        char = str(row.get("characteristics_ch1", "")).lower()
        if "bcr-abl1 at 3 month" in char:
            labels[sid] = 1 if "<10%" in char else 0
        else:
            labels[sid] = float("nan")
    return pd.Series(labels, name="label")

def parse_microarray_labels(meta_df):
    labels = {}
    for _, row in meta_df.iterrows():
        sid = row["sample_id"]
        title = str(row.get("title", "")).lower()
        if "nonresponder" in title:
            labels[sid] = 0
        elif "responder" in title:
            labels[sid] = 1
        else:
            labels[sid] = float("nan")
    return pd.Series(labels, name="label")

def log2_cpm_normalize(expr):
    cpm = expr.div(expr.sum(axis=0), axis=1) * 1e6
    return np.log2(cpm + 1)


def main():
    # RNA-seq
    log.info("Loading GSE130404 (RNA-seq)...")
    rna_expr   = load_expression("GSE130404")
    rna_meta   = load_metadata("GSE130404")
    rna_labels = parse_rnaseq_labels(rna_meta)
    valid_rna  = rna_labels.dropna().index
    rna_labels = rna_labels.loc[valid_rna].astype(int)
    rna_expr   = rna_expr.loc[:, rna_expr.columns.isin(valid_rna)]
    log.info(f"  Dopo label filtering: {rna_expr.shape[1]} samples ({(rna_labels==1).sum()} R / {(rna_labels==0).sum()} NR)")
    rna_expr   = log2_cpm_normalize(rna_expr)

    # Microarray
    log.info("Loading GSE14671 (microarray)...")
    mic_expr   = load_expression("GSE14671")
    mic_meta   = load_metadata("GSE14671")
    mic_labels = parse_microarray_labels(mic_meta)
    valid_mic  = mic_labels.dropna().index
    mic_labels = mic_labels.loc[valid_mic].astype(int)
    mic_expr   = mic_expr.loc[:, mic_expr.columns.isin(valid_mic)]
    log.info(f"  Dopo label filtering: {mic_expr.shape[1]} samples ({(mic_labels==1).sum()} R / {(mic_labels==0).sum()} NR)")

    # Geni comuni
    common_genes = rna_expr.index.intersection(mic_expr.index)
    log.info(f"Common genes: {len(common_genes)}")
    if len(common_genes) == 0:
        raise ValueError("Nessun gene comune — controlla la mappatura probe→symbol")
    rna_expr = rna_expr.loc[common_genes]
    mic_expr = mic_expr.loc[common_genes]

    # Z-score
    rna_scaled = StandardScaler().fit_transform(rna_expr.T)   # (n_rna, n_genes)
    mic_scaled = StandardScaler().fit_transform(mic_expr.T)   # (n_mic, n_genes)

    # Allinea sample order alle label
    rna_samples = [s for s in rna_labels.index if s in rna_expr.columns]
    mic_samples = [s for s in mic_labels.index if s in mic_expr.columns]
    rna_label_arr = rna_labels.loc[rna_samples].values.astype(np.int64)
    mic_label_arr = mic_labels.loc[mic_samples].values.astype(np.int64)

    # Salva
    np.save(os.path.join(PROC_DIR, "rnaseq_expr.npy"),       rna_scaled)
    np.save(os.path.join(PROC_DIR, "rnaseq_labels.npy"),     rna_label_arr)
    np.save(os.path.join(PROC_DIR, "microarray_expr.npy"),   mic_scaled)
    np.save(os.path.join(PROC_DIR, "microarray_labels.npy"), mic_label_arr)
    with open(os.path.join(PROC_DIR, "common_genes.txt"), "w") as f:
        f.write("\n".join(common_genes))

    sample_info = pd.concat([
        pd.DataFrame({"sample_id": rna_samples, "label": rna_label_arr, "domain": "rnaseq",     "dataset": "GSE130404"}),
        pd.DataFrame({"sample_id": mic_samples, "label": mic_label_arr, "domain": "microarray", "dataset": "GSE14671"}),
    ], ignore_index=True)
    sample_info.to_csv(os.path.join(PROC_DIR, "sample_info.csv"), index=False)

    log.info(f"Done. rnaseq: {rna_scaled.shape}, microarray: {mic_scaled.shape}, geni comuni: {len(common_genes)}")

if __name__ == "__main__":
    main()