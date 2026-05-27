#!/usr/bin/env python3
# 02_map_probes_to_genes.py
# Legge expression_matrix_raw.csv, mappa probe → gene symbol,
# salva expression_matrix_mapped.csv (raw mai toccato)

import os, GEOparse, pandas as pd

RAW_DIR = os.environ.get("RAW_DIR", "/home/jovyan/work/shared/raw")

DATASETS = {
    "GSE130404": ("GSE130404_family.soft.gz", "Symbol"),
    "GSE14671":  ("GSE14671_family.soft.gz",  "Gene Symbol"),
}

for gse_id, (soft_file, sym_col) in DATASETS.items():
    soft_path = os.path.join(RAW_DIR, gse_id, soft_file)
    print(f"\n=== {gse_id} ===")
    gse = GEOparse.get_GEO(filepath=soft_path, silent=True)
    gpl = gse.gpls[list(gse.gpls.keys())[0]]

    annot = gpl.table[["ID", sym_col]].copy()
    annot.columns = ["probe_id", "gene_symbol"]
    annot["probe_id"] = annot["probe_id"].astype(str)
    annot["gene_symbol"] = annot["gene_symbol"].str.split(" /// ").str[0].str.strip()
    annot = annot[annot["gene_symbol"].notna() & (annot["gene_symbol"] != "")]
    annot = annot.set_index("probe_id")

    # Legge RAW, non tocca mai raw
    expr = pd.read_csv(os.path.join(RAW_DIR, gse_id, "expression_matrix_raw.csv"), index_col=0)
    expr.index = expr.index.astype(str)
    print(f"  Prima: {expr.shape}")

    expr = expr[expr.index.isin(annot.index)]
    expr["gene_symbol"] = annot.loc[expr.index, "gene_symbol"].values
    expr["mean_expr"] = expr.drop(columns=["gene_symbol"]).mean(axis=1)
    expr = (expr.sort_values("mean_expr", ascending=False)
                .drop_duplicates(subset="gene_symbol")
                .drop(columns=["mean_expr"])
                .set_index("gene_symbol"))

    print(f"  Dopo: {expr.shape}")
    out = os.path.join(RAW_DIR, gse_id, "expression_matrix_mapped.csv")
    expr.to_csv(out)
    print(f"  Salvato: {out}")

print("\nDone.")