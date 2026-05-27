#!/usr/bin/env python3
# 01_download_data.py

import os, sys, subprocess, GEOparse, pandas as pd, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = os.environ.get("RAW_DIR", "/home/jovyan/work/shared/raw")

DATASETS = {
    "GSE130404": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130404/soft/GSE130404_family.soft.gz",
    "GSE14671":  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14671/soft/GSE14671_family.soft.gz",
}

def download(gse_id, url):
    dest_dir = os.path.join(RAW_DIR, gse_id)
    os.makedirs(dest_dir, exist_ok=True)
    soft_path = os.path.join(dest_dir, os.path.basename(url))
    if os.path.exists(soft_path):
        log.info(f"  {gse_id} già scaricato, skip.")
        return soft_path
    log.info(f"  Scarico {gse_id}...")
    subprocess.run(["wget", "-P", dest_dir, url], check=True)
    return soft_path

def parse_dataset(gse_id, soft_path):
    log.info(f"Parsing {gse_id}...")
    dest = os.path.dirname(soft_path)
    gse = GEOparse.get_GEO(filepath=soft_path, silent=True)
    for col in ("VALUE", "value", "VALUE_LOG2"):
        try:
            expr = gse.pivot_samples(col)
            out = os.path.join(dest, "expression_matrix_raw.csv")
            expr.to_csv(out)
            log.info(f"  Expression matrix ({col}): {expr.shape} -> {out}")
            break
        except Exception:
            continue
    rows = []
    for gsm_id, gsm in gse.gsms.items():
        row = {"sample_id": gsm_id}
        for k, v in gsm.metadata.items():
            row[k] = " | ".join(str(x) for x in v) if isinstance(v, list) else str(v)
        rows.append(row)
    meta = pd.DataFrame(rows)
    out_meta = os.path.join(dest, "metadata.csv")
    meta.to_csv(out_meta, index=False)
    log.info(f"  Metadata: {meta.shape}")

errors = []
for gse_id, url in DATASETS.items():
    try:
        soft_path = download(gse_id, url)
        parse_dataset(gse_id, soft_path)
    except Exception as e:
        log.error(f"Fallito {gse_id}: {e}")
        errors.append(gse_id)

if errors:
    sys.exit(1)
log.info("Done.")
