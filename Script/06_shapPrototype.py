#!/usr/bin/env python3
"""
06_shap_prototype.py
Analisi di interpretabilità sul modello ProtoDANN.

Cosa fa:
  1. Carica il modello salvato (proto_dann_best.pt) e il centroide (proto_centroid.pt)
  2. Calcola SHAP values sui 23 campioni del test set
     usando GradientExplainer di SHAP
  3. Salva:
     - shap_values.npy          (n_test x n_genes)
     - shap_top_genes.csv       (top 200 geni per |SHAP| medio)
     - top_shap_genes.txt       (lista geni per GSEA su Enrichr/MSigDB)
     - fig_shap_beeswarm.pdf    (beeswarm plot dei top 20 geni)
     - fig_shap_barplot.pdf     (barplot importanza media top 20)

Uso:
  pip install shap --break-system-packages
  export RESULTS_DIR=/home/jovyan/work/shared/results/v1_prototype
  export PROC_DIR=/home/jovyan/work/shared/processed
  export RAW_DIR=/home/jovyan/work/shared/raw
  python3 06_shap_prototype.py
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import Encoder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = os.environ.get("RESULTS_DIR",
              "/home/jovyan/work/shared/results/v1_prototype")
PROC_DIR    = os.environ.get("PROC_DIR", "/home/jovyan/work/shared/processed")
RAW_DIR     = os.environ.get("RAW_DIR",  "/home/jovyan/work/shared/raw")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = torch.device("cpu")   # SHAP funziona meglio su CPU

N_TOP_GENES    = 200   # geni da salvare nel CSV
N_PLOT_GENES   = 20    # geni da visualizzare nei plot
N_SHAP_BG      = 50    # campioni di background per GradientExplainer

C_NR  = "#D55E00"
C_R   = "#0072B2"
C_GREY = "#999999"

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})


# ── MODEL (stesso ProtoDANN di 04_train_prototype_v2.py) ─────────────────────

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    def forward(self, z, alpha):
        return self.net(GradientReversalFn.apply(z, alpha)).squeeze(1)


class ProtoDANN(nn.Module):
    def __init__(self, input_dim, encoder_hidden, latent_dim, dropout, input_dropout):
        super().__init__()
        self.encoder       = Encoder(input_dim, encoder_hidden, latent_dim,
                                     dropout, input_dropout)
        self.discriminator = DomainDiscriminator(latent_dim)

    def forward(self, x, alpha=1.0):
        z = self.encoder(x)
        return z, self.discriminator(z, alpha)


# ── SCORE WRAPPER per SHAP ────────────────────────────────────────────────────

class ProtoScorer(nn.Module):
    """
    Wrapper che espone una sola output: il prototype score (cosine similarity).
    SHAP userà questo come target da spiegare.
    """
    def __init__(self, model, centroid):
        super().__init__()
        self.model    = model
        self.centroid = centroid   # (latent_dim,)

    def forward(self, x):
        z, _ = self.model(x)
        z_norm = F.normalize(z, dim=1)
        sim    = (z_norm @ self.centroid)          # (batch,)
        score  = (sim + 1.0) / 2.0                 # [-1,1] -> [0,1]
        return score.unsqueeze(1)                   # (batch, 1) per SHAP


# ── CARICAMENTO ───────────────────────────────────────────────────────────────

def load_everything():
    # Config dal results_prototype.json
    results_path = os.path.join(RESULTS_DIR, "results_prototype.json")
    with open(results_path) as f:
        res = json.load(f)
    cfg = res["config"]

    # Dati preprocessati
    mic_X = np.load(os.path.join(PROC_DIR, "microarray_expr.npy"), allow_pickle=False)
    mic_y = np.load(os.path.join(PROC_DIR, "microarray_labels.npy"), allow_pickle=False)
    rna_X = np.load(os.path.join(PROC_DIR, "rnaseq_expr.npy"), allow_pickle=False)

    # Split train/test microarray
    sample_info = pd.read_csv(os.path.join(PROC_DIR, "sample_info.csv"))
    meta        = pd.read_csv(os.path.join(RAW_DIR, "GSE14671", "metadata.csv"))
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

    # Gene names
    genes_path = os.path.join(PROC_DIR, "common_genes.txt")
    with open(genes_path) as f:
        gene_names = [l.strip() for l in f.readlines()]

    input_dim = mic_X.shape[1]
    assert len(gene_names) == input_dim, \
        f"Gene names ({len(gene_names)}) != input_dim ({input_dim})"

    # Modello
    model = ProtoDANN(
        input_dim      = input_dim,
        encoder_hidden = cfg["encoder_hidden"],
        latent_dim     = cfg["latent_dim"],
        dropout        = cfg["dropout"],
        input_dropout  = cfg["input_dropout"],
    ).to(DEVICE)

    model_path = os.path.join(RESULTS_DIR, "proto_dann_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    log.info(f"Modello caricato: {model_path}")

    # Centroide
    centroid_path = os.path.join(RESULTS_DIR, "proto_centroid.pt")
    centroid = torch.load(centroid_path, map_location=DEVICE)
    log.info(f"Centroide caricato: {centroid_path}")

    return (model, centroid, mic_X_test, mic_y_test,
            rna_X, mic_X, gene_names, cfg)


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(model, centroid, X_test, X_background, gene_names):
    """
    Calcola SHAP values con GradientExplainer.

    Background: campioni RNA-seq (distribuzione di riferimento).
    Explained:  campioni microarray test set (23 campioni).

    Output: shap_values (n_test x n_genes)
    """
    scorer = ProtoScorer(model, centroid).to(DEVICE)
    scorer.eval()

    # Background: subset RNA-seq
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_background), size=N_SHAP_BG, replace=False)
    bg     = torch.tensor(X_background[bg_idx], dtype=torch.float32).to(DEVICE)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    log.info(f"Calcolo SHAP con GradientExplainer...")
    log.info(f"  Background: {bg.shape}  (RNA-seq)")
    log.info(f"  Explained:  {X_test_t.shape}  (microarray test set)")

    explainer   = shap.GradientExplainer(scorer, bg)
    shap_values = explainer.shap_values(X_test_t)   # (n_test, n_genes, 1)

    # Rimuovi dimensione output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    log.info(f"SHAP values shape: {shap_values.shape}")
    return shap_values


# ── RANKING E SALVATAGGIO ─────────────────────────────────────────────────────

def save_results(shap_values, gene_names, y_test):
    # Importanza media assoluta
    mean_abs = np.abs(shap_values).mean(axis=0)   # (n_genes,)
    ranked   = np.argsort(mean_abs)[::-1]

    df = pd.DataFrame({
        "gene":          [gene_names[i] for i in ranked[:N_TOP_GENES]],
        "mean_abs_shap": mean_abs[ranked[:N_TOP_GENES]],
        "mean_shap_R":   shap_values[y_test==1][:, ranked[:N_TOP_GENES]].mean(axis=0),
        "mean_shap_NR":  shap_values[y_test==0][:, ranked[:N_TOP_GENES]].mean(axis=0),
    })

    csv_path = os.path.join(RESULTS_DIR, "shap_top_genes.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Salvato: {csv_path}")

    # Lista geni per GSEA (Enrichr, MSigDB)
    txt_path = os.path.join(RESULTS_DIR, "top_shap_genes.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(df["gene"].tolist()))
    log.info(f"Salvato: {txt_path}  ({len(df)} geni)")

    # Salva tutti i SHAP values
    npy_path = os.path.join(RESULTS_DIR, "shap_values.npy")
    np.save(npy_path, shap_values)
    log.info(f"Salvato: {npy_path}")

    return df, ranked


# ── PLOT ──────────────────────────────────────────────────────────────────────

def plot_beeswarm(shap_values, gene_names, y_test, ranked):
    """
    Beeswarm plot: top N geni per importanza.
    Ogni punto = un campione. Colore = score SHAP (rosso = spinge verso NR, blu = verso R).
    """
    top_idx   = ranked[:N_PLOT_GENES]
    top_names = [gene_names[i] for i in top_idx]
    top_shap  = shap_values[:, top_idx]   # (n_test, N_PLOT_GENES)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for g_i, (gene, idx) in enumerate(zip(top_names, top_idx)):
        vals = top_shap[:, g_i]
        # Jitter verticale
        y_pos = N_PLOT_GENES - g_i - 1 + np.random.RandomState(g_i).uniform(
            -0.3, 0.3, len(vals))
        # Colore: rosso = SHAP negativo (spinge verso NR), blu = positivo (verso R)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            norm = (vals - vmin) / (vmax - vmin)
        else:
            norm = np.full_like(vals, 0.5)
        colors = plt.cm.coolwarm(norm)
        ax.scatter(vals, y_pos, c=colors, s=25, alpha=0.8, lw=0)

    ax.set_yticks(range(N_PLOT_GENES))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.axvline(0, color=C_GREY, lw=1, ls="--")
    ax.set_xlabel("SHAP value\n(positivo = spinge verso responder, negativo = verso non-responder)")
    ax.set_title(f"SHAP Beeswarm — Top {N_PLOT_GENES} geni predittivi\n"
                 "ProtoDANN applicato al test set (n = 23)", pad=8)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                                norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("SHAP value (normalizzato)", fontsize=8)

    out = os.path.join(FIG_DIR, "fig_shap_beeswarm.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"Salvato: {out}")


def plot_barplot(df):
    """Barplot importanza media assoluta top N geni."""
    top = df.head(N_PLOT_GENES).iloc[::-1]   # ordine crescente per barh

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    bars = ax.barh(top["gene"], top["mean_abs_shap"],
                   color=C_R, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {N_PLOT_GENES} geni per importanza media\n"
                 "ProtoDANN — Test set (n = 23)", pad=8)
    ax.tick_params(axis="y", labelsize=8)

    out = os.path.join(FIG_DIR, "fig_shap_barplot.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"Salvato: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Caricamento dati e modello...")
    model, centroid, mic_X_test, mic_y_test, rna_X, mic_X, gene_names, cfg = \
        load_everything()

    log.info(f"Test set: {mic_X_test.shape}  labels: {np.bincount(mic_y_test)}")
    log.info(f"Geni comuni: {len(gene_names)}")

    shap_values = compute_shap(model, centroid, mic_X_test, rna_X, gene_names)

    log.info("\nRanking geni per importanza SHAP...")
    df, ranked = save_results(shap_values, gene_names, mic_y_test)

    log.info("\nTop 20 geni predittivi:")
    log.info(df.head(20).to_string(index=False))

    log.info("\nGenerazione plot...")
    plot_beeswarm(shap_values, gene_names, mic_y_test, ranked)
    plot_barplot(df)

    log.info(f"\nCompletato. Output in: {RESULTS_DIR}")
    log.info(f"  shap_values.npy       — tutti i SHAP values (23 x {len(gene_names)})")
    log.info(f"  shap_top_genes.csv    — top {N_TOP_GENES} geni con scores")
    log.info(f"  top_shap_genes.txt    — lista geni per GSEA (Enrichr/MSigDB)")
    log.info(f"  fig_shap_beeswarm.pdf — beeswarm plot")
    log.info(f"  fig_shap_barplot.pdf  — barplot importanza")
    log.info("\nPer GSEA: carica top_shap_genes.txt su https://myfgsea.org o https://enrichr.qiagen.com")


if __name__ == "__main__":
    main()