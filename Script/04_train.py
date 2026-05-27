#!/usr/bin/env python3
"""
04_train_prototype.py — DANN + Responder Prototype Score

Fix applicati rispetto alla versione originale:
  1. Bug Youden: youden_threshold(y_calib, scores_calib) — non scores_calib, scores_calib
  2. Filtro degenerazione: round con score_range < 0.01 vengono ignorati nella
     selezione del best round. Previene che un round collassato (tutti gli score = 1.0)
     venga scelto come best per AUC alta ma fittizia.
"""

import os, sys, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import pandas as pd
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import Encoder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROC_DIR    = os.environ.get("PROC_DIR",    "/home/jovyan/work/shared/processed")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/home/jovyan/work/shared/results/v1_prototype")
RAW_DIR     = os.environ.get("RAW_DIR",     "/home/jovyan/work/shared/raw")
os.makedirs(RESULTS_DIR, exist_ok=True)

CFG = {
    "input_dropout":         0.5,
    "latent_dim":            64,
    "encoder_hidden":        [256],
    "dropout":               0.4,
    "pretrain_epochs":       500,
    "pretrain_lr":           1e-3,
    "pretrain_wd":           1e-5,
    "dann_epochs":           1000,
    "dann_lr":               5e-5,
    "dann_wd":               1e-4,
    "lambda_max":            1.0,
    "patience":              150,
    "alpha_proto":           0.5,
    "self_train_rounds":     20,
    "self_train_epochs":     500,
    "self_train_top_k_frac": 0.4,
    "score_range_min":       0.01,   # soglia anti-degenerazione
    "n_aug_copies":          5,
    "aug_noise_std":         0.05,
    "batch_size":            16,
    "seed":                  42,
    "n_bootstrap":           2000,
    "ci_alpha":              0.05,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])


# ── DATA ──────────────────────────────────────────────────────────────────────

def load_data():
    rna_X = np.load(os.path.join(PROC_DIR, "rnaseq_expr.npy"),       allow_pickle=False)
    rna_y = np.load(os.path.join(PROC_DIR, "rnaseq_labels.npy"),     allow_pickle=False)
    mic_X = np.load(os.path.join(PROC_DIR, "microarray_expr.npy"),   allow_pickle=False)
    mic_y = np.load(os.path.join(PROC_DIR, "microarray_labels.npy"), allow_pickle=False)
    log.info(f"RNA-seq:    {rna_X.shape}  labels: {np.bincount(rna_y)}")
    log.info(f"Microarray: {mic_X.shape}  labels: {np.bincount(mic_y)}")
    assert rna_X.shape[1] == mic_X.shape[1]

    meta        = pd.read_csv(os.path.join(RAW_DIR, "GSE14671", "metadata.csv"))
    sample_info = pd.read_csv(os.path.join(PROC_DIR, "sample_info.csv"))
    mic_samples = sample_info[sample_info["dataset"] == "GSE14671"]["sample_id"].values

    is_train = []
    for sid in mic_samples:
        row   = meta[meta["sample_id"] == sid]
        title = str(row["title"].values[0]).lower() if len(row) > 0 else ""
        is_train.append("training" in title)
    is_train      = np.array(is_train)
    mic_train_idx = np.where(is_train)[0]
    mic_test_idx  = np.where(~is_train)[0]

    log.info(f"Microarray Training Set: {len(mic_train_idx)} campioni  "
             f"labels: {np.bincount(mic_y[mic_train_idx])}")
    log.info(f"Microarray Test Set (mai toccato): "
             f"{len(mic_test_idx)} campioni  labels: {np.bincount(mic_y[mic_test_idx])}")

    return rna_X, rna_y, mic_X, mic_y, mic_train_idx, mic_test_idx


def augment_responders(X, y, n_copies=5, noise_std=0.05, seed=42):
    np.random.seed(seed)
    r_idx = np.where(y == 1)[0]
    aug_X, aug_y = [X], [y]
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_std, X[r_idx].shape)
        aug_X.append(X[r_idx] + noise)
        aug_y.append(y[r_idx])
    X_aug = np.vstack(aug_X)
    y_aug = np.concatenate(aug_y)
    log.info(f"Augmentation: {(y==1).sum()} R -> {(y_aug==1).sum()} R  "
             f"(NR presenti ma non usati per prototype: {(y==0).sum()})")
    return X_aug, y_aug


# ── PRETRAIN ──────────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, encoder, input_dim, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),        nn.ReLU(),
            nn.Linear(128, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


def pretrain_autoencoder(X, encoder, tag, epochs):
    log.info(f"\n{'='*50}\nAutoencoder pretraining — {tag}\n{'='*50}")
    ae  = Autoencoder(encoder, X.shape[1], CFG["latent_dim"]).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(),
                           lr=CFG["pretrain_lr"], weight_decay=CFG["pretrain_wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    fn  = nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32)
    dl  = DataLoader(TensorDataset(X_t), batch_size=CFG["batch_size"],
                     shuffle=True, drop_last=True)
    for epoch in range(1, epochs + 1):
        ae.train()
        total = 0.0
        for (b,) in dl:
            b = b.to(DEVICE)
            l = fn(ae(b), b)
            opt.zero_grad(); l.backward(); opt.step()
            total += l.item() * len(b)
        sch.step()
        if epoch % 100 == 0 or epoch == 1:
            log.info(f"  [{tag}] Epoch {epoch:4d}/{epochs}  "
                     f"MSE: {total/len(X_t):.4f}  lr: {sch.get_last_lr()[0]:.2e}")
    torch.save(ae.encoder.state_dict(),
               os.path.join(RESULTS_DIR, f"proto_encoder_pretrained_{tag}.pt"))
    return ae.encoder


# ── MODEL ─────────────────────────────────────────────────────────────────────

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
        z_rev = GradientReversalFn.apply(z, alpha)
        return self.net(z_rev).squeeze(1)


class ProtoDANN(nn.Module):
    def __init__(self, input_dim, encoder_hidden, latent_dim, dropout, input_dropout):
        super().__init__()
        self.encoder       = Encoder(input_dim, encoder_hidden, latent_dim,
                                     dropout, input_dropout)
        self.discriminator = DomainDiscriminator(latent_dim)

    def forward(self, x, alpha=1.0):
        z = self.encoder(x)
        return z, self.discriminator(z, alpha)


# ── PROTOTYPE ─────────────────────────────────────────────────────────────────

def compute_prototype(model, X_responders):
    model.eval()
    Xt = torch.tensor(X_responders, dtype=torch.float32)
    zs = []
    with torch.no_grad():
        for i in range(0, len(Xt), CFG["batch_size"]):
            b = Xt[i:i+CFG["batch_size"]].to(DEVICE)
            z, _ = model(b)
            zs.append(F.normalize(z, dim=1).cpu())
    centroid = torch.cat(zs, dim=0).mean(dim=0)
    return F.normalize(centroid.unsqueeze(0), dim=1).squeeze(0)


def prototype_score(model, X, centroid):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    scores = []
    with torch.no_grad():
        for i in range(0, len(Xt), CFG["batch_size"]):
            b = Xt[i:i+CFG["batch_size"]].to(DEVICE)
            z, _ = model(b)
            z_norm = F.normalize(z, dim=1)
            sim = (z_norm @ centroid.to(DEVICE)).cpu().numpy()
            scores.extend(sim)
    return (np.array(scores) + 1.0) / 2.0   # [-1,1] -> [0,1]


# ── TRAINING ──────────────────────────────────────────────────────────────────

def compute_alpha(epoch, total_epochs):
    p = epoch / total_epochs
    return CFG["lambda_max"] * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


def proto_loss(z_batch, y_batch, centroid):
    r_mask = (y_batch == 1)
    if r_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True).to(z_batch.device)
    z_r  = F.normalize(z_batch[r_mask], dim=1)
    sims = (z_r @ centroid.to(z_batch.device))
    return 1.0 - sims.mean()


def run_proto_dann(model, rna_X_r, rna_y_r,
                   mic_X_all, mic_X_val, mic_y_val,
                   centroid, epochs, tag="ProtoDANN"):
    log.info(f"\n  --- {tag} ---")
    log.info(f"  RNA train: {rna_X_r.shape}  R: {(rna_y_r==1).sum()}  NR: {(rna_y_r==0).sum()}")

    rna_Xt     = torch.tensor(rna_X_r,  dtype=torch.float32)
    rna_yt     = torch.tensor(rna_y_r,  dtype=torch.long)
    mic_Xt_all = torch.tensor(mic_X_all, dtype=torch.float32)
    mic_dt_all = torch.ones(len(mic_Xt_all))
    mic_Xt_val = torch.tensor(mic_X_val, dtype=torch.float32)
    mic_yt_val = torch.tensor(mic_y_val, dtype=torch.long)

    rna_loader     = DataLoader(TensorDataset(rna_Xt, rna_yt),
                                batch_size=CFG["batch_size"], shuffle=True, drop_last=True)
    mic_dom_loader = DataLoader(TensorDataset(mic_Xt_all, mic_dt_all),
                                batch_size=CFG["batch_size"], drop_last=True)
    mic_val_loader = DataLoader(TensorDataset(mic_Xt_val, mic_yt_val),
                                batch_size=CFG["batch_size"])

    opt = torch.optim.Adam(model.parameters(),
                           lr=CFG["dann_lr"], weight_decay=CFG["dann_wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    dom_loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc  = -1.0
    best_epoch    = 0
    patience_cnt  = 0
    best_state    = None
    best_centroid = centroid.clone()
    mic_iter      = iter(mic_dom_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        alpha = compute_alpha(epoch, epochs)
        epoch_dom = epoch_proto = 0.0

        for rna_bX, rna_by in rna_loader:
            rna_bX = rna_bX.to(DEVICE); rna_by = rna_by.to(DEVICE)
            z_rna, dom_rna = model(rna_bX, alpha)
            l_dom_rna = dom_loss_fn(dom_rna, torch.zeros(len(rna_bX)).to(DEVICE))
            l_proto   = proto_loss(z_rna, rna_by, centroid)

            try:
                mic_bX, _ = next(mic_iter)
            except StopIteration:
                mic_iter   = iter(mic_dom_loader)
                mic_bX, _ = next(mic_iter)

            mic_bX = mic_bX.to(DEVICE)
            _, dom_mic = model(mic_bX, alpha)
            l_dom_mic = dom_loss_fn(dom_mic, torch.ones(len(mic_bX)).to(DEVICE))
            l_total   = (l_dom_rna + l_dom_mic) * 0.5 + CFG["alpha_proto"] * l_proto
            opt.zero_grad(); l_total.backward(); opt.step()
            epoch_dom   += (l_dom_rna + l_dom_mic).item() * 0.5
            epoch_proto += l_proto.item()

        sch.step()

        rna_r_X      = rna_X_r[rna_y_r == 1]
        new_centroid = compute_prototype(model, rna_r_X)
        model.eval()
        val_scores   = prototype_score(model, mic_X_val, new_centroid)
        try:
            val_auc = roc_auc_score(mic_y_val, val_scores)
        except ValueError:
            val_auc = 0.0

        if val_auc > best_val_auc:
            best_val_auc  = val_auc
            best_epoch    = epoch
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_centroid = new_centroid.clone()
        else:
            patience_cnt += 1

        if epoch % 100 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:4d}/{epochs}  alpha={alpha:.2f}  "
                     f"dom={epoch_dom:.4f}  proto={epoch_proto:.4f}  "
                     f"val_AUC={val_auc:.4f}  best={best_val_auc:.4f}@{best_epoch}")

        if patience_cnt >= CFG["patience"]:
            log.info(f"  Early stopping a epoch {epoch}")
            break

    log.info(f"  Best epoch: {best_epoch}  val_AUC: {best_val_auc:.4f}")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, best_centroid, best_val_auc


# ── SELF-TRAINING ─────────────────────────────────────────────────────────────

def get_mic_candidates(model, mic_X, centroid, top_k_frac):
    scores  = prototype_score(model, mic_X, centroid)
    n_sel   = max(1, int(len(scores) * top_k_frac))
    top_idx = np.argsort(scores)[-n_sel:]
    log.info(f"  Candidati mic: {n_sel}  "
             f"score range: [{scores[top_idx].min():.3f}, {scores[top_idx].max():.3f}]")
    return mic_X[top_idx], scores


# ── VALUTAZIONE ───────────────────────────────────────────────────────────────

def bootstrap_auc_ci(y_true, y_score, n_bootstrap=2000, alpha=0.05, seed=42):
    rng       = np.random.RandomState(seed)
    auc_point = roc_auc_score(y_true, y_score)
    boot      = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        ys, yp = y_true[idx], y_score[idx]
        if len(np.unique(ys)) < 2:
            continue
        boot.append(roc_auc_score(ys, yp))
    boot = np.array(boot)
    return auc_point, float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


def youden_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = np.argmax(tpr + (1 - fpr) - 1)
    return float(thresholds[j]), float(tpr[j]), float(1 - fpr[j])


def evaluate_train_set(model, X, y, centroid, tag):
    scores = prototype_score(model, X, centroid)
    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        auc = 0.0
    score_range = float(scores.max() - scores.min())
    log.info(f"  [{tag}] AUC: {auc:.4f}  "
             f"score_R: {scores[y==1].mean():.3f}  score_NR: {scores[y==0].mean():.3f}  "
             f"score_range: {score_range:.4f}")
    return auc, scores, score_range


def evaluate_final(model, X_test, y_test, centroid, X_calib, y_calib):
    log.info(f"\n{'='*60}")
    log.info("VALUTAZIONE FINALE — test set (23 campioni, UNA SOLA VOLTA)")
    log.info(f"{'='*60}")

    scores_test  = prototype_score(model, X_test,  centroid)
    scores_calib = prototype_score(model, X_calib, centroid)

    auc, ci_low, ci_high = bootstrap_auc_ci(y_test, scores_test,
                                             CFG["n_bootstrap"], CFG["ci_alpha"])
    log.info(f"\n  AUC point estimate : {auc:.4f}")
    log.info(f"  95% CI bootstrap   : [{ci_low:.4f}, {ci_high:.4f}]")
    log.info(f"  N test             : {len(y_test)}  ({(y_test==0).sum()} NR, {(y_test==1).sum()} R)")
    log.info(f"\n  Score medio R  (test): {scores_test[y_test==1].mean():.4f} +/- {scores_test[y_test==1].std():.4f}")
    log.info(f"  Score medio NR (test): {scores_test[y_test==0].mean():.4f} +/- {scores_test[y_test==0].std():.4f}")

    preds_05 = (scores_test >= 0.5).astype(int)
    log.info(f"\n  --- Soglia fissa 0.5 ---")
    log.info("\n" + classification_report(y_test, preds_05,
             target_names=["NR", "R"], zero_division=0))

    # FIX: y_calib (label reali) e scores_calib (score del modello)
    thr_y, sens_y, spec_y = youden_threshold(y_calib, scores_calib)
    preds_y = (scores_test >= thr_y).astype(int)
    log.info(f"  --- Soglia Youden (calibrata su mic_train_set: {len(y_calib)} campioni) ---")
    log.info(f"  Threshold: {thr_y:.4f}  sens_calib: {sens_y:.3f}  spec_calib: {spec_y:.3f}")
    log.info("\n" + classification_report(y_test, preds_y,
             target_names=["NR", "R"], zero_division=0))

    nr_det = int((preds_y[y_test==0] == 0).sum())
    log.info(f"  NR identificati con soglia Youden: {nr_det}/{(y_test==0).sum()}")

    return {
        "auc":           float(auc),
        "ci_low":        float(ci_low),
        "ci_high":       float(ci_high),
        "ci_level":      "95%",
        "n_bootstrap":   CFG["n_bootstrap"],
        "score_R_mean":  float(scores_test[y_test==1].mean()),
        "score_NR_mean": float(scores_test[y_test==0].mean()),
        "scores_test":   [float(x) for x in scores_test],
        "labels_test":   [int(x)   for x in y_test],
        "threshold_05":      {"threshold": 0.5, "preds": [int(x) for x in preds_05]},
        "threshold_youden":  {"threshold": float(thr_y), "preds": [int(x) for x in preds_y]},
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    rna_X, rna_y, mic_X, mic_y, mic_train_idx, mic_test_idx = load_data()
    input_dim = rna_X.shape[1]

    mic_X_train = mic_X[mic_train_idx]; mic_y_train = mic_y[mic_train_idx]
    mic_X_test  = mic_X[mic_test_idx];  mic_y_test  = mic_y[mic_test_idx]

    rna_X_aug, rna_y_aug = augment_responders(
        rna_X, rna_y, CFG["n_aug_copies"], CFG["aug_noise_std"], CFG["seed"])

    # Pretrain
    enc_micro = Encoder(input_dim, CFG["encoder_hidden"], CFG["latent_dim"],
                        CFG["dropout"], CFG["input_dropout"])
    pretrain_autoencoder(mic_X, enc_micro, "microarray", CFG["pretrain_epochs"])

    enc_rna = Encoder(input_dim, CFG["encoder_hidden"], CFG["latent_dim"],
                      CFG["dropout"], CFG["input_dropout"])
    pretrain_autoencoder(rna_X, enc_rna, "rnaseq", CFG["pretrain_epochs"])

    model = ProtoDANN(input_dim, CFG["encoder_hidden"], CFG["latent_dim"],
                      CFG["dropout"], CFG["input_dropout"]).to(DEVICE)

    rna_pt = os.path.join(RESULTS_DIR, "proto_encoder_pretrained_rnaseq.pt")
    if os.path.exists(rna_pt):
        model.encoder.load_state_dict(torch.load(rna_pt, map_location=DEVICE))
        log.info("  Caricati pesi encoder pretrained RNA-seq")

    rna_r_X  = rna_X_aug[rna_y_aug == 1]
    centroid = compute_prototype(model, rna_r_X)
    log.info(f"  Centroide iniziale calcolato su {len(rna_r_X)} responder RNA-seq")

    log.info(f"\n{'='*50}\nStage 3: ProtoDANN iniziale\n{'='*50}")
    model, centroid, _ = run_proto_dann(
        model, rna_X_aug, rna_y_aug,
        mic_X, mic_X_train, mic_y_train,
        centroid, CFG["dann_epochs"], tag="ProtoDANN iniziale"
    )

    best_auc, _, best_range = evaluate_train_set(
        model, mic_X_train, mic_y_train, centroid, "mic_train [round 0]")
    best_state    = copy.deepcopy(model.state_dict())
    best_centroid = centroid.clone()
    best_round    = 0
    round_history = [{"round": 0, "train_auc": float(best_auc),
                      "score_range": float(best_range), "is_best": True}]

    log.info(f"\n{'='*50}\nStage 4: Self-training ({CFG['self_train_rounds']} rounds)\n{'='*50}")
    log.info(f"  FIX: round con score_range < {CFG['score_range_min']} vengono ignorati (modello collassato)")
    log.info("  Test set (23 campioni): NON toccato")

    for round_i in range(1, CFG["self_train_rounds"] + 1):
        log.info(f"\n  === Round {round_i}/{CFG['self_train_rounds']} ===")

        mic_candidates, _ = get_mic_candidates(
            model, mic_X_train, centroid, CFG["self_train_top_k_frac"])

        X_combined = np.vstack([rna_X_aug, mic_candidates])
        y_combined = np.concatenate([
            rna_y_aug,
            np.ones(len(mic_candidates), dtype=np.int64)
        ])
        log.info(f"  Combined: {X_combined.shape}  R: {(y_combined==1).sum()}")

        model, centroid, _ = run_proto_dann(
            model, X_combined, y_combined,
            mic_X, mic_X_train, mic_y_train,
            centroid, CFG["self_train_epochs"], tag=f"ST round {round_i}"
        )

        train_auc, _, score_range = evaluate_train_set(
            model, mic_X_train, mic_y_train, centroid,
            f"mic_train [round {round_i}]"
        )

        # FIX: ignora round degeneri (score_range < soglia)
        is_degenerate = score_range < CFG["score_range_min"]
        is_best       = (train_auc > best_auc) and not is_degenerate

        round_history.append({
            "round": round_i, "train_auc": float(train_auc),
            "score_range": float(score_range),
            "is_degenerate": is_degenerate, "is_best": is_best
        })

        if is_degenerate:
            log.info(f"  Round {round_i}: DEGENERE (score_range={score_range:.4f} < {CFG['score_range_min']}) -- ignorato")
        elif is_best:
            best_auc      = train_auc
            best_round    = round_i
            best_state    = copy.deepcopy(model.state_dict())
            best_centroid = centroid.clone()
            log.info(f"  Nuovo best round: {round_i}  AUC={best_auc:.4f}  score_range={score_range:.4f}")
        else:
            log.info(f"  Round {round_i}: AUC={train_auc:.4f}  score_range={score_range:.4f}  (best: {best_auc:.4f})")

    model.load_state_dict(best_state)
    centroid = best_centroid
    log.info(f"\nBest round: {best_round}  train_AUC: {best_auc:.4f}")

    torch.save(model.state_dict(),        os.path.join(RESULTS_DIR, "proto_dann_best.pt"))
    torch.save(model.encoder.state_dict(), os.path.join(RESULTS_DIR, "proto_encoder_final.pt"))
    torch.save(centroid,                  os.path.join(RESULTS_DIR, "proto_centroid.pt"))

    test_results = evaluate_final(
        model, mic_X_test, mic_y_test,
        centroid, mic_X_train, mic_y_train
    )

    results = {
        "approach":       "DANN + Responder Prototype (cosine similarity score)",
        "best_round":     best_round,
        "best_train_auc": float(best_auc),
        "round_history":  round_history,
        **test_results,
        "config":   CFG,
        "n_genes":  input_dim,
        "test_set": "GSE14671 Validation Set (23 campioni, mai visti durante training)",
    }
    with open(os.path.join(RESULTS_DIR, "results_prototype.json"), "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info("RISULTATO FINALE — ProtoDANN")
    log.info(f"  AUC:  {test_results['auc']:.4f}  95% CI [{test_results['ci_low']:.4f}, {test_results['ci_high']:.4f}]")
    log.info(f"  Score medio R:  {test_results['score_R_mean']:.4f}")
    log.info(f"  Score medio NR: {test_results['score_NR_mean']:.4f}")
    log.info(f"  Separazione R-NR: {test_results['score_R_mean'] - test_results['score_NR_mean']:.4f}")
    log.info(f"{'='*60}")
    log.info("Pipeline completa.")


if __name__ == "__main__":
    main()