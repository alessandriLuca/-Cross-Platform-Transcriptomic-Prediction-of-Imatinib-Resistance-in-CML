# CML Imatinib Resistance — ProtoDANN Cross-Platform Pipeline

Cross-platform prediction of imatinib resistance in CML using a
**Domain-Adversarial Neural Network with Responder Prototype Scoring (ProtoDANN)**
that harmonises bulk RNA-seq and microarray data into a shared latent space.

Instead of classifying non-responders vs. responders directly — which is
unstable with only 13 NR in the training set — ProtoDANN learns a centroid
of responders in latent space and scores each sample by cosine similarity to
that centroid. Non-responders emerge as outliers without ever being used in training.

## Datasets

| Dataset   | Platform   | N  | Compartment        | Label                        |
|-----------|------------|----|--------------------|------------------------------|
| GSE130404 | RNA-seq    | 96 | Whole blood        | EMR at 3 months (BCR-ABL1 >10%) |
| GSE14671  | Microarray | 59 | CD34+ sorted cells | Cytogenetic response (proxy) |

**Note:** GSE130404 is used for training; GSE14671 provides the independent
test set (23 samples, never seen during development). There is a known
endpoint mismatch (EMR vs. cytogenetic response) and a compartment difference
(whole blood vs. CD34+) — both declared as limitations.

## Results

| Model                  | AUC   | 95% CI        | p-value (permutation) |
|------------------------|-------|---------------|----------------------|
| ProtoDANN              | 0.863 | [0.603, 1.000]| 0.0038 (n=10,000)    |
| LASSO (no alignment)   | 0.569 | [0.191, 0.929]| —                    |
| Random Forest (no alignment) | 0.637 | [0.283, 0.944] | —             |

Domain alignment is the key performance driver: baselines without it collapse
on cross-platform transfer.

## Architecture

```
Input (16,622 common genes)
        |
  [input_dropout = 0.5]
        |
    [Encoder]              <- shared, platform-agnostic
    256 -> 64 dims
        |
  [Latent space z]
        |
  [L2 normalize z]
        |
  cosine_similarity(z, responder_centroid) -> prototype score [0, 1]
        |
  [Domain Discriminator + GRL]   <- alignment loss only, no classifier
    RNA-seq vs. microarray
```

The Gradient Reversal Layer (GRL) forces the latent space to be
platform-agnostic. The prototype score replaces the binary classifier:
high score = similar to responders, low score = candidate non-responder.

## Training pipeline

| Script                  | Description                                               |
|-------------------------|-----------------------------------------------------------|
| `01_download_data.py`   | Download GSE130404 and GSE14671 from GEO                  |
| `02_map_probes_to_genes.py` | Map Affymetrix probes to gene symbols, find common genes |
| `03_preprocess.py`      | Normalisation, z-scoring, train/test split                |
| `04_train.py`           | ProtoDANN training: autoencoder pretrain + DANN + self-training |
| `05_plotPrototype.py`   | Generate figures from results_prototype.json              |
| `06_shapPrototype.py`   | SHAP GradientExplainer interpretability analysis          |
| `7_.py`                 | Baseline comparison: LASSO + Random Forest                |
| `8_.py`                 | Permutation test (10,000 permutations)                    |
| `9_.py`                 | Endpoint concordance check (GSE14671 metadata inspection) |

## Quick start

```bash
export RESULTS_DIR=/path/to/results/v1_prototype
export PROC_DIR=/path/to/processed
export RAW_DIR=/path/to/raw

# Full pipeline
python3 scripts/01_download_data.py
python3 scripts/02_map_probes_to_genes.py
python3 scripts/03_preprocess.py

# Training (best run with nohup)
nohup python3 scripts/04_train.py > $RESULTS_DIR/training.log 2>&1 &

# Analysis (after training completes)
python3 scripts/05_plotPrototype.py
pip install shap --break-system-packages
python3 scripts/06_shapPrototype.py

# Validation
python3 scripts/7_.py   # baselines
python3 scripts/8_.py   # permutation test
python3 scripts/9_.py   # endpoint concordance
```

## Output files

```
results/v1_prototype/
├── proto_dann_best.pt          # full model weights (best round)
├── proto_encoder_final.pt      # encoder only (for deployment)
├── proto_centroid.pt           # responder centroid vector
├── results_prototype.json      # AUC, CI, round history, scores
├── baseline_results.json       # LASSO + RF comparison
├── permutation_test.json       # p-value, null distribution
├── shap_values.npy             # raw SHAP values (23 x 16622)
├── shap_top_genes.csv          # top 200 genes by mean |SHAP|
├── top_shap_genes.txt          # gene list for Enrichr/GSEA
└── figures/
    ├── fig1_roc_curve.pdf
    ├── fig2_score_distribution.pdf
    ├── fig3_selftraining.pdf
    ├── fig4_confusion.pdf
    ├── fig_shap_beeswarm.pdf
    ├── fig_shap_barplot.pdf
    └── fig_permutation.pdf
```

## Deployment

```python
import torch
import torch.nn.functional as F
from src.model import Encoder

# Load frozen model
encoder = Encoder(input_dim=16622, hidden_dims=[256], latent_dim=64,
                  dropout=0.4, input_dropout=0.5)
encoder.load_state_dict(torch.load("results/v1_prototype/proto_encoder_final.pt"))
encoder.eval()

centroid = torch.load("results/v1_prototype/proto_centroid.pt")

# Score a new patient sample (z-scored on common genes)
x = torch.tensor(patient_expr, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    z = encoder(x)
    z_norm = F.normalize(z, dim=1)
    score = ((z_norm @ centroid).item() + 1.0) / 2.0  # [0, 1]

# score close to 1.0 -> likely responder
# score well below 1.0 -> candidate non-responder
```

## File structure

```
cml_dann/
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_map_probes_to_genes.py
│   ├── 03_preprocess.py
│   ├── 04_train.py              # ProtoDANN (main)
│   ├── 05_plotPrototype.py
│   ├── 06_shapPrototype.py
│   ├── 7_.py                    # baselines
│   ├── 8_.py                    # permutation test
│   └── 9_.py                    # endpoint concordance
├── src/
│   └── model.py                 # Encoder, DANN architecture
└── README.md
```

## Known limitations

- Endpoint mismatch: training uses EMR (BCR-ABL1/ABL1 < 10% at 3 months);
  test set uses cytogenetic response as proxy. Concordance not quantifiable
  from public data.
- Compartment mismatch: GSE130404 uses whole peripheral blood; GSE14671
  uses CD34+ sorted cells.
- Test set N = 23 (6 NR): CI is wide [0.603, 1.000]. Permutation test
  p = 0.0038 confirms result is above chance.
- Score compression: all scores in [0.963, 1.000]. Model ranks correctly
  but absolute separation is small — use as a continuous risk score, not
  a binary classifier.
- For external validation with EMR labels: contact Kok/Hughes group
  (IMVS Adelaide) for the TIDEL-II held-out cohort (88 patients).
