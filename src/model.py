"""
src/model.py

Domain-Adversarial Neural Network (DANN) for cross-platform CML
imatinib resistance prediction.

Architecture:
  Input (n_genes)
      ↓
  [input_dropout]  — butta 90% delle feature ad ogni forward
      ↓
  [Encoder]  — shared, platform-agnostic feature extractor
      ↓
  [Latent space z]
     ↙          ↘
[Classifier]  [Domain discriminator]
  NR vs R       RNA-seq vs microarray

Training signal:
  L_total = L_classifier - λ * L_domain
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ------------------------------------------------------------------ #
# Gradient Reversal Layer
# ------------------------------------------------------------------ #

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha


# ------------------------------------------------------------------ #
# Network blocks
# ------------------------------------------------------------------ #

def mlp_block(in_dim: int, out_dim: int, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Encoder(nn.Module):
    """
    Shared encoder: maps high-dim gene expression → latent z.
    input_dropout: aggressivo (es. 0.9) per regolarizzare su dati ad alta dimensionalità.
    """
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 dropout: float = 0.3, input_dropout: float = 0.0):
        super().__init__()
        self.input_drop = nn.Dropout(p=input_dropout)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(mlp_block(prev, h, dropout))
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.input_drop(x))


class Classifier(nn.Module):
    """
    Classificatore a 2 classi: NR (0) vs R (1).
    Output: logits shape (batch, 2) — usa CrossEntropyLoss.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            mlp_block(latent_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, 2),   # 2 classi: NR=0, R=1
        )

    def forward(self, z):
        return self.net(z)   # logits (batch, 2)


class DomainDiscriminator(nn.Module):
    """Binary discriminator: RNA-seq (0) vs microarray (1)."""
    def __init__(self, latent_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.grl = GradientReversal(alpha=1.0)
        self.net = nn.Sequential(
            mlp_block(latent_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(self.grl(z)).squeeze(1)   # logits (batch,)


# ------------------------------------------------------------------ #
# Full DANN
# ------------------------------------------------------------------ #

class DANN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden: list = None,
        latent_dim: int = 64,
        clf_hidden: int = 64,
        disc_hidden: int = 64,
        dropout: float = 0.3,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        if encoder_hidden is None:
            encoder_hidden = [256]

        self.encoder       = Encoder(input_dim, encoder_hidden, latent_dim, dropout, input_dropout)
        self.classifier    = Classifier(latent_dim, clf_hidden, dropout)
        self.discriminator = DomainDiscriminator(latent_dim, disc_hidden, dropout)

    def forward(self, x, alpha: float = None):
        if alpha is not None:
            self.discriminator.grl.set_alpha(alpha)
        z         = self.encoder(x)
        clf_logit = self.classifier(z)   # (batch, 2)
        dom_logit = self.discriminator(z)
        return clf_logit, dom_logit, z

    def encode(self, x) -> torch.Tensor:
        """Inference-only: returns latent embedding."""
        with torch.no_grad():
            return self.encoder(x)

    def predict_proba(self, x) -> torch.Tensor:
        """Inference-only: returns softmax probabilities (batch, 2)."""
        with torch.no_grad():
            z = self.encoder(x)
            return torch.softmax(self.classifier(z), dim=1)

    def predict(self, x) -> torch.Tensor:
        """Inference-only: returns class predictions (batch,)."""
        return self.predict_proba(x).argmax(dim=1)
